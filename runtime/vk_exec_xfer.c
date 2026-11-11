// Transfers of the Vulkan op-stream replay: buffer and texture copies, texture
// clears, event signals, counter reports and conditional rendering. vk_exec.c calls vk_exec_op_xfer for these ops.
//
// Buffers are guest memory: vk_mem_resolve maps an address to the VkBuffer behind its pool, textures are the vk_texture.c images
// (linear texel layout, GENERAL layout; block-linear storage is only converted when a mirrored texture is uploaded). So a
// copy is one Vulkan transfer command and the pitch of a buffer side comes from the payload (OP_COPY_STRIDE or the
// tight row). Every copy ends the open pass and starts with a full barrier. Destinations: an image is marked
// GPU-written at record time; a CPU-visible buffer range is reported to vk_gpu_written once the GPU is done (write
// watch does not see GPU writes, textures stored there re-upload).
//
// Events and counters are CPU-visible side effects. Their values go into guest memory on the completion thread through
// vk_completion_add, in submit order, as gfx_null.c does: event WRITE / ADD, counter {0, timestamp} for every type but
// samples passed. A timestamp report puts 0 in the counter word: NX2NAVI gates its clears on two such words being unequal.
//
// Samples passed (NVN_COUNTER_TYPE_SAMPLES_PASSED) is a running count on the queue: RESET_COUNTER zeroes it, every
// draw after it adds the samples that passed the depth and stencil tests, REPORT_COUNTER writes the running value.
// NX2NAVI resets and reports it 3 to 8 times a frame in the field and its CPU reads the results back (f_70010c3298); a 0
// there culls what the query tested. The replay counts with Vulkan occlusion queries ("occlusion counts" below).
//
// Conditional rendering is a CPU-side skip, not VK_EXT_conditional_rendering: the extension compares one 32-bit word
// with zero, NVN compares the counter words at addr and addr + 16. The condition is evaluated when the op is recorded
// from guest memory; a counter reported by an earlier submission that is not done yet counts as 0 if it is a timestamp,
// as unknown if it is samples passed. An unknown count renders: the report usually sits in the command buffer being
// recorded, so its value cannot be waited for, and the word in memory still holds an older query's count. Rendering
// costs at most the work the condition would have saved; in the field
// NX2NAVI conditions only on timestamp reports. The condition sets VkExec.render_disabled, which vk_exec.c honours for
// clears and vk_exec_draw.c for draws and dispatches. A condition that cannot be read (address outside every pool)
// renders.
// Copies and CLEAR_TEX are not conditional.
//
// Contains code Copyright (c) 2026 jhaizhou, Copyright (c) 2026 Peiikai (MIT; see THIRD_PARTY_NOTICES.md).
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "platform/platform.h"
#include "host.h"
#include "nvn_api.h"
#include "nvn_tex.h"
#include "tex.h"
#include "vk.h"
#include "vk_exec.h"

enum { W_FORMAT, W_REGION, W_BUFFER, W_OVERLAP, W_COPY, W_CLEAR, W_QUERY, W_COUNT };

static void warn_once(int key, const char *fmt, ...) {
    static volatile int32_t seen[W_COUNT];
    if (__atomic_exchange_n(&seen[key], 1, __ATOMIC_RELAXED)) return;
    char msg[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    rt_log("vk: %s (logged once)", msg);
}

static uint32_t max1(uint32_t v) { return v ? v : 1; }

// ---- completion callbacks (completion thread)

typedef struct { uint64_t addr, size; int back; } Range;   // back: written into a VRAM mirror (vk_mem_mirrored)

static void guest_written_cb(void *arg) {
    Range *r = arg;
    if (r->back) vk_gpu_written_back(r->addr, r->size);
    else vk_gpu_written(r->addr, r->size);
    free(r);
}

// The GPU wrote [addr, addr+size) of CPU-visible memory: tell the texture cache once the fence is reached.
static void note_gpu_write(VkExec *ex, uint64_t addr, uint64_t size) {
    vk_usnap_drop(addr, size);
    if (nvn_gpu_only(addr)) return;
    Range *r = malloc(sizeof *r);
    if (!r) rt_fatal("vk: out of memory");
    *r = (Range){addr, size, vk_mem_mirrored(addr, size)};
    vk_mem_gpu_pending(addr, size);
    vk_completion_add(ex->fence, guest_written_cb, r);
}

typedef struct { volatile uint32_t *val; uint32_t mode, value; } EventWrite;

static void event_cb(void *arg) {
    EventWrite *e = arg;
    if (e->mode == NVN_EVENT_SIGNAL_MODE_ADD) __atomic_fetch_add(e->val, e->value, __ATOMIC_RELEASE);
    else __atomic_store_n(e->val, e->value, __ATOMIC_RELEASE);
    free(e);
}

// NVNcounterData = {counter, timestamp}; timestamps are Tegra 19.2 MHz ticks (the base of GetSystemTick).
static void write_counter(uint64_t addr, uint64_t counter) {
    uint64_t rec[2] = {counter, (uint64_t)((unsigned __int128)kern_now_ns() * 12 / 625)};
    memcpy((void *)(uintptr_t)addr, rec, sizeof rec);
}

// Reports other than samples passed (timestamps; the statistics counters are not modelled) write a counter of 0. The
// counter word matters: Ocarina writes {1, 0} into a record from the CPU, reports TIMESTAMP_TOP into it and gates
// clears on it differing from a zero record (mode 1), clears that must not run.
static void pending_written(uint64_t addr, uint64_t seq);   // below, with the pending table

typedef struct { uint64_t addr, seq; } CounterWrite;

static void counter_cb(void *arg) {
    CounterWrite *w = arg;
    write_counter(w->addr, 0);
    pending_written(w->addr, w->seq);
    free(w);
}

// Reports queued by the replay that the GPU has not finished. The condition of RENDER_COND reads a timestamp one as 0
// and a samples-passed one as unknown.
// The newest report per address, in a table probed over PENDING_PROBE slots from the address's hash: Ocarina issues
// hundreds of occlusion reports a frame and conditions draws on them in the same frame, so a ring of the last 256
// reports had dropped the report a condition referred to, which then read an older query's count (often 0) and culled
// visible ground and trees. It also rotates through tens of thousands of record addresses, so an entry whose value is
// in memory (written >= seq: a condition reads memory anyway) gives its slot to a new address; a table that only grew
// filled up after a minute and conditions on the newer records read the CPU's 1 again.
#define PENDING (1u << 16)
#define PENDING_PROBE 32
// seq: the newest report to addr; written: the newest whose value the completion thread has put in guest memory. A
// report counts as pending until then, not until its fence passes: the callback runs after the fence, and in between
// the record still held what the CPU wrote (Ocarina re-arms its condition records with 1 for the next frame), so a
// condition read 1 where the GPU had written 0, a depth clear ran after the depth copy and frames showed only sky.
// frame: the present count (vk_exec_cond_frame_end, in replay order) when the report was replayed. The game arms its
// condition records from the CPU between frames, so within a frame a timestamp report decides the record's value even
// after it reached memory: Ocarina's CPU, running a frame ahead, had re-armed a record with 1 by the time a later depth
// clear of the frame read it, the clear ran after the depth copy and the frame showed only sky (white flashes).
typedef struct { uint64_t addr, seq, frame; volatile uint64_t written; int unknown, type; } Pending;
static Pending pending[PENDING];
static uint64_t report_seq;
static uint64_t cond_frame;

// Per-frame counts of conditional work (NX2NAVI_LOG_RCOND=1, vk_exec_cond_frame_end logs them): clears and
// draws issued under a render condition, run or skipped.
unsigned vk_cond_clears_run, vk_cond_clears_skip, vk_cond_draws_skip, vk_cond_depth_run;

void vk_exec_cond_frame_end(uint64_t frame) {
    static int on = -1;
    if (on < 0) on = rt_knob("LOG_RCOND", "0");
    if (on && (vk_cond_clears_run || vk_cond_clears_skip || vk_cond_draws_skip))
        rt_log("vk: frame %llu conditions: clears run %u skipped %u, draws skipped %u, depth clears run %u",
               (unsigned long long)frame, vk_cond_clears_run, vk_cond_clears_skip, vk_cond_draws_skip, vk_cond_depth_run);
    vk_cond_clears_run = vk_cond_clears_skip = vk_cond_draws_skip = vk_cond_depth_run = 0;
    if (on == 2) rt_log("SEQ present %llu", (unsigned long long)frame);
    cond_frame++;
}

static Pending *pending_slot(uint64_t addr, int insert);

// Completion thread: the value of report seq to addr is in guest memory.
static void pending_written(uint64_t addr, uint64_t seq) {
    Pending *e = pending_slot(addr, 0);
    if (!e) return;
    uint64_t w = __atomic_load_n(&e->written, __ATOMIC_RELAXED);
    while (w < seq && !__atomic_compare_exchange_n(&e->written, &w, seq, 0, __ATOMIC_RELEASE, __ATOMIC_RELAXED)) {}
}

// Lookups run on the replay thread and the completion thread; only the replay thread inserts.
static Pending *pending_slot(uint64_t addr, int insert) {
    if (!addr) return NULL;
    uint32_t i = (uint32_t)((addr * 0x9E3779B97F4A7C15ull) >> 48) & (PENDING - 1);
    Pending *free_ = NULL, *victim = NULL;
    for (unsigned n = 0; n < PENDING_PROBE; n++, i = (i + 1) & (PENDING - 1)) {
        Pending *e = &pending[i];
        uint64_t a = __atomic_load_n(&e->addr, __ATOMIC_ACQUIRE);
        if (a == addr) return e;
        if (!insert) continue;
        if (!a) { if (!free_) free_ = e; continue; }
        if (__atomic_load_n(&e->written, __ATOMIC_ACQUIRE) >= e->seq && e->frame != cond_frame && (!victim || e->seq < victim->seq))
            victim = e;   // done, and not from this frame (whose reports still decide their records)
    }
    Pending *e = free_ ? free_ : victim;
    if (!e) return NULL;   // every probed entry still has a report in flight
    e->seq = 0, e->frame = 0, e->unknown = 0, e->type = -1;
    __atomic_store_n(&e->written, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&e->addr, addr, __ATOMIC_RELEASE);
    return e;
}

// ---- occlusion counts (samples passed)
//
// After a SAMPLES_PASSED reset, the first draw of a pass begins an occlusion query (precise when the device has
// occlusionQueryPrecise); the query ends with the pass, or at the next report or reset of the counter, and the next
// draw begins another. A Vulkan query begun inside a render pass must end inside it. NX2NAVI always resets, draws the
// tested proxies, reports, so counting stops at a report and restarts at the next reset: queries cover only the draws
// between the two, not every pass of the frame. Once the game reports twice without a reset in between, counting stays
// on for good (cumulative NVN semantics; that one report misses the draws since the previous one). The queries closed since the last
// report or reset form a segment, handed to the completion thread with that report or reset: in submit order it sums
// the segment's results into the running count (a reset drops the sum and zeroes the count) and writes the count of a
// report into guest memory. Slots come from one ring; the completion thread reads and host-resets them
// (hostQueryReset, Vulkan 1.2), so the replay records no reset commands. A full ring (no slot read back yet) leaves the
// draw uncounted and the segment's count at least 1, as does a query still unavailable after its fence. Without hostQueryReset the counts stay 0.
#define OQ_N 8192
#define LOG_ONCE(...) do { static volatile char once_; if (!__atomic_exchange_n(&once_, 1, __ATOMIC_RELAXED)) rt_log(__VA_ARGS__); } while (0)
static VkQueryPool oq_pool;
static int oq_state;                     // 0 not set up, 1 counting available, -1 unavailable
static VkQueryControlFlags oq_flags;
static volatile uint8_t oq_busy[OQ_N];   // from begin until the completion thread read and reset the slot
static uint32_t oq_next;                 // replay: next ring slot
static int oq_counting;                  // replay: draws are counted (a reset since the last report, or oq_keep)
static int oq_reported;                  // replay: a samples-passed report since the last reset
static int oq_keep;                      // replay: the game reported twice without a reset: count every draw
static int oq_open = -1;                 // replay: the slot whose query is active in the open pass
static uint32_t *oq_seg;                 // replay: slots closed since the last report or reset
static uint32_t oq_nseg, oq_capseg;
static int oq_lost;                      // replay: the segment missed a draw (ring full)
static uint64_t oq_count;                // completion thread: samples since the last reset

typedef struct { uint64_t addr, seq; int reset, lost; uint32_t n; uint32_t q[]; } OqSegment;

static int oq_ready(void) {
    if (oq_state) return oq_state > 0;
    VkCtx *v = vk_ctx();
    // vk_device.c enables every feature the device reports, so supported means enabled.
    VkPhysicalDeviceVulkan12Features f12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 f2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f12};
    vkGetPhysicalDeviceFeatures2(v->phys, &f2);
    oq_state = -1;
    if (!f12.hostQueryReset) {
        rt_log("vk: no hostQueryReset: samples-passed counters report 0");
        return 0;
    }
    VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_OCCLUSION,
                                .queryCount = OQ_N};
    VkResult r = vkCreateQueryPool(v->device, &qi, NULL, &oq_pool);
    if (r != VK_SUCCESS) {
        rt_log("vk: occlusion query pool creation failed (%d): samples-passed counters report 0", (int)r);
        return 0;
    }
    vkResetQueryPool(v->device, oq_pool, 0, OQ_N);
    oq_flags = f2.features.occlusionQueryPrecise ? VK_QUERY_CONTROL_PRECISE_BIT : 0;
    oq_state = 1;
    return 1;
}

static void oq_close(VkExec *ex) {
    if (oq_open < 0) return;
    vkCmdEndQuery(ex->cb, oq_pool, (uint32_t)oq_open);
    ex->recorded++;
    if (oq_nseg == oq_capseg) {
        oq_capseg = oq_capseg ? oq_capseg * 2 : 64;
        oq_seg = realloc(oq_seg, sizeof *oq_seg * oq_capseg);
        if (!oq_seg) rt_fatal("vk: out of memory");
    }
    oq_seg[oq_nseg++] = (uint32_t)oq_open;
    oq_open = -1;
}

void vk_exec_oq_draw(VkExec *ex) {
    if (!oq_counting || oq_open >= 0) return;
    uint32_t q = oq_next % OQ_N;
    if (__atomic_load_n(&oq_busy[q], __ATOMIC_ACQUIRE)) {
        warn_once(W_QUERY, "occlusion query ring full: draws left uncounted");
        oq_lost = 1;
        return;
    }
    oq_busy[q] = 1;
    oq_next++;
    vkCmdBeginQuery(ex->cb, oq_pool, q, oq_flags);
    ex->recorded++;
    oq_open = (int)q;
}

void vk_exec_oq_pass_end(VkExec *ex) { oq_close(ex); }

static void oq_batch_drop(void);   // the results kept by the completion thread (oq_take)

void vk_exec_xfer_shutdown(void) {
    if (oq_state > 0) vkDestroyQueryPool(vk_ctx()->device, oq_pool, NULL);
    oq_pool = VK_NULL_HANDLE, oq_state = 0, oq_next = 0, oq_counting = oq_reported = oq_keep = 0, oq_open = -1;
    oq_nseg = 0, oq_lost = 0, oq_count = 0;
    memset((void *)oq_busy, 0, sizeof oq_busy);
    oq_batch_drop();
}

// The completion thread reads the results in batches (NX2NAVI_OQ_BATCH, default 1; 0: every segment by itself, as
// before). The game tests one proxy per report, so a segment holds one query, and a frame of Hyrule Field has about a
// hundred of them: one vkGetQueryPoolResults and one vkResetQueryPool each. On an RX 7900 XT (26.10.1) each of those
// calls goes into the kernel, and the two hundred took 3.9 ms of every frame on the completion thread, which runs a
// submission's callbacks before it lets the submission's waiters go: the game's main thread waits for the previous
// frame's fence before it lets the next frame's GPU work start, so the frame after one replayed inside its submit
// call began 4 to 6 ms late with every thread idle. Slots come from the ring in order, so the slots of the segments
// that follow lie behind the one asked for: one call reads OQ_BATCH slots from there on with their availability, the
// available ones are kept (the results stay in the pool until the reset), and the segments after it take theirs from
// what is kept. A slot not available in the batch (its part had not run yet) is read again when its segment comes.
// The slots taken are reset in runs, one vkResetQueryPool for each: a run ends when a slot that does not continue it
// is taken or a new batch is read. Until then its slots stay busy for the replay, at most OQ_BATCH of the ring's
// OQ_N, also while the game reports nothing.
#define OQ_BATCH 256
static struct {
    uint32_t first, n;               // the slots kept: [first, first + n)
    uint64_t v[OQ_BATCH * 2];        // per slot: samples, available
    uint8_t taken[OQ_BATCH];
    uint32_t r0, rn;                 // the run of taken slots still to be reset
    uint64_t calls, slots, segs;     // since the last perf line: result calls, slots they read, segments served
} OQB;
static int oq_batch_mode = -1;

static void oq_batch_drop(void) { OQB.n = OQB.rn = 0; }

static void oq_reset_flush(VkDevice dev) {
    if (!OQB.rn) return;
    vkResetQueryPool(dev, oq_pool, OQB.r0, OQB.rn);
    for (uint32_t k = 0; k < OQB.rn; k++) __atomic_store_n(&oq_busy[OQB.r0 + k], 0, __ATOMIC_RELEASE);
    OQB.rn = 0;
}

static void oq_read(VkDevice dev, uint32_t first, uint32_t n) {
    VkResult r = vkGetQueryPoolResults(dev, oq_pool, first, n, sizeof OQB.v[0] * 2 * n, OQB.v, 2 * sizeof OQB.v[0],
                                       VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    if (r != VK_SUCCESS && r != VK_NOT_READY) {
        vk_device_fault_report(r);
        rt_fatal("vk: occlusion query results failed (%d)", (int)r);
    }
    OQB.first = first, OQB.n = n;
    memset(OQB.taken, 0, n);
    OQB.calls++, OQB.slots += n;
}

// The samples of slot q, whose submission has completed; *avail 0 when the query has no result.
static uint64_t oq_take(VkDevice dev, uint32_t q, int *avail) {
    uint32_t k = q - OQB.first;
    if (!(q >= OQB.first && k < OQB.n && !OQB.taken[k] && OQB.v[2 * k + 1])) {
        // not kept, or kept as unavailable (read before its part ran): from this slot on, up to the replay's next slot
        oq_reset_flush(dev);   // before the kept results go: the run's slots are among them
        uint32_t next = __atomic_load_n(&oq_next, __ATOMIC_RELAXED) % OQ_N, n = OQ_N - q;
        if (next > q && next - q < n) n = next - q;
        if (n > OQ_BATCH) n = OQ_BATCH;
        if (oq_batch_mode == 0) n = 1;
        oq_read(dev, q, n);
        k = 0;
    }
    OQB.taken[k] = 1;
    if (OQB.rn && q != OQB.r0 + OQB.rn) oq_reset_flush(dev);
    if (!OQB.rn) OQB.r0 = q;
    OQB.rn++;
    *avail = OQB.v[2 * k + 1] != 0;
    return OQB.v[2 * k];
}

static void oq_cb(void *arg) {
    OqSegment *s = arg;
    VkDevice dev = vk_ctx()->device;
    if (oq_batch_mode < 0) oq_batch_mode = cfg_bool("NX2NAVI_OQ_BATCH");
    uint64_t sum = 0;
    for (uint32_t i = 0; i < s->n; i++) {
        // The fence has passed, so a query that is still unavailable will not become available: waiting would block the
        // completion thread, and with it every later callback, for good. It counts as visible (one sample).
        int avail;
        uint64_t v = oq_take(dev, s->q[i], &avail);
        if (avail) sum += v;
        else {
            sum += 1;
            LOG_ONCE("vk: occlusion query slot %u not available after its fence, counted as visible", s->q[i]);
        }
    }
    if (s->n && ++OQB.segs >= 12000) {
        if (rt_trace & RT_TRACE_PERF)
            rt_log("perf: occlusion queries: %llu segments read with %llu result calls of %.1f slots on average",
                   (unsigned long long)OQB.segs, (unsigned long long)OQB.calls,
                   OQB.calls ? (double)OQB.slots / (double)OQB.calls : 0.0);
        OQB.calls = OQB.slots = OQB.segs = 0;
    }
    if (s->reset) oq_count = 0;
    else {
        oq_count += sum;
        if (s->lost && !oq_count) oq_count = 1;
        if (s->addr) write_counter(s->addr, oq_count), pending_written(s->addr, s->seq);
    }
    free(s);
}

// Hands the segment to the completion thread with a report (addr; 0: no CPU-visible memory) or a reset.
static void oq_segment(VkExec *ex, uint64_t addr, int reset) {
    oq_close(ex);
    OqSegment *s = malloc(sizeof *s + sizeof *s->q * oq_nseg);
    if (!s) rt_fatal("vk: out of memory");
    s->addr = addr, s->seq = report_seq, s->reset = reset, s->lost = oq_lost, s->n = oq_nseg;
    memcpy(s->q, oq_seg, sizeof *s->q * oq_nseg);
    oq_nseg = 0, oq_lost = 0;
    vk_completion_add(ex->fence, oq_cb, s);
}

static void reset_counter(VkExec *ex, const p_reset_counter *p) {
    if (p->type != NVN_COUNTER_TYPE_SAMPLES_PASSED || !oq_ready()) return;
    oq_segment(ex, 0, 1);
    oq_counting = 1, oq_reported = 0;
}

// A samples-passed report: the segment goes with it (addr 0: no CPU-visible memory) and counting stops until the next
// reset, unless reports came without resets before.
static void oq_report(VkExec *ex, uint64_t addr) {
    oq_segment(ex, addr, 0);
    if (oq_reported && !oq_keep) {
        oq_keep = 1;
        rt_log("vk: samples passed reported twice without a reset: counting every draw from now on");
    }
    oq_reported = 1;
    if (!oq_keep) oq_counting = 0;
}

// ---- event order check
// The game orders work between its queues with events: a command buffer waits (WaitEvent) until another's SignalEvent
// has set the value. Every submission replays here on one queue in the order it came, and a wait records nothing, so a
// wait whose signal comes in a later submission does not hold its work back: that work runs before what it waits for.
// The replay keeps the value each event reaches in replay order (ev_seen, the signals recorded so far, on top of the
// value in memory) and counts the waits it reaches unmet; NX2NAVI_TRACE=perf logs the count, and the first few are
// logged as they happen. Replay thread only.
#define EV_SEEN 64
static struct { volatile uint32_t *val; uint32_t v; } ev_seen[EV_SEEN];
static int ev_n, ev_next;
static uint64_t ev_waits, ev_unmet, ev_late, ev_late_subs;
// Waits reached unmet, until a replayed signal meets them (ev_late: the signal came that many submissions later, so
// the work after the wait ran before the work it waited for) or they are pushed out by newer ones.
#define EV_OPEN 64
static struct { volatile uint32_t *val; uint32_t mode, value; uint64_t sub; } ev_open[EV_OPEN];
static int ev_open_n;

static uint32_t *ev_slot(volatile uint32_t *val, int create) {
    for (int i = 0; i < ev_n; i++)
        if (ev_seen[i].val == val) return &ev_seen[i].v;
    if (!create) return NULL;
    int i = ev_n < EV_SEEN ? ev_n++ : ev_next++ % EV_SEEN;
    ev_seen[i].val = val, ev_seen[i].v = __atomic_load_n(val, __ATOMIC_ACQUIRE);
    return &ev_seen[i].v;
}

static int ev_holds(uint32_t have, uint32_t mode, uint32_t want) {
    return mode == NVN_EVENT_WAIT_MODE_GEQUAL_WRAP ? (int32_t)(have - want) >= 0 : have == want;
}

// ---- events the CPU signals
// Ocarina gates its main queue on the CPU: each frame the game sets one of two events to 0 (they take turns frame by
// frame), submits the worker queues' command buffers and the main queue's, whose first command waits for the event
// to be 1, and signals 1 from the CPU (nvnEventSignal) once it has finished writing what the frame's commands read.
// The signal comes before the main queue's submit call, after it, or while that call is replaying. On the console
// the GPU starts the queue at the signal and reads memory from then on.
// Here a submission's recording begins with the copies of what it reads: the VRAM mirrors' sync and the uniform
// snapshot (vk_exec_record), and the wait is tested when the walk reaches it, some hundred microseconds later. A
// signal that came in between passed the wait, and the frame was drawn from copies taken before the game had
// finished writing: single frames with the scene's shadowing missing or the whole scene black, the interface as it
// was, most of them when other work keeps the CPU busy, since the game's threads then finish later and the signal
// falls into the replay more often.
// So a wait is tested against the value the event had when the submission took its copies: every nvnEventSignal is
// numbered and keeps the value it replaced (vk_exec_event_cpu, before the write), vk_exec_record notes the number at
// its start (vk_exec_event_sync), and a signal with a later number does not count for that submission. The wait then
// stops the submission like any wait reached before its signal, and vk_backend.c submit_sync lets it go on at once
// in a submission of its own, whose copies are taken after the signal.
// NX2NAVI_EVENT_AT_SYNC=0: the value in memory when the walk reaches the wait, as before.
#define CPU_EV 16
#define CPU_EV_HIST 8
static struct { const void *val; uint64_t seq[CPU_EV_HIST]; uint32_t before[CPU_EV_HIST]; unsigned n; } cpu_ev[CPU_EV];
static int cpu_ev_n;                             // cpu_ev_lock
static uint64_t cpu_ev_seq;                      // cpu_ev_lock: nvnEventSignal calls so far
static uint64_t ev_sync_seq = UINT64_MAX;        // cpu_ev_lock: cpu_ev_seq at the recording's start; UINT64_MAX outside one
static PlatMutex cpu_ev_lock = PLAT_MUTEX_INIT;
static uint64_t ev_after_sync;                   // replay: waits only a signal after the copies would have passed

int vk_exec_event_at_sync(void) {
    static int on = -1;
    if (on < 0) on = rt_knob("EVENT_AT_SYNC", "1");
    return on;
}

void vk_exec_event_cpu(const void *event) {
    plat_mutex_lock(&cpu_ev_lock);
    int i = 0;
    while (i < cpu_ev_n && cpu_ev[i].val != event) i++;
    if (i == cpu_ev_n && cpu_ev_n < CPU_EV) cpu_ev[cpu_ev_n++].val = event, cpu_ev[i].n = 0;
    cpu_ev_seq++;
    if (i < cpu_ev_n) {
        unsigned k = cpu_ev[i].n++ % CPU_EV_HIST;
        cpu_ev[i].seq[k] = cpu_ev_seq, cpu_ev[i].before[k] = __atomic_load_n((const volatile uint32_t *)event, __ATOMIC_ACQUIRE);
    }
    plat_mutex_unlock(&cpu_ev_lock);
}

// 1 when the CPU has signalled this event before (nvnEventSignal): its waits are gates the game opens from the CPU.
int vk_exec_event_cpu_known(const void *event) {
    plat_mutex_lock(&cpu_ev_lock);
    int i = 0;
    while (i < cpu_ev_n && cpu_ev[i].val != event) i++;
    int known = i < cpu_ev_n;
    plat_mutex_unlock(&cpu_ev_lock);
    return known;
}

// The wait's condition on the value in memory now, whatever the replay has seen. Any thread.
int vk_exec_event_holds_now(const void *payload) {
    const p_wait_event *p = payload;
    return ev_holds(__atomic_load_n(p->val, __ATOMIC_ACQUIRE), p->mode, p->value);
}

void vk_exec_event_sync(int begin) {
    plat_mutex_lock(&cpu_ev_lock);
    ev_sync_seq = begin ? cpu_ev_seq : UINT64_MAX;
    plat_mutex_unlock(&cpu_ev_lock);
}

uint64_t vk_exec_event_late_take(void) {
    uint64_t n = ev_after_sync;
    ev_after_sync = 0;
    return n;
}

// The event's value in memory, and in *at_sync the value it had when the submission being recorded took its copies
// (the same outside a recording, and for an event the CPU has not signalled since).
static uint32_t ev_mem(volatile uint32_t *val, uint32_t *at_sync) {
    uint32_t mem = __atomic_load_n(val, __ATOMIC_ACQUIRE);
    *at_sync = mem;
    plat_mutex_lock(&cpu_ev_lock);
    if (ev_sync_seq != UINT64_MAX)
        for (int i = 0; i < cpu_ev_n; i++) {
            if (cpu_ev[i].val != (const void *)val) continue;
            uint64_t first = UINT64_MAX;
            unsigned n = cpu_ev[i].n < CPU_EV_HIST ? cpu_ev[i].n : CPU_EV_HIST;
            for (unsigned k = 0; k < n; k++)   // the earliest signal after the copies replaced the value they saw
                if (cpu_ev[i].seq[k] > ev_sync_seq && cpu_ev[i].seq[k] < first) first = cpu_ev[i].seq[k], *at_sync = cpu_ev[i].before[k];
            break;
        }
    plat_mutex_unlock(&cpu_ev_lock);
    return mem;
}

// Whether the wait holds. *late: it holds only by a CPU signal that came after the submission's copies.
static int ev_met(const p_wait_event *p, uint32_t *mem, const uint32_t **seen, int *late) {
    uint32_t at_sync;
    *mem = ev_mem(p->val, &at_sync);
    *seen = ev_slot(p->val, 0);
    int by_seen = *seen && ev_holds(**seen, p->mode, p->value);
    int now = ev_holds(*mem, p->mode, p->value) || by_seen, then = ev_holds(at_sync, p->mode, p->value) || by_seen;
    *late = now && !then;
    return vk_exec_event_at_sync() ? then : now;
}

int vk_exec_event_met(const void *payload) {
    uint32_t mem;
    const uint32_t *seen;
    int late;
    return ev_met(payload, &mem, &seen, &late);
}

int vk_exec_signalled;

int vk_exec_wait_event(const void *payload) {
    const p_wait_event *p = payload;
    ev_waits++;
    uint32_t mem;
    const uint32_t *seen;
    int late;
    int met = ev_met(p, &mem, &seen, &late);
    if (late) {
        ev_after_sync++;
        vk_exec_evt('L', (const void *)p->val, 0);
    }
    if (met) return 1;
    ev_unmet++;
    if (ev_open_n == EV_OPEN) memmove(ev_open, ev_open + 1, sizeof ev_open - sizeof *ev_open), ev_open_n--;
    ev_open[ev_open_n].val = p->val, ev_open[ev_open_n].mode = p->mode, ev_open[ev_open_n].value = p->value;
    ev_open[ev_open_n++].sub = vk_exec_submit_seq;
    static int logged;
    if (logged++ < 8)
        rt_log("vk: event wait reached before its signal: event %p, mode %u, wants %#x, memory %#x, replayed signals %s%#x",
               (void *)p->val, p->mode, p->value, mem, seen ? "" : "none ", seen ? *seen : 0);
    return 0;
}

void vk_exec_event_stats(uint64_t *waits, uint64_t *unmet, uint64_t *late, uint64_t *late_subs) {
    *waits = ev_waits, *unmet = ev_unmet, *late = ev_late, *late_subs = ev_late_subs;
    ev_waits = ev_unmet = ev_late = ev_late_subs = 0;
}

static void signal_event(VkExec *ex, const p_signal_event *p) {
    uint32_t *seen = ev_slot(p->val, 1);
    *seen = p->mode == NVN_EVENT_SIGNAL_MODE_ADD ? *seen + p->value : p->value;
    vk_exec_evt('s', (const void *)p->val, *seen);
    vk_exec_signalled = 1;
    for (int i = 0; i < ev_open_n;)
        if (ev_open[i].val == p->val && ev_holds(*seen, ev_open[i].mode, ev_open[i].value)) {
            ev_late++, ev_late_subs += vk_exec_submit_seq - ev_open[i].sub;
            ev_open[i] = ev_open[--ev_open_n];
        } else i++;
    EventWrite *e = malloc(sizeof *e);
    if (!e) rt_fatal("vk: out of memory");
    *e = (EventWrite){p->val, p->mode, p->value};
    vk_completion_add(ex->fence, event_cb, e);
}

static void report_counter(VkExec *ex, const p_report_counter *p) {
    int samples = p->type == NVN_COUNTER_TYPE_SAMPLES_PASSED && oq_ready();
    report_seq++;   // before oq_report / counter_cb take it
    if (nvn_gpu_only(p->addr)) {   // CPU-invisible pool: no host memory behind it
        if (samples) oq_report(ex, 0);
        return;
    }
    VkMemSpan m;
    if (!vk_mem_resolve(p->addr, sizeof(NVNcounterData), &m))
        rt_fatal("vk: counter report at %#llx is in no memory pool", (unsigned long long)p->addr);
    vk_mem_count(m.cls, VK_MUSE_GPU_WRITE, sizeof(NVNcounterData));
    if (samples) oq_report(ex, p->addr);
    else {
        CounterWrite *a = malloc(sizeof *a);
        if (!a) rt_fatal("vk: out of memory");
        *a = (CounterWrite){p->addr, report_seq};
        vk_completion_add(ex->fence, counter_cb, a);
    }
    Pending *e = pending_slot(p->addr, 1);
    if (e) e->seq = report_seq, e->frame = cond_frame, e->unknown = samples, e->type = (int)p->type;
    else LOG_ONCE("vk: a counter report's table neighbourhood is all in flight: conditions on %#llx read memory",
                  (unsigned long long)p->addr);
}

// ---- conditional rendering

// 1: *v read, 0: not readable, 2: a samples-passed report not in memory yet (unknown)
static int counter_word(uint64_t addr, uint64_t *v) {
    const Pending *e = pending_slot(addr, 0);   // the last report to addr decides
    if (e && !e->unknown && e->frame == cond_frame) return *v = 0, 1;   // a timestamp/statistics report of this frame
    if (e && __atomic_load_n(&e->written, __ATOMIC_ACQUIRE) < e->seq) return *v = 0, e->unknown ? 2 : 1;
    VkMemSpan m;
    if (nvn_gpu_only(addr) || !vk_mem_resolve(addr, sizeof *v, &m)) return 0;
    memcpy(v, (const void *)(uintptr_t)addr, sizeof *v);
    vk_mem_count(m.cls, VK_MUSE_CPU_READ, sizeof *v);
    return 1;
}

// The counter type of the newest report to addr the ring still holds, -1 when there is none (NX2NAVI_LOG_RCOND).
static int last_type(uint64_t addr) {
    const Pending *e = pending_slot(addr, 0);
    return e ? e->type : -1;
}

// NX2NAVI_LOG_RCOND=ab (diagnosis): the words a condition compares and where each came from: "f" a report replayed
// this frame (0), "p" a pending report (0 or unknown), "m" memory, with the newest report's type and frame.
void vk_exec_cond_why(const p_render_cond *c, char *buf, size_t cap) {
    int o = snprintf(buf, cap, "%#llx mode %d:", (unsigned long long)c->addr, c->mode);
    for (int k = 0; k < 2 && o < (int)cap; k++) {
        uint64_t a = c->addr + 16 * (uint64_t)k;
        const Pending *e = pending_slot(a, 0);
        const char *src = !e ? "m" : !e->unknown && e->frame == cond_frame ? "f"
                        : __atomic_load_n(&e->written, __ATOMIC_ACQUIRE) < e->seq ? "p" : "m";
        o += snprintf(buf + o, cap - (size_t)o, " %llu %s(type %d, frame %lld of %llu)",
                      (unsigned long long)*(const uint64_t *)(uintptr_t)a, src, e ? e->type : -1,
                      e ? (long long)e->frame : -1LL, (unsigned long long)cond_frame);
    }
}

static int cond_enabled(const p_render_cond *c) {
    uint64_t x, y;
    int rx = counter_word(c->addr, &x), ry = counter_word(c->addr + 16, &y);
    if (!rx || !ry) {
        warn_once(W_BUFFER, "conditional render counters at %#llx are not readable: rendering", (unsigned long long)c->addr);
        return 1;
    }
    if (rx == 2 || ry == 2) return 1;   // a samples-passed count not done yet
    int on = c->mode == 0 ? x == y : x != y;
    // NX2NAVI_LOG_RCOND=1: one condition in 499 (up to 300 lines), on or off, with the counter types last reported there
    static int log_on = -1, nlog;
    static unsigned ncond;
    if (log_on < 0) log_on = rt_knob("LOG_RCOND", "0");
    if (log_on && ncond++ % 499 == 0 && nlog < 300) {
        nlog++;
        // (pending: the report value has not reached guest memory yet)
        int px = 0, py = 0;   // the word came from a report not done yet (forced to 0) rather than memory
        const Pending *ex_ = pending_slot(c->addr, 0), *ey_ = pending_slot(c->addr + 16, 0);
        px = ex_ && ex_->written < ex_->seq, py = ey_ && ey_->written < ey_->seq;
        const uint64_t *m = (const uint64_t *)(uintptr_t)c->addr;
        rt_log("vk: render condition %s at %#llx mode %d: %llu (type %d%s) vs %llu (type %d%s); memory %llx %llx %llx %llx",
               on ? "on" : "off", (unsigned long long)c->addr, c->mode, (unsigned long long)x, last_type(c->addr), px ? ", pending" : "",
               (unsigned long long)y, last_type(c->addr + 16), py ? ", pending" : "", (unsigned long long)m[0],
               (unsigned long long)m[1], (unsigned long long)m[2], (unsigned long long)m[3]);
    }
    return on;
}

// ---- textures

typedef struct {
    const VkTexImage *img;
    VkImageAspectFlags aspect;
    uint32_t level, layer0;
    int bpb, bw, bh;
    int32_t x, y, z;           // region origin
    uint32_t w, h, nz;         // region extent; nz: slices (3D) or layers
} Sub;

// Resolves tex, its view and a copy region to image coordinates. 0 (warned) when the copy cannot be replayed.
static int sub_resolve(const tex_t *t, const view_t *v, int has_v, const NVNcopyRegion *r, int defaults_to_level,
                       Sub *s) {
    memset(s, 0, sizeof *s);
    if (!t) return 0;
    TexVkFormat f;
    if (!tex_vk_format(t->d.format, &f) || f.vk != f.ideal || f.expand_rgb || f.astc_bw) {
        warn_once(W_FORMAT, "copy of format %#x texture (decoded or expanded on upload): skipped", t->d.format);
        return 0;
    }
    s->bpb = nvn_fmt_bpb(t->d.format, &s->bw, &s->bh);
    s->img = vk_texture_get(t);
    if (!s->img || !s->bpb) {
        warn_once(W_FORMAT, "copy of format %#x texture without an image: skipped", t->d.format);
        return 0;
    }
    const VkTexImage *img = s->img;
    if (img->aspect & VK_IMAGE_ASPECT_COLOR_BIT) s->aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    else if (img->aspect == VK_IMAGE_ASPECT_DEPTH_BIT) s->aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    else {
        warn_once(W_FORMAT, "copy of depth-stencil format %#x texture: skipped", t->d.format);
        return 0;
    }
    s->level = has_v && v->set_levels ? v->base_level : 0;
    s->layer0 = has_v && v->set_layers ? v->min_layer : 0;
    uint32_t lw = max1(img->width >> s->level), lh = max1(img->height >> s->level);
    int k = img->scale > 1 ? img->scale : 1;   // a larger shadow map: the game's region in its texels
    s->x = r->xoffset * k, s->y = r->yoffset * k, s->z = r->zoffset;
    s->nz = r->depth > 0 ? (uint32_t)r->depth : 1;
    s->w = r->width > 0 ? (uint32_t)r->width * k : defaults_to_level ? (s->x < (int32_t)lw ? lw - (uint32_t)s->x : 0) : 0;
    s->h = r->height > 0 ? (uint32_t)r->height * k : defaults_to_level ? (s->y < (int32_t)lh ? lh - (uint32_t)s->y : 0) : 0;
    int is3d = img->type == VK_IMAGE_TYPE_3D;
    uint32_t layers = is3d ? 1 : (uint32_t)s->z + s->nz, first = is3d ? 0 : s->layer0;
    if (s->level >= img->levels || s->x < 0 || s->y < 0 || s->z < 0 || !s->w || !s->h ||
        (uint64_t)s->x + s->w > lw || (uint64_t)s->y + s->h > lh || first + layers > img->layers ||
        (is3d && (uint64_t)s->z + s->nz > max1(img->depth >> s->level))) {
        warn_once(W_REGION, "copy region %d,%d,%d %ux%ux%u outside level %u of a %ux%u texture: skipped", s->x, s->y,
                  s->z, s->w, s->h, s->nz, s->level, img->width, img->height);
        return 0;
    }
    return 1;
}

static VkImageSubresourceLayers sub_layers(const Sub *s) {
    int is3d = s->img->type == VK_IMAGE_TYPE_3D;
    return (VkImageSubresourceLayers){s->aspect, s->level, is3d ? 0 : s->layer0 + (uint32_t)s->z, is3d ? 1 : s->nz};
}

static VkOffset3D sub_offset(const Sub *s) {
    return (VkOffset3D){s->x, s->y, s->img->type == VK_IMAGE_TYPE_3D ? s->z : 0};
}

static VkExtent3D sub_extent(const Sub *s) {
    return (VkExtent3D){s->w, s->h, s->img->type == VK_IMAGE_TYPE_3D ? s->nz : 1};
}

// Buffer side of a texture <-> buffer copy: bytes covered, pitches, Vulkan buffer span (a VRAM mirror when the pool
// has one).
static int buffer_side(const VkExec *ex, const Sub *s, uint64_t addr, uint64_t *size, VkBufferImageCopy *c,
                       VkMemSpan *span) {
    uint32_t wb = (s->w + (uint32_t)s->bw - 1) / (uint32_t)s->bw, hb = (s->h + (uint32_t)s->bh - 1) / (uint32_t)s->bh;
    uint64_t row = ex->copy_stride > 0 ? (uint64_t)ex->copy_stride : (uint64_t)wb * (uint64_t)s->bpb;
    if (row % (uint64_t)s->bpb || row < (uint64_t)wb * (uint64_t)s->bpb) {
        warn_once(W_REGION, "buffer copy row stride %llu does not fit %u blocks of %d bytes: skipped",
                  (unsigned long long)row, wb, s->bpb);
        return 0;
    }
    uint64_t img = row * hb;
    *size = img * (s->nz - 1) + row * (hb - 1) + (uint64_t)wb * (uint64_t)s->bpb;
    VkMemSpan m;
    if (!addr || !vk_mem_resolve(addr, *size, &m) || m.avail < *size) {
        warn_once(W_BUFFER, "texture copy buffer %#llx (%llu bytes) is not one backed span: skipped",
                  (unsigned long long)addr, (unsigned long long)*size);
        return 0;
    }
    VK_MEM_PICK(m);
    *span = m;
    *c = (VkBufferImageCopy){.bufferOffset = m.offset, .bufferRowLength = (uint32_t)(row / (uint64_t)s->bpb) * (uint32_t)s->bw,
                             .bufferImageHeight = hb * (uint32_t)s->bh, .imageSubresource = sub_layers(s),
                             .imageOffset = sub_offset(s), .imageExtent = sub_extent(s)};
    return 1;
}

static void copy_buffer_image(VkExec *ex, const p_copy_b2t *p, int to_image) {
    Sub s;
    if (!sub_resolve(p->tex, &p->view, p->has_view, &p->r, 1, &s)) return;
    vk_gpu_gap_note(to_image ? VK_GAP_COPY_B2T : VK_GAP_COPY_T2B, (uint64_t)s.w * s.h);
    if (s.img->scale > 1) {   // the buffer holds the game's texel count, the image more
        warn_once(W_REGION, "buffer copy of a shadow map enlarged by NX2NAVI_SHADOW_SCALE: skipped");
        return;
    }
    uint64_t size;
    VkBufferImageCopy c;
    VkMemSpan m;
    if (!buffer_side(ex, &s, p->src, &size, &c, &m)) return;
    vk_mem_count(m.cls, to_image ? VK_MUSE_COPY_SRC : VK_MUSE_GPU_WRITE, size);
    vk_exec_pass_end(ex);
    vk_exec_barrier(ex);
    if (to_image) vkCmdCopyBufferToImage(ex->cb, m.buffer, s.img->image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    else vkCmdCopyImageToBuffer(ex->cb, s.img->image, VK_IMAGE_LAYOUT_GENERAL, m.buffer, 1, &c);
    ex->recorded++;
    if (to_image) vk_texture_gpu_written(p->tex);
    else vk_mem_gpu_wrote(&m, size), note_gpu_write(ex, p->src, size);
}

static void copy_t2t(VkExec *ex, const p_copy_t2t *p) {
    Sub s, d;
    if (!sub_resolve(p->src, &p->sv, p->has_sv, &p->sr, 1, &s)) return;
    if (!sub_resolve(p->dst, &p->dv, p->has_dv, &p->dr, 1, &d)) return;
    d.nz = s.nz;
    vk_gpu_gap_note(VK_GAP_COPY_T2T, (uint64_t)s.w * s.h);
    int same_size = s.w == d.w && s.h == d.h;
    int same_format = s.img->format == d.img->format;
    if (s.img->samples != d.img->samples && !(s.img->samples > 1 && d.img->samples == 1 && same_size && same_format)) {
        warn_once(W_COPY, "texture copy between %u and %u samples: skipped", s.img->samples, d.img->samples);
        return;
    }
    if (s.aspect != d.aspect || (s.aspect != VK_IMAGE_ASPECT_COLOR_BIT && !same_format)) {
        warn_once(W_COPY, "texture copy between formats %#x and %#x: skipped", p->src->d.format, p->dst->d.format);
        return;
    }
    VkImageSubresourceLayers sl = sub_layers(&s), dl = sub_layers(&d);
    vk_exec_pass_end(ex);
    vk_exec_barrier(ex);
    if (s.img->samples > 1 && d.img->samples == 1) {
        VkImageResolve rv = {sl, sub_offset(&s), dl, sub_offset(&d), sub_extent(&s)};
        vkCmdResolveImage(ex->cb, s.img->image, VK_IMAGE_LAYOUT_GENERAL, d.img->image, VK_IMAGE_LAYOUT_GENERAL, 1, &rv);
    } else if (same_size && same_format) {
        VkImageCopy cp = {sl, sub_offset(&s), dl, sub_offset(&d), sub_extent(&s)};
        vkCmdCopyImage(ex->cb, s.img->image, VK_IMAGE_LAYOUT_GENERAL, d.img->image, VK_IMAGE_LAYOUT_GENERAL, 1, &cp);
    } else {   // scaled or converting copy: nearest, or linear when asked for (colour only)
        int is3d = s.img->type == VK_IMAGE_TYPE_3D;
        VkImageBlit bl = {.srcSubresource = sl, .dstSubresource = dl};
        bl.srcOffsets[0] = sub_offset(&s);
        bl.srcOffsets[1] = (VkOffset3D){s.x + (int32_t)s.w, s.y + (int32_t)s.h, is3d ? s.z + (int32_t)s.nz : 1};
        bl.dstOffsets[0] = sub_offset(&d);
        bl.dstOffsets[1] = (VkOffset3D){d.x + (int32_t)d.w, d.y + (int32_t)d.h, is3d ? d.z + (int32_t)d.nz : 1};
        int linear = (p->flags & NVN_COPY_FLAGS_LINEAR_FILTER_BIT) && s.aspect == VK_IMAGE_ASPECT_COLOR_BIT;
        vkCmdBlitImage(ex->cb, s.img->image, VK_IMAGE_LAYOUT_GENERAL, d.img->image, VK_IMAGE_LAYOUT_GENERAL, 1, &bl,
                       linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
    }
    ex->recorded++;
    vk_texture_gpu_written(p->dst);
    static int lg = -1;
    if (lg < 0) lg = rt_knob("LOG_RCOND", "0");
    if (lg == 2 && (d.img->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) && d.w >= 1600)
        rt_log("SEQ depth copy %#llx -> %#llx", (unsigned long long)s.img->addr, (unsigned long long)d.img->addr);
}

// ---- buffer to buffer

static void copy_b2b(VkExec *ex, const p_copy_b2b *p) {
    uint64_t src = p->src, dst = p->dst, left = p->size;
    vk_gpu_gap_note(VK_GAP_COPY_B2B, p->size);
    int started = 0;
    while (left) {
        VkMemSpan s, d;
        if (!vk_mem_resolve(src, 1, &s) || !vk_mem_resolve(dst, 1, &d)) {
            warn_once(W_BUFFER, "buffer copy %#llx -> %#llx (%llu bytes) leaves backed memory: rest skipped",
                      (unsigned long long)p->src, (unsigned long long)p->dst, (unsigned long long)p->size);
            break;
        }
        uint64_t n = left < s.avail ? left : s.avail;
        if (n > d.avail) n = d.avail;
        VK_MEM_PICK(s);
        VK_MEM_PICK(d);
        if (s.buffer == d.buffer && s.offset < d.offset + n && d.offset < s.offset + n) {
            warn_once(W_OVERLAP, "buffer copy %#llx -> %#llx overlaps within one buffer: skipped",
                      (unsigned long long)p->src, (unsigned long long)p->dst);
            break;
        }
        if (!started) vk_exec_pass_end(ex), vk_exec_barrier(ex), started = 1;
        VkBufferCopy bc = {s.offset, d.offset, n};
        vkCmdCopyBuffer(ex->cb, s.buffer, d.buffer, 1, &bc);
        vk_mem_gpu_wrote(&d, n);
        ex->recorded++;
        vk_mem_count(s.cls, VK_MUSE_COPY_SRC, n);
        vk_mem_count(d.cls, VK_MUSE_GPU_WRITE, n);
        src += n, dst += n, left -= n;
    }
    if (started) note_gpu_write(ex, p->dst, p->size - left);
}

// ---- buffer fills (ClearBuffer, and ReportValue as a 4-byte one): vkCmdFillBuffer in GPU order after a full
// barrier, so the value lands after earlier work, as NVN's semaphore release does; CPU-visible ranges go to
// vk_guest_written at the fence, as copies do. Works in CPU-invisible pools.

static void clear_buf(VkExec *ex, const p_clear_buf *p) {
    if ((p->addr | p->size) & 3) {
        warn_once(W_BUFFER, "buffer fill at %#llx of %llu bytes is not 4-byte aligned: skipped", (unsigned long long)p->addr,
                  (unsigned long long)p->size);
        return;
    }
    uint64_t dst = p->addr, left = p->size;
    vk_gpu_gap_note(VK_GAP_CLEAR_BUF, p->size);
    int started = 0;
    while (left) {
        VkMemSpan d;
        if (!vk_mem_resolve(dst, 1, &d) || d.offset % 4) {
            warn_once(W_BUFFER, "buffer fill at %#llx (%llu bytes) leaves backed memory: rest skipped",
                      (unsigned long long)p->addr, (unsigned long long)p->size);
            break;
        }
        uint64_t n = left < d.avail ? left : d.avail & ~3ull;
        if (!n) break;
        if (!started) vk_exec_pass_end(ex), vk_exec_barrier(ex), started = 1;
        VK_MEM_PICK(d);
        vkCmdFillBuffer(ex->cb, d.buffer, d.offset, n, p->value);
        // A VRAM mirror's guest copy is filled too, at once: atomics through the device address table count in both
        // copies (translator patch 0036), and a counter reset here must start from the same value in both.
        VkBuffer twin = vk_mem_guest_twin(&d);
        if (twin) vkCmdFillBuffer(ex->cb, twin, d.offset, n, p->value), ex->recorded++;
        vk_mem_gpu_wrote(&d, n);
        ex->recorded++;
        vk_mem_count(d.cls, VK_MUSE_GPU_WRITE, n);
        dst += n, left -= n;
    }
    if (started) note_gpu_write(ex, p->addr, p->size - left);
}

// ---- texture clears

static int int_format(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8_UINT: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8B8A8_UINT: case VK_FORMAT_R8_SINT:
    case VK_FORMAT_R8G8_SINT: case VK_FORMAT_R8G8B8A8_SINT: case VK_FORMAT_R16_UINT: case VK_FORMAT_R16G16_UINT:
    case VK_FORMAT_R16G16B16A16_UINT: case VK_FORMAT_R16_SINT: case VK_FORMAT_R16G16_SINT:
    case VK_FORMAT_R16G16B16A16_SINT: case VK_FORMAT_R32_UINT: case VK_FORMAT_R32G32_UINT:
    case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_R32_SINT: case VK_FORMAT_R32G32_SINT:
    case VK_FORMAT_R32G32B32A32_SINT: case VK_FORMAT_A2B10G10R10_UINT_PACK32:
        return 1;
    default: return 0;
    }
}

// Whole levels only (vkCmdClear*Image has no rectangle); partial regions, channel masks and a view that reinterprets
// the format are skipped.
static void clear_tex(VkExec *ex, const p_clear_tex *p) {
    const tex_t *t = p->tex;
    if (!t) return;
    Sub s;
    if (!sub_resolve(t, &p->view, p->has_view, &p->r, 1, &s)) return;
    const VkTexImage *img = s.img;
    int is3d = img->type == VK_IMAGE_TYPE_3D;
    uint32_t lw = max1(img->width >> s.level), lh = max1(img->height >> s.level);
    vk_gpu_gap_note(VK_GAP_CLEAR_TEX, (uint64_t)lw * lh);
    int reformat = p->has_view && p->view.set_format && p->view.format != t->d.format;
    if (s.x || s.y || s.w != lw || s.h != lh || (p->mask & 0xf) != 0xf || reformat ||
        (is3d && (s.z || s.nz != max1(img->depth >> s.level)))) {
        warn_once(W_CLEAR, "texture clear of region %d,%d %ux%u mask %#x (level %ux%u, view format change %d): skipped",
                  s.x, s.y, s.w, s.h, p->mask, lw, lh, reformat);
        return;
    }
    VkImageSubresourceRange rng = {img->aspect, s.level, 1, is3d ? 0 : s.layer0 + (uint32_t)s.z, is3d ? 1 : s.nz};
    if (img->aspect & VK_IMAGE_ASPECT_COLOR_BIT) {
        rng.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        VkClearColorValue cv;
        memcpy(&cv, &p->c, sizeof cv);
        if (int_format(img->format) != (p->is_uint != 0)) {
            // A value of the other kind that is all zero bits (0.0f is) is the same clear either way: Ocarina clears its
            // rg16ui edge map (480x270) with a float 0 each frame, and skipped, the post chain's edge classification
            // read the previous frames' edges. Other values stay skipped.
            if (cv.uint32[0] | cv.uint32[1] | cv.uint32[2] | cv.uint32[3]) {
                warn_once(W_CLEAR, "texture clear value kind (uint %d) does not match format %d: skipped", p->is_uint,
                          (int)img->format);
                return;
            }
        }
        vk_exec_pass_end(ex);
        vk_exec_barrier(ex);
        vkCmdClearColorImage(ex->cb, img->image, VK_IMAGE_LAYOUT_GENERAL, &cv, 1, &rng);
    } else {
        int both = (img->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) && (img->aspect & VK_IMAGE_ASPECT_STENCIL_BIT);
        VkClearDepthStencilValue dv = {0, 0};
        if (img->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) dv.depth = p->c.f[0];
        if (img->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) dv.stencil = both ? p->c.u[1] : p->c.u[0];
        vk_exec_pass_end(ex);
        vk_exec_barrier(ex);
        vkCmdClearDepthStencilImage(ex->cb, img->image, VK_IMAGE_LAYOUT_GENERAL, &dv, 1, &rng);
    }
    ex->recorded++;
    vk_texture_gpu_written(t);
}

static void op_xfer_impl(VkExec *ex, uint32_t op, const void *payload) {
    if (op != OP_RENDER_ENABLE && op != OP_SIGNAL_EVENT) vk_contact_flush(ex);
    switch (op) {
    case OP_COPY_B2B: copy_b2b(ex, payload); break;
    case OP_COPY_B2T: copy_buffer_image(ex, payload, 1); break;
    case OP_COPY_T2B: copy_buffer_image(ex, payload, 0); break;   // p_copy_b2t: src holds the destination address
    case OP_COPY_T2T: copy_t2t(ex, payload); break;
    case OP_CLEAR_TEX: clear_tex(ex, payload); break;
    case OP_SIGNAL_EVENT: signal_event(ex, payload); break;
    case OP_REPORT_COUNTER: report_counter(ex, payload); break;
    case OP_RESET_COUNTER: reset_counter(ex, payload); break;
    case OP_CLEAR_BUF: clear_buf(ex, payload); break;
    case OP_RENDER_ENABLE: ex->render_disabled = !((const p_flag *)payload)->value; break;
    case OP_RENDER_COND:
        ex->render_cond = *(const p_render_cond *)payload;
        ex->has_render_cond = 1;
        ex->render_disabled = !cond_enabled(payload);
        break;
    default: rt_fatal("vk: op %s is not a transfer", nvn_op_name(op));
    }
}

// NX2NAVI_REPLAY_TIME: see the note over vk_exec_op_draw in vk_exec_draw.c; transfers are the third kind.
void vk_exec_op_xfer(VkExec *ex, uint32_t op, const void *payload) {
    static int on = -1;
    if (on < 0) on = cfg_bool("NX2NAVI_REPLAY_TIME");
    if (!on) { op_xfer_impl(ex, op, payload); return; }
    uint64_t t0 = plat_now_ns();
    op_xfer_impl(ex, op, payload);
    perf_replay_op_at(PERF_RP_XFER, op, plat_now_ns() - t0);
}
