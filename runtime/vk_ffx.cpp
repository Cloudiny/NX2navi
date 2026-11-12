// AMD FidelityFX through its API library (amd_fidelityfx_vk.dll on Windows for FSR 3.1, or amd_fidelityfx_upscaler_vk.so
// next to the executable on Linux for FSR 4): temporal upscaling here, in the place of DLSS in vk_dlaa.c, and frame
// generation (vk_ffx_fg_*, below). The API's headers are in third_party/ffx_api (MIT); they are C++ in places, hence
// this file. Nothing of it runs unless the upscaler or frame generation asks for FSR 3.1 or FSR 4.
//
// FSR 3.1 / FSR 4 take the same inputs vk_dlaa.c makes for DLSS: the scene's light target (linear HDR), its D32F depth (0
// near, 1 far, with the near and far planes of the frame's own projection, vk_dlaa.c proj_planes) and the camera
// motion vectors in render pixels from the current frame to the previous one, the convention both share, so the
// vectors go in unscaled.
#ifdef _WIN32
#include <windows.h>
#define FFX_LIBRARY "amd_fidelityfx_vk.dll"
#define FFX_FSR4_LIBRARY NULL
#else
#include <dlfcn.h>
#include <unistd.h>
#define FFX_LIBRARY NULL
#define FFX_FSR4_LIBRARY "amd_fidelityfx_upscaler_vk.so"
#endif
#include <atomic>
#include <chrono>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <wchar.h>

#include "third_party/ffx_api/ffx_api.h"
#include "third_party/ffx_api/ffx_upscale.h"
#include "third_party/ffx_api/vk/ffx_api_vk.h"

extern "C" {
#include "cfg.h"
#include "host.h"
#include "platform/platform.h"
#include "vk.h"
#include "vk_ffx.h"
}

namespace {

struct Api {
    void *lib = nullptr;
    PfnFfxCreateContext create = nullptr;
    PfnFfxDestroyContext destroy = nullptr;
    PfnFfxConfigure configure = nullptr;
    PfnFfxQuery query = nullptr;
    PfnFfxDispatch dispatch = nullptr;
    int tried = 0, ok = 0;
} A, A4;

// What the settings menu may ask before anything has used FSR 3.1 or FSR 4, from any thread.
std::atomic<int> g_loaded{-1};        // what vk_ffx_load found: 1 or 0; -1 while it has not run
std::atomic<int> g_present{-1};       // whether amd_fidelityfx_vk.dll is there: 1 or 0; -1 while not asked
std::atomic<int> g_fsr4_loaded{-1};   // what vk_ffx_load_fsr4 found: 1 or 0; -1 while it has not run
std::atomic<int> g_fsr4_present{-1};  // whether amd_fidelityfx_upscaler_vk.so is there: 1 or 0; -1 while not asked

int library_path(const char *libname, char *path, size_t size) {
    if (!libname) return 0;
#ifdef _WIN32
    DWORD n = GetModuleFileNameA(NULL, path, (DWORD)size);
    while (n && path[n - 1] != '\\') n--;
    return snprintf(path + n, size - n, "%s", libname) < (int)(size - n);
#else
    if (!plat_exe_path(path, size)) return 0;
    char *sep = plat_last_sep(path);
    if (!sep) return 0;
    sep[1] = 0;
    size_t used = (size_t)(sep + 1 - path);
    return snprintf(path + used, size - used, "%s", libname) < (int)(size - used);
#endif
}

void message(uint32_t type, const wchar_t *msg) {
    char s[512];
    size_t n = 0;
    for (; msg && msg[n] && n < sizeof s - 1; n++) s[n] = msg[n] < 128 ? (char)msg[n] : '?';
    s[n] = 0;
    static int told;
    if (told++ < 40) rt_log("ffx: %s %s", type == FFX_API_MESSAGE_TYPE_ERROR ? "error:" : "warning:", s);
}

// The library's pipeline creates (vk_ffx_proc_addr). It passes no pipeline cache, so one that names none goes through
// the runtime's (vk_pipeline.c vk_pipeline_create_foreign); one that names a cache is the library's own business.
VKAPI_ATTR VkResult VKAPI_CALL compute_pipelines(VkDevice dev, VkPipelineCache cache, uint32_t n,
                                                 const VkComputePipelineCreateInfo *ci, const VkAllocationCallbacks *a,
                                                 VkPipeline *out) {
    if (cache) return vkCreateComputePipelines(dev, cache, n, ci, a, out);
    return vk_pipeline_create_foreign(dev, 1, n, ci, a, out);
}

VKAPI_ATTR VkResult VKAPI_CALL graphics_pipelines(VkDevice dev, VkPipelineCache cache, uint32_t n,
                                                  const VkGraphicsPipelineCreateInfo *ci, const VkAllocationCallbacks *a,
                                                  VkPipeline *out) {
    if (cache) return vkCreateGraphicsPipelines(dev, cache, n, ci, a, out);
    return vk_pipeline_create_foreign(dev, 0, n, ci, a, out);
}

int pipe_cache_on(void) {   // NX2NAVI_FFX_PIPE_CACHE, read once
    static std::atomic<int> on{-1};
    int v = on.load(std::memory_order_relaxed);
    if (v < 0) on.store(v = cfg_bool("NX2NAVI_FFX_PIPE_CACHE"), std::memory_order_relaxed);
    return v;
}

// ffxCreateContext with what it made in pipelines: the library compiles every pass of a context inside this call, on
// the thread that asked, so the line says what a context cost and where its pipelines came from.
ffxReturnCode_t create_counted(const Api &api, ffxContext *ctx, ffxCreateContextDescHeader *desc) {
    VkForeignPipes p0, p1;
    vk_pipeline_foreign_counts(&p0);
    auto t0 = std::chrono::steady_clock::now();
    ffxReturnCode_t r = api.create(ctx, desc, nullptr);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    vk_pipeline_foreign_counts(&p1);
    unsigned long long made = p1.made - p0.made, hit = p1.from_cache - p0.from_cache, comp = p1.compiled - p0.compiled,
                       none = p1.uncached - p0.uncached;
    if (r != FFX_API_RETURN_OK) return r;
    if (!pipe_cache_on())
        rt_log("ffx: context created in %.0f ms; its pipelines are made without the runtime's pipeline caches "
               "(NX2NAVI_FFX_PIPE_CACHE=0)", ms);
    else if (!made)
        rt_log("ffx: context created in %.0f ms; it made no pipeline through the runtime's creates", ms);
    else
        rt_log("ffx: context created in %.0f ms; its %llu pipelines: %llu from the pipeline caches without a compile, "
               "%llu compiled and kept in the play cache for the next start, %llu through the one pipeline cache (what "
               "it held is not told apart), %llu with no pipeline cache (none is up yet, or pipeline binaries are on)",
               ms, made, hit, comp, made - hit - comp - none, none);
    return r;
}

// One upscaling context per render size and output size, as AMD's own sample keeps them: it waits for the GPU and makes
// the context anew whenever the render resolution changes, and sets FFX_UPSCALE_ENABLE_DYNAMIC_RESOLUTION only for its
// dynamic resolution mode. One context per output size that every mode of the menu ran on, with another render size at
// each dispatch and without that flag, preceded an access violation in AMD's driver inside FSR 4's dispatch at the
// first frame after a switch from 67 % to 100 % (vk_ffx4.cpp keeps its contexts the same way). With the game's dynamic
// resolution on (NX2NAVI_DYN_RES, dynres.c), whose steps change the render size every few seconds, one context with
// that flag takes them all. A context of another render size goes once no command buffer can still hold it
// (UP_RECENT), so a switch keeps at most two of an output size for a few frames, and the game's second scenes, of
// another output size, keep their own. When all slots are taken, the least recently used one is destroyed for the new
// size once the GPU is idle, as vk_dlaa.c free_size does for DLSS.
struct Up {
    ffxContext ctx = nullptr;
    uint32_t mw = 0, mh = 0, ow = 0, oh = 0;   // the render size it was made for (the largest it takes with drs); the output size
    bool drs = false;                          // made for dynamic resolution (FFX_UPSCALE_ENABLE_DYNAMIC_RESOLUTION)
    int is_fsr4 = 0;
    uint64_t used = 0, used_ns = 0;            // up_tick and the time when context_for last returned it
} U[6];
uint64_t up_tick;
// A context used in the last few calls may sit in a command buffer that is not submitted yet, where waiting for the
// GPU does not cover it: a frame makes one or two calls (the scene, and a second scene after it), so this is a few
// frames back.
const uint64_t UP_RECENT = 8;
// A context no upscale used for this long goes (trim): FSR 3.1's after a switch to FSR 4, and a second scene's (the
// pause screen's) a while after it closed. Unused that long, no command buffer names it.
const uint64_t UP_IDLE_NS = 90ull * 1000000000ull;

void destroy_up(Up *u, const char *why) {
    rt_log("ffx: the %s context for %ux%u (render sizes up to %ux%u) destroyed: %s", u->is_fsr4 ? "FSR 4" : "FSR 3.1",
           u->ow, u->oh, u->mw, u->mh, why);
    (u->is_fsr4 ? A4 : A).destroy(&u->ctx, nullptr);
    *u = Up();
}

void trim(uint64_t now_ns) {
    for (Up &u : U)
        if (u.ctx && now_ns - u.used_ns > UP_IDLE_NS) destroy_up(&u, "no upscale used it for 90 s");
}

FfxApiResource image(const VkFfxImage *i, uint32_t state, uint32_t usage) {
    VkImageCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = i->format;
    ci.extent = {i->width, i->height, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.usage = i->storage ? VK_IMAGE_USAGE_STORAGE_BIT : 0;
    return ffxApiGetResourceVK((void *)i->image, ffxApiGetImageResourceDescriptionVK(i->image, ci, usage), state);
}

Up *context_for(uint32_t w, uint32_t h, uint32_t ow, uint32_t oh, int fsr4) {
    Api &api = fsr4 ? A4 : A;
    up_tick++;
    uint64_t now = plat_now_ns();
    const bool drs = cfg_bool("NX2NAVI_DYN_RES");
    Up *found = nullptr;
    for (Up &u : U)
        if (u.ctx && u.is_fsr4 == fsr4 && u.ow == ow && u.oh == oh && u.drs == drs &&
            (drs ? w <= u.mw && h <= u.mh : w == u.mw && h == u.mh))
            found = &u;
    const VkCtx *c = vk_ctx();
    int waited = 0;
    for (Up &u : U)   // the contexts of this output size for another render size, once no command buffer holds them
        if (u.ctx && &u != found && u.is_fsr4 == fsr4 && u.ow == ow && u.oh == oh && up_tick - u.used >= UP_RECENT) {
            if (!waited++) vk_queue_lock(), vkDeviceWaitIdle(c->device), vk_queue_unlock();   // their work may be in flight
            destroy_up(&u, "the render size changed");
        }
    if (found) {
        found->used = up_tick, found->used_ns = now;
        return found;
    }
    Up *slot = nullptr, *lru = nullptr;
    for (Up &u : U) {
        if (!u.ctx) {
            if (!slot) slot = &u;
            continue;
        }
        if (up_tick - u.used >= UP_RECENT && (!lru || u.used < lru->used)) lru = &u;
    }
    if (!slot && lru) {
        vk_queue_lock(), vkDeviceWaitIdle(c->device), vk_queue_unlock();   // its work may still be in flight
        destroy_up(lru, "its slot went to another output size");
        slot = lru;
    }
    if (!slot) return nullptr;
    // the render size itself; for dynamic resolution, every size up to the output's (and a scene at the output size)
    uint32_t mw = !drs ? w : w > ow ? w : ow, mh = !drs ? h : h > oh ? h : oh;
    ffxCreateBackendVKDesc be = {};
    be.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
    be.vkDevice = c->device;
    be.vkPhysicalDevice = c->phys;
    be.vkDeviceProcAddr = vk_ffx_proc_addr;
    ffxCreateContextDescUpscale d = {};
    d.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    d.header.pNext = &be.header;
    // no infinite-depth flag: the game's far plane is finite (vk_dlaa.c proj_planes) and each dispatch gives it
    d.flags = FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE | FFX_UPSCALE_ENABLE_AUTO_EXPOSURE;
    // NX2NAVI_FFX_DEBUG=1 (test): the library checks the values of each dispatch and says what it finds (message)
    if (rt_knob("FFX_DEBUG", "0")) d.flags |= FFX_UPSCALE_ENABLE_DEBUG_CHECKING;
    if (drs) d.flags |= FFX_UPSCALE_ENABLE_DYNAMIC_RESOLUTION;
    d.maxRenderSize = {mw, mh};
    d.maxUpscaleSize = {ow, oh};
    d.fpMessage = message;
    ffxReturnCode_t r = create_counted(api, &slot->ctx, &d.header);
    if (r != FFX_API_RETURN_OK) {
        rt_log("ffx: %s context creation for %ux%u (render sizes up to %ux%u) failed (%u)",
               fsr4 ? "FSR 4" : "FSR 3.1", ow, oh, mw, mh, (unsigned)r);
        slot->ctx = nullptr;
        return nullptr;
    }
    slot->mw = mw, slot->mh = mh, slot->ow = ow, slot->oh = oh, slot->drs = drs, slot->is_fsr4 = fsr4;
    slot->used = up_tick, slot->used_ns = now;
    ffxQueryGetProviderVersion v = {};
    v.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
    api.query(&slot->ctx, &v.header);
    rt_log("ffx: %s upscaling to %ux%u from render sizes up to %ux%u (%s)", fsr4 ? "FSR 4" : "FSR 3.1", ow, oh, mw, mh,
           v.versionName ? v.versionName : "provider unknown");
    return slot;
}

int load_api(Api &api, std::atomic<int> &loaded, const char *libname, const char *label) {
    if (api.tried) return api.ok;
    api.tried = 1;
    if (!libname) {
        rt_log("ffx: this build has no %s: %s off", label, label);
        loaded.store(0, std::memory_order_release);
        return 0;
    }
    char path[PLAT_PATH_MAX];
    if (!library_path(libname, path, sizeof path)) {
        rt_log("ffx: cannot resolve path for %s: %s off", libname, label);
        loaded.store(0, std::memory_order_release);
        return 0;
    }
#ifdef _WIN32
    HMODULE m = LoadLibraryA(path);
    if (!m) {
        rt_log("ffx: %s not found: %s off", path, label);
        loaded.store(0, std::memory_order_release);
        return 0;
    }
    api.lib = m;
    api.create = (PfnFfxCreateContext)(void *)GetProcAddress(m, "ffxCreateContext");
    api.destroy = (PfnFfxDestroyContext)(void *)GetProcAddress(m, "ffxDestroyContext");
    api.configure = (PfnFfxConfigure)(void *)GetProcAddress(m, "ffxConfigure");
    api.query = (PfnFfxQuery)(void *)GetProcAddress(m, "ffxQuery");
    api.dispatch = (PfnFfxDispatch)(void *)GetProcAddress(m, "ffxDispatch");
#else
    void *m = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!m) {
        rt_log("ffx: %s not loaded (%s): %s off", path, dlerror(), label);
        loaded.store(0, std::memory_order_release);
        return 0;
    }
    api.lib = m;
    api.create = (PfnFfxCreateContext)dlsym(m, "ffxCreateContext");
    api.destroy = (PfnFfxDestroyContext)dlsym(m, "ffxDestroyContext");
    api.configure = (PfnFfxConfigure)dlsym(m, "ffxConfigure");
    api.query = (PfnFfxQuery)dlsym(m, "ffxQuery");
    api.dispatch = (PfnFfxDispatch)dlsym(m, "ffxDispatch");
#endif
    api.ok = api.create && api.destroy && api.configure && api.query && api.dispatch;
    rt_log("ffx: %s %s", libname, api.ok ? "loaded" : "lacks the API entry points: upscaling off");
    loaded.store(api.ok, std::memory_order_release);
    return api.ok;
}

int file_available(std::atomic<int> &loaded, std::atomic<int> &present_flag, const char *libname) {
    int l = loaded.load(std::memory_order_acquire);
    if (l >= 0) return l;
    if (!libname) return 0;
    int present = present_flag.load(std::memory_order_relaxed);
    if (present < 0) {
        char path[PLAT_PATH_MAX];
        if (!library_path(libname, path, sizeof path)) {
            present = 0;
        } else {
#ifdef _WIN32
            DWORD attr = GetFileAttributesA(path);
            present = attr != INVALID_FILE_ATTRIBUTES && !(attr & FILE_ATTRIBUTE_DIRECTORY);
#else
            present = access(path, R_OK) == 0;
#endif
        }
        present_flag.store(present, std::memory_order_relaxed);
    }
    return present;
}

VkImageAspectFlags image_aspect(VkFormat fmt) {
    switch (fmt) {
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_X8_D24_UNORM_PACK32:
    case VK_FORMAT_D32_SFLOAT:
        return VK_IMAGE_ASPECT_DEPTH_BIT;
    case VK_FORMAT_D16_UNORM_S8_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
        return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    default:
        return VK_IMAGE_ASPECT_COLOR_BIT;
    }
}

void transition_inputs(VkCommandBuffer cb, const VkFfxImage *color, const VkFfxImage *depth, const VkFfxImage *mv,
                       int to_read) {
    const VkImageLayout src_layout = to_read ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    const VkImageLayout dst_layout = to_read ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL;
    const VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    const VkPipelineStageFlags dst_stage = to_read ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    const VkAccessFlags src_access = to_read ? VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT : VK_ACCESS_SHADER_READ_BIT;
    const VkAccessFlags dst_access = to_read ? VK_ACCESS_SHADER_READ_BIT : VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;

    VkImageMemoryBarrier b[3] = {};
    const VkFfxImage *imgs[3] = {color, depth, mv};
    for (int i = 0; i < 3; i++) {
        b[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b[i].srcAccessMask = src_access;
        b[i].dstAccessMask = dst_access;
        b[i].oldLayout = src_layout;
        b[i].newLayout = dst_layout;
        b[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b[i].image = imgs[i]->image;
        b[i].subresourceRange = {image_aspect(imgs[i]->format), 0, 1, 0, 1};
    }
    vkCmdPipelineBarrier(cb, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 3, b);
}

}  // namespace

extern "C" int vk_ffx_load(void) { return load_api(A, g_loaded, FFX_LIBRARY, "FSR 3.1"); }
extern "C" int vk_ffx_available(void) { return file_available(g_loaded, g_present, FFX_LIBRARY); }
extern "C" int vk_ffx_load_fsr4(void) {
    if (!FFX_FSR4_LIBRARY) return load_api(A4, g_fsr4_loaded, FFX_FSR4_LIBRARY, "FSR 4");
    const VkCtx *c = vk_ctx();
    if (!c || !c->device || !c->has_descriptor_buffer) {
        if (!A4.tried) {
            A4.tried = 1;
            rt_log("ffx: VK_EXT_descriptor_buffer is absent: FSR 4 off");
            g_fsr4_loaded.store(0, std::memory_order_release);
        }
        return 0;
    }
    return load_api(A4, g_fsr4_loaded, FFX_FSR4_LIBRARY, "FSR 4");
}
extern "C" int vk_ffx_fsr4_available(void) {
    if (!FFX_FSR4_LIBRARY) return 0;
    const VkCtx *c = vk_ctx();
    if (c && c->device && !c->has_descriptor_buffer) return 0;
    return file_available(g_fsr4_loaded, g_fsr4_present, FFX_FSR4_LIBRARY);
}

extern "C" int vk_ffx_framegen_available(void) { return vk_ffx_available(); }   // the same library, its API calls

// NX2NAVI_FSR4 (default 0, live): FSR 4 where it runs (vk_ffx4.cpp) and has not failed in a row, else FSR 3.1.
extern "C" int vk_ffx_fsr4(void) { return cfg_bool("NX2NAVI_FSR4") && vk_ffx4_available() && !vk_ffx4_failed(); }

extern "C" int vk_ffx_trim_idle(uint64_t recent) {
    int n = 0;
    for (Up &u : U)
        if (u.ctx && up_tick - u.used >= recent) destroy_up(&u, "room for an FSR 4 context"), n++;
    return n;
}

extern "C" int vk_ffx_upscale(VkCommandBuffer cb, const VkFfxImage *color, const VkFfxImage *depth, const VkFfxImage *mv,
                              const VkFfxImage *out, float jx, float jy, int reset, float dt_ms, const VkFfxCamera *cam,
                              int fsr4) {
    // Each upscaler keeps a history of its own: the frame after a change from one to the other starts it anew, or it
    // would blend this picture with whatever that upscaler last saw.
    static int last;   // 3 or 4: which one made the frame before
    uint64_t now = plat_now_ns();
    if (!fsr4) {
        if (!vk_ffx_load()) return 0;
        trim(now);
        vk_ffx4_trim(now);
        if (vk_ffx_fsr4()) {
            if (vk_ffx4_upscale(cb, color, depth, mv, out, jx, jy, reset || last != 4, dt_ms, cam)) {
                if (last != 4) rt_log("ffx: the upscaler is %s", vk_ffx4_name());
                last = 4;
                return 1;
            }
            // it failed and said so: FSR 3.1 makes this frame
        }
        if (last != 3) {
            if (last) rt_log("ffx: the upscaler is AMD FSR 3.1");
            reset = 1, last = 3;
        }
    } else {
        if (!vk_ffx_load_fsr4()) return 0;
        trim(now);
        vk_ffx4_trim(now);
        if (last != 4) {
            if (last) rt_log("ffx: the upscaler is AMD FSR 4");
            reset = 1, last = 4;
        }
    }
    Up *u = context_for(color->width, color->height, out->width, out->height, fsr4);
    if (!u) return 0;
    if (fsr4) transition_inputs(cb, color, depth, mv, 1);
    const uint32_t in_state = fsr4 ? FFX_API_RESOURCE_STATE_COMPUTE_READ : FFX_API_RESOURCE_STATE_UNORDERED_ACCESS;
    ffxDispatchDescUpscale d = {};
    d.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    d.commandList = cb;
    // FSR 3.1 reads inputs in GENERAL (unordered-access state); FSR 4 (.so) expects SHADER_READ_ONLY_OPTIMAL (compute read)
    d.color = image(color, in_state, 0);
    d.depth = image(depth, in_state, 0);
    d.motionVectors = image(mv, in_state, 0);
    d.output = image(out, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS, FFX_API_RESOURCE_USAGE_UAV);
    d.jitterOffset = {jx, jy};
    d.motionVectorScale = {1.0f, 1.0f};
    d.renderSize = {color->width, color->height};
    d.upscaleSize = {out->width, out->height};
    // The sharpness slider (0-100) drives FSR's own RCAS here, as it does for FSR 1
    int sharp = cfg_int("NX2NAVI_SHARPNESS");
    d.enableSharpening = sharp > 0;
    d.sharpness = sharp > 100 ? 1.0f : sharp / 100.0f;
    d.frameTimeDelta = dt_ms;
    d.preExposure = 1.0f;
    d.reset = reset != 0;
    // the frame's camera (vk_dlaa.c proj_planes)
    d.cameraNear = cam && cam->znear > 0 ? cam->znear : 0.1f;
    d.cameraFar = cam && cam->zfar > 0 ? cam->zfar : 1e8f;
    d.cameraFovAngleVertical = cam && cam->fov_y > 0 ? cam->fov_y : 1.0f;
    d.viewSpaceToMetersFactor = cam && cam->meters > 0 ? cam->meters : 1.0f;
    Api &api = fsr4 ? A4 : A;
    ffxReturnCode_t r = api.dispatch(&u->ctx, &d.header);
    if (fsr4) transition_inputs(cb, color, depth, mv, 0);
    if (r != FFX_API_RETURN_OK) {
        static int told;
        if (told++ < 8) rt_log("ffx: %s dispatch failed (%u)", fsr4 ? "FSR 4" : "FSR 3.1", (unsigned)r);
        return 0;
    }
    return 1;
}

extern "C" ffxReturnCode_t vk_ffx_create(ffxContext *ctx, ffxCreateContextDescHeader *desc) {
    return vk_ffx_load() ? create_counted(A, ctx, desc) : FFX_API_RETURN_ERROR;
}

// What the library is handed as ffxCreateBackendVKDesc.vkDeviceProcAddr: the device's functions, with the two
// pipeline creates replaced by ours (compute_pipelines, graphics_pipelines) unless NX2NAVI_FFX_PIPE_CACHE is 0.
extern "C" VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vk_ffx_proc_addr(VkDevice dev, const char *name) {
    if (name && pipe_cache_on()) {
        if (!strcmp(name, "vkCreateComputePipelines")) return (PFN_vkVoidFunction)compute_pipelines;
        if (!strcmp(name, "vkCreateGraphicsPipelines")) return (PFN_vkVoidFunction)graphics_pipelines;
    }
    return vkGetDeviceProcAddr(dev, name);
}

// --probe-ffx-cache (host.h): one FSR 3.1 upscaling context on the device, without a window or the game, and where
// its pipelines came from. The pipeline caches are those beside the executable, or NX2NAVI_PIPELINE_CACHE's, and are
// saved as after a run, so a second run of the probe shows what the first one's files give
// (tests/ffx_cache_check.py runs it twice on a folder of its own). Fails when the context is not created, or when
// the library made no pipeline through the runtime's creates although they are on.
extern "C" int vk_ffx_probe_cache(void) {
    if (!vk_init()) {
        printf("probe-ffx-cache: SKIP (no Vulkan device)\n");
        return 0;
    }
    if (!vk_ffx_load()) {
        printf("probe-ffx-cache: SKIP (no amd_fidelityfx_vk.dll beside the executable, or a build without FSR 3.1)\n");
        return 0;
    }
    vk_pipeline_push_fn();   // the first vk_pipeline_* call reads the pipeline caches
    VkForeignPipes p0, p1;
    vk_pipeline_foreign_counts(&p0);
    auto t0 = std::chrono::steady_clock::now();
    Up *u = context_for(1280, 720, 2560, 1440, 0);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    vk_pipeline_foreign_counts(&p1);
    unsigned long long made = p1.made - p0.made, hit = p1.from_cache - p0.from_cache, comp = p1.compiled - p0.compiled,
                       none = p1.uncached - p0.uncached;
    int on = pipe_cache_on(), fails = 0;
    printf("probe-ffx-cache: context %s in %.0f ms; pipelines made through the runtime's creates: %llu (from the "
           "pipeline caches %llu, compiled into the play cache %llu, through the one cache %llu, with no cache %llu); "
           "NX2NAVI_FFX_PIPE_CACHE %d\n", u ? "created" : "not created", ms, made, hit, comp, made - hit - comp - none,
           none, on);
    if (!u) fails++, printf("probe-ffx-cache: FAIL: no context\n");
    if (u && on && !made) fails++, printf("probe-ffx-cache: FAIL: the library created its pipelines past the runtime\n");
    if (u && !on && made) fails++, printf("probe-ffx-cache: FAIL: pipelines came through the runtime with the setting 0\n");
    vk_queue_lock(), vkDeviceWaitIdle(vk_ctx()->device), vk_queue_unlock();
    vk_ffx_release();
    vk_pipeline_save();
    printf("probe-ffx-cache: %s\n", fails ? "FAILED" : "passed");
    return fails ? 1 : 0;
}
extern "C" ffxReturnCode_t vk_ffx_destroy(ffxContext *ctx) { return A.ok ? A.destroy(ctx, nullptr) : FFX_API_RETURN_ERROR; }
extern "C" ffxReturnCode_t vk_ffx_configure(ffxContext *ctx, const ffxConfigureDescHeader *desc) {
    return A.ok ? A.configure(ctx, desc) : FFX_API_RETURN_ERROR;
}
extern "C" ffxReturnCode_t vk_ffx_query(ffxContext *ctx, ffxQueryDescHeader *desc) {
    return A.ok ? A.query(ctx, desc) : FFX_API_RETURN_ERROR;
}
extern "C" ffxReturnCode_t vk_ffx_dispatch(ffxContext *ctx, const ffxDispatchDescHeader *desc) {
    return A.ok ? A.dispatch(ctx, desc) : FFX_API_RETURN_ERROR;
}

extern "C" void vk_ffx_release(void) {
    vk_ffx4_release();
    for (Up &u : U)
        if (u.ctx) {
            Api &api = u.is_fsr4 ? A4 : A;
            if (api.ok) api.destroy(&u.ctx, nullptr);
            u = Up();
        }
}
