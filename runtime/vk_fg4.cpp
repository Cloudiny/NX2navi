// Frame generation through AMD's FSR SDK 2.4: amd_fidelityfx_loader_vk.dll (vk_ffx4.cpp loads it) with
// amd_fidelityfx_framegeneration_vk.dll next to the executable. vk_fg.cpp takes this path instead of the FSR 3.1
// library's where the setting asks for it while FSR 4 is the upscaler (vk_fg.cpp use4, NX2NAVI_FSR4_FG), and its
// functions here are called by vk_fg.cpp alone. The
// loader picks the frame generation and the swapchain for the card: the machine-learned FSR Frame Generation 4.x on
// Radeon RX 9000 cards with Windows 11, FSR Frame Generation 3.1.x on every other. The log names both versions.
//
// What happens each frame is what vk_fg.cpp does (the configuration, then the preparation from the frame's depth and
// motion vectors, then the generation from the swapchain's callback at present, the order FSR Frame Generation 4
// requires), in this SDK's layouts and with what it adds: the camera's position and axes in the preparation, which
// FSR Frame Generation 4 needs, and the version each context is made for.
//
// The swapchain's life differs from the FSR 3.1 library's:
// - A new size remakes the swapchain in place: its create function takes the swapchain's own handle, and the handle
//   stays. vk_fg4_destroy_swapchain therefore does nothing while the swapchain context lives.
// - Taking it down (vk_fg4_release) waits for its presents, destroys the frame generation context, then the swapchain
//   context, and calls its destroy function last, as the SDK asks.
#define VK_FFX_NO_API   // vk_ffx.h: without the API declarations of the other set of headers
#ifndef _WIN32
// AMD publishes the SDK's libraries for Windows only: in any other build this path is never available.
extern "C" {
#include "vk.h"
#include "vk_ffx.h"
#include "vk_fg.h"
}
extern "C" int vk_fg4_available(void) { return 0; }
extern "C" int vk_fg4_failed(void) { return 0; }
extern "C" const char *vk_fg4_name(void) { return ""; }
extern "C" int vk_fg4_active(void) { return 0; }
extern "C" int vk_fg4_generating(void) { return 0; }
extern "C" void vk_fg4_queues(VkQueue, uint32_t, VkQueue, uint32_t) {}
extern "C" VkResult vk_fg4_create_swapchain(const VkSwapchainCreateInfoKHR *, VkSwapchainKHR *) {
    return VK_ERROR_FEATURE_NOT_PRESENT;
}
extern "C" void vk_fg4_destroy_swapchain(VkSwapchainKHR) {}
extern "C" VkResult vk_fg4_get_images(VkSwapchainKHR, uint32_t *, VkImage *) { return VK_ERROR_FEATURE_NOT_PRESENT; }
extern "C" VkResult vk_fg4_acquire(VkSwapchainKHR, uint64_t, VkSemaphore, uint32_t *) { return VK_ERROR_FEATURE_NOT_PRESENT; }
extern "C" VkResult vk_fg4_present(const VkPresentInfoKHR *) { return VK_ERROR_FEATURE_NOT_PRESENT; }
extern "C" uint64_t vk_fg4_presents(VkSwapchainKHR) { return 0; }
extern "C" PFN_vkSetHdrMetadataEXT vk_fg4_hdr_metadata_fn(void) { return nullptr; }
extern "C" void vk_fg4_frame(VkCommandBuffer, VkSwapchainKHR, VkFormat, VkExtent2D, VkImage, const int *) {}
extern "C" void vk_fg4_release(void) {}
#else
#include <windows.h>
#include <mutex>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

#include "third_party/ffx_sdk/framegeneration/include/ffx_framegeneration.h"
#include "third_party/ffx_sdk/framegeneration/include/vk/ffx_api_framegeneration_vk.h"
#include "vk_ffx4_api.h"

extern "C" {
#include "cfg.h"
#include "host.h"
#include "platform/platform.h"
#include "vk.h"
#include "vk_dlaa.h"
#include "vk_ffx.h"
#include "vk_fg.h"
#include "ui/menu.h"
}

namespace {

struct {
    VkQueue present = VK_NULL_HANDLE, acquire = VK_NULL_HANDLE;
    uint32_t present_family = 0, acquire_family = 0;
    ffxContext sc = nullptr;   // the swapchain context
    ffxContext fg = nullptr;   // the frame generation context
    VkSwapchainKHR handle = VK_NULL_HANDLE;   // the swapchain the context made: remade in place, destroyed after the context
    VkExtent2D fg_ext = {0, 0};
    VkExtent2D fg_max = {0, 0};   // the context's largest render size
    VkFormat fg_fmt = VK_FORMAT_UNDEFINED;
    ffxQueryDescSwapchainReplacementFunctionsVK fn = {};
    // what the contexts were made from: the API asks that it stays until the context is destroyed
    ffxCreateContextDescFrameGenerationSwapChainVK sc_desc = {};
    ffxCreateContextDescFrameGenerationSwapChainVersionVK sc_ver = {};
    ffxOverrideVersion sc_pin = {};
    ffxCreateContextDescFrameGeneration fg_desc = {};
    ffxCreateBackendVKDesc fg_be = {};
    ffxCreateContextDescFrameGenerationHudless fg_hl = {};
    ffxCreateContextDescFrameGenerationVersion fg_ver = {};
    ffxOverrideVersion fg_pin = {};
    uint64_t frame = 0;
    int failed = 0;
    uint64_t generated = 0, paused = 0, no_inputs = 0, hudless = 0, no_hudless = 0;
    int on = 0;   // the last configured frame had generation on (vk_fg4_generating)
} F;

std::mutex g_lock;       // the probe: the menu asks from its thread, the present from its own
int g_avail = -1;        // -1 not asked yet; 0 this path does not run here; 1 it does
char g_sc_list[96];      // the swapchain versions the loader listed, for the log
char g_fg_name[24];      // the frame generation version a context gets: the one asked for, else the first listed
uint64_t g_fg_pin, g_sc_pin;   // NX2NAVI_FG4_VERSION, NX2NAVI_FG4_SWAPCHAIN (tests): the version asked for by its id; 0 the loader's choice

VkResult submit_game(uint32_t n, const VkSubmitInfo *si, VkFence fence) {
    vk_queue_lock();   // the game's queue is shared with the recorder and the present path
    VkResult r = vkQueueSubmit(vk_ctx()->gfx_queue, n, si, fence);
    vk_queue_unlock();
    return r;
}

ffxReturnCode_t generate(ffxDispatchDescFrameGeneration *d, void *user) {
    F.generated++;
    return vk_ffx4_api_dispatch((ffxContext *)user, &d->header);
}

FfxApiResource image(const VkFfxImage *i) {
    VkImageCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = i->format;
    ci.extent = {i->width, i->height, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    return ffxApiGetResourceVK((void *)i->image, ffxApiGetImageResourceDescriptionVK(i->image, ci, 0),
                               FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
}

// After this the FSR 3.1 library generates the frames: vk_fg.cpp's vk_fg_wanted stops wanting this swapchain, the
// present code takes it down as it does for a switch in the menu, and the next one is made by the other path.
void fail(const char *what, unsigned r) {
    rt_log("fg4: %s failed (%u): the FSR SDK's frame generation is off for this run, FSR 3.1's takes its place", what, r);
    F.failed = 1;
}

// The list the loader gives for one kind of context, as text, and the id of the version a test asks for by the start
// of its name (0: none asked, or not listed).
int list_versions(uint64_t type, const char *want, char *out, size_t size, uint64_t *pin) {
    uint64_t ids[8] = {};
    const char *names[8] = {};
    int n = vk_ffx4_api_versions(type, ids, names, 8);
    size_t len = 0;
    out[0] = 0;
    for (int i = 0; i < n; i++) {
        const char *nm = names[i] ? names[i] : "?";
        len += (size_t)snprintf(out + len, size - len, "%s%s", i ? ", " : "", nm);
        if (len >= size) len = size - 1;
        if (want && *want && !*pin && !strncmp(nm, want, strlen(want))) *pin = ids[i];
    }
    return n;
}

// The version a context runs, as its provider names it ("3.1.6"); "?" when it does not say.
const char *provider(ffxContext *ctx) {
    ffxQueryGetProviderVersion v = {};
    v.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
    vk_ffx4_api_query(ctx, &v.header);
    return v.versionName ? v.versionName : "?";
}

// An HDR10 swapchain (10-bit PQ or 16-bit float): frame generation is told its inputs are HDR.
bool hdr_format(VkFormat f) {
    return f == VK_FORMAT_A2B10G10R10_UNORM_PACK32 || f == VK_FORMAT_A2R10G10B10_UNORM_PACK32 ||
           f == VK_FORMAT_R16G16B16A16_SFLOAT;
}

// The swapchain's own presents are done before it is remade or taken down.
void wait_presents(void) {
    ffxDispatchDescFrameGenerationSwapChainWaitForPresentsVK w = {};
    w.header.type = FFX_API_DISPATCH_DESC_TYPE_FGSWAPCHAIN_WAIT_FOR_PRESENTS_VK;
    vk_ffx4_api_dispatch(&F.sc, &w.header);
}

}  // namespace

// The loader and its frame generation library are beside the executable and list a frame generation and a swapchain
// for the device. Asked once, with the device; 0 before it exists.
extern "C" int vk_fg4_available(void) {
    int a = __atomic_load_n(&g_avail, __ATOMIC_ACQUIRE);   // asked several times a frame
    if (a >= 0) return a;
    std::lock_guard<std::mutex> hold(g_lock);
    if (g_avail >= 0) return g_avail;
    const VkCtx *c = vk_ctx();
    if (!c || !c->device) return 0;
    if (!vk_ffx4_api_file("amd_fidelityfx_framegeneration_vk.dll")) {   // not shipped beside this build
        __atomic_store_n(&g_avail, 0, __ATOMIC_RELEASE);
        return 0;
    }
    char fg[96];
    char *sc = g_sc_list;
    int nf = list_versions(FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION, cfg_get("NX2NAVI_FG4_VERSION"), fg, sizeof fg, &g_fg_pin);
    int ns = list_versions(FFX_API_CREATE_CONTEXT_DESC_TYPE_FGSWAPCHAIN_VK, cfg_get("NX2NAVI_FG4_SWAPCHAIN"), sc, sizeof g_sc_list,
                           &g_sc_pin);
    a = nf > 0 && ns > 0;
    // the menu's name for it: the first of the list, which the loader takes when none is asked for (a test's version
    // changes the context, not this name); a version the driver brings carries a star
    snprintf(g_fg_name, sizeof g_fg_name, "%.*s", (int)strcspn(fg, ",*"), fg);
    for (size_t k = strlen(g_fg_name); k && g_fg_name[k - 1] == ' '; k--) g_fg_name[k - 1] = 0;
    __atomic_store_n(&g_avail, a, __ATOMIC_RELEASE);
    if (a) rt_log("fg4: the FSR SDK lists for this device: frame generation %s; its swapchain %s", fg, sc);
    else rt_log("fg4: the FSR SDK lists no frame generation for this device: FSR 3.1's library generates the frames");
    return a;
}

extern "C" int vk_fg4_failed(void) { return F.failed; }

extern "C" const char *vk_fg4_name(void) { return g_fg_name; }

extern "C" void vk_fg4_queues(VkQueue present, uint32_t present_family, VkQueue acquire, uint32_t acquire_family) {
    F.present = present, F.present_family = present_family;
    F.acquire = acquire, F.acquire_family = acquire_family;
}

// The SDK's swapchain is in place: the window's swapchain handle is its object, and every swapchain call has to go
// through its functions, after a failure too, until the present code has taken it down.
extern "C" int vk_fg4_active(void) { return F.sc != nullptr; }

extern "C" int vk_fg4_generating(void) { return vk_fg4_active() && !F.failed && __atomic_load_n(&F.on, __ATOMIC_RELAXED); }

extern "C" VkResult vk_fg4_create_swapchain(const VkSwapchainCreateInfoKHR *ci, VkSwapchainKHR *sc) {
    const VkCtx *c = vk_ctx();
    if (F.sc) {   // a new size: the swapchain is remade in place and keeps its handle
        wait_presents();
        VkSwapchainCreateInfoKHR in = *ci;
        in.oldSwapchain = VK_NULL_HANDLE;
        VkSwapchainKHR h = F.handle;
        VkResult r = F.fn.pOutCreateSwapchainFFXAPI(c->device, &in, nullptr, &h, (void *)F.handle);
        if (r == VK_SUCCESS) *sc = F.handle = h;
        return r;
    }
    if (F.failed || !F.present || !vk_fg4_available()) return VK_ERROR_FEATURE_NOT_PRESENT;
    F.handle = VK_NULL_HANDLE;
    F.sc_desc = {};
    F.sc_desc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FGSWAPCHAIN_VK;
    F.sc_desc.physicalDevice = c->phys;
    F.sc_desc.device = c->device;
    F.sc_desc.swapchain = &F.handle;
    F.sc_desc.createInfo = *ci;
    F.sc_desc.createInfo.oldSwapchain = VK_NULL_HANDLE;
    F.sc_desc.gameQueue = {c->gfx_queue, c->gfx_family, submit_game};
    F.sc_desc.asyncComputeQueue = {VK_NULL_HANDLE, c->gfx_family, nullptr};
    F.sc_desc.presentQueue = {F.present, F.present_family, nullptr};
    F.sc_desc.imageAcquireQueue = {F.acquire, F.acquire_family, nullptr};
    // the version this code was built against, which the SDK asks for with every context, and a test's version
    F.sc_ver = {};
    F.sc_ver.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_VERSION_VK;
    F.sc_ver.version = FFX_FRAMEGENERATION_SWAPCHAIN_VK_VERSION;
    F.sc_desc.header.pNext = &F.sc_ver.header;
    if (g_sc_pin) {
        F.sc_pin = {};
        F.sc_pin.header.type = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
        F.sc_pin.versionId = g_sc_pin;
        F.sc_ver.header.pNext = &F.sc_pin.header;
    }
    ffxReturnCode_t r = vk_ffx4_api_create(&F.sc, &F.sc_desc.header);
    if (r != FFX_API_RETURN_OK || !F.sc) {
        F.sc = nullptr, F.handle = VK_NULL_HANDLE;
        fail("the frame generation swapchain", r);
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    F.fn = {};
    F.fn.header.type = FFX_API_QUERY_DESC_TYPE_FGSWAPCHAIN_FUNCTIONS_VK;
    // The swapchain the driver brings (listed with a star, the loader's choice where it is listed) hands out no destroy
    // function: its context's destruction is the swapchain's too (vk_fg4_release).
    if (vk_ffx4_api_query(&F.sc, &F.fn.header) != FFX_API_RETURN_OK || !F.fn.pOutQueuePresentKHR || !F.fn.pOutCreateSwapchainFFXAPI ||
        !F.fn.pOutGetSwapchainImagesKHR || !F.fn.pOutAcquireNextImageKHR || !F.handle) {
        VkSwapchainKHR h = F.handle;
        PFN_vkDestroySwapchainFFXAPI destroy = F.fn.pOutDestroySwapchainFFXAPI;
        vk_ffx4_api_destroy(&F.sc), F.sc = nullptr, F.handle = VK_NULL_HANDLE;
        if (h && destroy) destroy(c->device, h, nullptr, nullptr);   // nothing was handed out yet
        fail("the swapchain functions", 0);
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    *sc = F.handle;
    rt_log("fg4: AMD FSR frame generation swapchain, %ux%u (%s%s)", ci->imageExtent.width, ci->imageExtent.height,
           g_sc_pin ? "the version a test asked for, of " : "the loader's choice of ", g_sc_list);
    return VK_SUCCESS;
}

// Nothing while the swapchain context lives: a new size remakes the swapchain in place (vk_fg4_create_swapchain), and
// vk_fg4_release destroys it after its context.
extern "C" void vk_fg4_destroy_swapchain(VkSwapchainKHR sc) { (void)sc; }

extern "C" VkResult vk_fg4_get_images(VkSwapchainKHR sc, uint32_t *n, VkImage *img) {
    return F.fn.pOutGetSwapchainImagesKHR(vk_ctx()->device, sc, n, img);
}

extern "C" VkResult vk_fg4_acquire(VkSwapchainKHR sc, uint64_t timeout, VkSemaphore sem, uint32_t *idx) {
    return F.fn.pOutAcquireNextImageKHR(vk_ctx()->device, sc, timeout, sem, VK_NULL_HANDLE, idx);
}

extern "C" VkResult vk_fg4_present(const VkPresentInfoKHR *pi) { return F.fn.pOutQueuePresentKHR(vk_ctx()->gfx_queue, pi); }

extern "C" uint64_t vk_fg4_presents(VkSwapchainKHR sc) {
    return vk_fg4_active() && F.fn.pOutGetLastPresentCountFFXAPI && sc ? F.fn.pOutGetLastPresentCountFFXAPI(sc) : 0;
}

extern "C" PFN_vkSetHdrMetadataEXT vk_fg4_hdr_metadata_fn(void) { return vk_fg4_active() ? F.fn.pOutSetHdrMetadataEXT : nullptr; }

extern "C" void vk_fg4_frame(VkCommandBuffer cb, VkSwapchainKHR sc, VkFormat fmt, VkExtent2D ext, VkImage hudless, const int pic[4]) {
    if (!vk_fg4_active() || F.failed) return;
    vk_dlaa_hudless_wanted = 1;   // the upscaling pass keeps a copy of the picture before the HUD (vk_dlaa.c hudless)
    const VkCtx *c = vk_ctx();
    VkFfxImage depth, mv;
    float jx = 0, jy = 0;
    int reset = 0;
    int have = vk_dlaa_fg_inputs(&depth, &mv, &jx, &jy, &reset);
    // The context's largest render size covers the depth and motion vectors it is given: the picture's size, which
    // the scene never exceeds (vk_fg.cpp vk_fg_frame has the history of this rule).
    unsigned pw, ph;
    wide_picture(&pw, &ph);
    VkExtent2D max_render = {ext.width > pw ? ext.width : pw, ext.height > ph ? ext.height : ph};
    if (F.fg && (F.fg_ext.width != ext.width || F.fg_ext.height != ext.height || F.fg_fmt != fmt)) {
        vk_queue_lock(), vkDeviceWaitIdle(c->device), vk_queue_unlock();   // a new window size: the context is made for its display size
        vk_ffx4_api_destroy(&F.fg);
        F.fg = nullptr;
    }
    if (!F.fg) {
        F.fg_be = {};
        F.fg_be.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
        F.fg_be.deviceContext = {c->device, c->phys, vkGetDeviceProcAddr};
        // the HUD-less picture has the swapchain's format (gfx_present.c hudless_record)
        F.fg_hl = {};
        F.fg_hl.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION_HUDLESS;
        F.fg_hl.header.pNext = &F.fg_be.header;
        F.fg_hl.hudlessBackBufferFormat = ffxApiGetSurfaceFormatVK(fmt);
        F.fg_ver = {};
        F.fg_ver.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION_VERSION;
        F.fg_ver.header.pNext = &F.fg_hl.header;
        F.fg_ver.version = FFX_FRAMEGENERATION_VERSION;
        if (g_fg_pin) {
            F.fg_pin = {};
            F.fg_pin.header.type = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
            F.fg_pin.versionId = g_fg_pin;
            F.fg_be.header.pNext = &F.fg_pin.header;
        }
        F.fg_desc = {};
        F.fg_desc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
        F.fg_desc.header.pNext = &F.fg_ver.header;
        F.fg_desc.flags = FFX_FRAMEGENERATION_ENABLE_DEPTH_INFINITE | (hdr_format(fmt) ? FFX_FRAMEGENERATION_ENABLE_HIGH_DYNAMIC_RANGE : 0);
        F.fg_desc.displaySize = {ext.width, ext.height};
        F.fg_desc.maxRenderSize = {max_render.width, max_render.height};
        F.fg_desc.backBufferFormat = ffxApiGetSurfaceFormatVK(fmt);
        uint64_t t0 = plat_now_ns();
        ffxReturnCode_t r = vk_ffx4_api_create(&F.fg, &F.fg_desc.header);
        if (r != FFX_API_RETURN_OK || !F.fg) return F.fg = nullptr, fail("the frame generation context", r);
        F.fg_ext = ext, F.fg_fmt = fmt, F.fg_max = max_render;
        FfxApiEffectMemoryUsage mem = {};
        ffxQueryDescFrameGenerationGetGPUMemoryUsage mq = {};
        mq.header.type = FFX_API_QUERY_DESC_TYPE_FRAMEGENERATION_GPU_MEMORY_USAGE;
        mq.gpuMemoryUsageFrameGeneration = &mem;
        vk_ffx4_api_query(&F.fg, &mq.header);
        rt_log("fg4: AMD FSR frame generation %s for %ux%u%s, render size up to %ux%u, made in %.0f ms, %.0f MiB of video memory",
               provider(&F.fg), ext.width, ext.height, hdr_format(fmt) ? " (HDR)" : "", max_render.width, max_render.height,
               (double)(plat_now_ns() - t0) / 1e6, (double)mem.totalUsageInBytes / 1048576.0);
    }
    if (have && (depth.width > F.fg_max.width || depth.height > F.fg_max.height)) {
        static int told;
        if (!told++)
            rt_log("fg4: a %ux%u scene is larger than the context's %ux%u: those frames are not generated", depth.width,
                   depth.height, F.fg_max.width, F.fg_max.height);
        have = 0;
    }
    // Only with the picture without the HUD: generated from the presented picture alone, the HUD would be warped along
    // with the scene behind it and trail when the camera turns. A frame without that picture is shown as it is.
    int on = have && hudless && !menu_is_open();
    __atomic_store_n(&F.on, on, __ATOMIC_RELAXED);
    if (!have) F.no_inputs++;
    else if (!hudless) F.no_hudless++;
    else if (!on) F.paused++;
    F.frame++;
    ffxConfigureDescFrameGeneration cfg = {};
    cfg.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
    cfg.swapChain = (void *)sc;
    cfg.frameGenerationCallback = generate;
    cfg.frameGenerationCallbackUserContext = &F.fg;
    cfg.frameGenerationEnabled = on != 0;
    cfg.allowAsyncWorkloads = false;
    // the picture's rectangle: with black bars beside or below it (a window of another shape) a generation over the
    // whole swapchain warped the picture's edges into the bars
    cfg.generationRect = pic && pic[2] > 0 && pic[3] > 0 ? FfxApiRect2D{pic[0], pic[1], pic[2], pic[3]}
                                                         : FfxApiRect2D{0, 0, (int32_t)ext.width, (int32_t)ext.height};
    if (on) {
        VkFfxImage h = {hudless, fmt, ext.width, ext.height, 1};
        cfg.HUDLessColor = image(&h);
        F.hudless++;
    }
    cfg.frameID = F.frame;
    ffxReturnCode_t r = vk_ffx4_api_configure(&F.fg, &cfg.header);
    if (r != FFX_API_RETURN_OK) return fail("frame generation configuration", r);
    if (!on) return;
    ffxDispatchDescFrameGenerationPrepareV2 p = {};
    p.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE_V2;
    p.frameID = F.frame;
    p.commandList = cb;
    p.renderSize = {depth.width, depth.height};
    p.jitterOffset = {jx, jy};
    p.motionVectorScale = {1.0f, 1.0f};   // render pixels, current to previous (vk_dlaa.c mv_pass), as FSR takes them
    static uint64_t last_ns;
    uint64_t now = plat_now_ns();
    double dt = last_ns ? (now - last_ns) / 1e6 : 16.67;
    last_ns = now;
    p.frameTimeDelta = dt > 0 && dt < 100 ? (float)dt : 16.67f;
    p.reset = reset != 0;
    p.cameraNear = 0.1f;   // Ocarina's projection: near 0.1, far at infinity (vk_ffx.cpp)
    p.cameraFar = 1e8f;
    p.cameraFovAngleVertical = 1.0f;
    // The game's world is taken to be in metres (its near plane is at 0.1). With 0.01 here, the value vk_fg.cpp gives
    // the FSR 3.1 library, frame generation 3.1.6 drew grey translucent shapes into the sky of its frames while the
    // camera turned, and with 0 it smeared them; with 1 its frames were clean in the same camera sweep.
    p.viewSpaceToMetersFactor = 1.0f;
    p.depth = image(&depth);
    p.motionVectors = image(&mv);
    // the camera in the world: the game's own, or one at the origin looking down -z on a frame without it
    VkDlaaCamera cam;
    if (vk_dlaa_fg_camera(&cam)) {
        memcpy(p.cameraPosition, cam.pos, sizeof p.cameraPosition), memcpy(p.cameraUp, cam.up, sizeof p.cameraUp);
        memcpy(p.cameraRight, cam.right, sizeof p.cameraRight), memcpy(p.cameraForward, cam.fwd, sizeof p.cameraForward);
        if (cam.fov_y > 0) p.cameraFovAngleVertical = cam.fov_y;
    } else {
        p.cameraUp[1] = 1.0f, p.cameraRight[0] = 1.0f, p.cameraForward[2] = -1.0f;
    }
    r = vk_ffx4_api_dispatch(&F.fg, &p.header);
    if (r != FFX_API_RETURN_OK) return fail("frame generation preparation", r);
    if (F.frame == 2 || F.frame % 3600 == 0)
        rt_log("fg4: %llu frames presented, %llu generated (%llu with the HUD kept out); paused %llu (menu), %llu without depth "
               "and motion, %llu without the HUD-less picture", (unsigned long long)F.frame, (unsigned long long)F.generated,
               (unsigned long long)F.hudless, (unsigned long long)F.paused, (unsigned long long)F.no_inputs,
               (unsigned long long)F.no_hudless);
}

extern "C" void vk_fg4_release(void) {
    __atomic_store_n(&F.on, 0, __ATOMIC_RELAXED);
    if (!F.sc && !F.fg) return;
    if (F.sc) wait_presents();
    if (F.fg) vk_ffx4_api_destroy(&F.fg), F.fg = nullptr;
    if (F.sc) {
        VkSwapchainKHR h = F.handle;
        vk_ffx4_api_destroy(&F.sc), F.sc = nullptr, F.handle = VK_NULL_HANDLE;
        if (h && F.fn.pOutDestroySwapchainFFXAPI) F.fn.pOutDestroySwapchainFFXAPI(vk_ctx()->device, h, nullptr, nullptr);
    }
    F.fn = {};
}
#endif   // _WIN32
