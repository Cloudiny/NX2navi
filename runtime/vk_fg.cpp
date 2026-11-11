// Frame generation with AMD FSR 3.1 (NX2NAVI_FRAMEGEN = fsr3; applies after a restart): AMD's frame generation
// swapchain (amd_fidelityfx_vk.dll, vk_ffx.cpp loads it) takes the place of gfx_present.c's swapchain. Every frame
// the game presents, the swapchain presents a generated frame between it and the one before, on its own present
// queue and pacing threads, so the screen shows twice the game's frame rate.
//
// The generated frame is built from the presented picture and the scene's depth and camera motion vectors, which
// vk_dlaa.c makes for DLSS and FSR 3.1 upscaling: frame generation needs one of those on (DLAA or Native AA included).
// The HUD is interpolated with the picture (it hardly moves); the menu is not, since generation pauses while it is
// open. The device creates two more queues of the graphics family for it (vk_device.c, vk_fg_queue_count).
#ifdef _WIN32
#include <windows.h>
#endif
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <wchar.h>

#include "third_party/ffx_api/ffx_api.h"
#include "third_party/ffx_api/ffx_framegeneration.h"
#include "third_party/ffx_api/vk/ffx_api_vk.h"

extern "C" {
#include "cfg.h"
#include "host.h"
#include "platform/platform.h"
#include "vk.h"
#include "vk_dlaa.h"
#include "vk_dlssg.h"
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
    VkExtent2D fg_ext = {0, 0};
    VkExtent2D fg_max = {0, 0};   // the context's largest render size
    VkFormat fg_fmt = VK_FORMAT_UNDEFINED;
    ffxQueryDescSwapchainReplacementFunctionsVK fn = {};
    uint64_t frame = 0;
    int failed = 0;
    uint64_t generated = 0, paused = 0, no_inputs = 0, hudless = 0, no_hudless = 0;
    int on = 0;   // the last configured frame had generation on (vk_fg_generating)
} F;

VkResult submit_game(uint32_t n, const VkSubmitInfo *si, VkFence fence) {
    vk_queue_lock();   // the game's queue is shared with the recorder and the present path
    VkResult r = vkQueueSubmit(vk_ctx()->gfx_queue, n, si, fence);
    vk_queue_unlock();
    return r;
}

ffxReturnCode_t generate(ffxDispatchDescFrameGeneration *d, void *user) {
    F.generated++;
    return vk_ffx_dispatch((ffxContext *)user, &d->header);
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

void fail(const char *what, unsigned r) {
    rt_log("fg: %s failed (%u): frame generation off for this run", what, r);
    F.failed = 1;
}

}  // namespace

// Follows the setting: gfx_present.c's window_settings_live remakes the swapchain when this stops matching vk_fg_active.
// It needs DLSS's or FSR 3.1's motion vectors, so it also needs one of them on, and the FSR 3.1 library: a build
// without it keeps the plain swapchain whatever the setting says, with one line in the log. Not on a device made through
// Streamline's proxies (NVIDIA's frame generation set at the start, vk_dlssg_dpa): AMD's swapchain there crashed in
// NVIDIA's driver when the menu switched from NVIDIA's frame generation to AMD's. That switch waits for a restart,
// as the menu says (menu.cpp row_framegen). Not after a failure (fail): the present code then takes AMD's swapchain
// down as it does for a switch in the menu, and a plain one follows.
//
// Two libraries can make the swapchain: the FSR 3.1 library, here, or AMD's FSR SDK 2.4 (vk_fg4.cpp) where the
// setting asks for it while FSR 4 is the upscaler (use4). The swapchain in place is wanted only while its library is
// still the one to use: a change of that setting or of the upscaler between FSR 3.1 and FSR 4, or a failure of the SDK
// 2.4 path, takes it down as switching frame generation off does, and the next present makes it with the other library.
static int use4(void) {   // NX2NAVI_FSR4_FG (default 0, live): 1 takes the SDK 2.4's frame generation with FSR 4
    return cfg_bool("NX2NAVI_FSR4_FG") && vk_ffx_fsr4() && !vk_fg4_failed() && vk_fg4_available();
}

extern "C" int vk_fg_wanted(void) {
    if (strcmp(cfg_value("NX2NAVI_FRAMEGEN"), "fsr3") || !cfg_bool("NX2NAVI_DLAA") || !vk_dlaa_wanted() || vk_dlssg_dpa())
        return 0;
    int four = use4();
    if (vk_fg4_active()) return four;
    if (F.sc) return !F.failed && !four;
    if (four) return 1;
    if (F.failed) return 0;
    if (vk_ffx_framegen_available()) return 1;
    static int told;
    if (!__atomic_exchange_n(&told, 1, __ATOMIC_RELAXED))
        rt_log("fg: AMD FSR 3.1 frame generation is set, but FSR 3.1 is not available here: off");
    return 0;
}

// Always the three queues: the setting can be switched on during a run, and idle queues cost nothing.
extern "C" uint32_t vk_fg_queue_count(void) { return 3; }

extern "C" void vk_fg_queues(VkQueue present, uint32_t present_family, VkQueue acquire, uint32_t acquire_family) {
    F.present = present, F.present_family = present_family;
    F.acquire = acquire, F.acquire_family = acquire_family;
    vk_fg4_queues(present, present_family, acquire, acquire_family);
}

// AMD's swapchain is in place: the window's swapchain handle is AMD's object, and every swapchain call has to go
// through AMD's functions. That holds after a failure too, until the present code has replaced the swapchain
// (vk_fg_wanted): the plain Vulkan calls must never get that handle.
extern "C" int vk_fg_active(void) { return F.sc != nullptr || vk_fg4_active(); }

extern "C" int vk_fg_generating(void) {
    if (vk_fg4_active()) return vk_fg4_generating();
    return F.sc && !F.failed && __atomic_load_n(&F.on, __ATOMIC_RELAXED);
}

extern "C" VkResult vk_fg_create_swapchain(const VkSwapchainCreateInfoKHR *ci, VkSwapchainKHR *sc) {
    if (vk_fg4_active()) return vk_fg4_create_swapchain(ci, sc);
    if (!F.sc && use4()) {
        VkResult r = vk_fg4_create_swapchain(ci, sc);
        if (r == VK_SUCCESS || vk_fg4_active()) return r;
        // it failed before its swapchain existed and said so (use4 is 0 from here on): the FSR 3.1 library's below
    }
    if (F.sc) return F.fn.pOutCreateSwapchainFFXAPI(vk_ctx()->device, ci, nullptr, sc, F.sc);
    if (!vk_fg_wanted() || F.failed || !F.present || !vk_ffx_load()) return VK_ERROR_FEATURE_NOT_PRESENT;
    const VkCtx *c = vk_ctx();
    ffxCreateContextDescFrameGenerationSwapChainVK d = {};
    d.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FGSWAPCHAIN_VK;
    d.physicalDevice = c->phys;
    d.device = c->device;
    VkSwapchainKHR none = VK_NULL_HANDLE;
    d.swapchain = &none;
    d.createInfo = *ci;
    d.createInfo.oldSwapchain = VK_NULL_HANDLE;
    d.gameQueue = {c->gfx_queue, c->gfx_family, submit_game};
    d.asyncComputeQueue = {VK_NULL_HANDLE, c->gfx_family, nullptr};
    d.presentQueue = {F.present, F.present_family, nullptr};
    d.imageAcquireQueue = {F.acquire, F.acquire_family, nullptr};
    ffxReturnCode_t r = vk_ffx_create(&F.sc, &d.header);
    if (r != FFX_API_RETURN_OK) {
        F.sc = nullptr;
        fail("the frame generation swapchain", r);
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    F.fn.header.type = FFX_API_QUERY_DESC_TYPE_FGSWAPCHAIN_FUNCTIONS_VK;
    if (vk_ffx_query(&F.sc, &F.fn.header) != FFX_API_RETURN_OK || !F.fn.pOutQueuePresentKHR) {
        vk_ffx_destroy(&F.sc), F.sc = nullptr;   // with it goes the swapchain it made: nothing was handed out yet
        fail("the swapchain functions", 0);
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    *sc = none;
    rt_log("fg: AMD FSR 3.1 frame generation swapchain, %ux%u", ci->imageExtent.width, ci->imageExtent.height);
    return VK_SUCCESS;
}

extern "C" void vk_fg_destroy_swapchain(VkSwapchainKHR sc) {
    if (vk_fg4_active()) return vk_fg4_destroy_swapchain(sc);
    if (F.sc && sc) F.fn.pOutDestroySwapchainFFXAPI(vk_ctx()->device, sc, nullptr, F.sc);
}

extern "C" VkResult vk_fg_get_images(VkSwapchainKHR sc, uint32_t *n, VkImage *img) {
    if (vk_fg4_active()) return vk_fg4_get_images(sc, n, img);
    return F.fn.pOutGetSwapchainImagesKHR(vk_ctx()->device, sc, n, img);
}

extern "C" VkResult vk_fg_acquire(VkSwapchainKHR sc, uint64_t timeout, VkSemaphore sem, uint32_t *idx) {
    if (vk_fg4_active()) return vk_fg4_acquire(sc, timeout, sem, idx);
    return F.fn.pOutAcquireNextImageKHR(vk_ctx()->device, sc, timeout, sem, VK_NULL_HANDLE, idx);
}

extern "C" VkResult vk_fg_present(const VkPresentInfoKHR *pi) {
    if (vk_fg4_active()) return vk_fg4_present(pi);
    return F.fn.pOutQueuePresentKHR(vk_ctx()->gfx_queue, pi);
}

extern "C" uint64_t vk_fg_presents(VkSwapchainKHR sc) {
    if (vk_fg4_active()) return vk_fg4_presents(sc);
    return F.sc && F.fn.pOutGetLastPresentCountFFXAPI && sc ? F.fn.pOutGetLastPresentCountFFXAPI(sc) : 0;
}

// The HDR10 metadata goes to AMD's swapchain through its own call (vk_hdr.c vk_hdr_metadata).
extern "C" PFN_vkSetHdrMetadataEXT vk_fg_hdr_metadata_fn(void) {
    if (vk_fg4_active()) return vk_fg4_hdr_metadata_fn();
    return F.sc ? F.fn.pOutSetHdrMetadataEXT : nullptr;
}

// An HDR10 swapchain (10-bit PQ or 16-bit float): frame generation is told its inputs are HDR.
static bool hdr_format(VkFormat f) {
    return f == VK_FORMAT_A2B10G10R10_UNORM_PACK32 || f == VK_FORMAT_A2R10G10B10_UNORM_PACK32 ||
           f == VK_FORMAT_R16G16B16A16_SFLOAT;
}

extern "C" void vk_fg_frame(VkCommandBuffer cb, VkSwapchainKHR sc, VkFormat fmt, VkExtent2D ext, VkImage hudless,
                            VkImageView hudless_view, const int pic[4]) {
    (void)hudless_view;
    if (vk_fg4_active()) return vk_fg4_frame(cb, sc, fmt, ext, hudless, pic);
    if (!F.sc || F.failed) return;
    vk_dlaa_hudless_wanted = 1;   // the upscaling pass keeps a copy of the picture before the HUD (vk_dlaa.c hudless)
    const VkCtx *c = vk_ctx();
    VkFfxImage depth, mv;
    float jx = 0, jy = 0;
    int reset = 0;
    int have = vk_dlaa_fg_inputs(&depth, &mv, &jx, &jy, &reset);
    // The context's largest render size covers the depth and motion vectors it is given. They have the scene's size,
    // which DLAA and the widescreen pictures make larger than the window, and a context made for the window's size alone
    // wrote past its own resources: a crash in amd_fidelityfx_vk.dll, device faults, striped frames. The scene is never
    // larger than the picture (widescreen.c wide_picture), so the context is made for that once; remaking it while the
    // swapchain still generated from the old one hung the present. A frame with a larger depth anyway is not generated.
    unsigned pw, ph;
    wide_picture(&pw, &ph);
    VkExtent2D max_render = {ext.width > pw ? ext.width : pw, ext.height > ph ? ext.height : ph};
    if (F.fg && (F.fg_ext.width != ext.width || F.fg_ext.height != ext.height || F.fg_fmt != fmt)) {
        vk_queue_lock(), vkDeviceWaitIdle(c->device), vk_queue_unlock();   // a new window size: the context is made for its display size
        vk_ffx_destroy(&F.fg);
        F.fg = nullptr;
    }
    if (!F.fg) {
        ffxCreateBackendVKDesc be = {};
        be.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
        be.vkDevice = c->device;
        be.vkPhysicalDevice = c->phys;
        be.vkDeviceProcAddr = vk_ffx_proc_addr;
        // the HUD-less picture has the swapchain's format (gfx_present.c hudless_record)
        ffxCreateContextDescFrameGenerationHudless hl = {};
        hl.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION_HUDLESS;
        hl.header.pNext = &be.header;
        hl.hudlessBackBufferFormat = ffxApiGetSurfaceFormatVK(fmt);
        ffxCreateContextDescFrameGeneration d = {};
        d.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
        d.header.pNext = &hl.header;
        // no infinite-depth flag: the game's far plane is finite, and each preparation gives it (vk_fg_frame)
        d.flags = hdr_format(fmt) ? FFX_FRAMEGENERATION_ENABLE_HIGH_DYNAMIC_RANGE : 0;
        d.displaySize = {ext.width, ext.height};
        d.maxRenderSize = {max_render.width, max_render.height};
        d.backBufferFormat = ffxApiGetSurfaceFormatVK(fmt);
        ffxReturnCode_t r = vk_ffx_create(&F.fg, &d.header);
        if (r != FFX_API_RETURN_OK) return F.fg = nullptr, fail("the frame generation context", r);
        F.fg_ext = ext, F.fg_fmt = fmt, F.fg_max = max_render;
        rt_log("fg: frame generation context for %ux%u%s, render size up to %ux%u", ext.width, ext.height,
               hdr_format(fmt) ? " (HDR)" : "", max_render.width, max_render.height);
    }
    if (have && (depth.width > F.fg_max.width || depth.height > F.fg_max.height)) {
        static int told;
        if (!told++)
            rt_log("fg: a %ux%u scene is larger than the context's %ux%u: those frames are not generated", depth.width,
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
    ffxReturnCode_t r = vk_ffx_configure(&F.fg, &cfg.header);
    if (r != FFX_API_RETURN_OK) return fail("frame generation configuration", r);
    if (!on) return;
    ffxDispatchDescFrameGenerationPrepare p = {};
    p.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE;
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
    // the frame's camera (vk_dlaa.c proj_planes), as the upscaler gets it (vk_ffx.cpp)
    VkFfxCamera cam;
    if (!vk_dlaa_ffx_camera(&cam)) memset(&cam, 0, sizeof cam);
    p.cameraNear = cam.znear > 0 ? cam.znear : 0.1f;
    p.cameraFar = cam.zfar > 0 ? cam.zfar : 1e8f;
    p.cameraFovAngleVertical = cam.fov_y > 0 ? cam.fov_y : 1.0f;
    p.viewSpaceToMetersFactor = cam.meters > 0 ? cam.meters : 1.0f;
    p.depth = image(&depth);
    p.motionVectors = image(&mv);
    r = vk_ffx_dispatch(&F.fg, &p.header);
    if (r != FFX_API_RETURN_OK) return fail("frame generation preparation", r);
    if (F.frame == 2 || F.frame % 3600 == 0)
        rt_log("fg: %llu frames presented, %llu generated (%llu with the HUD kept out); paused %llu (menu), %llu without depth "
               "and motion, %llu without the HUD-less picture", (unsigned long long)F.frame, (unsigned long long)F.generated,
               (unsigned long long)F.hudless, (unsigned long long)F.paused, (unsigned long long)F.no_inputs,
               (unsigned long long)F.no_hudless);
}

extern "C" void vk_fg_release(void) {
    vk_fg4_release();
    if (F.fg) vk_ffx_destroy(&F.fg), F.fg = nullptr;
    if (F.sc) vk_ffx_destroy(&F.sc), F.sc = nullptr;
    __atomic_store_n(&F.on, 0, __ATOMIC_RELAXED);
}
