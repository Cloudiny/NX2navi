// AMD FidelityFX (vk_ffx.cpp): FSR 3.1 temporal upscaling through amd_fidelityfx_vk.dll, loaded at run time.
#pragma once
#include <vulkan/vulkan.h>

typedef struct {
    VkImage image;   // level 0, layer 0, in VK_IMAGE_LAYOUT_GENERAL
    VkFormat format;
    uint32_t width, height;
    int storage;     // created with VK_IMAGE_USAGE_STORAGE_BIT (FSR writes it)
} VkFfxImage;

// The scene camera FSR takes, from the frame's projection (vk_dlaa.c proj_planes): the vertical field of view in
// radians, the near and far planes' view distances (far: VK_DLAA_FAR_INFINITE, 1e8, for a projection without a far
// plane) and the metres in one unit of view space. A field of 0 is not known, and FSR then gets 1 radian, 0.1, 1e8 and
// 1.
typedef struct { float fov_y, znear, zfar, meters; } VkFfxCamera;

int vk_ffx_load(void);        // 1 when the library is there with its entry points (loaded once)
int vk_ffx_load_fsr4(void);   // 1 when amd_fidelityfx_upscaler_vk.so (FSR 4 on Linux) is loaded and descriptorBuffer is on
// Whether FSR 3.1 / Linux FSR 4 upscaling can run in this build, for the settings menu: after load ran, what it found;
// before, whether the library file is there, asked once without loading it. 0 where the build has no FSR to load.
int vk_ffx_available(void);
int vk_ffx_fsr4_available(void);
// The same for frame generation (vk_fg.cpp), which runs through the library's own API calls (vk_ffx_create and the
// rest below), not only an upscaler.
int vk_ffx_framegen_available(void);
// Records FSR 3.1 or Linux FSR 4 (fsr4 != 0) into cb: color (the scene light, render size), depth (D32F), mv
// (R16G16F, render pixels, current to previous), out (the output size, RGBA16F, storage). jx, jy: the jitter the scene
// took, render pixels. cam: the frame's camera. 0 on failure (the caller falls back as it does when DLSS fails).
int vk_ffx_upscale(VkCommandBuffer cb, const VkFfxImage *color, const VkFfxImage *depth, const VkFfxImage *mv,
                   const VkFfxImage *out, float jx, float jy, int reset, float dt_ms, const VkFfxCamera *cam, int fsr4);
void vk_ffx_release(void);   // before the device goes away
// FSR 3.1's contexts that no upscale used in the last `recent` calls, destroyed with the device idle (the caller waited
// for it): room for an FSR 4 context whose creation failed. Their count.
int vk_ffx_trim_idle(uint64_t recent);
// The device's functions as the library is given them (ffxCreateBackendVKDesc.vkDeviceProcAddr): its pipeline creates
// go through the runtime's pipeline caches (vk_pipeline.c vk_pipeline_create_foreign; NX2NAVI_FFX_PIPE_CACHE=0: not).
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vk_ffx_proc_addr(VkDevice dev, const char *name);

// AMD FSR 4 (vk_ffx4.cpp): the ML upscaler of AMD's FSR SDK 2.4, through amd_fidelityfx_loader_vk.dll and
// amd_fidelityfx_upscaler_vk.dll beside the executable. vk_ffx_upscale runs it in FSR 3.1's place when the setting
// asks (NX2NAVI_FSR4) and it runs here, and falls back to FSR 3.1 when it fails.
int vk_ffx4_available(void);      // the libraries are there and list a 4.x version for the device; asked once, with the
                                  // device, from any thread (0 before the device exists)
const char *vk_ffx4_name(void);   // "AMD FSR 4.1.2" once vk_ffx4_available found it
int vk_ffx4_upscale(VkCommandBuffer cb, const VkFfxImage *color, const VkFfxImage *depth, const VkFfxImage *mv,
                    const VkFfxImage *out, float jx, float jy, int reset, float dt_ms,
                    const VkFfxCamera *cam);   // as vk_ffx_upscale
void vk_ffx4_release(void);
int vk_ffx4_failed(void);         // it failed three times in a row: FSR 3.1 serves until vk_ffx4_retry
void vk_ffx4_retry(void);         // the menu chose FSR 4 again: try it anew after such failures
void vk_ffx4_trim(uint64_t now_ns);   // at each upscale: FSR 4's contexts unused for a while go
int vk_ffx_fsr4(void);            // the setting asks for FSR 4 and it runs here: what vk_ffx_upscale uses, for the menu

#if defined(__cplusplus) && !defined(VK_FFX_NO_API)
// The API's calls themselves, for vk_fg.cpp (0 = FFX_API_RETURN_OK): the library loaded first (vk_ffx_load).
#include "third_party/ffx_api/ffx_api.h"
extern "C" {
ffxReturnCode_t vk_ffx_create(ffxContext *ctx, ffxCreateContextDescHeader *desc);
ffxReturnCode_t vk_ffx_destroy(ffxContext *ctx);
ffxReturnCode_t vk_ffx_configure(ffxContext *ctx, const ffxConfigureDescHeader *desc);
ffxReturnCode_t vk_ffx_query(ffxContext *ctx, ffxQueryDescHeader *desc);
ffxReturnCode_t vk_ffx_dispatch(ffxContext *ctx, const ffxDispatchDescHeader *desc);
}
#endif
