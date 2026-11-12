// Vulkan instance, physical device choice, logical device, VMA, timeline semaphore and the validation messenger.
// Contains code Copyright (c) 2026 Peiikai (MIT; see THIRD_PARTY_NOTICES.md).
// The VMA implementation is runtime/vk_vma.cpp (VMA is C++ only).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __linux__
#include <errno.h>
#include <unistd.h>
#endif

#include "cfg.h"
#include "host.h"
#include "vk.h"
#include "platform/platform.h"
#include "vk_dlaa.h"
#include "vk_fg.h"
#include "vk_dlssg.h"

// The loader's vkGetDeviceProcAddr until Streamline is on the device (see vk.h).
PFN_vkGetDeviceProcAddr vk_dpa = vkGetDeviceProcAddr;

static uint32_t g_gfx_queues = 1;   // queues the graphics family offers (pick_device)

static VkCtx g_vk;
static int g_ready;
static PlatMutex g_lock = PLAT_MUTEX_INIT;
// The fence vk_device_lost_poll asks about. Nothing submits with it, so its status is "not ready" for as long as the
// device lives.
static VkFence lost_fence;
static PlatMutex lost_fence_lock = PLAT_MUTEX_INIT;

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_cb(VkDebugUtilsMessageSeverityFlagBitsEXT sev, VkDebugUtilsMessageTypeFlagsEXT type,
                                               const VkDebugUtilsMessengerCallbackDataEXT *d, void *user) {
    (void)type, (void)user;
    int err = (sev & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0;
    if (err) plat_inc32(&g_vk.validation_errors);
    static volatile int32_t logged;
    if (plat_inc32(&logged) <= 500) rt_log("vk validation %s: %s", err ? "error" : "warning", d->pMessage);
    return VK_FALSE;
}

static int has_ext(const VkExtensionProperties *e, uint32_t n, const char *name) {
    for (uint32_t i = 0; i < n; i++) if (!strcmp(e[i].extensionName, name)) return 1;
    return 0;
}

static int layer_present(const char *name) {
    uint32_t n = 0;
    vkEnumerateInstanceLayerProperties(&n, NULL);
    VkLayerProperties *l = calloc(n ? n : 1, sizeof *l);
    vkEnumerateInstanceLayerProperties(&n, l);
    int found = 0;
    for (uint32_t i = 0; i < n; i++) if (!strcmp(l[i].layerName, name)) found = 1;
    free(l);
    return found;
}

static int instance_ext_present(const char *name) {
    uint32_t n = 0;
    vkEnumerateInstanceExtensionProperties(NULL, &n, NULL);
    VkExtensionProperties *e = calloc(n ? n : 1, sizeof *e);
    vkEnumerateInstanceExtensionProperties(NULL, &n, e);
    int found = 0;
    for (uint32_t i = 0; i < n; i++) if (!strcmp(e[i].extensionName, name)) found = 1;
    free(e);
    return found;
}

// Mesa's drivers (RADV on Linux and the Steam Deck) keep a shader cache of their own beside our pipeline cache, by
// default in ~/.cache and up to 1 GB: about 35 MB in a minute of the title screen. A Deck's /home is small and the game
// usually sits on the SD card, so unless MESA_SHADER_CACHE_DIR names a folder, it goes to mesa-cache beside the exe.
// That folder is ours alone, so its limit is raised to MESA_CACHE_SIZE unless MESA_SHADER_CACHE_MAX_SIZE is set: the
// stage libraries the library warm builds in play live only in the driver's cache (vk_pipeline.c, "the play cache"),
// and a preparation of every shader variant does not fit in 1 GB, so the rest would be compiled again in play.
#define MESA_CACHE_SIZE "8G"
static void mesa_cache_beside_exe(void) {
    char dir[PLAT_PATH_MAX * 2], size[32];
    if (plat_getenv("MESA_SHADER_CACHE_DIR", dir, sizeof dir)) return;
    int len = plat_exe_path(dir, sizeof dir - 16);
    char *cut = len ? plat_last_sep(dir) : NULL;
    if (!cut) return;
    strcpy(cut + 1, "mesa-cache");
    plat_setenv("MESA_SHADER_CACHE_DIR", dir);
    if (!plat_getenv("MESA_SHADER_CACHE_MAX_SIZE", size, sizeof size)) plat_setenv("MESA_SHADER_CACHE_MAX_SIZE", MESA_CACHE_SIZE);
}

#ifdef __linux__
// NVIDIA's Linux driver keeps the pipelines it compiles in a disk cache of its own, by default in ~/.cache/nvidia,
// shared with every program. One process adds at most __GL_SHADER_DISK_CACHE_SIZE to it (1 GiB unless set), and a start
// that finds more than that in several files deletes the later ones. nx2navi.exe --prepare-shaders builds the 42,142
// variants of game 1.1.1 into it in one process: with the default limit it stored about 30,700 of them and the rest
// were compiled again by the first start after it. So unless the environment sets them, the cache goes to nvidia-cache
// beside the exe, which nothing else writes, with a limit of NV_CACHE_BYTES. The driver makes GLCache inside that
// folder but not the folder, so it is made here; pointed at a folder that is missing or not writable it caches nothing
// at all, so when that folder cannot be made or written neither variable is set and the driver keeps its own folder
// and limit. Without NVIDIA's kernel driver (no /proc/driver/nvidia/version, the test the AppImage's build makes for
// DLSS) the variables do nothing and the folder would stay empty, so nothing is done.
#define NV_CACHE_BYTES "8589934592"   // 8 GiB
static void nvidia_cache_beside_exe(void) {
    if (plat_fs_kind("/proc/driver/nvidia/version") != PLAT_FS_FILE) return;
    char dir[PLAT_PATH_MAX * 2], size[32];
    int own_dir = !plat_getenv("__GL_SHADER_DISK_CACHE_PATH", dir, sizeof dir);
    if (own_dir) {
        int len = plat_exe_path(dir, sizeof dir - 16);
        char *cut = len ? plat_last_sep(dir) : NULL;
        if (!cut) return;
        strcpy(cut + 1, "nvidia-cache");
        int made = plat_fs_mkdir(dir), there = made == PLAT_FS_OK || made == PLAT_FS_EXISTS;
        int err = !there ? plat_fs_os_error() : access(dir, W_OK | X_OK) ? errno : 0;
        if (err) {
            rt_log("vk: NVIDIA's shader disk cache stays in the driver's own folder with its own limit: %s %s (%s)", dir,
                   there ? "is not writable" : "cannot be made", strerror(err));
            return;
        }
        plat_setenv("__GL_SHADER_DISK_CACHE_PATH", dir);
    }
    int own_size = !plat_getenv("__GL_SHADER_DISK_CACHE_SIZE", size, sizeof size);
    if (own_size) plat_setenv("__GL_SHADER_DISK_CACHE_SIZE", NV_CACHE_BYTES);
    rt_log("vk: NVIDIA's shader disk cache in %s (%s), up to %s bytes (%s)", dir,
           own_dir ? "beside the exe" : "__GL_SHADER_DISK_CACHE_PATH", own_size ? NV_CACHE_BYTES : size,
           own_size ? "the game's limit" : "__GL_SHADER_DISK_CACHE_SIZE");
}
#endif

// A driver may read these variables when it loads, and a thread can load the Vulkan driver before create_instance:
// widescreen.c's wide_init asks for the monitor's size (fullscreen with a monitor chosen and the picture at the
// window's size), and SDL's load of the Vulkan library there lists the instance extensions, which loads the drivers,
// while the prewarm thread may be in create_instance; setenv beside other threads' getenv is not safe either. main
// therefore calls this for a game run before any thread of the game starts; create_instance calls it for the other
// modes. Both calls happen before other threads exist or under vk_init's lock, so the flag needs no atomics.
void vk_driver_cache_env(void) {
    static int done;
    if (done) return;
    done = 1;
    mesa_cache_beside_exe();
#ifdef __linux__
    nvidia_cache_beside_exe();
#endif
}

static int create_instance(void) {
    vk_driver_cache_env();
    vk_dlssg_early();   // DLSS Frame Generation: Streamline before any Vulkan call of the process (its slInit asks so)
    const char *ext[12];
    uint32_t n_ext = (uint32_t)gfx_window_vk_extensions(ext, 7);
    const char *layers[] = {"VK_LAYER_KHRONOS_validation"};
    int validate = cfg_bool("NX2NAVI_VK_VALIDATION");
    if (validate && !layer_present(layers[0])) {
        rt_log("vk: NX2NAVI_VK_VALIDATION set but %s is not installed; continuing without", layers[0]);
        validate = 0;
    }
    // DLSS Frame Generation: Streamline's plugins call its naming functions even when they say it is off
    if (validate || (vk_dlssg_wanted() && instance_ext_present(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)))
        ext[n_ext++] = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    // armed, not wanted: the menu switches HDR on during the run (vk_hdr.c), and the extensions are fixed here
    if (vk_hdr_armed() && instance_ext_present(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME))   // HDR10 surface formats
        ext[n_ext++] = VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME;
    vk_dlaa_instance_exts(ext, &n_ext, 12);
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "NX2navi",
                             .pEngineName = "NX2navi", .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app,
                               .enabledExtensionCount = n_ext, .ppEnabledExtensionNames = ext,
                               .enabledLayerCount = validate ? 1 : 0, .ppEnabledLayerNames = layers};
    rdoc_init();
    VkResult r = vk_dlssg_create_instance(&ci, &g_vk.instance);
    if (r != VK_SUCCESS) { rt_log("vk: vkCreateInstance failed (%d); is a Vulkan 1.3 driver installed?", (int)r); return 0; }
    g_vk.validation = validate;
    if (validate) {
        PFN_vkCreateDebugUtilsMessengerEXT f =
            (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(g_vk.instance, "vkCreateDebugUtilsMessengerEXT");
        VkDebugUtilsMessengerCreateInfoEXT mi = {
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            .pfnUserCallback = debug_cb};
        if (f) f(g_vk.instance, &mi, NULL, &g_vk.messenger);
    }
    return 1;
}

static const char *const k_required_ext[] = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME, VK_EXT_DEPTH_CLIP_CONTROL_EXTENSION_NAME,
    VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME};

// Name of the first missing required extension, or NULL.
static const char *missing_required(VkPhysicalDevice pd) {
    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(pd, NULL, &n, NULL);
    VkExtensionProperties *e = calloc(n ? n : 1, sizeof *e);
    vkEnumerateDeviceExtensionProperties(pd, NULL, &n, e);
    const char *miss = NULL;
    for (size_t i = 0; i < sizeof k_required_ext / sizeof *k_required_ext && !miss; i++)
        if (!has_ext(e, n, k_required_ext[i])) miss = k_required_ext[i];
    free(e);
    return miss;
}

// Graphics+compute family that can present to the window.
static uint32_t find_gfx_family(VkPhysicalDevice pd, const VkQueueFamilyProperties *qp, uint32_t nq) {
    for (uint32_t i = 0; i < nq; i++)
        if ((qp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            gfx_window_vk_presentable(g_vk.instance, pd, i))
            return i;
    return UINT32_MAX;
}

static int name_has(const char *s, const char *sub) {
    for (size_t n = strlen(sub); *s; s++)
        if (!strncasecmp(s, sub, n)) return 1;
    return 0;
}

// First discrete GPU with Vulkan 1.3, the required extensions and a graphics+present queue; NX2NAVI_GPU_INDEX (index) or
// NX2NAVI_GPU_NAME (case-insensitive part of the name) overrides.
static int pick_device(void) {
    uint32_t n = 0;
    vk_dlssg_enumerate(g_vk.instance, &n, NULL);
    if (!n) { rt_log("vk: no Vulkan device"); return 0; }
    VkPhysicalDevice *pd = calloc(n, sizeof *pd);
    vk_dlssg_enumerate(g_vk.instance, &n, pd);
    const char *e = cfg_get("NX2NAVI_GPU_INDEX");   // no fixed default: the best device when unset
    int want = e && *e ? atoi(e) : -1;
    const char *name = cfg_get("NX2NAVI_GPU_NAME");
    int best = -1, best_score = -1, listed = 0;
retry:
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(pd[i], &p);
        if (!listed)
            rt_log("vk: gpu %u: %s (type %d, api %u.%u.%u)", i, p.deviceName, (int)p.deviceType,
                   VK_API_VERSION_MAJOR(p.apiVersion), VK_API_VERSION_MINOR(p.apiVersion), VK_API_VERSION_PATCH(p.apiVersion));
        if (want >= 0 && (int)i != want) continue;
        if (name && *name && !name_has(p.deviceName, name)) continue;
        if (p.apiVersion < VK_API_VERSION_1_3) { rt_log("vk: gpu %u skipped: Vulkan below 1.3", i); continue; }
        const char *miss = missing_required(pd[i]);
        if (miss) { rt_log("vk: gpu %u skipped: missing extension %s", i, miss); continue; }
        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd[i], &nq, NULL);
        VkQueueFamilyProperties *qp = calloc(nq ? nq : 1, sizeof *qp);
        vkGetPhysicalDeviceQueueFamilyProperties(pd[i], &nq, qp);
        uint32_t fam = find_gfx_family(pd[i], qp, nq);
        free(qp);
        if (fam == UINT32_MAX) { rt_log("vk: gpu %u skipped: no graphics+compute+present queue family", i); continue; }
        int score = (int)i == want ? 100 : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 3
                  : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 2 : 1;
        if (score > best_score) { best_score = score; best = (int)i; }
    }
    // A chosen GPU that is not usable or no longer there (the menu's GPU row, an unplugged card, a driver that lists the
    // devices in another order): the best usable one instead, so the game still starts and the choice can be changed
    if (best < 0 && (want >= 0 || (name && *name))) {
        if (want >= 0) rt_log("vk: NX2NAVI_GPU_INDEX=%d is not usable (see above): the best usable gpu instead", want);
        else rt_log("vk: no usable gpu matches NX2NAVI_GPU_NAME=%s (see above): the best usable gpu instead", name);
        want = -1, name = NULL, listed = 1;
        goto retry;
    }
    if (best < 0) {
        rt_log("vk: no device with Vulkan 1.3 and the required extensions (see above)");
        free(pd);
        return 0;
    }
    g_vk.phys = pd[best];
    g_vk.device_index = best;
    free(pd);
    vkGetPhysicalDeviceProperties(g_vk.phys, &g_vk.props);
    vkGetPhysicalDeviceMemoryProperties(g_vk.phys, &g_vk.mem_props);
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_vk.phys, &nq, NULL);
    VkQueueFamilyProperties *qp = calloc(nq ? nq : 1, sizeof *qp);
    vkGetPhysicalDeviceQueueFamilyProperties(g_vk.phys, &nq, qp);
    g_vk.gfx_family = find_gfx_family(g_vk.phys, qp, nq);
    g_gfx_queues = qp[g_vk.gfx_family].queueCount;
    g_vk.xfer_family = g_vk.gfx_family;
    for (uint32_t i = 0; i < nq; i++)   // a family that does transfer only (async DMA engine)
        if ((qp[i].queueFlags & VK_QUEUE_TRANSFER_BIT) && !(qp[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))) {
            g_vk.xfer_family = i;
            break;
        }
    free(qp);
    rt_log("vk: using gpu %d: %s", best, g_vk.props.deviceName);
    vk_caps_driver(g_vk.phys);   // before vk_mem_class_log, which chooses the pools' storage by driver
    vk_mem_class_log(&g_vk.props, &g_vk.mem_props);
    return 1;
}

typedef struct {
    VkPhysicalDeviceFeatures2 f2;
    VkPhysicalDeviceVulkan11Features f11;
    VkPhysicalDeviceVulkan12Features f12;
    VkPhysicalDeviceVulkan13Features f13;
    VkPhysicalDeviceDepthClipControlFeaturesEXT clip;
    VkPhysicalDeviceCustomBorderColorFeaturesEXT cbc;
    VkPhysicalDeviceExtendedDynamicStateFeaturesEXT eds;
    VkPhysicalDeviceExtendedDynamicState2FeaturesEXT eds2;
    VkPhysicalDeviceExtendedDynamicState3FeaturesEXT eds3;
    VkPhysicalDeviceVertexInputDynamicStateFeaturesEXT vids;
    VkPhysicalDeviceConditionalRenderingFeaturesEXT cond;
    VkPhysicalDeviceTransformFeedbackFeaturesEXT xfb;
    VkPhysicalDeviceProvokingVertexFeaturesEXT pv;
    VkPhysicalDeviceRobustness2FeaturesEXT rob2;
    VkPhysicalDeviceFaultFeaturesEXT fault;
    VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT fsi;
    VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT gpl;
    VkPhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT fbl;
    VkPhysicalDeviceDynamicRenderingLocalReadFeaturesKHR lr;
    VkPhysicalDeviceMeshShaderFeaturesEXT mesh;
    VkPhysicalDeviceFragmentShadingRateFeaturesKHR vrs;
    VkPhysicalDeviceShaderMaximalReconvergenceFeaturesKHR mrc;
    VkPhysicalDeviceMaintenance5FeaturesKHR m5;
    VkPhysicalDevicePipelineBinaryFeaturesKHR pbin;
    VkPhysicalDeviceDescriptorBufferFeaturesEXT desc_buf;
} FeatureChain;

// One line for a fatal error's report (dispatch.h rt_fatal_note_add): the card, its driver, and the settings in effect
// that change how the game's memory behaves. A report often arrives as the end of a log alone, or as a picture of it,
// and the lines that say these things are at its start. No lock is taken: the failing thread may hold any.
static void fatal_note(void) {
    char mem[512];
    vk_mem_fatal_note(mem, sizeof mem);
    rt_log("fatal: gpu %s; %s; texture detail %s", vk_caps_summary(), mem, vk_texture_detail_name());
}

static int create_device(void) {
    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(g_vk.phys, NULL, &ne, NULL);
    VkExtensionProperties *ep = calloc(ne ? ne : 1, sizeof *ep);
    vkEnumerateDeviceExtensionProperties(g_vk.phys, NULL, &ne, ep);
    const char *ext[96];
    uint32_t n_ext = 0;
    for (size_t i = 0; i < sizeof k_required_ext / sizeof *k_required_ext; i++) ext[n_ext++] = k_required_ext[i];

    FeatureChain *c = calloc(1, sizeof *c);
    void **tail = &c->f2.pNext;
#define CHAIN(member, stype) do { c->member.sType = stype; *tail = &c->member; tail = &c->member.pNext; } while (0)
    c->f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    CHAIN(f11, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES);
    CHAIN(f12, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES);
    CHAIN(f13, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES);
    CHAIN(clip, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT);
#define OPTIONAL_EXT(flag, name, member, stype) do { \
        if (has_ext(ep, ne, name)) { g_vk.flag = 1; ext[n_ext++] = name; CHAIN(member, stype); } } while (0)
    OPTIONAL_EXT(has_custom_border_color, VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME, cbc, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT);
    OPTIONAL_EXT(has_eds, VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME, eds, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT);
    OPTIONAL_EXT(has_eds2, VK_EXT_EXTENDED_DYNAMIC_STATE_2_EXTENSION_NAME, eds2, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT);
    OPTIONAL_EXT(has_eds3, VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME, eds3, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT);
    OPTIONAL_EXT(has_vertex_input_dynamic_state, VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME, vids, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_INPUT_DYNAMIC_STATE_FEATURES_EXT);
    OPTIONAL_EXT(has_conditional_rendering, VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME, cond, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT);
    OPTIONAL_EXT(has_transform_feedback, VK_EXT_TRANSFORM_FEEDBACK_EXTENSION_NAME, xfb, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT);
    OPTIONAL_EXT(has_provoking_vertex, VK_EXT_PROVOKING_VERTEX_EXTENSION_NAME, pv, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROVOKING_VERTEX_FEATURES_EXT);
    OPTIONAL_EXT(has_robustness2, VK_EXT_ROBUSTNESS_2_EXTENSION_NAME, rob2, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT);
    OPTIONAL_EXT(has_device_fault, VK_EXT_DEVICE_FAULT_EXTENSION_NAME, fault, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT);
    // The translator's SPIR-V declares fragment shader interlock whenever the host reports it (vk_pipeline.c spirv_fixup).
    OPTIONAL_EXT(has_fragment_shader_interlock, VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME, fsi,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT);
    // NVN2 task and mesh programs (gameplay geometry) draw through VK_EXT_mesh_shader (vk_exec_draw.c op_draw_mesh)
    OPTIONAL_EXT(has_mesh_shader, VK_EXT_MESH_SHADER_EXTENSION_NAME, mesh, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT);
    // The foliage's far shadow cascades shade one fragment per 2x2 pixels (vk_exec_draw.c mesh_shading_rate)
    OPTIONAL_EXT(has_vrs, VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME, vrs, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR);
    // Guest shaders that use warp-wide operations after divergent code run with their threads reconverged
    // (vk_pipeline.c spirv_reconverge)
    OPTIONAL_EXT(has_maximal_reconvergence, VK_KHR_SHADER_MAXIMAL_RECONVERGENCE_EXTENSION_NAME, mrc,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MAXIMAL_RECONVERGENCE_FEATURES_KHR);
#ifndef _WIN32
    // FSR 4 on Linux (vk_ffx.cpp): amd_fidelityfx_upscaler_vk.so uses descriptor buffers on Vulkan.
    OPTIONAL_EXT(has_descriptor_buffer, VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME, desc_buf,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT);
#endif
    // Compiled pipelines kept under the runtime's own keys (vk_pipeline.c pipe_create): VK_KHR_pipeline_binary, which
    // needs VK_KHR_maintenance5 (core only from Vulkan 1.4; the instance asks for 1.3). NX2NAVI_PIPE_BINARIES=0 leaves
    // both off.
    if (cfg_bool("NX2NAVI_PIPE_BINARIES") && has_ext(ep, ne, VK_KHR_PIPELINE_BINARY_EXTENSION_NAME) &&
        has_ext(ep, ne, VK_KHR_MAINTENANCE_5_EXTENSION_NAME)) {
        g_vk.has_pipeline_binary = 1;
        ext[n_ext++] = VK_KHR_MAINTENANCE_5_EXTENSION_NAME;
        ext[n_ext++] = VK_KHR_PIPELINE_BINARY_EXTENSION_NAME;
        CHAIN(m5, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR);
        CHAIN(pbin, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_BINARY_FEATURES_KHR);
    }
    // Feedback loops (vk.h VkCtx.feedback): the layout, and barriers inside dynamic rendering
    int has_fbl = 0, has_lr = 0;
    if (has_ext(ep, ne, VK_EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_EXTENSION_NAME) &&
        has_ext(ep, ne, VK_KHR_DYNAMIC_RENDERING_LOCAL_READ_EXTENSION_NAME)) {
        has_fbl = has_lr = 1;
        ext[n_ext++] = VK_EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_EXTENSION_NAME;
        ext[n_ext++] = VK_KHR_DYNAMIC_RENDERING_LOCAL_READ_EXTENSION_NAME;
        CHAIN(fbl, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_FEATURES_EXT);
        CHAIN(lr, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_LOCAL_READ_FEATURES_KHR);
    }
    vk_dlaa_device_exts(ep, ne, ext, &n_ext, 40);   // NGX's; the eight slots after it stay for the eight below
    // FSR 3.1 (vk_ffx.cpp): its Vulkan back end looks vkGetBufferMemoryRequirements2KHR up by that name, which resolves
    // only with the extension enabled (the core function has no suffix); without it the back end called address 0.
    // Always asked for: FSR 3.1 can be switched on in the menu during a run
    if (has_ext(ep, ne, VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME))
        ext[n_ext++] = VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME;
    // DLSS Frame Generation (vk_dlssg.c): Streamline's plugins look functions of extensions core in Vulkan 1.3 up by
    // their suffixed names too, which resolve only with the extension itself enabled
    if (vk_dlssg_wanted()) {
        static const char *const promoted[] = {VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
                                               VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME, VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME,
                                               VK_KHR_BIND_MEMORY_2_EXTENSION_NAME, VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
                                               VK_KHR_COPY_COMMANDS_2_EXTENSION_NAME, VK_KHR_MAINTENANCE_1_EXTENSION_NAME,
                                               VK_KHR_MAINTENANCE_2_EXTENSION_NAME, VK_KHR_MAINTENANCE_3_EXTENSION_NAME};
        for (size_t i = 0; i < sizeof promoted / sizeof *promoted && n_ext < 88; i++)
            if (has_ext(ep, ne, promoted[i])) ext[n_ext++] = promoted[i];
    }
    if (has_ext(ep, ne, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) { g_vk.has_memory_budget = 1; ext[n_ext++] = VK_EXT_MEMORY_BUDGET_EXTENSION_NAME; }
    // GPU timestamps on the CPU clock (vk_sync.c): the perf line splits GPU idle time into waiting for a submission or not.
    if (has_ext(ep, ne, VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME)) { g_vk.has_calibrated = 2; ext[n_ext++] = VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME; }
    else if (has_ext(ep, ne, VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME)) { g_vk.has_calibrated = 1; ext[n_ext++] = VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME; }
    if (vk_hdr_armed() && has_ext(ep, ne, VK_EXT_HDR_METADATA_EXTENSION_NAME)) {   // the display's peak (vk_hdr_metadata)
        g_vk.has_hdr_metadata = 1;
        ext[n_ext++] = VK_EXT_HDR_METADATA_EXTENSION_NAME;
    }
    // GPU breadcrumbs for a lost device (vk_crumbs.c): AMD's drivers and RADV report it.
    if (has_ext(ep, ne, VK_AMD_BUFFER_MARKER_EXTENSION_NAME)) { g_vk.has_buffer_marker = 1; ext[n_ext++] = VK_AMD_BUFFER_MARKER_EXTENSION_NAME; }
    // Pass marks on NVIDIA (vk_crumbs.c): checkpoints only with NX2NAVI_PASS_CRUMBS=1, since NVIDIA documents a cost for
    // the extension. NVIDIA's drivers report it, AMD's do not.
    if (vk_pass_marks_want_nv() && has_ext(ep, ne, VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME)) {
        g_vk.has_nv_checkpoints = 1;
        ext[n_ext++] = VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME;
    }
    // Graphics pipeline library: chained last so it can be dropped again when vk_caps_gpl declines it.
    void **gpl_link = NULL;
    if (has_ext(ep, ne, VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME) && has_ext(ep, ne, VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME)) {
        ext[n_ext++] = VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME;
        ext[n_ext++] = VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME;
        gpl_link = tail;
        CHAIN(gpl, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT);
    }
    free(ep);

    // The driver fills the chain with what it supports; the same chain then enables all of it (robustBufferAccess
    // stays on: guest shaders index constant buffers freely and out-of-range reads must stay harmless).
    vkGetPhysicalDeviceFeatures2(g_vk.phys, &c->f2);
    vk_caps_features(&c->f2.features);
    if (gpl_link) {
        VkPhysicalDeviceGraphicsPipelineLibraryPropertiesEXT gp = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_PROPERTIES_EXT};
        VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &gp};
        vkGetPhysicalDeviceProperties2(g_vk.phys, &p2);
        if (!vk_caps_gpl(&c->gpl, &gp)) { *gpl_link = NULL; n_ext -= 2; }
    }
    const VkPhysicalDeviceFeatures *h = &c->f2.features;
    const char *miss = NULL;
    if (!c->f12.timelineSemaphore) miss = "timelineSemaphore";
    else if (!c->f13.dynamicRendering) miss = "dynamicRendering";
    else if (!c->f13.synchronization2) miss = "synchronization2";
    else if (!h->shaderStorageImageWriteWithoutFormat) miss = "shaderStorageImageWriteWithoutFormat";
    else if (!c->clip.depthClipControl) miss = "depthClipControl";
    else if (!h->shaderInt64) miss = "shaderInt64";
    else if (!h->samplerAnisotropy) miss = "samplerAnisotropy";
    else if (!h->independentBlend) miss = "independentBlend";
    else if (!h->geometryShader) miss = "geometryShader";
    else if (!h->textureCompressionBC) miss = "textureCompressionBC";
    if (miss) { rt_log("vk: required feature missing on %s: %s", g_vk.props.deviceName, miss); free(c); return 0; }
    if (!c->fault.deviceFault) g_vk.has_device_fault = 0;
    if (g_vk.has_vrs) {   // the rate is set per draw; no rate attachment and no per-primitive rates
        c->vrs.attachmentFragmentShadingRate = c->vrs.primitiveFragmentShadingRate = VK_FALSE;
        if (!c->vrs.pipelineFragmentShadingRate) g_vk.has_vrs = 0;
        rt_log("vk: fragment shading rate: %s", g_vk.has_vrs ? "per draw" : "not supported");
    }
    if (g_vk.has_mesh_shader) {
        // these two need features that are not enabled (multiview, primitive fragment shading rate)
        c->mesh.multiviewMeshShader = c->mesh.primitiveFragmentShadingRateMeshShader = VK_FALSE;
        // Mesh queries for two diagnoses: NX2NAVI_GPU_PASSES (vk_crumbs.c) and the dump's primitive counts
        // (vk_dump.c vk_dump_mesh_begin), which also need pipeline statistics and host query reset.
        const char *gp = cfg_get("NX2NAVI_GPU_PASSES"), *dmp = cfg_get("NX2NAVI_DUMP");
        int want = (gp && *gp && strcmp(gp, "0")) || (dmp && *dmp);
        g_vk.has_mesh_queries = want && c->mesh.meshShaderQueries && h->pipelineStatisticsQuery && c->f12.hostQueryReset;
        c->mesh.meshShaderQueries = g_vk.has_mesh_queries ? VK_TRUE : VK_FALSE;
        if (!c->mesh.meshShader || !c->mesh.taskShader) g_vk.has_mesh_shader = 0;
        rt_log("vk: mesh shaders: %s", g_vk.has_mesh_shader ? "VK_EXT_mesh_shader" : "the extension is there, its task or mesh feature is not");
    }   // vkGetDeviceFaultInfoEXT needs the feature, not just the extension
    {   // mesh_fallback: auto (on a device without mesh shaders), 1 (on every device, for testing) or 0
        const char *mf = cfg_value("NX2NAVI_MESH_FALLBACK");
        g_vk.mesh_fallback = !strcmp(mf, "1") || (strcmp(mf, "0") && !g_vk.has_mesh_shader);
        if (g_vk.mesh_fallback)
            rt_log("vk: mesh fallback: task and mesh programs run as compute passes and are drawn from their buffers (%s)",
                   g_vk.has_mesh_shader ? "mesh_fallback = 1" : "this device has no mesh shaders");
        else if (!g_vk.has_mesh_shader)
            rt_log("vk: mesh shaders: not supported and mesh_fallback = 0 (task/mesh draws are skipped)");
    }
    if (g_vk.has_maximal_reconvergence && !c->mrc.shaderMaximalReconvergence) g_vk.has_maximal_reconvergence = 0;
#ifndef _WIN32
    if (g_vk.has_descriptor_buffer) {
        c->desc_buf.descriptorBufferCaptureReplay = VK_FALSE;
        c->desc_buf.descriptorBufferPushDescriptors = VK_FALSE;
        if (!c->desc_buf.descriptorBuffer) g_vk.has_descriptor_buffer = 0;
    }
#endif
    if (g_vk.has_pipeline_binary && !(c->pbin.pipelineBinaries && c->m5.maintenance5)) g_vk.has_pipeline_binary = 0;
    g_vk.has_buffer_device_address = c->f12.bufferDeviceAddress;
    g_vk.has_astc_ldr = h->textureCompressionASTC_LDR;
    {
        const char *e = cfg_get("NX2NAVI_FEEDBACK");
        int layout_ok = has_fbl && has_lr && c->fbl.attachmentFeedbackLoopLayout && c->lr.dynamicRenderingLocalRead;
        g_vk.feedback = e && !strcmp(e, "0") ? VK_FEEDBACK_OFF
                      : layout_ok && !(e && !strcmp(e, "restart")) ? VK_FEEDBACK_LAYOUT : VK_FEEDBACK_RESTART;
        rt_log("vk: feedback loops: %s (attachmentFeedbackLoopLayout %d, dynamicRenderingLocalRead %d)",
               g_vk.feedback == VK_FEEDBACK_LAYOUT ? "feedback layout + by-region barrier"
               : g_vk.feedback == VK_FEEDBACK_RESTART ? "pass restart + full barrier" : "off (NX2NAVI_FEEDBACK=0)",
               has_fbl && c->fbl.attachmentFeedbackLoopLayout, has_lr && c->lr.dynamicRenderingLocalRead);
    }

    // Frame generation (vk_fg.cpp) presents on a second queue of the graphics family and acquires on a third
    static const float prio[3] = {1.0f, 1.0f, 1.0f};
    uint32_t nq_gfx = vk_fg_queue_count() < g_gfx_queues ? vk_fg_queue_count() : g_gfx_queues;
    VkDeviceQueueCreateInfo qi[2] = {
        {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = g_vk.gfx_family, .queueCount = nq_gfx,
         .pQueuePriorities = prio}};
    uint32_t nqi = 1;
    if (g_vk.xfer_family != g_vk.gfx_family) {
        qi[1] = (VkDeviceQueueCreateInfo){.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                          .queueFamilyIndex = g_vk.xfer_family, .queueCount = 1, .pQueuePriorities = prio};
        nqi = 2;
    }
    VkDeviceCreateInfo di = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &c->f2, .queueCreateInfoCount = nqi,
                             .pQueueCreateInfos = qi, .enabledExtensionCount = n_ext, .ppEnabledExtensionNames = ext};
    VkResult r = vk_dlssg_create_device(g_vk.instance, g_vk.phys, &di, &g_vk.device);
    free(c);
    if (r != VK_SUCCESS) { rt_log("vk: vkCreateDevice failed (%d)", (int)r); return 0; }
    vk_dlssg_device_ready(g_vk.phys, g_vk.device);
    if (vk_dlssg_dpa()) {   // Streamline's, so its plugins hook every entry point the replay fetches
        vk_dpa = vk_dlssg_dpa();
        rt_log("vk: device entry points come through Streamline's interposer (DLSS Frame Generation)");
    }
    vkGetDeviceQueue(g_vk.device, g_vk.gfx_family, 0, &g_vk.gfx_queue);
    if (nq_gfx >= 2) {
        VkQueue present, acquire;
        vkGetDeviceQueue(g_vk.device, g_vk.gfx_family, 1, &present);
        vkGetDeviceQueue(g_vk.device, g_vk.gfx_family, nq_gfx - 1, &acquire);
        vk_fg_queues(present, acquire);
    }
    if (g_vk.xfer_family != g_vk.gfx_family) vkGetDeviceQueue(g_vk.device, g_vk.xfer_family, 0, &g_vk.xfer_queue);
    else { g_vk.xfer_queue = g_vk.gfx_queue; g_vk.xfer_shared = 1; }

    VkPhysicalDeviceExternalMemoryHostPropertiesEXT emh = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &emh};
    vkGetPhysicalDeviceProperties2(g_vk.phys, &p2);
    g_vk.min_imported_host_alignment = (uint32_t)emh.minImportedHostPointerAlignment;
    vk_caps_init(&g_vk);
    vk_mem_lazy_decide();   // the driver is known: the lazy write-back's default depends on it
    rt_fatal_note_add(fatal_note);

    VmaAllocatorCreateInfo ai = {.physicalDevice = g_vk.phys, .device = g_vk.device, .instance = g_vk.instance,
                                 .vulkanApiVersion = VK_API_VERSION_1_3,
                                 .flags = (g_vk.has_buffer_device_address ? VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT : 0) |
                                          (g_vk.has_memory_budget ? VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT : 0),
                                 .preferredLargeHeapBlockSize = vk_vma_block_size(),
                                 .pDeviceMemoryCallbacks = vk_headroom_callbacks()};
    if ((r = vmaCreateAllocator(&ai, &g_vk.vma)) != VK_SUCCESS) { rt_log("vk: vmaCreateAllocator failed (%d)", (int)r); return 0; }
    vk_mem_ballast(g_vk.vma);

    VkSemaphoreTypeCreateInfo st = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                    .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE, .initialValue = 0};
    VkSemaphoreCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &st};
    if ((r = vkCreateSemaphore(g_vk.device, &si, NULL, &g_vk.timeline)) != VK_SUCCESS) {
        rt_log("vk: timeline semaphore creation failed (%d)", (int)r);
        return 0;
    }
    vk_crumbs_init(&g_vk);
    return 1;
}

static long g_errors_total;     // outlives the device so the probe can print it after vk_shutdown

static void destroy_all(void) {
    gfx_window_release_vk();
    if (g_vk.device) vkDeviceWaitIdle(g_vk.device);
    if (g_vk.device) vk_dlaa_release();   // the DLSS features and NGX: the allocator and the device still live
    if (g_vk.device) vk_astc_release();
    if (g_vk.device) vk_crumbs_release();
    if (g_vk.timeline) vkDestroySemaphore(g_vk.device, g_vk.timeline, NULL);
    if (lost_fence) vkDestroyFence(g_vk.device, lost_fence, NULL);   // vk_device_lost_poll
    lost_fence = VK_NULL_HANDLE;
    vk_headroom_stop();
    if (g_vk.vma) vk_mem_ballast_release(g_vk.vma);
    if (g_vk.vma) vmaDestroyAllocator(g_vk.vma);
    if (g_vk.device) vkDestroyDevice(g_vk.device, NULL);
    if (g_vk.messenger) {
        PFN_vkDestroyDebugUtilsMessengerEXT f =
            (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(g_vk.instance, "vkDestroyDebugUtilsMessengerEXT");
        if (f) f(g_vk.instance, g_vk.messenger, NULL);
    }
    if (g_vk.instance) vkDestroyInstance(g_vk.instance, NULL);
    g_errors_total += g_vk.validation_errors;
    memset(&g_vk, 0, sizeof g_vk);
}

long vk_validation_errors_total(void) { return g_errors_total + (g_ready ? g_vk.validation_errors : 0); }

int vk_init(void) {
    plat_mutex_lock(&g_lock);
    if (!g_ready) {
        memset(&g_vk, 0, sizeof g_vk);
        if (create_instance() && pick_device() && create_device()) g_ready = 1, vk_headroom_start(), vk_dlaa_init();
        else destroy_all();
    }
    int ok = g_ready;
    plat_mutex_unlock(&g_lock);
    return ok;
}

void vk_shutdown(void) {
    plat_mutex_lock(&g_lock);
    if (g_ready) { destroy_all(); g_ready = 0; }
    plat_mutex_unlock(&g_lock);
}

VkCtx *vk_ctx(void) { return g_ready ? &g_vk : NULL; }

static const char *fault_address_type(VkDeviceFaultAddressTypeEXT t) {
    switch (t) {
    case VK_DEVICE_FAULT_ADDRESS_TYPE_NONE_EXT: return "none";
    case VK_DEVICE_FAULT_ADDRESS_TYPE_READ_INVALID_EXT: return "read invalid";
    case VK_DEVICE_FAULT_ADDRESS_TYPE_WRITE_INVALID_EXT: return "write invalid";
    case VK_DEVICE_FAULT_ADDRESS_TYPE_EXECUTE_INVALID_EXT: return "execute invalid";
    case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_UNKNOWN_EXT: return "instruction pointer (unknown)";
    case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_INVALID_EXT: return "instruction pointer (invalid)";
    case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_FAULT_EXT: return "instruction pointer (fault)";
    default: return "unknown type";
    }
}

// The report of a lost device: the breadcrumbs, the pass marks, the video memory and the driver's fault information.
// Once per process, on the thread that saw the loss first (vk_device_lost_check).
static void device_fault_report(void) {
    vk_crumbs_report();
    vk_pass_marks_report();
    vk_mem_lost_report();
    VkCtx *c = vk_ctx();
    PFN_vkGetDeviceFaultInfoEXT get =
        c && c->has_device_fault ? (PFN_vkGetDeviceFaultInfoEXT)vk_dpa(c->device, "vkGetDeviceFaultInfoEXT")
                                 : NULL;
    if (!get) {
        rt_log("vk: device lost; no fault report (VK_EXT_device_fault %s)", c && c->has_device_fault ? "has no entry point"
                                                                                                    : "not enabled");
        return;
    }
    VkDeviceFaultCountsEXT n = {.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT};
    VkResult q = get(c->device, &n, NULL);
    VkDeviceFaultAddressInfoEXT *ai = calloc(n.addressInfoCount + 1, sizeof *ai);
    VkDeviceFaultVendorInfoEXT *vi = calloc(n.vendorInfoCount + 1, sizeof *vi);
    VkDeviceFaultInfoEXT info = {.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT, .pAddressInfos = ai, .pVendorInfos = vi};
    uint64_t binary = n.vendorBinarySize;
    n.vendorBinarySize = 0;   // the vendor binary is not fetched; its size alone is logged
    if (q == VK_SUCCESS && ai && vi) q = get(c->device, &n, &info);
    if (q != VK_SUCCESS && q != VK_INCOMPLETE) {
        rt_log("vk: device lost; vkGetDeviceFaultInfoEXT failed (%d)", (int)q);
    } else {
        rt_log("vk: device fault: \"%s\" (%u address infos, %u vendor infos, vendor binary %llu bytes)", info.description,
               n.addressInfoCount, n.vendorInfoCount, (unsigned long long)binary);
        // The faulting access lies in the precision-aligned range around the reported address.
        for (uint32_t i = 0; i < n.addressInfoCount; i++)
            rt_log("vk: device fault address %u: %s at %#llx, precision %#llx", i, fault_address_type(ai[i].addressType),
                   (unsigned long long)ai[i].reportedAddress, (unsigned long long)ai[i].addressPrecision);
        for (uint32_t i = 0; i < n.vendorInfoCount; i++)
            rt_log("vk: device fault vendor info %u: \"%s\" code %#llx data %#llx", i, vi[i].description,
                   (unsigned long long)vi[i].vendorFaultCode, (unsigned long long)vi[i].vendorFaultData);
    }
    free(ai);
    free(vi);
}

// ---- the end of a run whose device is lost (vk.h vk_device_lost_check)
// A lost device stays lost: every later call on it fails or, worse, answers as if it had worked, and a driver that is
// called on from several threads after the loss can fault inside itself. So the first thread that sees the loss ends
// the run, and nothing else touches the device meanwhile: the paths that would read vk_lost and return, and a thread
// that gets the loss as a result of its own stops here for good.
// LOST_EXIT_MS bounds the end. The first thread writes the report and waits for a save commit (fs_host.c
// fs_save_crash_wait, 2 s at most); should it hang on a lock that a stopped thread holds, the watchdog ends the
// process with the same code, and the lines already written are in the log (rt_fatal_code wrote them itself).
#define LOST_EXIT_MS 10000
volatile int32_t vk_lost;

static void lost_wait_exit(void *arg) {
    (void)arg;
    for (int t = 0; t < LOST_EXIT_MS; t += 50) plat_sleep_ms(50);
    _Exit(RT_LOST_EXIT);
}

void vk_device_lost_check(VkResult r, const char *where) {
    if (r != VK_ERROR_DEVICE_LOST) return;
    if (plat_xchg32(&vk_lost, 1)) lost_wait_exit(NULL);   // another thread is ending the run: this one stops here
    plat_thread_spawn(lost_wait_exit, NULL, "vk lost exit", 0);
    device_fault_report();
    rt_fatal_code(RT_LOST_EXIT, "vk: the GPU device was lost (seen at: %s). The run ends here; the lines above name what "
                                "the GPU was doing", where);
}


void vk_device_lost_poll(void) {
    VkCtx *c = vk_ctx();
    if (!c || vk_device_is_lost()) return;
    VkFence f = __atomic_load_n(&lost_fence, __ATOMIC_ACQUIRE);
    if (!f) {
        plat_mutex_lock(&lost_fence_lock);
        if (!lost_fence) {
            VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            VkFence n = VK_NULL_HANDLE;
            VkResult r = vkCreateFence(c->device, &fi, NULL, &n);
            if (r == VK_SUCCESS) __atomic_store_n(&lost_fence, n, __ATOMIC_RELEASE);
            else vk_device_lost_check(r, "the fence that asks whether the device lives");
        }
        f = lost_fence;
        plat_mutex_unlock(&lost_fence_lock);
        if (!f) return;
    }
    vk_device_lost_check(vkGetFenceStatus(c->device, f), "a poll of the device");
}

int vk_selftest(void) {
    if (!vk_init()) return 1;
    VkCtx *c = &g_vk;
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 64 << 10,
                             .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
    VmaAllocationCreateInfo aci = {.usage = VMA_MEMORY_USAGE_AUTO, .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT};
    VkBuffer buf;
    VmaAllocation alloc;
    VkResult r = vmaCreateBuffer(c->vma, &bi, &aci, &buf, &alloc, NULL);
    if (r != VK_SUCCESS) { rt_log("vk selftest: vmaCreateBuffer failed (%d)", (int)r); return 2; }
    vmaDestroyBuffer(c->vma, buf, alloc);

    VkSemaphoreSignalInfo sig = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO, .semaphore = c->timeline, .value = 1};
    if ((r = vkSignalSemaphore(c->device, &sig)) != VK_SUCCESS) { rt_log("vk selftest: vkSignalSemaphore failed (%d)", (int)r); return 3; }
    uint64_t value = 1;
    VkSemaphoreWaitInfo wi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, .semaphoreCount = 1, .pSemaphores = &c->timeline,
                              .pValues = &value};
    if ((r = vkWaitSemaphores(c->device, &wi, 1000000000ull)) != VK_SUCCESS) { rt_log("vk selftest: vkWaitSemaphores failed (%d)", (int)r); return 4; }
    uint64_t now = 0;
    vkGetSemaphoreCounterValue(c->device, c->timeline, &now);
    if (now != 1) { rt_log("vk selftest: timeline value %llu, expected 1", (unsigned long long)now); return 5; }
    return 0;
}

// --probe-vkdevice (host.h)
int probe_vkdevice(void) {
    int rc = vk_selftest();
    VkCtx *c = vk_ctx();
    if (rc || !c) { printf("probe-vkdevice: FAILED (selftest %d)\n", rc ? rc : 1); vk_shutdown(); return 1; }
    uint32_t v = c->props.apiVersion;
    printf("probe-vkdevice: device %d \"%s\" api %u.%u.%u, gfx queue family %u, transfer family %u%s, host import alignment %#x\n",
           c->device_index, c->props.deviceName, VK_API_VERSION_MAJOR(v), VK_API_VERSION_MINOR(v), VK_API_VERSION_PATCH(v),
           c->gfx_family, c->xfer_family, c->xfer_shared ? " (shared queue)" : "", c->min_imported_host_alignment);
    printf("probe-vkdevice: optional: custom_border=%d eds=%d eds2=%d eds3=%d vertex_input=%d cond_render=%d xfb=%d provoking=%d "
           "robustness2=%d fault=%d budget=%d bda=%d interlock=%d buffer_marker=%d nv_checkpoints=%d validation=%d\n",
           c->has_custom_border_color, c->has_eds, c->has_eds2, c->has_eds3, c->has_vertex_input_dynamic_state,
           c->has_conditional_rendering, c->has_transform_feedback, c->has_provoking_vertex, c->has_robustness2,
           c->has_device_fault, c->has_memory_budget, c->has_buffer_device_address, c->has_fragment_shader_interlock,
           c->has_buffer_marker, c->has_nv_checkpoints, c->validation);
    fatal_note();   // the line a fatal error or a crash would add to the log, to read it without one
    vk_shutdown();
    printf("probe-vkdevice: %s (validation errors %ld)\n", g_errors_total ? "FAILED" : "passed", g_errors_total);
    return g_errors_total ? 1 : 0;
}
