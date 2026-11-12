// DLAA: NVIDIA DLSS at the scene's own size (NX2NAVI_DLAA, nx2navi.ini dlaa). NGX is loaded at run time from the driver's
// NGX core: on Windows _nvngx.dll, found through HKLM\SOFTWARE\NVIDIA Corporation\Global\NGXCore; on Linux
// libnvidia-ngx.so.1, which the NVIDIA driver installs on the library path. NGX loads the DLSS model from the folder of
// the executable: nvngx_dlss.dll on Windows, libnvidia-ngx-dlss.so.<version> on Linux. No NVIDIA SDK code or header is
// compiled in: the few NGX types used are declared here from the SDK ABI and the Linux core's shutdown interface.
//
// The core's parameter object is a C++ class whose virtual functions are called through its vtable: the Set overloads
// in slots 0-7 and the Get overloads in 8-15. MSVC (the Windows core) lays out each group of overloads in reverse
// declaration order, the Itanium C++ ABI (the Linux core, built with GCC) in declaration order. vtable_ok checks the
// layout with a set-then-get round trip before anything relies on it.
//
// NGX and DLAA's resources live as long as the device: vk_dlaa_init brings NGX up on the new device, and
// vk_dlaa_release (Linux) frees everything DLAA made and shuts NGX down before the device is destroyed.
//
// Where DLAA runs in the frame, the jitter and the motion vectors: see the sections below.
#ifdef _WIN32
#include <windows.h>
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

#include "cfg.h"
#include "gfx.h"
#include "host.h"
#include "nvn_api.h"
#include "platform/platform.h"
#include "vk.h"
#include "vk_dlaa.h"
#include "vk_exec.h"
#include "vk_ffx.h"
#include "vk_rec.h"
#include "dlaa_mv_spv.h"

typedef int NgxResult;   // NVSDK_NGX_Result: 1 success, 0xBAD00000 | n failures
#define NGX_OK(r) (((unsigned)(r) & 0xFFF00000u) != 0xBAD00000u)
#define NGX_VERSION_API 0x15
#define NGX_FEATURE_SUPERSAMPLING 1
#define NGX_PERFQ_DLAA 5   // NVSDK_NGX_PerfQuality_Value: 0 performance, 1 balanced, 2 quality, 3 ultra performance
typedef struct NgxParam NgxParam;
typedef struct NgxHandle NgxHandle;
typedef struct { void *slot[17]; } NgxParamVtbl;
struct NgxParam { const NgxParamVtbl *vt; };
typedef struct {   // NVSDK_NGX_ImageViewInfo_VK
    VkImageView view;
    VkImage image;
    VkImageSubresourceRange range;
    VkFormat format;
    unsigned width, height;
} NgxImageView;
typedef struct {   // NVSDK_NGX_Resource_VK
    union { NgxImageView img; struct { VkBuffer buffer; unsigned bytes; } buf; } r;
    int type;      // 0 image view, 1 buffer
    _Bool rw;
} NgxResourceVK;

typedef NgxResult (*PFN_Init)(unsigned long long app, const wchar_t *path, VkInstance, VkPhysicalDevice, VkDevice,
                              PFN_vkGetInstanceProcAddr, PFN_vkGetDeviceProcAddr, int version, const void *info);
typedef NgxResult (*PFN_ReqExt)(unsigned *ni, const char ***ie, unsigned *nd, const char ***de);
typedef NgxResult (*PFN_Params)(NgxParam **);
typedef NgxResult (*PFN_DestroyParams)(NgxParam *);
typedef NgxResult (*PFN_Create)(VkCommandBuffer, int feature, NgxParam *, NgxHandle **);
typedef NgxResult (*PFN_Evaluate)(VkCommandBuffer, const NgxHandle *, const NgxParam *, void *callback);
typedef NgxResult (*PFN_Release)(NgxHandle *);
#ifndef _WIN32
// NVSDK_NGX_VULKAN_Shutdown1 as the Linux core exports it: the device and a pointer that receives the devices NGX still
// serves afterwards (vk_dlaa_release).
typedef NgxResult (*PFN_Shutdown1)(VkDevice, unsigned *remaining);
#endif

static struct {
    void *core;   // the NGX core library (an HMODULE on Windows)
    PFN_Init init;
    PFN_ReqExt req_ext;
    PFN_Params caps, alloc_params;
    PFN_DestroyParams destroy_params;
    PFN_Create create;
    PFN_Evaluate evaluate;
    PFN_Release release;
    void *feat_inst_req;   // NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements (vk_ngxfg), NULL when the core lacks it
    int tried, loaded, available;
    int ngx_up;   // vk_dlaa_init brought NGX up: DLSS serves this run
    NgxParam *capsp;   // the capability parameters (vk_dlaa_init), which DLSS's statistics callback reads and fills
    int wanted;   // -1 not read yet
#ifndef _WIN32
    // Linux teardown (vk_dlaa_release)
    PFN_Shutdown1 shutdown1;   // required: core_load leaves DLAA off without it
    VkDevice device;           // what NGX was initialised with: set while vk_dlaa_release owes it a shutdown
    NgxParam *caps_map;        // GetCapabilityParameters' map: allocated for the caller, destroyed by vk_dlaa_release
#endif
} N = {.wanted = -1};

// DLSS Frame Generation through NGX itself (vk_ngxfg_*, at the end of this file): NX2NAVI_FRAMEGEN = dlss where
// Streamline does not run, which is every build but the Windows one.
static struct {
    int max;                     // generated frames per rendered one this GPU allows (MultiFrameCountMax), 0: none
    NgxHandle *feat;
    NgxParam *p;
    uint32_t w, h;               // the picture the feature was created for
    VkFormat fmt;
    int failed, reset, evals;
} FG;

static int taa_is_fsr4(void) { return !strcmp(cfg_value("NX2NAVI_TAA"), "fsr4"); }
static int taa_is_fsr(void) { return !strcmp(cfg_value("NX2NAVI_TAA"), "fsr3") || taa_is_fsr4(); }

// Whether this run can generate frames through NGX: DLSS upscaling on at the start, as NGX comes up only then. The
// swapchain then gets the presenter's extra images and NGX is asked about frame generation; whether frames are
// generated is NX2NAVI_FRAMEGEN = dlss, read live (vk_ngxfg_on).
int vk_ngxfg_wanted(void) {
#ifdef _WIN32
    return 0;   // Streamline's path there (vk_dlssg.c)
#else
    static int w = -1;
    if (w < 0) w = cfg_bool("NX2NAVI_DLAA") && !taa_is_fsr();
    return w;
#endif
}

typedef void (*SetI)(NgxParam *, const char *, int);
typedef void (*SetU)(NgxParam *, const char *, unsigned);
typedef void (*SetF)(NgxParam *, const char *, float);
typedef void (*SetP)(NgxParam *, const char *, void *);
typedef NgxResult (*GetI)(const NgxParam *, const char *, int *);
typedef NgxResult (*GetU)(const NgxParam *, const char *, unsigned *);
typedef NgxResult (*GetF)(const NgxParam *, const char *, float *);
typedef NgxResult (*GetP)(const NgxParam *, const char *, void **);
typedef NgxResult (*GetU64)(const NgxParam *, const char *, unsigned long long *);
// The SDK declares Set(u64), Set(float), Set(double), Set(u32), Set(int), Set(ID3D11Resource *), Set(ID3D12Resource *),
// Set(void *), then the Get overloads in the same order, then Reset.
#ifdef _WIN32
enum { S_P = 0, S_D12, S_D11, S_I, S_U32, S_D, S_F, S_U64, G_P, G_D12, G_D11, G_I, G_U32, G_D, G_F, G_U64 };
#else
enum { S_U64 = 0, S_F, S_D, S_U32, S_I, S_D11, S_D12, S_P, G_U64, G_F, G_D, G_U32, G_I, G_D11, G_D12, G_P };
#endif
static void p_seti(NgxParam *p, const char *n, int v) { ((SetI)p->vt->slot[S_I])(p, n, v); }
static void p_setu(NgxParam *p, const char *n, unsigned v) { ((SetU)p->vt->slot[S_U32])(p, n, v); }
static void p_setf(NgxParam *p, const char *n, float v) { ((SetF)p->vt->slot[S_F])(p, n, v); }
static void p_setp(NgxParam *p, const char *n, void *v) { ((SetP)p->vt->slot[S_P])(p, n, v); }
static int p_geti(const NgxParam *p, const char *n, int *v) { return NGX_OK(((GetI)p->vt->slot[G_I])(p, n, v)); }
static int p_getu(const NgxParam *p, const char *n, unsigned *v) { return NGX_OK(((GetU)p->vt->slot[G_U32])(p, n, v)); }
static int p_getf(const NgxParam *p, const char *n, float *v) { return NGX_OK(((GetF)p->vt->slot[G_F])(p, n, v)); }
static int p_getp(const NgxParam *p, const char *n, void **v) { return NGX_OK(((GetP)p->vt->slot[G_P])(p, n, v)); }
static int p_getull(const NgxParam *p, const char *n, unsigned long long *v) {
    return NGX_OK(((GetU64)p->vt->slot[G_U64])(p, n, v));
}

#ifdef _WIN32
#define DLSS_MODEL "nvngx_dlss.dll"
#else
#define DLSS_MODEL "libnvidia-ngx-dlss.so.<version>"
#endif

// NGX is brought up only when DLSS is wanted at start: its device extensions can only be asked for at device creation,
// and loading the driver's NGX core on every NVIDIA machine would cost start time and change the device with DLAA off.
// So DLSS applies after a restart. FSR 3.1 (amd_fidelityfx_vk.dll, vk_ffx.cpp) asks nothing of the device: NX2NAVI_TAA
// and dlaa follow the setting at once when NGX is not up, and vk_dlaa_frame loads the library the first time it is used.
int vk_dlaa_wanted(void) {
    static int at_start = -1;
    if (at_start < 0) at_start = cfg_bool("NX2NAVI_DLAA");
    return at_start || (!N.ngx_up && cfg_bool("NX2NAVI_DLAA") && taa_is_fsr());
}

// The temporal upscaler behind dlaa: NVIDIA DLSS through NGX, once NGX is up, or else AMD FSR 3.1 / FSR 4, which take
// the same inputs. The DLSS mode setting (NX2NAVI_DLSS) is the render scale for both.
static int fsr3(void) { return !N.ngx_up && taa_is_fsr(); }
static int fsr_frame;   // fsr3() as vk_dlaa_init and then vk_dlaa_frame read it: the upscaler this frame runs with
// FSR 3.1 / FSR 4 is the upscaler and its library loads (vk_ffx_available / vk_ffx_fsr4_available): widescreen.c
// shrinks the scene only for an upscaler that runs.
int vk_dlaa_fsr3(void) {
    return vk_dlaa_wanted() && fsr3() && (taa_is_fsr4() ? vk_ffx_fsr4_available() : vk_ffx_available());
}

// The upscaler the menu has set cannot start in this run: DLSS when NGX was never tried (a device made without its
// extensions), FSR while NGX is up and serves the run. A DLSS this PC could not bring up is not waiting for a restart.
int vk_dlaa_needs_restart(void) {
    if (!cfg_bool("NX2NAVI_DLAA")) return 0;
    return !taa_is_fsr() ? !N.ngx_up && !N.tried : N.ngx_up;
}

#ifdef _WIN32
// The NGX core's folder when the driver has not written NGXCore\FullPath (seen with driver 617.42, whose key held
// only DLSS override values): the core sits in the driver's own DriverStore folder, the one its Vulkan driver is in.
// That is the display adapter's VulkanDriverName, where the Vulkan loader itself finds the driver.
static int ngx_dir_from_adapter(char *dir, DWORD size) {
    HKEY cls;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, "SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}",
                      0, KEY_READ, &cls) != ERROR_SUCCESS)
        return 0;
    int found = 0;
    for (DWORD i = 0; !found; i++) {
        char sub[64], val[MAX_PATH + 1] = {0}, probe[MAX_PATH + 32];
        DWORD sn = sizeof sub, vn = MAX_PATH;
        if (RegEnumKeyExA(cls, i, sub, &sn, NULL, NULL, NULL, NULL) != ERROR_SUCCESS) break;
        if (RegGetValueA(cls, sub, "VulkanDriverName", RRF_RT_REG_SZ | RRF_RT_REG_MULTI_SZ, NULL, val, &vn) != ERROR_SUCCESS)
            continue;
        char *slash = strrchr(val, '\\');
        if (!slash) continue;
        *slash = 0;
        snprintf(probe, sizeof probe, "%s\\_nvngx.dll", val);
        if (GetFileAttributesA(probe) == INVALID_FILE_ATTRIBUTES) continue;
        snprintf(dir, size, "%s", val);
        found = 1;
    }
    RegCloseKey(cls);
    return found;
}
#endif

static int core_load(void) {
    if (N.tried) return N.loaded;
    N.tried = 1;
#ifdef _WIN32
    char dir[MAX_PATH] = {0}, path[MAX_PATH + 32];
    DWORD n = sizeof dir;
    if (RegGetValueA(HKEY_LOCAL_MACHINE, "SOFTWARE\\NVIDIA Corporation\\Global\\NGXCore", "FullPath", RRF_RT_REG_SZ,
                     NULL, dir, &n) != ERROR_SUCCESS && !ngx_dir_from_adapter(dir, sizeof dir)) {
        rt_log("dlaa: the driver has no NGX core (registry NGXCore missing, none beside its Vulkan driver): DLAA off");
        return 0;
    }
    snprintf(path, sizeof path, "%s\\_nvngx.dll", dir);
    N.core = LoadLibraryA(path);
    if (!N.core) {
        rt_log("dlaa: cannot load %s (error %lu): DLAA off", path, GetLastError());
        return 0;
    }
#define FN(f, name) N.f = (void *)GetProcAddress((HMODULE)N.core, name)
#else
    // The driver's NGX core; the dynamic loader finds it where the driver put it (/usr/lib, /usr/lib/x86_64-linux-gnu).
    const char *path = "libnvidia-ngx.so.1";
    N.core = plat_lib_open(path);
    if (!N.core) {
        rt_log("dlaa: cannot load %s (not an NVIDIA driver, or one without NGX): DLAA off", path);
        return 0;
    }
#define FN(f, name) N.f = plat_lib_sym(N.core, name)
#endif
    FN(init, "NVSDK_NGX_VULKAN_Init_Ext2");
#ifndef _WIN32
    FN(shutdown1, "NVSDK_NGX_VULKAN_Shutdown1");   // Linux: checked below
#endif
    FN(req_ext, "NVSDK_NGX_VULKAN_RequiredExtensions");
    FN(caps, "NVSDK_NGX_VULKAN_GetCapabilityParameters");
    FN(alloc_params, "NVSDK_NGX_VULKAN_AllocateParameters");
    FN(destroy_params, "NVSDK_NGX_VULKAN_DestroyParameters");
    FN(create, "NVSDK_NGX_VULKAN_CreateFeature");
    FN(evaluate, "NVSDK_NGX_VULKAN_EvaluateFeature");
    FN(release, "NVSDK_NGX_VULKAN_ReleaseFeature");
    FN(feat_inst_req, "NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements");   // optional: frame generation's
#undef FN
    if (!N.init || !N.req_ext || !N.caps || !N.alloc_params || !N.create || !N.evaluate || !N.release) {
        rt_log("dlaa: the NGX core %s lacks Vulkan entry points: DLAA off", path);
        return 0;
    }
#ifndef _WIN32
    // Without Shutdown1 the device cannot be taken back from NGX (vk_dlaa_release), and the process then segfaults after
    // the Vulkan teardown (NVIDIA 615.78.08): DLAA stays off rather than run with a teardown known to crash. The
    // parameter maps are destroyed there too.
    if (!N.shutdown1 || !N.destroy_params) {
        rt_log("dlaa: the NGX core %s lacks NVSDK_NGX_VULKAN_Shutdown1 or DestroyParameters, which its teardown needs: "
               "DLAA off", path);
        return 0;
    }
#endif
    N.loaded = 1;
    rt_log("dlaa: NGX core %s", path);
    return 1;
}

static int listed(const char *const *list, uint32_t n, const char *s) {
    for (uint32_t i = 0; i < n; i++)
        if (!strcmp(list[i], s)) return 1;
    return 0;
}

// NVSDK_NGX_FeatureDiscoveryInfo with an application id (the layout of the SDK's struct, 64-bit), for the question
// which instance extensions frame generation wants.
typedef struct {
    int sdk, feature;
    struct { int type; union { struct { const char *id; int engine; const char *ver; } project; unsigned long long app; } v; } ident;
    const wchar_t *path;
    const void *info;
} NgxDiscovery;
typedef NgxResult (*PFN_FeatInstReq)(const NgxDiscovery *, uint32_t *, VkExtensionProperties **);
#define NGX_FEATURE_FRAME_GENERATION 11   // NVSDK_NGX_Feature_FrameGeneration
// The names NGX lists are copied: its list may not outlive the call, and the device creation keeps the pointers.
static char fg_names[16][VK_MAX_EXTENSION_NAME_SIZE];
static int fg_nnames;
static const char *fg_keep(const char *name) {
    for (int i = 0; i < fg_nnames; i++)
        if (!strcmp(fg_names[i], name)) return fg_names[i];
    if (fg_nnames >= 16) return NULL;
    snprintf(fg_names[fg_nnames], sizeof fg_names[0], "%s", name);
    return fg_names[fg_nnames++];
}

void vk_dlaa_instance_exts(const char **ext, uint32_t *n, uint32_t max) {
    unsigned ni = 0, nd = 0;
    const char **ie = NULL, **de = NULL;
    if (!vk_dlaa_wanted() || fsr3() || !core_load() || !NGX_OK(N.req_ext(&ni, &ie, &nd, &de))) return;
    for (unsigned i = 0; i < ni; i++)
        if (!listed(ext, *n, ie[i]) && *n < max) ext[(*n)++] = ie[i];
    // Frame generation's own instance extensions (vk_ngxfg_evaluate). Its device extensions are not asked for: the
    // core's GetFeatureDeviceExtensionRequirements answered "not implemented" (0xBAD00012) with NVIDIA's 615 Linux
    // driver, and NGX's general list (vk_dlaa_device_exts) served the feature.
    if (vk_ngxfg_wanted() && N.feat_inst_req) {
        static const wchar_t here[] = L".";
        NgxDiscovery d;
        memset(&d, 0, sizeof d);
        d.sdk = NGX_VERSION_API, d.feature = NGX_FEATURE_FRAME_GENERATION;
        d.ident.type = 0, d.ident.v.app = 231313132ull;   // the application id vk_dlaa_init gives NGX
        d.path = here;
        uint32_t k = 0;
        VkExtensionProperties *pr = NULL;
        if (NGX_OK(((PFN_FeatInstReq)N.feat_inst_req)(&d, &k, &pr)) && pr)
            for (uint32_t i = 0; i < k; i++) {
                const char *nm = fg_keep(pr[i].extensionName);
                if (nm && !listed(ext, *n, nm) && *n < max) ext[(*n)++] = nm, rt_log("ngxfg: instance extension %s", nm);
            }
    }
}

void vk_dlaa_device_exts(const VkExtensionProperties *have, uint32_t nhave, const char **ext, uint32_t *n,
                         uint32_t max) {
    unsigned ni = 0, nd = 0;
    const char **ie = NULL, **de = NULL;
    if (!vk_dlaa_wanted() || fsr3() || !core_load() || !NGX_OK(N.req_ext(&ni, &ie, &nd, &de))) return;
    for (unsigned i = 0; i < nd; i++) {
        int ok = 0;
        for (uint32_t k = 0; k < nhave && !ok; k++) ok = !strcmp(have[k].extensionName, de[i]);
        if (!ok) rt_log("dlaa: the device lacks %s, which NGX asks for", de[i]);
        else if (!listed(ext, *n, de[i]) && *n < max) ext[(*n)++] = de[i];
    }
}

static int vtable_ok(void) {
    NgxParam *p = NULL;
    if (!NGX_OK(N.alloc_params(&p)) || !p) return 0;
    int i = 0;
    unsigned u = 0;
    float f = 0;
    void *v = NULL;
    p_seti(p, "NX2navi.Test.I", -12345);
    p_setu(p, "NX2navi.Test.U", 4000000000u);
    p_setf(p, "NX2navi.Test.F", 1.25f);
    p_setp(p, "NX2navi.Test.P", (void *)p);
    int ok = p_geti(p, "NX2navi.Test.I", &i) && i == -12345 && p_getu(p, "NX2navi.Test.U", &u) && u == 4000000000u &&
             p_getf(p, "NX2navi.Test.F", &f) && f == 1.25f && p_getp(p, "NX2navi.Test.P", &v) && v == (void *)p;
    if (N.destroy_params) N.destroy_params(p);
    if (!ok) rt_log("dlaa: NGX parameter layout check failed (i=%d u=%u f=%g p=%p): DLAA off", i, u, f, v);
    return ok;
}

#ifndef _WIN32
// UTF-8 to UTF-32 wchar_t, without the C library's locale (the runtime never sets one, so mbstowcs would stop at the
// first non-ASCII byte). Returns 0 on malformed input or when out has no room.
static int utf8_to_wide(const char *in, wchar_t *out, size_t cap) {
    const unsigned char *u = (const unsigned char *)in;
    size_t n = 0;
    while (*u) {
        unsigned c = *u++, more = c < 0x80 ? 0 : c >= 0xF0 ? 3 : c >= 0xE0 ? 2 : c >= 0xC0 ? 1 : 4;
        if (more == 4) return 0;
        if (more) c &= 0x3F >> more;
        for (unsigned k = 0; k < more; k++, u++) {
            if ((*u & 0xC0) != 0x80) return 0;
            c = c << 6 | (*u & 0x3F);
        }
        if (n + 1 >= cap) return 0;
        out[n++] = (wchar_t)c;
    }
    out[n] = 0;
    return 1;
}
#endif

void vk_dlaa_init(void) {
    if (vk_dlaa_wanted() && fsr3()) {
        fsr_frame = 1;
        int is4 = taa_is_fsr4();
        N.available = is4 ? vk_ffx_load_fsr4() : vk_ffx_load();
        rt_log("dlaa: the temporal upscaler is AMD FSR %s%s", is4 ? "4" : "3.1",
               N.available ? "" : ", which is not available: the game's own scaling");
        return;
    }
    if (!vk_dlaa_wanted() || !core_load()) return;
#ifdef _WIN32
    wchar_t wdir[MAX_PATH];
    DWORD len = GetModuleFileNameW(NULL, wdir, MAX_PATH);
    while (len && wdir[len - 1] != L'\\') len--;
    wdir[len] = 0;
#else
    // The executable's folder, where NGX looks for libnvidia-ngx-dlss.so and writes its logs, as a wchar_t string:
    // 32-bit code points here, as the Linux core expects.
    char dir[PLAT_PATH_MAX];
    wchar_t wdir[PLAT_PATH_MAX];
    char *cut = plat_exe_path(dir, sizeof dir) ? plat_last_sep(dir) : NULL;
    if (!cut || !utf8_to_wide((cut[1] = 0, dir), wdir, PLAT_PATH_MAX)) {
        rt_log("dlaa: cannot name the executable's folder for NGX: DLAA off");
        return;
    }
#endif
    // 231313132: the application id NVIDIA's DLSS samples use for builds without an NVIDIA-issued id
    const VkCtx *c = vk_ctx();
    NgxResult r = N.init(231313132ull, wdir, c->instance, c->phys, c->device, vkGetInstanceProcAddr,
                         vkGetDeviceProcAddr, NGX_VERSION_API, NULL);
    if (!NGX_OK(r)) {
        rt_log("dlaa: NGX init failed (%08x): DLAA off", (unsigned)r);
        return;
    }
#ifndef _WIN32
    N.device = c->device;   // NGX holds the device from here, also if the layout check below turns DLAA off
#endif
    if (!vtable_ok()) return;
    NgxParam *caps = NULL;
    int avail = 0, driver = 0;
    if (NGX_OK(N.caps(&caps)) && caps) {
#ifndef _WIN32
        N.caps_map = caps;   // allocated for us (the SDK header: free it with DestroyParameters): vk_dlaa_release does
#endif
        p_geti(caps, "SuperSampling.Available", &avail);
        p_geti(caps, "SuperSampling.NeedsUpdatedDriver", &driver);
    }
    N.available = avail != 0;
    N.capsp = caps;
    N.ngx_up = 1;
    rt_log("dlaa: NGX up; DLSS %s%s", avail ? "available" : "NOT available (is " DLSS_MODEL " next to the executable?)",
           driver ? "; the driver needs an update" : "");
    if (vk_ngxfg_wanted() && caps) {
        int fga = 0;
        unsigned mx = 0;
        p_geti(caps, "FrameGeneration.Available", &fga);
        if (!p_getu(caps, "DLSSG.MultiFrameCountMax", &mx)) mx = 1;
        // MultiFrameCountMax said 3 (4x) with the 310.9.16 library on an RTX 5090, yet that library evaluated 5
        // generated frames without an error on NVIDIA's 615 driver, and NVIDIA ships up to 6x on RTX 50 cards: a GPU
        // with multi-frame generation gets up to 5 (6x).
        FG.max = fga ? (mx >= 3 ? 5 : mx < 1 ? 1 : (int)mx) : 0;
        rt_log("ngxfg: DLSS Frame Generation %s (up to %dx)",
               fga ? "available" : "NOT available (is libnvidia-ngx-dlssg.so.<version> next to the executable?)", FG.max + 1);
    }
}

int vk_dlaa_available(void) { return vk_dlaa_wanted() && N.available; }

// ---------------------------------------------------------------- where DLAA runs in NX2navi's frame
// Measured (NX2NAVI_DUMP, frame 2100 at Lookout Landing, 2560x1440 scene): the G-buffer passes draw into five targets with
// the scene depth (d32f at the scene size), one of them the R11G11B10F light target; lighting and the transparents
// then draw into that light target alone, the last time with the scene depth attached; the first pass narrower than a
// quarter of the scene starts the bloom chain, which reads it. After the bloom come the tonemap and post passes, then
// the HUD is drawn into the same target and a last pass copies it to the 8-bit output.
// So: after the G-buffer, a pass at the scene size with exactly one colour target (R11G11B10F or RGBA16F) and the D32F
// depth arms the frame with that pair; the next pass at most a quarter as wide ends the scene (no jitter after it) and,
// when the scene is as wide as the output, runs DLAA on the armed target first. Bloom, tonemap, post and HUD then all
// work on the anti-aliased picture.
// Upscaling (a scene smaller than the output: NX2NAVI_DLSS quality/balanced/performance, or the game's own 5/6): measured
// at a 960x536 scene, 1920x1080 output, the bloom and its composite write back into the light target at the scene size,
// then one full-screen draw (the FXAA shader below, which also scales) reads it into an output-sized R11G11B10F target
// with no depth, and the HUD is drawn over that target at the output size. DLSS runs at the begin of that pass: it
// reads the light target and its result is blitted into the pass's target, whose scaling draw is then skipped. The
// HUD, drawn afterwards into the same target, is never jittered nor passed through DLSS.
// The scaling draw is not always that FXAA shader: at a 1280x720 scene and 1920x1080 output (1.4.3, dlss = quality)
// the game's tonemap (guest fs 7e114c4330dc212a) reads the light target and writes the output-sized target itself,
// over DLSS's picture, so the jittered scene reached the screen and the whole picture shook. So from DLSS's evaluation
// to the frame's end, draws in passes larger than the light target sample DLSS's output in its place
// (vk_dlaa_swap_img, applied by vk_exec_draw.c tex_info): whichever shader scales, it scales DLSS's picture.
#define OCAM_BLOCK 0x3500u   // Ocarina's scene context block (vk_dlaa_draw)
#define OCAM_BYTES 0xb0u     // its rows 0-10: the view and the projection
#define DLSS_SIZES 5         // DLSS features kept, one per scene and output size (D.sizes)
// A size no evaluation used for this many goes with its feature (stale_sizes): DLSS's history buffers and the work
// images at the output size held their VRAM after a switch of the DLSS mode, until four more sizes had come and gone.
#define SIZE_STALE 3600
static struct {
    VkExecTarget color, depth;   // the armed pair, valid in the frame it was armed
    int armed, done, gbuf;       // this frame: armed on the light target, DLAA ran, the G-buffer was drawn
    int ended, up;               // this frame: the scene is drawn (jitter over), DLSS wrote the scaled output
    struct Size {                // a DLSS feature and its work images per scene size (dynamic resolution flips it)
        uint32_t w, h, ow, oh;   // render size, output size
        VkImage out, mv;
        VmaAllocation out_mem, mv_mem;
        VkImageView out_view, mv_view;
        VkImage mask, chars[2];  // the bias mask DLSS reads, the character class mask of this and last frame (R8)
        VmaAllocation mask_mem, chars_mem[2];
        VkImageView mask_view, chars_view[2];
        int chars_cur, chars_ok; // chars[chars_cur] is written next; chars_ok: the other holds last frame's
        VkImage dist[2];         // each pixel's view distance, this and last frame (R32F, dlaa_mv.comp depth_out)
        VmaAllocation dist_mem[2];
        VkImageView dist_view[2];
        int dist_cur, dist_ok;   // as chars_cur and chars_ok
        NgxHandle *feat;
        uint64_t used;           // evals count when it last ran
    } sizes[DLSS_SIZES], *last;  // last: the one the previous DLAA frame ran with
    VkImage last_color;          // the target DLAA ran on last frame: another one means a new history
    NgxParam *params;
    int failed, reset;
    uint64_t evals, frames_missed;
    int share_off;   // frames left that pick the output pass without the mode's share (vk_dlaa_pass, vk_dlaa_frame)
    int shape_logged;   // passes of another shape than the scene's turned down as the output pass (vk_dlaa_pass)
    // jitter (vk_dlaa_viewport)
    VkImage jdepth;              // the scene depth of the last frame DLAA ran on: passes drawing with it are jittered
    uint32_t jw, jh;             // its size: on Ocarina's frame the camera's draws into any D32F depth of it are (jitter_draw)
    float jx, jy;                // the offset the next scene draws take, render pixels (window x right, y down)
    float ux, uy;                // the offset this frame's scene draws took (DLSS and the motion vectors get these)
    uint32_t jframe;
    int jit_ok, jit_used, hold;  // jitter allowed this frame / a draw used it / frames left without it
    uint64_t holds, jittered;   // jitter holds; evaluations of a jittered scene
    uint64_t jdraws, jfixed;    // draws into a scene-size depth that took the jitter / that did not (full-screen)
    // camera (vk_dlaa_draw) and motion vectors (mv_pass)
    uint64_t cam_addr;           // this frame's scene camera block (vk_dlaa_draw), 0: none seen
    uint32_t cam_size, cam_vp;   // its size and the offset of its view-projection rows
    struct {                     // Ocarina: this frame's scene contexts of light-target draws
        uint64_t addr, seq;      // seq: vk_exec_submit_seq at the last draw that used it
        uint32_t n;              // draws
        int got;                 // cam holds its rows 0-10 as the first draw read them (vk_usnap_read: 1, 2), 0 not yet
        float cam[OCAM_BYTES / 4];
    } views[8];
    int nviews;
    VkBuffer camb;               // two CAM_SLOT slots: this frame's block and last frame's
    VmaAllocation camb_mem;
    VkBuffer mvm;                // MVM_SLOTS pairs of blocks for dlaa_mv.comp: the identity and the reprojection (mv_pass)
    VmaAllocation mvm_mem;
    uint8_t *mvm_map;
    uint32_t mvm_slot;
    double cam_prev[16];         // last frame's view-projection (rows 3-6 of its block), for the reprojection
    int cam_slot, cam_prev_ok;   // the slot this frame writes / the other slot holds last frame's camera
    VkDescriptorSetLayout mv_dsl;
    VkPipelineLayout mv_layout;
    VkPipeline mv_pipe;
    VkSampler mv_samp;
    int mv_ok;                   // -1 not built, 0 failed, 1 ready
    uint64_t mv_frames, mv_none, masked;   // masked: frames with the characters' bias mask
    uint64_t why[3];             // history resets: scene size changed, no camera, scene target changed
    uint64_t target_kept;        // scene target changed and the history stayed (target_changed)
    // frames without a camera and what mv_pass did about them (dlaa_cam_keep, cam_note)
    uint64_t cam_none_draw, cam_none_snap;   // no scene-context draw this frame / one, not in the evaluating submission's copy
    uint64_t cam_at_draw;        // camera taken from the bytes the draw read, where the evaluating submission lacks them
    uint64_t cam_kept, cam_kept_max;   // no camera, history kept with zero motion / runs cut short at KEEP_MAX
    float cut_pos[3], cut_fwd[3];      // last frame's camera position and direction (cam_cut)
    uint64_t cut_ns, cuts;             // when it was read; camera cuts that started a new history
    int cut_ok, cut_logged;
    int keep_run;                // frames in a row kept without a camera
    uint64_t gap_logged, miss_logged, target_logged, second_logged;
    uint64_t seq_present;        // vk_exec_submit_seq at the last present (vk_dlaa_frame)
    uint64_t seconds;            // present intervals with a second scene after DLSS ran
    int on;                      // the dlaa setting, read at each present (the menu switches it live)
    int preset, preset_dirty;    // the DLSS model features are created with (preset_value); a change pending
    VkExecTarget ids;            // this frame's G-buffer target 0 (object classes in green), valid with ids_ok
    int ids_ok;
    unsigned cfg_seen;           // cfg_version when preset was read
    uint64_t last_eval_ns;       // plat_now_ns of the previous evaluation (FrameTimeDeltaInMsec)
    VkFfxImage fg_depth, fg_mv;  // the last evaluation's depth and motion vectors, for frame generation
    VkImageView fg_depth_view, fg_mv_view;
    VkDlaaCamera fg_cam;         // this frame's camera (mv_pass), fg_cam_ok when known
    int fg_cam_ok;
    float fg_jx, fg_jy;
    int fg_fresh, fg_reset;      // fresh: not handed out yet (vk_dlaa_fg_inputs)
    uint64_t presents;           // vk_dlaa_frame calls so far
    uint64_t fg_at, hl_at;       // presents when the inputs and the HUD-less picture were made (fg_current)
    // the picture without the HUD for frame generation (vk_dlaa_fg_hudless): DLSS's output encoded to sRGB as the game's
    // last pass encodes its picture, then copied bit for bit into an RGBA8 UNORM image, the format the present reads; with
    // HDR on, DLSS's output as RGBA16F, as the game's float picture the present encodes (hl_srgb unused)
    VkImage hl_srgb, hl;
    VmaAllocation hl_srgb_mem, hl_mem;
    VkImageView hl_view;
    uint32_t hl_w, hl_h;
    VkFormat hl_fmt;
    int hl_fresh;
} D = {.mv_ok = -1, .cfg_seen = ~0u};

// What this frame drew into a D32F depth at least DLAA_SCENE_MIN_W wide before DLSS ran (vk_dlaa_draw), for the line a
// frame without a camera logs (cam_note); cleared at each present (vk_dlaa_frame).
static struct {
    uint32_t zdraws, zother;      // draws into the scene depth / into another such depth than last frame's (jdepth)
    uint32_t zother_w, zother_h;
    uint32_t ocam, ocam_off;      // of the scene-depth draws: with the scene context at c3 / of those, not into a scene
    VkFormat ocam_off_fmt;        //   colour target (the format of the last such)
    uint32_t c3_other, c3_size;   // with another block at c3 (the size of the last such; 0: none bound)
    uint64_t seq0;                // vk_exec_submit_seq at the frame's first scene-depth draw
    uint32_t one_n, one_w, one_h; // after arming: passes with one colour target and no depth (vk_dlaa_pass), the last's
    VkFormat one_fmt;             //   size and format, for a frame without the DLAA point (vk_dlaa_frame)
    uint32_t second;              // after DLSS ran: passes shaped like the light target's (another scene)
    VkImage second_img;
} F;

// What an evaluation left for frame generation is for the present of its own frame. Frame generation takes it only
// while its swapchain is in place, so with frame generation off the last evaluation's images stayed marked fresh for
// as long as it was off. Among them is the game's own depth target, which the game replaces when the scene's size
// changes: switching the upscaler off and on again with frame generation set (the menu's Upscaling row passes through
// its choices without one) handed that destroyed image to frame generation's preparation, and AMD's driver ended the
// process. Images made more than FG_INPUT_PRESENTS presents ago are therefore never handed out, and vk_dlaa_frame
// drops them when the upscaler goes off or on.
#define FG_INPUT_PRESENTS 4   // an evaluation is taken at its own present, or the one after when the present thread is behind
static int fg_current(uint64_t made) { return D.presents - made <= FG_INPUT_PRESENTS; }

int vk_dlaa_fg_hudless(VkImage *img, VkImageView *view, uint32_t *w, uint32_t *h, VkFormat *fmt) {
    if (!D.hl_fresh) return 0;
    D.hl_fresh = 0;
    if (!fg_current(D.hl_at)) return 0;
    *img = D.hl, *view = D.hl_view, *w = D.hl_w, *h = D.hl_h, *fmt = D.hl_fmt;
    return 1;
}

int vk_dlaa_fg_views(VkImageView *depth, VkImageView *mv) {
    *depth = D.fg_depth_view, *mv = D.fg_mv_view;
    return D.fg_depth_view && D.fg_mv_view;
}

int vk_dlaa_fg_camera(VkDlaaCamera *c) {
    if (!D.fg_cam_ok) return 0;
    *c = D.fg_cam;
    return 1;
}

// Metres in one unit of Ocarina's view space, for FSR's distance heuristics: one. Measured in Hyrule Field, the camera
// orbits Link about 4 units behind him (the nearest scene depth, 0.975, is 1 - 0.1 / 4) at about 3 units a second, and
// travels about 4.7 units a second while he runs; in centimetres (the 0.01 FSR had) he would be 4 cm from the camera.
#define VIEW_METERS 1.0f

int vk_dlaa_ffx_camera(VkFfxCamera *c) {
    if (!D.fg_cam_ok) return 0;
    *c = (VkFfxCamera){D.fg_cam.fov_y, D.fg_cam.znear, D.fg_cam.zfar, VIEW_METERS};
    return 1;
}

int vk_dlaa_fg_inputs(VkFfxImage *depth, VkFfxImage *mv, float *jx, float *jy, int *reset) {
    if (!D.fg_fresh || D.failed) return 0;
    D.fg_fresh = 0;
    if (!fg_current(D.fg_at)) return 0;
    // reset: DLSS started a new history on this frame (evaluate). Handed out as 0 before, so frame generation was never
    // told of a camera cut, another scene size or another target, and interpolated across them: an area load
    // generated frames between two unrelated pictures.
    *depth = D.fg_depth, *mv = D.fg_mv, *jx = D.fg_jx, *jy = D.fg_jy, *reset = D.fg_reset;
    return 1;
}

const VkTexImage *vk_dlaa_swap_img;
VkImageView vk_dlaa_swap_view;

static int dl_active(void) { return N.available && !D.failed && D.on; }
// The upscaler this frame runs with, as vk_dlaa_frame read it. The menu can set NX2NAVI_TAA to dlss between that and
// evaluate; read again there while NGX is not up (DLSS starts only after a restart), it sent evaluate to DLSS's path
// with NGX's functions not loaded, a call to address 0 (issue #325).
static int use_fsr(void) { return !N.ngx_up && fsr_frame; }

static void dl_fail(const char *what, NgxResult r) {
    rt_log("dlaa: %s failed (%08x): DLAA off for this run", what, (unsigned)r);
    D.failed = 1;
}

static int image2d(uint32_t w, uint32_t h, VkFormat f, VkImageUsageFlags usage, VkImage *img, VmaAllocation *mem,
                   VkImageView *view) {
    const VkCtx *c = vk_ctx();
    VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = f,
                            .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                            .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage,
                            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(c->vma, &ii, &ai, img, mem, NULL) != VK_SUCCESS) return 0;
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = *img,
                                .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = f,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    return vkCreateImageView(c->device, &vi, NULL, view) == VK_SUCCESS;
}

// Which features may stay alive when another is made (NX2NAVI_DLSS_IDLE_FREE). A feature made while one with a larger
// input size is alive leaves that larger one drawing alternating bands of rows black or noise from then on, for as long
// as it is used; seen for the pause screen's feature (the picture's size in and out), in two ways:
// - a pause of 3,600 frames frees the gameplay feature (stale_sizes) and the unpause makes it again: every later pause
//   was striped, in each of 8 scripted runs of pause, unpause, pause in which it had been freed and in none of 12 in
//   which it had not, whatever the texture, write-back, HDR and frame generation settings;
// - a DLSS mode picked in the menu makes a feature of a third size at the unpause, with nothing freed: the next pause
//   was striped (performance to ultra performance in play).
// Giving the kept feature its own sizes again at each evaluation changed nothing. A feature made while a smaller one
// is alive leaves the smaller one as it was (the pause screen's after the gameplay one, at every first pause). So when
// a feature is made, the alive ones with a larger input are released first, and after a release since the last
// creation all of them (release_kept); their images stay and each is made again when its size is next evaluated.
// Scripted on one build, each way with the rule and with 2: the long pause and the mode change were striped without it
// and clean with it. The input size may not be what matters: in every striped case the feature made was of another
// model (performance's or ultra performance's) than the pause screen's, and the one run in which balanced's feature,
// of the pause screen's model, was made again beside it without the rule was clean. The rule covers both readings.
//   1 (default)  idle features are freed
//   0            idle features are never freed (about 0.5 GB more video memory after the first pause)
//   2            idle features are freed and nothing is released when a feature is made (as before; shows the fault)
static int feat_released;

static int idle_free(void) {
    static int m = -1;
    if (m < 0) m = cfg_int("NX2NAVI_DLSS_IDLE_FREE");
    return m;
}

// The images and views of a size, then its feature. The device must be idle: free_size waits for it, vk_dlaa_release
// has. Whatever was not made is skipped, so a half-made size goes the same way.
static void destroy_size(struct Size *e) {
    const VkCtx *c = vk_ctx();
    if (e->out_view) vkDestroyImageView(c->device, e->out_view, NULL);
    if (e->mv_view) vkDestroyImageView(c->device, e->mv_view, NULL);
    if (e->out) vmaDestroyImage(c->vma, e->out, e->out_mem);
    if (e->mv) vmaDestroyImage(c->vma, e->mv, e->mv_mem);
    if (e->mask_view) vkDestroyImageView(c->device, e->mask_view, NULL);
    if (e->mask) vmaDestroyImage(c->vma, e->mask, e->mask_mem);
    for (int i = 0; i < 2; i++) {
        if (e->chars_view[i]) vkDestroyImageView(c->device, e->chars_view[i], NULL);
        if (e->chars[i]) vmaDestroyImage(c->vma, e->chars[i], e->chars_mem[i]);
        if (e->dist_view[i]) vkDestroyImageView(c->device, e->dist_view[i], NULL);
        if (e->dist[i]) vmaDestroyImage(c->vma, e->dist[i], e->dist_mem[i]);
    }
    if (e->feat && N.release) N.release(e->feat), feat_released = 1;
    memset(e, 0, sizeof *e);
}

static void free_size(struct Size *e) {
    vk_queue_lock(), vkDeviceWaitIdle(vk_ctx()->device), vk_queue_unlock();   // a third scene size: the evicted images may still be in flight
    destroy_size(e);
}

// The entry for a scene and output size: its output (RGBA16F at the output size, written by DLSS, then blitted into
// the target) and motion vectors (RG16F at the scene size, render pixels), both kept in GENERAL like every image of the
// back end. NX2navi's dynamic resolution switches between a few sizes; each keeps its feature, the least recently used
// goes for a new one.
static struct Size *size_for(VkCommandBuffer cb, uint32_t w, uint32_t h, uint32_t ow, uint32_t oh) {
    struct Size *e = NULL, *lru = &D.sizes[0];
    for (int i = 0; i < DLSS_SIZES; i++) {
        if (D.sizes[i].w == w && D.sizes[i].h == h && D.sizes[i].ow == ow && D.sizes[i].oh == oh) return &D.sizes[i];
        if (!D.sizes[i].out && !e) e = &D.sizes[i];
        if (D.sizes[i].used < lru->used) lru = &D.sizes[i];
    }
    if (!e) {
        if (D.last == lru) D.last = NULL;
        free_size(e = lru);
    }
    VkImageUsageFlags u = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                          VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VkImageUsageFlags u8 = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    if (!image2d(ow, oh, VK_FORMAT_R16G16B16A16_SFLOAT, u, &e->out, &e->out_mem, &e->out_view) ||
        !image2d(w, h, VK_FORMAT_R16G16_SFLOAT, u | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &e->mv, &e->mv_mem, &e->mv_view) ||
        !image2d(w, h, VK_FORMAT_R8_UNORM, u8, &e->mask, &e->mask_mem, &e->mask_view) ||
        !image2d(w, h, VK_FORMAT_R8_UNORM, u8, &e->chars[0], &e->chars_mem[0], &e->chars_view[0]) ||
        !image2d(w, h, VK_FORMAT_R8_UNORM, u8, &e->chars[1], &e->chars_mem[1], &e->chars_view[1]) ||
        !image2d(w, h, VK_FORMAT_R32_SFLOAT, u8, &e->dist[0], &e->dist_mem[0], &e->dist_view[0]) ||
        !image2d(w, h, VK_FORMAT_R32_SFLOAT, u8, &e->dist[1], &e->dist_mem[1], &e->dist_view[1]))
        return NULL;
    VkImageMemoryBarrier2 ib[7];
    VkImage im[7] = {e->out, e->mv, e->mask, e->chars[0], e->chars[1], e->dist[0], e->dist[1]};
    for (int i = 0; i < 7; i++)
        ib[i] = (VkImageMemoryBarrier2){.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                        .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                        .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = im[i],
                                        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 7,
                           .pImageMemoryBarriers = ib};
    vkCmdPipelineBarrier2(cb, &di);
    e->w = w, e->h = h, e->ow = ow, e->oh = oh;
    return e;
}

static void full_barrier(VkCommandBuffer cb) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}

static NgxResourceVK res_image(VkImageView v, VkImage img, VkFormat f, uint32_t w, uint32_t h, VkImageAspectFlags a,
                               int rw) {
    NgxResourceVK r;
    memset(&r, 0, sizeof r);
    r.r.img = (NgxImageView){v, img, {a, 0, 1, 0, 1}, f, w, h};
    r.type = 0;
    r.rw = rw;
    return r;
}

// ---------------------------------------------------------------- timing (NX2NAVI_DLAA_TIMES=1)
// GPU timestamps around the camera motion dispatch, the per-object motion pass and DLSS's evaluation, and the CPU
// time of NGX's call and of draining the recorder before it, summed and logged every TIMES_SUMMARY frames:
//   dlaa times: gpu camera motion a ms, object motion b ms, DLSS c ms (frames timed); cpu: recorder drain d ms, NGX
//   evaluate e ms, motion pass recording f ms
// Each frame takes its own slot of TIMES_SLOTS; a slot's results are read when it comes round again (by then the GPU is
// long done with it; not ready: that frame is left out).
enum { T_MV0, T_MV1, T_OBJ, T_DL0, T_DL1, T_N };
#define TIMES_SLOTS 8u
#define TIMES_SUMMARY 600
static struct {
    int on;                      // -1 not read
    VkQueryPool pool;
    uint32_t slot;
    int used[TIMES_SLOTS];       // the slot holds a frame's timestamps (all T_N written)
    double gpu[3];               // ms summed: camera motion, object motion, DLSS
    uint64_t frames, cpu_drain, cpu_ngx, cpu_mv;   // ns summed
} T = {.on = -1};

static int times_on(void) {
    if (T.on < 0) {
        T.on = rt_knob("DLAA_TIMES", "0");
        VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                    .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = TIMES_SLOTS * T_N};
        if (T.on && vkCreateQueryPool(vk_ctx()->device, &qi, NULL, &T.pool) != VK_SUCCESS) T.on = 0;
    }
    return T.on;
}

static void times_mark(VkCommandBuffer cb, int which) {
    if (times_on()) vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, T.pool, T.slot * T_N + (uint32_t)which);
}

// At the start of a frame's timing (mv_pass): the slot's last results are read and it is reset.
static void times_begin(VkCommandBuffer cb) {
    if (!times_on()) return;
    T.slot = (T.slot + 1) % TIMES_SLOTS;
    if (T.used[T.slot]) {
        uint64_t q[T_N];
        if (vkGetQueryPoolResults(vk_ctx()->device, T.pool, T.slot * T_N, T_N, sizeof q, q, sizeof q[0],
                                  VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
            double ns = vk_ctx()->props.limits.timestampPeriod / 1e6;   // ticks to ms
            T.gpu[0] += (double)(q[T_MV1] - q[T_MV0]) * ns, T.gpu[1] += (double)(q[T_OBJ] - q[T_MV1]) * ns;
            T.gpu[2] += (double)(q[T_DL1] - q[T_DL0]) * ns;
            if (++T.frames % TIMES_SUMMARY == 0) {
                rt_log("dlaa times: gpu camera motion %.3f ms, object motion %.3f ms, DLSS %.3f ms (%llu frames timed); "
                       "cpu: recorder drain %.3f ms, NGX evaluate %.3f ms, motion pass recording %.3f ms",
                       T.gpu[0] / TIMES_SUMMARY, T.gpu[1] / TIMES_SUMMARY, T.gpu[2] / TIMES_SUMMARY,
                       (unsigned long long)T.frames, T.cpu_drain / 1e6 / TIMES_SUMMARY, T.cpu_ngx / 1e6 / TIMES_SUMMARY,
                       T.cpu_mv / 1e6 / TIMES_SUMMARY);
                T.gpu[0] = T.gpu[1] = T.gpu[2] = 0, T.cpu_drain = T.cpu_ngx = T.cpu_mv = 0;
            }
        }
    }
    T.used[T.slot] = 0;
    vkCmdResetQueryPool(cb, T.pool, T.slot * T_N, T_N);
}

// ---------------------------------------------------------------- motion vectors
// The camera's motion first (runtime/shaders/dlaa_mv.comp): this frame's scene camera block is copied out of the
// replay's uniform snapshot (vk_usnap.c, as of the submit call) into one slot of camb, last frame's stays in the other,
// and the compute pass reprojects every pixel's depth through the two (inv4 and mv_pass below). Then the per-object
// pass (vk_exec_draw.c, vk_mvobj_execute) writes the motion of whatever moves on its own over it.
#define CAM_BYTES 256u   // vec4 rows 0-15: view, view-projection (3-6), projection, inverse view
#define CAM_BLOCK 0x940u   // the whole block, kept for the per-object motion pass's vertex shaders (vk_exec_draw.c)
#define CAM_SLOT 0xa00u    // camb slot stride
#define MVM_SLOTS 16u      // frames the GPU may still be reading the reprojection blocks of

// dlaa_mv.comp's push constants in its layout (std430: dz at offset 32)
typedef struct {
    float size[2], jitter[2], bias, prev_ok, zfit, zprev_ok;
    float dz[4];   // the projection's [2][2], [2][3], [3][2], [3][3]
    float zmask;
} MvPush;

static int mv_init(void) {
    if (D.mv_ok >= 0) return D.mv_ok;
    D.mv_ok = 0;
    const VkCtx *c = vk_ctx();
    VkDescriptorSetLayoutBinding b[10] = {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {8, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {9, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}};
    VkDescriptorSetLayoutCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                           .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                           .bindingCount = 10, .pBindings = b};
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MvPush)};
    VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_NEAREST,
                              .minFilter = VK_FILTER_NEAREST, .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                              .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE};
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sizeof dlaa_mv_spv,
                                   .pCode = dlaa_mv_spv};
    VkShaderModule m = VK_NULL_HANDLE;
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 2 * CAM_SLOT,
                             .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vkCreateDescriptorSetLayout(c->device, &dci, NULL, &D.mv_dsl) != VK_SUCCESS) goto fail;
    VkBufferCreateInfo mbi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = MVM_SLOTS * 2 * CAM_BYTES,
                              .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT};
    VmaAllocationCreateInfo mai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                   .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                            VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo minfo;
    if (vmaCreateBuffer(c->vma, &mbi, &mai, &D.mvm, &D.mvm_mem, &minfo) != VK_SUCCESS) goto fail;
    D.mvm_map = minfo.pMappedData;
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                     .pSetLayouts = &D.mv_dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
    if (vkCreatePipelineLayout(c->device, &li, NULL, &D.mv_layout) != VK_SUCCESS ||
        vkCreateSampler(c->device, &si, NULL, &D.mv_samp) != VK_SUCCESS ||
        vkCreateShaderModule(c->device, &mi, NULL, &m) != VK_SUCCESS ||
        vmaCreateBuffer(c->vma, &bi, &ai, &D.camb, &D.camb_mem, NULL) != VK_SUCCESS)
        goto fail;
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
                                      .layout = D.mv_layout};
    VkResult r = vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, NULL, &D.mv_pipe);
    vkDestroyShaderModule(c->device, m, NULL);
    m = VK_NULL_HANDLE;
    if (r != VK_SUCCESS) goto fail;
    return D.mv_ok = 1;
fail:
    if (m) vkDestroyShaderModule(c->device, m, NULL);   // made before a later step failed
    rt_log("dlaa: the motion-vector pass could not be set up: DLAA runs with zero motion");
    return 0;
}

// The reprojection from this frame's clip space to last frame's, last * inverse(this), in double precision: the shader
// took each pixel back to world space through a float inverse of the view-projection, and with NX2navi's world
// coordinates in the thousands and its far plane the result was noise of a fraction of a pixel and more on distant
// surfaces, which DLSS followed into a shimmer like heat haze (roofs and trees, most of all in the upscaling modes).
// The shader is unchanged: its "this frame" block gets the identity and its "last frame" block this matrix, so it
// applies just the one matrix to each pixel. A still camera gives the identity, and zero motion exactly.
static int inv4(const double m[16], double o[16]) {
    double a[4][8];
    for (int r = 0; r < 4; r++)
        for (int k = 0; k < 8; k++) a[r][k] = k < 4 ? m[r * 4 + k] : (k - 4 == r);
    for (int col = 0; col < 4; col++) {
        int piv = col;
        for (int r = col + 1; r < 4; r++)
            if (fabs(a[r][col]) > fabs(a[piv][col])) piv = r;
        if (fabs(a[piv][col]) < 1e-300) return 0;
        for (int k = 0; k < 8; k++) { double t = a[col][k]; a[col][k] = a[piv][k]; a[piv][k] = t; }
        double d = a[col][col];
        for (int k = 0; k < 8; k++) a[col][k] /= d;
        for (int r = 0; r < 4; r++)
            if (r != col) {
                double f = a[r][col];
                for (int k = 0; k < 8; k++) a[r][k] -= f * a[col][k];
            }
    }
    for (int r = 0; r < 4; r++)
        for (int k = 0; k < 4; k++) o[r * 4 + k] = a[r][k + 4];
    return 1;
}

// Ocarina's camera from its scene context block (vk_dlaa_draw): its scene vertex shaders take the world position to
// the camera through rows 0-2 (a 3x4 view) and then to clip space through rows 7-10 (the projection, 4x4), so the
// view-projection is rows 7-10 times rows 0-2.
static void ocam_vp(const float *v, double out[16]) {
    const float *P = v + 7 * 4;
    double V[16] = {v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], 0, 0, 0, 1};
    for (int r = 0; r < 4; r++)
        for (int k = 0; k < 4; k++) {
            double acc = 0;
            for (int j = 0; j < 4; j++) acc += (double)P[r * 4 + j] * V[j * 4 + k];
            out[r * 4 + k] = acc;
        }
}

// Frame generation's camera (vk_dlaa_fg_camera) from the Ocarina block: the projection is rows 7-10, the view rows 0-2
// (rows: right, up, back; the camera looks down -z), m the reprojection from this frame's clip space to last frame's.
// Each matrix is handed out transposed, for row vectors.
static void fg_store(float out[16], const double m[16]) {
    for (int r = 0; r < 4; r++)
        for (int k = 0; k < 4; k++) out[r * 4 + k] = (float)m[k * 4 + r];
}
// The view distance in front of the camera at which the projection P (row-major, column vectors, looking down -z) gives
// clip depth z_ndc = k: (P[10] z + P[11]) / (P[14] z + P[15]) = k, solved for z. INFINITY where no distance gives it.
static double plane_dist(const double P[16], double k) {
    double den = P[10] - k * P[14];
    if (fabs(den) <= 1e-9 * (fabs(P[10]) + fabs(P[14]))) return INFINITY;
    return -(k * P[15] - P[11]) / den;
}

// What the depth buffer's 0 and 1 stand for, from the frame's projection and the device's depth mode. With NVN's
// NEAR_IS_MINUS_W (clip depth -w..w, which the pipelines keep through negativeOneToOne, vk_pipeline.c) a fragment
// stores z_ndc * 0.5 + 0.5, so 0 is z_ndc = -1; with NEAR_IS_ZERO it stores z_ndc. Both are 0..1 in the D32F target
// (the viewport's depth range, vk_exec_draw.c), and the -w..w mapping stores exactly what a 0..w projection with the
// same planes would: (f / (f - n)) (1 - n / distance), the convention DLSS, Streamline and FSR take. Ocarina's planes
// are not fixed: in Hyrule Field its z row is (-1.00001, -0.200001), near 0.1 and far about 25,000 (the view's units
// are metres), and the camera before it had far about 1,000. Taken as 0.1 and 1e8 (no far plane), a distance d reads
// as (f - n) d / (f - d): 1,000 as 1,042, 10,000 as 16,650, and the far plane's neighbourhood without bound. So the
// planes come from the projection every frame; a projection whose stored 0 is not in front of the camera keeps the last
// frame's planes.
static void proj_planes(const double P[16], VkDlaaCamera *c) {
    int minus_w = nvn_depth_minus_w();
    double at0 = plane_dist(P, minus_w ? -1.0 : 0.0), at1 = plane_dist(P, 1.0);   // the distances stored as 0 and 1
    int inverted = !(at0 < at1);   // reversed depth: the near plane is stored as 1
    double n = inverted ? at1 : at0, f = inverted ? at0 : at1;
    if (!(n > 0) || !isfinite(n)) return;
    c->inverted = inverted;
    c->znear = (float)n;
    c->infinite = !(f < VK_DLAA_FAR_INFINITE);
    c->zfar = c->infinite ? VK_DLAA_FAR_INFINITE : (float)f;
    // one line at the start and whenever the planes change (the first 20 times)
    static float logged_n, logged_f;
    static int logged_inv = -1, lines;
    if (lines < 20 && (inverted != logged_inv || fabsf(c->znear - logged_n) > 1e-4f * logged_n ||
                       fabsf(c->zfar - logged_f) > 1e-4f * logged_f)) {
        lines++, logged_inv = inverted, logged_n = c->znear, logged_f = c->zfar;
        char fz[32];
        if (c->infinite) snprintf(fz, sizeof fz, "infinite");
        else snprintf(fz, sizeof fz, "%.6g", c->zfar);
        rt_log("dlaa: the scene's projection: near %.6g, far %s, depth %s, clip depth %s (z row %.6g %.6g %.6g %.6g, "
               "w row %.6g %.6g %.6g %.6g)", c->znear, fz, inverted ? "inverted (1 near)" : "0 near, 1 far",
               minus_w ? "-w..w" : "0..w", P[8], P[9], P[10], P[11], P[12], P[13], P[14], P[15]);
    }
}

static void fg_camera(const float *v, const double m[16]) {
    double P[16], Pi[16], mi[16];
    for (int i = 0; i < 16; i++) P[i] = v[28 + i];
    if (!inv4(P, Pi) || !inv4(m, mi)) return;
    VkDlaaCamera *c = &D.fg_cam;
    fg_store(c->view_to_clip, P), fg_store(c->clip_to_view, Pi);
    fg_store(c->clip_to_prev_clip, m), fg_store(c->prev_clip_to_clip, mi);
    for (int i = 0; i < 3; i++) {
        c->right[i] = v[i], c->up[i] = v[4 + i], c->fwd[i] = -v[8 + i];
        // the position: -R^T t, R the view's 3x3 and t its last column
        c->pos[i] = -(v[i] * v[3] + v[4 + i] * v[7] + v[8 + i] * v[11]);
    }
    c->fov_y = P[5] != 0 ? 2.0f * (float)atan(1.0 / fabs(P[5])) : 1.0f;
    c->aspect = P[0] != 0 ? (float)fabs(P[5] / P[0]) : 1.777f;
    if (!(c->znear > 0)) c->znear = 0.1f, c->zfar = VK_DLAA_FAR_INFINITE, c->infinite = 1;   // until a projection gives them
    proj_planes(P, c);
    D.fg_cam_ok = 1;
    // NX2NAVI_MOUSE_TRACE=1 (test runs, mouse_direct.c): the camera's direction each frame
    static int trace = -1;
    if (trace < 0) {
        const char *e = cfg_get("NX2NAVI_MOUSE_TRACE");
        trace = e && *e && strcmp(e, "0");
    }
    if (trace)
        rt_log("mousetrace: cam t=%.3f fwd=%.6f,%.6f,%.6f pos=%.3f,%.3f,%.3f", plat_now_ns() / 1e6, c->fwd[0], c->fwd[1],
               c->fwd[2], c->pos[0], c->pos[1], c->pos[2]);
}

// Frames without a camera (dlaa_cam_keep, NX2NAVI_DLAA_CAM_KEEP; 1 by default, 0 restores the old handling). Measured in
// Kokiri Forest during dialogue (Saria, dlss = performance): about 1 frame in 20 had no camera, and each started a new
// history twice, on that frame and on the next (nothing to reproject from): 24 to 26 resets per 300 frames. Those were
// frames the event order held past their present (vk_exec.c), which then ran in a submission without a uniform copy
// (the present thread's empty one), so the lookup at DLSS found nothing. With the change: no camera reset there in
// 3600 frames but the first. Two things change:
// - Ocarina's camera rows are taken from the bytes the scene draws themselves read (vk_dlaa_draw, views[].cam), not
//   looked up again in the uniform copy of the submission DLSS runs in, which on some frames has none or lost the
//   block to a GPU write.
// - A frame that has no camera at all keeps DLSS's history: its vectors are zero (the camera taken as still) and last
//   frame's camera stays the one the next frame reprojects from. At most KEEP_MAX frames in a row: a longer stretch
//   is something else (a cut scene with a camera of its own), and a still history there would trail what moves.
#define KEEP_MAX 6

static const char *const usnap_why[] = {"found", "no copy in this submission", "dropped by a GPU write in this submission",
                                        "left out: a GPU write in flight", "not memory the CPU writes",
                                        "not bound in this submission"};

// One line for each of the first frames without a camera: what the frame drew into the scene depth, which submission
// the scene context was seen in and why the evaluating one lacks it, and what was done.
static void cam_note(int view, int kept) {
    uint64_t seq = vk_exec_submit_seq;
    char snap[200] = "";
    if (D.cam_addr)
        snprintf(snap, sizeof snap, "; scene context %#llx last drawn %llu submission(s) before DLSS's, there: %s",
                 (unsigned long long)D.cam_addr, (unsigned long long)(view >= 0 ? seq - D.views[view].seq : 0),
                 usnap_why[vk_usnap_why(D.cam_addr, D.cam_size == OCAM_BLOCK ? OCAM_BYTES : D.cam_size)]);
    rt_log("dlaa: frame %llu without a camera: %llu scene-depth draws over %llu submission(s) (%u into another D32 "
           "depth %ux%u), %u with the scene context (%u not into a scene colour target, format %d), %u with another c3 "
           "block (last size %#x)%s; %s",
           (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1, (unsigned long long)F.zdraws,
           (unsigned long long)(F.zdraws + F.zother ? seq - F.seq0 + 1 : 0), F.zother, F.zother_w, F.zother_h, F.ocam,
           F.ocam_off, (int)F.ocam_off_fmt, F.c3_other, F.c3_size, snap,
           kept == 1 ? "history kept, zero motion" : kept == 2 ? "new history (more than KEEP_MAX in a row)"
                                                 : "new history");
}

// Camera cuts (NX2NAVI_DLAA_CUT, default 1; 0: off). Only a lost camera (mv_pass), another scene size or another
// target started a new history, so DLSS and frame generation (vk_dlaa_fg_inputs) carried their history across a camera
// that jumped: an area load or a spawn moves it 5 to 180 m in one frame. Such a frame now starts a new history, as
// NVIDIA's DLSS guide asks for a camera cut: a move over 5 m, or a turn over 10 degrees together with a move over 0.5 m,
// both per 60 Hz frame and scaled by the time since the last camera. Camera traces of play in the Fire Temple
// (NX2NAVI_MOUSE_TRACE) had a median of 0.05 m and 0.08 degrees a frame. A turn alone (the mouse camera, which orbits
// about 3 m from Link) does not count, and the turn with a move counts only from CUT_TURN degrees: a quick swing of the
// orbiting camera turns 11 to 18 degrees and moves 0.9 to 1.4 m a frame, and taken for a cut at 10 degrees it started a
// new history at every frame of the swing (nine in a row in Hyrule Field), which at FSR's or DLSS's smaller modes showed
// the scene at its render size for those frames, blocky and flickering. Its motion vectors carry it as well as any other
// turn. Each cut is logged, up to 24.
#define CUT_TURN 30.0
// v: the Ocarina scene context's rows 0-2, the view (fg_camera).
static void cam_cut(const float *v) {
    static int on = -1;
    if (on < 0) on = rt_knob("DLAA_CUT", "1");
    float pos[3], fwd[3];
    for (int i = 0; i < 3; i++) {
        fwd[i] = -v[8 + i];
        pos[i] = -(v[i] * v[3] + v[4 + i] * v[7] + v[8 + i] * v[11]);
    }
    uint64_t now = plat_now_ns();
    if (on && D.cut_ok) {
        double dot = fwd[0] * D.cut_fwd[0] + fwd[1] * D.cut_fwd[1] + fwd[2] * D.cut_fwd[2];
        double turn = acos(dot > 1 ? 1 : dot < -1 ? -1 : dot) * (180.0 / 3.14159265358979);
        double dx = pos[0] - D.cut_pos[0], dy = pos[1] - D.cut_pos[1], dz = pos[2] - D.cut_pos[2];
        double move = sqrt(dx * dx + dy * dy + dz * dz);
        double k = (now - D.cut_ns) / 1e6 / (1000.0 / 60);   // 60 Hz frames since the last camera
        if (k < 0.25) k = 0.25;
        if (k > 4) k = 4;
        if (move > 5.0 * k || (turn > CUT_TURN * k && move > 0.5 * k)) {
            D.reset = 1, D.cuts++;
            if (D.cut_logged < 24)
                D.cut_logged++, rt_log("dlaa: frame %llu: camera cut (moved %.2f m, turned %.1f degrees in %.1f ms): new "
                                       "history (%llu so far)",
                                       (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1,
                                       move, turn, (now - D.cut_ns) / 1e6, (unsigned long long)D.cuts);
        }
    }
    memcpy(D.cut_pos, pos, sizeof pos), memcpy(D.cut_fwd, fwd, sizeof fwd);
    D.cut_ns = now, D.cut_ok = 1;
}

// The pass, recorded through the recorder like the replay's own commands. 0: no camera this frame (the vectors stay
// zero and DLSS starts a new history); 2: no camera, but the history is kept (the vectors stay zero, dlaa_cam_keep).
// bias > 0 with this frame's G-buffer classes (D.ids) also writes e->mask, the bias mask (dlaa_mv.comp); *masked says
// whether it did.
static int mv_pass(VkExec *ex, struct Size *e, VkImageView zv, float jx, float jy, float bias, int *masked) {
    *masked = 0;
    VkCommandBuffer cb = ex->cb;
    uint32_t w = e->w, h = e->h;
    static int keep = -1;
    if (keep < 0) keep = rt_knob("DLAA_CAM_KEEP", "1");
    VkDescriptorBufferInfo cam;
    int view = -1;   // Ocarina: the views[] entry of the camera
    for (int i = 0; D.cam_size == OCAM_BLOCK && D.cam_addr && i < D.nviews; i++)
        if (D.views[i].addr == D.cam_addr) view = i;
    const float *held = keep && view >= 0 && D.views[view].got ? D.views[view].cam : NULL;
    int found = mv_init() && D.cam_addr && vk_usnap_find(D.cam_addr, D.cam_size == OCAM_BLOCK ? OCAM_BYTES : D.cam_size, 4, &cam);
    if (!found && held && D.mv_ok == 1) {
        D.cam_at_draw++;
        if (D.gap_logged < 40) {
            D.gap_logged++;
            rt_log("dlaa: frame %llu: camera read at the draw: scene context %#llx last drawn %llu submission(s) before "
                   "DLSS's, there: %s; the frame's scene-depth draws over %llu submission(s); %d held submissions",
                   (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1,
                   (unsigned long long)D.cam_addr, (unsigned long long)(vk_exec_submit_seq - D.views[view].seq),
                   usnap_why[vk_usnap_why(D.cam_addr, OCAM_BYTES)],
                   (unsigned long long)(F.zdraws + F.zother ? vk_exec_submit_seq - F.seq0 + 1 : 0), vk_exec_held());
        }
    }
    else if (!found) {
        D.mv_none++;
        if (D.cam_addr) D.cam_none_snap++;
        else D.cam_none_draw++;
        e->chars_ok = e->dist_ok = 0;
        int kept = 0;
        if (keep && D.mv_ok == 1 && D.cam_prev_ok) {
            if (D.keep_run < KEEP_MAX) kept = 1, D.keep_run++, D.cam_kept++;
            else kept = 2, D.cam_kept_max++;
        }
        if (D.gap_logged < 40) D.gap_logged++, cam_note(view, kept);
        if (kept == 1) return 2;
        D.cam_prev_ok = 0;
        return 0;
    }
    D.keep_run = 0;
    // this frame's view-projection as the scene draws read it (Ocarina, dlaa_cam_keep) or as the snapshot holds it, for
    // the reprojection; kept as last frame's for the next
    const float *cr = held ? held
                      : D.cam_size == OCAM_BLOCK ? (const float *)vk_usnap_bytes(D.cam_addr, OCAM_BYTES)
                                                 : (const float *)vk_usnap_bytes(D.cam_addr + D.cam_vp, 64);
    double cur_vp[16], m[16], ci[16];
    int have_m = 0;
    if (cr && D.cam_size == OCAM_BLOCK) ocam_vp(cr, cur_vp);
    else if (cr)
        for (int i = 0; i < 16; i++) cur_vp[i] = cr[i];
    if (cr) {
        if (D.cam_prev_ok && inv4(cur_vp, ci)) {
            for (int r = 0; r < 4; r++)
                for (int k = 0; k < 4; k++) {
                    double v = 0;
                    for (int j = 0; j < 4; j++) v += D.cam_prev[r * 4 + j] * ci[j * 4 + k];
                    m[r * 4 + k] = v;
                }
            have_m = 1;
        }
    }
    D.fg_cam_ok = 0;
    if (cr && have_m && D.cam_size == OCAM_BLOCK) fg_camera(cr, m);
    if (cr && D.cam_size == OCAM_BLOCK) cam_cut(cr);
    int s = D.cam_slot, had_prev = D.cam_prev_ok;
    if (D.cam_size == CAM_BLOCK) {   // the whole block, for the per-object motion pass (the deferred engine's only)
        VkBufferCopy bc = {cam.offset, (VkDeviceSize)s * CAM_SLOT, CAM_BLOCK};
        vkCmdCopyBuffer(cb, cam.buffer, D.camb, 1, &bc);
        full_barrier(cb);
    }
    D.cam_slot ^= 1, D.cam_prev_ok = cr != NULL;
    if (cr) memcpy(D.cam_prev, cur_vp, sizeof cur_vp);
    if (!had_prev || !have_m) {   // nothing to reproject into: zero motion, new history
        e->chars_ok = e->dist_ok = 0;
        return 0;
    }
    VkDescriptorImageInfo zi = {D.mv_samp, zv, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo oi = {VK_NULL_HANDLE, e->mv_view, VK_IMAGE_LAYOUT_GENERAL};
    uint32_t ms = D.mvm_slot++ % MVM_SLOTS;
    float *blk = (float *)(D.mvm_map + (size_t)ms * 2 * CAM_BYTES);   // rows 3-6 of each block are read
    memset(blk, 0, 2 * CAM_BYTES);
    for (int r = 0; r < 4; r++)
        for (int k = 0; k < 4; k++)
            blk[12 + r * 4 + k] = r == k ? 1.0f : 0.0f, blk[64 + 12 + r * 4 + k] = (float)m[r * 4 + k];
    vmaFlushAllocation(vk_ctx()->vma, D.mvm_mem, (VkDeviceSize)ms * 2 * CAM_BYTES, 2 * CAM_BYTES);
    VkDescriptorBufferInfo cur = {D.mvm, (VkDeviceSize)ms * 2 * CAM_BYTES, CAM_BYTES};
    VkDescriptorBufferInfo prev = {D.mvm, (VkDeviceSize)ms * 2 * CAM_BYTES + CAM_BYTES, CAM_BYTES};
    // The class mask needs the G-buffer at the scene size; otherwise bindings 4-7 get the mask images as stand-ins
    // (the shader returns before reading them with bias 0).
    const VkExecTarget *ids = &D.ids;
    int use = bias > 0 && D.ids_ok && ids->image && ids->width == w && ids->height == h && !ids->is3d;
    VkImageView iv = use ? vk_exec_target_view(ids->img, ids->format, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT,
                                               ids->level, ids->base_layer, 1, 0)
                         : e->mask_view;
    int cur_c = e->chars_cur;
    VkDescriptorImageInfo ii = {D.mv_samp, iv, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo mki = {VK_NULL_HANDLE, e->mask_view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo cpi = {VK_NULL_HANDLE, e->chars_view[cur_c ^ 1], VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo coi = {VK_NULL_HANDLE, e->chars_view[cur_c], VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo dpi = {VK_NULL_HANDLE, e->dist_view[e->dist_cur ^ 1], VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo doi = {VK_NULL_HANDLE, e->dist_view[e->dist_cur], VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet wr[10] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &zi},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 1, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &oi},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 2, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = &cur},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 3, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = &prev},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 4, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 5, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &mki},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 6, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &cpi},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 7, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &coi},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 8, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &dpi},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 9, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &doi}};
    times_begin(cb);
    times_mark(cb, T_MV0);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, D.mv_pipe);
    rec_push(vk_pipeline_push_fn(), cb, VK_PIPELINE_BIND_POINT_COMPUTE, D.mv_layout, 0, 10, wr);
    // The depth check (dlaa_mv.comp, depth_prev) where nothing else gives characters their own motion: Ocarina's frame,
    // which has neither the per-object pass nor the class buffer. Its distances take the projection's depth terms from
    // rows 7-10 of the scene context (ocam_vp). NX2NAVI_DLAA_ZFIT=0 (test): the camera's motion alone, as before.
    static int zfit_on = -1;
    if (zfit_on < 0) zfit_on = rt_knob("DLAA_ZFIT", "1");
    int zfit = zfit_on && !use && cr && D.cam_size == OCAM_BLOCK;
    MvPush pc = {{(float)w, (float)h}, {jx, jy}, use ? bias : 0.0f, use && e->chars_ok ? 1.0f : 0.0f,
                 zfit ? 1.0f : 0.0f, zfit && e->dist_ok ? 1.0f : 0.0f,
                 {zfit ? cr[38] : 0.0f, zfit ? cr[39] : 0.0f, zfit ? cr[42] : 0.0f, zfit ? cr[43] : 0.0f},
                 zfit ? bias : 0.0f};
    vkCmdPushConstants(cb, D.mv_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
    vkCmdDispatch(cb, (w + 7) / 8, (h + 7) / 8, 1);
    full_barrier(cb);
    times_mark(cb, T_MV1);
    vk_mvcheck_camera(cb, e->mv, w, h, jx, jy);   // NX2NAVI_MV_CHECK (test)
    // the characters' own motion over the camera's (vk_exec_draw.c, per-object motion)
    uint64_t c0 = T.on > 0 ? kern_now_ns() : 0;
    if (D.cam_size == CAM_BLOCK && vk_mvobj_execute(ex, D.cam_addr, e->mv_view, zv, D.depth.format, w, h)) full_barrier(cb);
    if (T.on > 0) T.cpu_mv += kern_now_ns() - c0;
    times_mark(cb, T_OBJ);
    if (T.on > 0) T.used[T.slot] = 1;   // the DLSS marks follow in evaluate
    vk_mvcheck_final(cb, ex->fence, e->mv, D.color.image, D.color.format, D.color.level, D.color.base_layer, w, h, jx, jy);
    D.mv_frames++;
    if (use) e->chars_cur ^= 1, e->chars_ok = 1, D.masked++;
    else e->chars_ok = 0;
    if (zfit && bias > 0) D.masked++;
    e->dist_cur ^= 1, e->dist_ok = zfit;
    *masked = use || (zfit && bias > 0);
    return 1;
}

// Last frame's camera block for the per-object motion pass, recorded before this frame's mv_pass: that pass writes
// this frame's into slot cam_slot and leaves the other slot, last frame's, as it is. 0: none (no history).
int vk_dlaa_prev_camera(VkDescriptorBufferInfo *out, float jitter[2]) {
    if (!dl_active() || !D.cam_prev_ok || !D.camb) return 0;
    if (D.cam_size != CAM_BLOCK) return 0;   // the per-object pass knows the deferred engine's block only
    *out = (VkDescriptorBufferInfo){D.camb, (VkDeviceSize)(D.cam_slot ^ 1) * CAM_SLOT, CAM_BLOCK};
    jitter[0] = D.jit_ok ? D.jx : 0.0f, jitter[1] = D.jit_ok ? D.jy : 0.0f;
    return 1;
}

// ---------------------------------------------------------------- reactive mask
// Particles, smoke, sparks and other transparents have no motion vectors of their own: the motion image holds the
// surface's behind them, and DLSS carried their old colour along that surface (trails behind sparks, smoke smeared
// with the camera). DLSS takes a mask of where to trust this frame's colour over the history (NGX
// "DLSS.Input.Bias.Current.Color.Mask"). It is made from the picture itself, at no cost on the CPU: the light target is
// copied when the transparents start (the pass that arms DLAA, vk_dlaa_pass) and compared with what DLSS gets
// (runtime/dlaa_mask_spv.h, tools/make_dlaa_mask.py): the share of a pixel's colour that changed, from MASK_THR on,
// times MASK_GAIN. Off by default (NX2NAVI_DLSS_REACTIVE=1 turns it on, NX2NAVI_DLAA_MASKDEBUG=1 shows the mask instead of
// the picture): the pass that arms also draws the sky and blended fog layers over the solid scene, and the mask
// covered the sky, sunlit trees and roofs and a wide fog region besides the particles (seen with MASKDEBUG at 19:45 in
// the first save), where DLSS would give up its anti-aliasing. It needs the particle draws alone (drawn again into the
// mask, like the motion draws) before it can be on. In the upscaling modes the light target DLSS reads also holds the
// bloom by then; the copy would have to be compared at the end of the scene instead.
#include "dlaa_mask_spv.h"
#define MASK_THR 0.1f
#define MASK_GAIN 2.5f
#define SNAPS 3

static struct {
    struct Snap {                // per light target size and format: its copy at arming, and the mask
        uint32_t w, h;
        VkFormat f;
        VkImage before, mask;
        VmaAllocation before_mem, mask_mem;
        VkImageView before_view, mask_view;
        uint64_t used;
    } s[SNAPS];
    struct Snap *cur;            // this frame's copy, NULL: none
    VkDescriptorSetLayout dsl;
    VkPipelineLayout layout;
    VkPipeline pipe;
    int ok, knob;                // ok: -1 not built, 0 failed, 1 ready; knob: -1 not read
    uint64_t made, missed;       // frames with a mask; frames DLSS ran without the copy
} R = {.ok = -1, .knob = -1};

static int mask_init(void) {
    if (R.knob < 0) R.knob = rt_knob("DLSS_REACTIVE", "0");
    if (!R.knob) return 0;
    if (R.ok >= 0) return R.ok;
    R.ok = 0;
    const VkCtx *c = vk_ctx();
    VkDescriptorSetLayoutBinding b[3] = {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}};
    VkDescriptorSetLayoutCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                           .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                           .bindingCount = 3, .pBindings = b};
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sizeof dlaa_mask_spv,
                                   .pCode = dlaa_mask_spv};
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(c->device, &dci, NULL, &R.dsl) != VK_SUCCESS) goto fail;
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                     .pSetLayouts = &R.dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
    if (vkCreatePipelineLayout(c->device, &li, NULL, &R.layout) != VK_SUCCESS ||
        vkCreateShaderModule(c->device, &mi, NULL, &m) != VK_SUCCESS)
        goto fail;
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
                                      .layout = R.layout};
    VkResult r = vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, NULL, &R.pipe);
    vkDestroyShaderModule(c->device, m, NULL);
    m = VK_NULL_HANDLE;
    if (r != VK_SUCCESS) goto fail;
    return R.ok = 1;
fail:
    if (m) vkDestroyShaderModule(c->device, m, NULL);   // made before a later step failed
    rt_log("dlaa: the reactive mask pass could not be set up: DLSS runs without it");
    return 0;
}

// A snapshot's images and views. The device must be idle: free_snap waits for it, vk_dlaa_release has.
static void destroy_snap(struct Snap *s) {
    const VkCtx *c = vk_ctx();
    if (s->before_view) vkDestroyImageView(c->device, s->before_view, NULL);
    if (s->mask_view) vkDestroyImageView(c->device, s->mask_view, NULL);
    if (s->before) vmaDestroyImage(c->vma, s->before, s->before_mem);
    if (s->mask) vmaDestroyImage(c->vma, s->mask, s->mask_mem);
    memset(s, 0, sizeof *s);
}

static void free_snap(struct Snap *s) {
    vk_queue_lock(), vkDeviceWaitIdle(vk_ctx()->device), vk_queue_unlock();   // a fourth size: the evicted images may still be in flight
    destroy_snap(s);
}

static struct Snap *snap_for(VkCommandBuffer cb, uint32_t w, uint32_t h, VkFormat f) {
    static uint64_t tick;
    struct Snap *e = NULL, *lru = &R.s[0];
    for (int i = 0; i < SNAPS; i++) {
        if (R.s[i].w == w && R.s[i].h == h && R.s[i].f == f) return R.s[i].used = ++tick, &R.s[i];
        if (!R.s[i].before && !e) e = &R.s[i];
        if (R.s[i].used < lru->used) lru = &R.s[i];
    }
    if (!e) free_snap(e = lru);
    VkImageUsageFlags u = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (!image2d(w, h, f, u, &e->before, &e->before_mem, &e->before_view) ||
        !image2d(w, h, VK_FORMAT_R8_UNORM, u | VK_IMAGE_USAGE_STORAGE_BIT, &e->mask, &e->mask_mem, &e->mask_view)) {
        free_snap(e);
        return NULL;
    }
    VkImageMemoryBarrier2 ib[2];
    VkImage im[2] = {e->before, e->mask};
    for (int i = 0; i < 2; i++)
        ib[i] = (VkImageMemoryBarrier2){.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                        .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                        .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = im[i],
                                        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 2,
                           .pImageMemoryBarriers = ib};
    vkCmdPipelineBarrier2(cb, &di);
    e->w = w, e->h = h, e->f = f, e->used = ++tick;
    return e;
}

// At arming: the light target as it is before the transparents, outside any pass.
static void mask_snapshot(VkExec *ex) {
    R.cur = NULL;
    if (mask_init() <= 0) return;
    const VkExecTarget *t = &D.color;
    struct Snap *s = snap_for(ex->cb, t->width, t->height, t->format);
    if (!s) return;
    vk_exec_barrier(ex);
    VkImageCopy c = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, t->level, t->base_layer, 1},
                     .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .extent = {t->width, t->height, 1}};
    vkCmdCopyImage(ex->cb, t->image, VK_IMAGE_LAYOUT_GENERAL, s->before, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    full_barrier(ex->cb);
    ex->recorded += 2;
    R.cur = s;
}

// Before DLSS: the mask from the copy and the light target (view cv). NULL: none this frame.
static struct Snap *mask_pass(VkCommandBuffer cb, VkImageView cv, uint32_t w, uint32_t h) {
    struct Snap *s = R.cur;
    R.cur = NULL;
    if (mask_init() <= 0) return NULL;
    if (!s || s->w != w || s->h != h || s->f != D.color.format) return R.missed++, NULL;
    VkDescriptorImageInfo bi = {D.mv_samp, s->before_view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo ai = {D.mv_samp, cv, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo mi = {VK_NULL_HANDLE, s->mask_view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet wr[3] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &bi},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 1, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ai},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 2, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &mi}};
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, R.pipe);
    rec_push(vk_pipeline_push_fn(), cb, VK_PIPELINE_BIND_POINT_COMPUTE, R.layout, 0, 3, wr);
    float pc[4] = {(float)w, (float)h, MASK_THR, MASK_GAIN};
    vkCmdPushConstants(cb, R.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    vkCmdDispatch(cb, (w + 7) / 8, (h + 7) / 8, 1);
    full_barrier(cb);
    R.made++;
    return s;
}

// ---------------------------------------------------------------- teardown
// The HUD-less picture's images, a half-made pair too. The device must be idle, or the images not yet used.
static void hl_destroy(void) {
    const VkCtx *c = vk_ctx();
    if (D.hl_view) vkDestroyImageView(c->device, D.hl_view, NULL);
    if (D.hl) vmaDestroyImage(c->vma, D.hl, D.hl_mem);
    if (D.hl_srgb) vmaDestroyImage(c->vma, D.hl_srgb, D.hl_srgb_mem);
    D.hl_view = VK_NULL_HANDLE, D.hl = D.hl_srgb = VK_NULL_HANDLE;
}

// The device is idle here. Release NGX features and parameters before their Vulkan resources, then shut down NGX even
// if initialization succeeded but the layout check failed. The Linux core's Shutdown1 takes an extra output pointer
// compared with the SDK's public wrapper; it reports the number of devices still using NGX.
// ---- the free camera's view of the ground (debug_tools.c dbg_pad). While it flies, nine of the view distances
// mv_pass wrote (a grid over the lower part of the view, where the ground is) go back to the host, and the camera rises
// where the closest is nearer than its clearance: flown through a hill, it showed the void below the map. A slot is
// read FC_SLOTS evaluations after its copy, when the GPU has long been past it.
int dbg_freecam_flying(void);                          // debug_tools.c
void dbg_freecam_ground(const float *dist, int n);
#define FC_N 9
#define FC_SLOTS 4
static VkBuffer fc_buf;
static VmaAllocation fc_mem;
static float *fc_map;
static uint32_t fc_next;
static uint8_t fc_have[FC_SLOTS];

static void fcam_probe(VkCommandBuffer cb, const struct Size *e, uint32_t w, uint32_t h) {
    if (!dbg_freecam_flying() || !e->dist_ok) {   // dist_ok: mv_pass wrote this frame's distances (Ocarina's frame)
        memset(fc_have, 0, sizeof fc_have);
        return;
    }
    const VkCtx *c = vk_ctx();
    if (!fc_buf) {
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = FC_SLOTS * FC_N * sizeof(float),
                                 .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
        VmaAllocationInfo info;
        if (vmaCreateBuffer(c->vma, &bi, &ai, &fc_buf, &fc_mem, &info) != VK_SUCCESS) {
            fc_buf = VK_NULL_HANDLE;
            return;
        }
        fc_map = info.pMappedData;
    }
    uint32_t s = fc_next++ % FC_SLOTS;
    if (fc_have[s]) {
        vmaInvalidateAllocation(c->vma, fc_mem, s * FC_N * sizeof(float), FC_N * sizeof(float));
        dbg_freecam_ground(fc_map + s * FC_N, FC_N);
    }
    // the replay's viewports have negative heights: row 0 is the top of the view
    static const float fx[3] = {0.2f, 0.5f, 0.8f}, fy[3] = {0.6f, 0.78f, 0.96f};
    VkBufferImageCopy r[FC_N];
    for (int i = 0; i < FC_N; i++)
        r[i] = (VkBufferImageCopy){.bufferOffset = (s * FC_N + i) * sizeof(float),
                                   .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                                   .imageOffset = {(int32_t)(fx[i % 3] * (w - 1)), (int32_t)(fy[i / 3] * (h - 1)), 0},
                                   .imageExtent = {1, 1, 1}};
    vkCmdCopyImageToBuffer(cb, e->dist[e->dist_cur ^ 1], VK_IMAGE_LAYOUT_GENERAL, fc_buf, FC_N, r);
    fc_have[s] = 1;
}

void vk_dlaa_release(void) {
#ifndef _WIN32
    const VkCtx *c = vk_ctx();
    if (!c) return;
    for (int i = 0; i < 5; i++)
        if (D.sizes[i].feat) N.release(D.sizes[i].feat), D.sizes[i].feat = NULL;
    if (FG.feat) N.release(FG.feat);   // frame generation's feature and parameters (vk_ngxfg_evaluate), before NGX goes
    if (FG.p) N.destroy_params(FG.p);
    memset(&FG, 0, sizeof FG);
    if (D.params) N.destroy_params(D.params);
    if (N.caps_map) N.destroy_params(N.caps_map);
    for (int i = 0; i < 5; i++) destroy_size(&D.sizes[i]);
    for (int i = 0; i < SNAPS; i++) destroy_snap(&R.s[i]);
    hl_destroy();
    if (D.mv_pipe) vkDestroyPipeline(c->device, D.mv_pipe, NULL);
    if (D.mv_layout) vkDestroyPipelineLayout(c->device, D.mv_layout, NULL);
    if (D.mv_dsl) vkDestroyDescriptorSetLayout(c->device, D.mv_dsl, NULL);
    if (D.mv_samp) vkDestroySampler(c->device, D.mv_samp, NULL);
    if (D.camb) vmaDestroyBuffer(c->vma, D.camb, D.camb_mem);
    if (D.mvm) vmaDestroyBuffer(c->vma, D.mvm, D.mvm_mem);   // mapped for good: the destroy unmaps it
    if (fc_buf) vmaDestroyBuffer(c->vma, fc_buf, fc_mem), fc_buf = VK_NULL_HANDLE, fc_map = NULL;
    if (R.pipe) vkDestroyPipeline(c->device, R.pipe, NULL);
    if (R.layout) vkDestroyPipelineLayout(c->device, R.layout, NULL);
    if (R.dsl) vkDestroyDescriptorSetLayout(c->device, R.dsl, NULL);
    if (T.pool) vkDestroyQueryPool(c->device, T.pool, NULL);
    memset(&D, 0, sizeof D), D.mv_ok = -1, D.cfg_seen = ~0u;
    memset(&R, 0, sizeof R), R.ok = -1, R.knob = -1;
    memset(&T, 0, sizeof T), T.on = -1;
    memset(&F, 0, sizeof F);
    vk_dlaa_swap_img = NULL, vk_dlaa_swap_view = VK_NULL_HANDLE;
    if (!N.device) return;   // NGX never came up on this device: DLAA off, FSR 3.1, or its init failed
    unsigned left = 0;
    vk_queue_lock();   // NGX may wait for the device itself
    NgxResult r = N.shutdown1(N.device, &left);
    vk_queue_unlock();
    rt_log("dlaa: NGX shut down (%08x, %u devices left)", (unsigned)r, left);
    N.device = NULL, N.caps_map = NULL, N.ngx_up = 0, N.available = 0;
#endif
}

// ---------------------------------------------------------------- jitter
// The scene's draws (every pass drawing with the scene depth, until the DLAA point) have their viewport moved by a
// sub-pixel Halton(2,3) offset, which shifts what they rasterise as a jittered projection would; full-screen passes
// that shade by window position are unaffected. Jitter only follows a frame DLSS ran on: a jittered frame that DLSS
// does not resolve shakes on screen, so a missed DLAA point stops the jitter for a second.
static float halton(uint32_t i, uint32_t b) {
    float f = 1, r = 0;
    for (; i; i /= b) f /= (float)b, r += f * (float)(i % b);
    return r;
}

// Whether a draw takes the jitter, before DLSS ran this frame. The deferred frame: every draw into the scene depth.
// Ocarina's frame: every draw that projects through the camera (it binds the scene context at c3, vk_dlaa_draw) into a
// D32F depth of the scene's size, whichever image that is. Ocarina draws its depth pre-pass into the scene depth, copies
// that depth into a second image of the same size, and then shades the same geometry into the light target against
// the copy with the depth test EQUAL. Jittering only the draws into the first image left the shading draws unjittered
// against jittered depth: EQUAL failed on nearly every pixel, the light target kept its clear colour, and the ground
// and characters went white while the sky stayed right. Its full-screen passes drawn with a scene depth
// attached (the bloom composite DLSS reads) do not take the jitter either: they read the jittered scene through their
// own texture coordinates, and moving them too would shift the picture DLSS gets by a second offset.
static int jitter_draw(const VkExec *ex) {
    const VkExecTarget *z = &ex->depth;
    if (!D.jit_ok || D.ended || !z->image) return 0;
    if (D.cam_size != OCAM_BLOCK) return z->image == D.jdepth;
    return z->format == VK_FORMAT_D32_SFLOAT && z->width == D.jw && z->height == D.jh &&
           ex->ubo[0][0].size == OCAM_BLOCK && ex->ubo[0][0].addr;
}

void vk_dlaa_viewport(VkExec *ex, VkViewport *v) {
    ex->vp_jittered = jitter_draw(ex);
    if (!ex->vp_jittered) return;
    ex->pass_jit = 1;
    v->x += D.jx, v->y += D.jy;
    D.ux = D.jx, D.uy = D.jy, D.jit_used = 1;
}

void vk_dlaa_jitter_check(VkExec *ex) {
    int j = jitter_draw(ex);
    if (j != ex->vp_jittered) ex->dirty |= VK_EXEC_DIRTY_VIEWPORT;
    const VkExecTarget *z = &ex->depth;   // the draws into a scene-size depth, with the jitter or without
    if (D.jit_ok && !D.ended && z->image && z->format == VK_FORMAT_D32_SFLOAT && z->width == D.jw && z->height == D.jh) {
        if (j) D.jdraws++;
        else D.jfixed++;
    }
}

static int scene_color(VkFormat f);

static void dlaa_draw_locked(VkExec *ex);
void vk_dlaa_draw(VkExec *ex) {
    if (!dl_active() || D.ended) return;
    static PlatMutex lock = PLAT_MUTEX_INIT;   // the replay workers' scene draws count at once (vk_exec.c chunks)
    int lk = vk_exec_shared();
    if (lk) plat_mutex_lock(&lock);
    dlaa_draw_locked(ex);
    if (lk) plat_mutex_unlock(&lock);
}

static void dlaa_draw_locked(VkExec *ex) {
    static int survey = -1;   // NX2NAVI_DLAA_SURVEY=1 (test): what is drawn into the armed light target
    if (survey < 0) survey = rt_knob("DLAA_SURVEY", "0");
    static uint64_t dfn[16];   // scene draws writing depth, by depth test (reversed depth: GREATER/GEQUAL)
    if (survey && ex->depth.image && ex->depth.image == D.jdepth && ex->pipe.ds.dt && ex->pipe.ds.dw)
        dfn[ex->pipe.ds.df & 15]++;
    if (survey && D.armed && ex->color[0].image == D.color.image) {
        static uint64_t n[2][2][2], arms, frames, last_frame;   // [blend][depth write][depth test]
        const VkPipeState *s = &ex->pipe;
        n[s->color.blend_en & 1][s->ds.dw][s->ds.dt]++;
        uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
        if (f != last_frame) last_frame = f, frames++;
        static uint64_t ps;
        if (ex->pass_serial != ps) ps = ex->pass_serial, arms++;
        if (frames % 300 == 0) {
            static uint64_t logged;
            if (logged != frames) {
                logged = frames;
                rt_log("dlaa survey: scene draws writing depth by depth test (NVN func 1-8: never less equal lequal "
                       "greater notequal gequal always): %llu %llu %llu %llu %llu %llu %llu %llu",
                       (unsigned long long)dfn[1], (unsigned long long)dfn[2], (unsigned long long)dfn[3],
                       (unsigned long long)dfn[4], (unsigned long long)dfn[5], (unsigned long long)dfn[6],
                       (unsigned long long)dfn[7], (unsigned long long)dfn[8]);
                rt_log("dlaa survey: %llu frames, %.1f passes a frame into the light target after arming; draws a frame "
                       "(blend/depth write/depth test): 000 %.1f 001 %.1f 010 %.1f 011 %.1f 100 %.1f 101 %.1f 110 %.1f 111 %.1f",
                       (unsigned long long)frames, (double)arms / frames, (double)n[0][0][0] / frames, (double)n[0][0][1] / frames,
                       (double)n[0][1][0] / frames, (double)n[0][1][1] / frames, (double)n[1][0][0] / frames,
                       (double)n[1][0][1] / frames, (double)n[1][1][0] / frames, (double)n[1][1][1] / frames);
            }
        }
    }
    // The scene camera of a draw into the scene depth: the deferred engine binds a 0x940-byte block at c4 with the
    // view-projection in rows 3-6 (0x30); Ocarina's opaque scene draws bind a 0x3500-byte scene context at c3, whose
    // rows 0-2 and 7-10 their vertex shaders use as the view and the projection (ocam_vp). Its full-screen passes bind
    // a 0x200-byte block at c3 instead, which is not the camera. Without a camera there is no camera motion and no
    // jitter (next_jitter), so DLSS runs on each frame alone without shaking the picture.
    const p_bind_buf *b = &ex->ubo[0][1];   // c4: the replay keeps user banks from c3 on
    const VkExecTarget *z = &ex->depth;
    if (!z->image || z->width < DLAA_SCENE_MIN_W || z->format != VK_FORMAT_D32_SFLOAT) return;
    if (!F.zdraws && !F.zother) F.seq0 = vk_exec_submit_seq;
    if (D.jdepth && z->image != D.jdepth) {
        F.zother++, F.zother_w = z->width, F.zother_h = z->height;
        return;
    }
    F.zdraws++;
    if (ex->ubo[0][0].size == OCAM_BLOCK && ex->ubo[0][0].addr) {
        F.ocam++;
        if (!ex->color[0].image || !scene_color(ex->color[0].format)) F.ocam_off++, F.ocam_off_fmt = ex->color[0].format;
    } else F.c3_other++, F.c3_size = ex->ubo[0][0].addr ? ex->ubo[0][0].size : 0;
    if (b->size == CAM_BLOCK && b->addr) D.cam_addr = b->addr, D.cam_size = CAM_BLOCK, D.cam_vp = 0x30;
    else if (ex->ubo[0][0].size == OCAM_BLOCK && ex->ubo[0][0].addr && ex->color[0].image &&
             scene_color(ex->color[0].format)) {   // Ocarina: the scene context most light-target draws use this frame
        uint64_t a = ex->ubo[0][0].addr;
        int i = 0, best = 0;
        while (i < D.nviews && D.views[i].addr != a) i++;
        if (i == D.nviews && D.nviews < 8) D.views[D.nviews].addr = a, D.views[D.nviews].n = 0, D.views[D.nviews++].got = 0;
        if (i < D.nviews) {
            D.views[i].n++, D.views[i].seq = vk_exec_submit_seq;
            // The camera rows as this draw reads them, from its own submission's copy: DLSS runs in a later submission
            // on some frames, whose copy may not hold the block (mv_pass, dlaa_cam_keep).
            if (!D.views[i].got) D.views[i].got = vk_usnap_read(a, OCAM_BYTES, D.views[i].cam);
        }
        for (int k = 1; k < D.nviews; k++)
            if (D.views[k].n > D.views[best].n) best = k;
        D.cam_addr = D.views[best].addr, D.cam_size = OCAM_BLOCK, D.cam_vp = 0;
    }
}

// NX2NAVI_DLAA_JITTER=0 (test): no jitter. Read once.
static int jitter_on(void) {
    static int on = -1;
    if (on < 0) on = rt_knob("DLAA_JITTER", "1");
    return on;
}

// The jitter moves on right after DLAA ran, not at the present: the replay can record the next frame's scene before
// the present of this one is handled, and a step taken there landed in the middle of a frame (worse after an off/on
// switch, which shifted that order). Jitter needs a frame DLSS ran on; a frame without the DLAA point stops it.
// The sequence is as long as NVIDIA's DLSS guide asks for the upscaling ratio, 8 x (output / scene)^2 phases (18 in
// Quality, 24 in Balanced, 32 in Performance, 72 in Ultra Performance), and 16 at least, the length DLAA has used from the start.
static void next_jitter(uint32_t w, uint32_t ow) {
    int nj = !jitter_on();
    double r = w ? (double)ow / w : 1.0;
    uint32_t phases = (uint32_t)ceil(8.0 * r * r - 1e-6);
    if (phases < 16) phases = 16;
    if (phases > 72) phases = 72;
    uint32_t i = D.jframe++ % phases + 1;
    D.jx = halton(i, 2) - 0.5f, D.jy = halton(i, 3) - 0.5f;
    // Jitter only where the camera is known (vk_dlaa_draw): the deferred engine's block or Ocarina's scene context. On
    // Ocarina's frame the draws that take it are the camera's into every scene-size depth (jitter_draw); taking only the
    // draws into DLSS's own depth image turned the picture white, and jitter was off there until that was found.
    D.jit_ok = !D.hold && !nj && (D.cam_size == CAM_BLOCK || D.cam_size == OCAM_BLOCK);
}

// The DLSS model (NX2NAVI_DLSS_PRESET, live): default takes NVIDIA's model per mode (evaluate: K for DLAA, Quality and
// Balanced, M for Performance, L for Ultra Performance), 0 here. A letter a..m takes that model in every mode: k pins
// K, j is the other transformer model, which NVIDIA describes as slightly less ghosting at the cost of extra flicker, e
// the older CNN model, cheaper on slower RTX cards. The value is an NVSDK_NGX_DLSS_Hint_Render_Preset (1 A ... 13 M),
// read when a feature is created; a change recreates the features (preset_check).
static int preset_value(void) {
    const char *v = cfg_get("NX2NAVI_DLSS_PRESET");
    int c = v && v[0] && !v[1] ? v[0] | 0x20 : 0;
    return c >= 'a' && c <= 'm' ? c - 'a' + 1 : 0;
}

// How strongly DLSS favours the current frame on characters and where they just were (NX2NAVI_DLSS_CHAR_MASK, percent,
// live; 0 off). Where a character's vectors are the camera's only (dlaa_objmv = 0, or a draw the per-object pass
// skips), one that moves on its own drags a faint copy of itself through DLSS's history; the bias mask makes DLSS take
// less of the history there, at the cost of some of its anti-aliasing on those pixels. Ocarina's frame has no class
// buffer to find characters by: there the mask covers the pixels whose depth the camera's motion does not explain
// (dlaa_mv.comp, the depth check).
static float char_bias(void) {
    const char *v = cfg_get("NX2NAVI_DLSS_CHAR_MASK");
    long n = v && *v ? strtol(v, NULL, 10) : 65;
    return n <= 0 ? 0.0f : n >= 100 ? 1.0f : (float)n / 100.0f;
}

// The quality setting DLSS is created with, from the scene's share of the output width.
static int perf_quality(uint32_t w, uint32_t ow) {
    if (w >= ow) return NGX_PERFQ_DLAA;
    double k = (double)w / ow;
    return k >= 0.66 ? 2 : k >= 0.57 ? 1 : k >= 0.49 ? 0 : 3;
}

// Every 300 frames DLSS ran, the history resets and camera gaps of those frames, while there were any (at most 400 lines a
// run): the measure for dlaa_cam_keep at one spot of the game.
static void window_line(void) {
    enum { N_ = 11 };
    static uint64_t last[N_];
    static int lines;
    if (D.evals % 300) return;
    uint64_t now[N_] = {D.why[0], D.why[1], D.why[2], D.mv_none, D.cam_none_draw, D.cam_none_snap, D.cam_kept,
                        D.cam_at_draw, D.frames_missed, D.seconds, D.target_kept}, d[N_], any = 0;
    for (int i = 0; i < N_; i++) d[i] = now[i] - last[i], last[i] = now[i];
    for (int i = 0; i < N_; i++) any |= d[i];
    if (!any || lines >= 400) return;
    lines++;
    rt_log("dlaa: frames %llu-%llu: resets %llu size, %llu camera, %llu target (%llu target changes with the history "
           "kept); %llu without a camera (%llu without a "
           "scene-context draw, %llu not in DLSS's submission), %llu kept with zero motion; %llu cameras read at the draw; "
           "%llu frames without the DLAA point, %llu with a second scene after DLSS ran",
           (unsigned long long)(D.evals - 299), (unsigned long long)D.evals, (unsigned long long)d[0],
           (unsigned long long)d[1], (unsigned long long)d[2], (unsigned long long)d[10], (unsigned long long)d[3],
           (unsigned long long)d[4],
           (unsigned long long)d[5], (unsigned long long)d[6], (unsigned long long)d[7], (unsigned long long)d[8],
           (unsigned long long)d[9]);
}

// dst: the target DLSS's picture goes to, NULL for the light target itself (DLAA in place).
static void hudless(VkCommandBuffer cb, struct Size *e, uint32_t w, uint32_t h);

// Another light target than last frame's. The game draws its scene into one of a few pooled targets of the same size
// and moves to another whenever its pass list changes, in the middle of play with the camera and the scene carrying
// on: 35 times in a walk of a minute through Kokiri Forest, three of them within eight frames on the ladder below
// the tree house.
// The history is the upscaler's own and not in the target, so where this frame's motion vectors are valid (cont: the
// camera of this frame and of the last one, mv_pass 1) the history stays. A new history at each change showed as a few
// soft, grainy frames every time, a flicker over the whole picture. Without valid vectors, and with
// NX2NAVI_DLAA_TARGET_KEEP=0 (test), a new history as before. The first few are logged with the new target's size, to
// tell a target the game made anew from the ones it alternates between.
static void target_changed(const VkExecTarget *t, int cont) {
    static int keep = -1;
    if (keep < 0) keep = rt_knob("DLAA_TARGET_KEEP", "1");
    int kept = keep && cont && D.last_color;
    if (D.last_color && D.target_logged < 16)
        D.target_logged++, rt_log("dlaa: frame %llu: the light target changed from %p to %p (%ux%u format %d): %s",
                                  (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1,
                                  (void *)D.last_color, (void *)t->image, t->width, t->height, (int)t->format,
                                  kept ? "the history stays (the camera carries on)" : "new history");
    if (kept) D.target_kept++;
    else D.reset = 1, D.why[2]++;
    D.last_color = t->image;
}

// DLSS's own video memory, the features and their history (NGX's DLSS statistics callback with the capability
// parameters), for the log's VRAM breakdown (vk_dlaa_vram); -1 until known. Asked on the replay thread, which makes every
// other NGX call, every 600 evaluations.
static volatile int64_t dlss_vram = -1;
int64_t vk_dlaa_vram(void) { return __atomic_load_n(&dlss_vram, __ATOMIC_RELAXED); }

static void query_vram(void) {
    void *cb = NULL;
    unsigned long long n = 0;
    if (fsr3() || !N.capsp || !p_getp(N.capsp, "DLSSGetStatsCallback", &cb) || !cb) return;
    if (NGX_OK(((NgxResult (*)(NgxParam *))cb)(N.capsp)) && p_getull(N.capsp, "SizeInBytes", &n))
        __atomic_store_n(&dlss_vram, (int64_t)n, __ATOMIC_RELAXED);
}

// Frees the sizes other than cur that no evaluation used for SIZE_STALE evaluations: the GPU finished with them long
// ago (the CPU is never that many presents ahead), so their images and feature go at once.
static void stale_sizes(const struct Size *cur) {
    if (!idle_free()) return;
    for (int i = 0; i < DLSS_SIZES; i++) {
        struct Size *s = &D.sizes[i];
        if (s == cur || !s->out || D.evals < s->used + SIZE_STALE) continue;
        rt_log("dlaa: the DLSS feature at %ux%u -> %ux%u went unused for %d frames: freed", s->w, s->h, s->ow, s->oh,
               SIZE_STALE);
        if (D.last == s) D.last = NULL;
        free_size(s);
    }
}

// Before a feature is made: the features of the other sizes with a larger input go, or all of them (all: a feature was
// released since the last creation, feat_released); their images stay. The GPU may still be running an evaluation of
// theirs, so it finishes first, as in free_size.
static void release_kept(const struct Size *making, int all) {
    int n = 0;
    for (int i = 0; i < DLSS_SIZES; i++) {
        struct Size *s = &D.sizes[i];
        if (s == making || !s->feat) continue;
        if (!all && (uint64_t)s->w * s->h <= (uint64_t)making->w * making->h) continue;
        if (!n++) vk_queue_lock(), vkDeviceWaitIdle(vk_ctx()->device), vk_queue_unlock();
        N.release(s->feat), s->feat = NULL;
        if (D.last == s) D.last = NULL;
    }
    if (n) rt_log("dlaa: %d DLSS feature(s) released before the one for %ux%u is made (%s); made again when used", n,
                  making->w, making->h, all ? "after a release" : "larger inputs");
}

// The outer columns and rows of what a jittered pass draws into. The jitter moves the scene's viewport by under half a
// pixel (vk_dlaa_viewport), which keeps every pixel centre inside it, yet on an RX 7900 XT (AMD's Windows driver) the draws
// of a viewport moved left left the last column out, and of one moved up the last row: the driver evidently limits
// drawing to the whole pixels inside the viewport. Those pixels keep what was there before the pass: in a frame dump
// (NX2NAVI_DUMP, Hyrule Field) the depth's last row kept its clear value through every scene pass and the light target the
// sky's colour, and in a village the light target's clear colour, white. With NX2NAVI_DLAA_JITTER=0, and without DLSS or
// FSR, every row is drawn.
// So when a pass that took the jitter ends (vk_exec.c vk_exec_pass_end), the outermost row and column on each side of
// its colour targets and its depth are filled from the ones next to them, before anything reads them. Filling only the
// armed pair right before the upscaler read it (evaluate, which still does it for a target a dispatch wrote) was too
// late for whatever reads the scene in between: water reads the scene behind it and showed bright patches along the
// bottom in a fifth of the shots of a run through a stream (none with the jitter off), and the bloom and the
// anti-aliasing pass read the row too. (The white line in a player's clips had a second cause, the upscaler's picture
// never reaching the screen with the HUD hidden: vk_dlaa_upscaler.)
// A copied edge is one render pixel at the picture's border, which the upscaler blends with its history.
// NX2NAVI_DLAA_EDGE_FILL: 1 (default) at each jittered pass's end and before the upscaler; 2 only before the upscaler (as
// 0.9.1 did, for comparison); 0 not at all.
static int edge_fill_mode(void) {
    static int on = -1;
    if (on < 0) on = rt_knob("DLAA_EDGE_FILL", "1");
    return on;
}

// One image's border from the texels next to it, in one copy: the columns and rows without their ends, then the four
// corners from their diagonal neighbours. Every source is inside the border and every destination on it, so no region
// reads what another writes. The image is in GENERAL and nothing is still drawing into it.
static void edge_copy(VkCommandBuffer cb, const VkExecTarget *g, VkImageAspectFlags aspect) {
    int32_t w = (int32_t)g->width, h = (int32_t)g->height;
    if (w < 4 || h < 4) return;
    const VkImageSubresourceLayers sl = {aspect, g->level, g->base_layer, 1};
#define EDGE(sx, sy, dx, dy, ew, eh)                                                                                      \
    {.srcSubresource = sl, .srcOffset = {sx, sy, 0}, .dstSubresource = sl, .dstOffset = {dx, dy, 0},                     \
     .extent = {(uint32_t)(ew), (uint32_t)(eh), 1}}
    const VkImageCopy r[8] = {EDGE(1, 1, 0, 1, 1, h - 2),         EDGE(w - 2, 1, w - 1, 1, 1, h - 2),
                              EDGE(1, 1, 1, 0, w - 2, 1),         EDGE(1, h - 2, 1, h - 1, w - 2, 1),
                              EDGE(1, 1, 0, 0, 1, 1),             EDGE(w - 2, 1, w - 1, 0, 1, 1),
                              EDGE(1, h - 2, 0, h - 1, 1, 1),     EDGE(w - 2, h - 2, w - 1, h - 1, 1, 1)};
#undef EDGE
    vkCmdCopyImage(cb, g->image, VK_IMAGE_LAYOUT_GENERAL, g->image, VK_IMAGE_LAYOUT_GENERAL, 8, r);
}

static int edge_target(const VkExecTarget *g) { return g->image && g->samples <= 1 && !g->is3d && !g->blocks; }

void vk_dlaa_pass_end(VkExec *ex) {
    if (!ex->pass_jit) return;
    ex->pass_jit = 0;
    if (edge_fill_mode() != 1) return;
    int any = edge_target(&ex->depth) && (ex->depth.aspect & VK_IMAGE_ASPECT_DEPTH_BIT);
    for (uint32_t i = 0; i < ex->pass_ncolor && i < 8; i++) any |= edge_target(&ex->color[i]);
    if (!any) return;
    static int told;
    if (!told++) rt_log("dlaa: the outer rows and columns of each jittered pass's targets are filled from the ones next to them");
    vk_exec_barrier(ex);   // the pass's draws are done, and its targets are in GENERAL
    for (uint32_t i = 0; i < ex->pass_ncolor && i < 8; i++)
        if (edge_target(&ex->color[i])) edge_copy(ex->cb, &ex->color[i], VK_IMAGE_ASPECT_COLOR_BIT), ex->recorded++;
    if (edge_target(&ex->depth) && (ex->depth.aspect & VK_IMAGE_ASPECT_DEPTH_BIT))
        edge_copy(ex->cb, &ex->depth, VK_IMAGE_ASPECT_DEPTH_BIT), ex->recorded++;
}

// Before the upscaler reads the armed pair (evaluate): once more, for a light target a dispatch wrote after its last
// pass. The pair is at level 0 of the scene's size.
static void edge_fill(VkCommandBuffer cb, const VkExecTarget *t, const VkExecTarget *z, int jittered) {
    if (!edge_fill_mode() || !jittered) return;
    if (edge_target(t)) edge_copy(cb, t, VK_IMAGE_ASPECT_COLOR_BIT);
    if (edge_target(z) && z->width == t->width && z->height == t->height) edge_copy(cb, z, VK_IMAGE_ASPECT_DEPTH_BIT);
    full_barrier(cb);
}

// Set when the scene the upscaler runs on has another size than the last one; vk_dlaa_resized_take reads it.
static volatile int32_t size_changed;

static void evaluate(VkExec *ex, const VkExecTarget *dst) {
    const VkExecTarget *t = &D.color, *z = &D.depth;
    if (!dst) dst = t;
    uint32_t w = t->width, h = t->height, ow = dst->width, oh = dst->height;
    VkCommandBuffer cb = ex->cb;
    vk_exec_barrier(ex);   // what drew the scene is done (no pass is open here)
    vk_gpu_gap_note(VK_GAP_UPSCALER, (uint64_t)ow * oh);
    if (D.preset_dirty) {   // another model: features are created with it, so every size gets a new one
        for (int i = 0; i < DLSS_SIZES; i++)
            if (D.sizes[i].out) free_size(&D.sizes[i]);
        D.last = NULL, D.preset_dirty = 0;
    }
    struct Size *e = size_for(cb, w, h, ow, oh);
    if (!e) return dl_fail("work image creation", 0);
    if (e != D.last) {   // another size's history is from another moment
        D.reset = 1, D.why[0]++, e->chars_ok = 0;
        __atomic_store_n(&size_changed, 1, __ATOMIC_RELEASE);
    }
    // Read once: vk_dlaa_frame runs on the present's thread and can clear fsr_frame while this replay is between the
    // checks below, which would send the rest of this function to DLSS's path without NGX after all.
    const int fsr = use_fsr();
    if (!fsr && !N.ngx_up) return;   // neither runs this frame: vk_dlaa_frame turns DLAA off at the next present
    if (!fsr && !D.params && !NGX_OK(N.alloc_params(&D.params))) return dl_fail("parameter allocation", 0);
    NgxParam *p = D.params;
    VkImageView zv = vk_exec_target_view(z->img, z->format, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_DEPTH_BIT, z->level,
                                         z->base_layer, 1, 0);
    float jx = D.jit_used ? D.ux : 0.0f, jy = D.jit_used ? D.uy : 0.0f;
    edge_fill(cb, t, z, D.jit_used);   // before the motion vectors read the depth
    static int no_mv = -1;
    if (no_mv < 0) no_mv = !rt_knob("DLAA_MV", "1");   // NX2NAVI_DLAA_MV=0: zero motion (test)
    int masked = 0, mv = no_mv ? 0 : mv_pass(ex, e, zv, jx, jy, char_bias(), &masked);
    if (mv != 1) {
        VkClearColorValue zero = {{0, 0, 0, 0}};
        VkImageSubresourceRange all = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cb, e->mv, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &all);
        full_barrier(cb);
        // no camera: the history cannot be moved along (2: kept as it is for a short gap, mv_pass dlaa_cam_keep)
        if (!no_mv && mv == 0) D.reset = 1, D.why[1]++;
    }
    VkImageView cv = vk_exec_target_view(t->img, t->format, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, t->level,
                                         t->base_layer, 1, 0);
    struct Snap *mk = D.mv_samp ? mask_pass(cb, cv, w, h) : NULL;
    int timed = T.on > 0 && T.used[T.slot];   // mv_pass began this frame's slot
    if (timed) times_mark(cb, T_DL0);
    uint64_t c0 = timed ? kern_now_ns() : 0;
    rec_drain();   // NGX records into the command buffer itself: the recorder thread must have made every call so far
    if (timed) T.cpu_drain += kern_now_ns() - c0;
    if (mv == 1) fcam_probe(cb, e, w, h);   // mv_pass's distances, behind its barrier
    if (fsr) {   // FSR 3.1 / FSR 4 (vk_ffx.cpp) in DLSS's place: the same images, then the same blit below
        int is_fsr4 = taa_is_fsr4();
        const char *label = is_fsr4 ? "FSR 4" : "FSR 3.1";
        if (t->level || t->base_layer || z->level || z->base_layer) return dl_fail("FSR on a target that is not level 0, layer 0", 0);
        if (D.last_color != t->image) target_changed(t, mv == 1);
        uint64_t now = plat_now_ns();
        double dt = D.last_eval_ns ? (now - D.last_eval_ns) / 1e6 : 0;
        D.last_eval_ns = now;
        VkFfxImage fc = {t->image, t->format, w, h, 0}, fz = {z->image, z->format, w, h, 0},
                   fm = {e->mv, VK_FORMAT_R16G16_SFLOAT, w, h, 1}, fo = {e->out, VK_FORMAT_R16G16B16A16_SFLOAT, ow, oh, 1};
        VkFfxCamera fcam;
        if (!vk_dlaa_ffx_camera(&fcam)) memset(&fcam, 0, sizeof fcam);   // no camera this frame: FSR's fixed values
        if (!vk_ffx_upscale(cb, &fc, &fz, &fm, &fo, jx, jy, D.reset, dt > 0 && dt < 100 ? (float)dt : 16.67f, &fcam, is_fsr4))
            return dl_fail(label, 0);
        static int told;
        if (!told++) rt_log("dlaa: %s at %ux%u -> %ux%u (scene format %d, depth format %d)", label, w, h, ow, oh,
                            (int)t->format, (int)z->format);
        goto evaluated;
    }
    if (!e->feat) {
        // NX2NAVI_TRACE=perf: the release of the kept features (a device idle wait) and the create, as one mark
        uint64_t t0 = perf_mark_begin();
        if (idle_free() != 2) release_kept(e, feat_released);
        feat_released = 0;
        p_setu(p, "CreationNodeMask", 1);
        p_setu(p, "VisibilityNodeMask", 1);
        p_setu(p, "Width", w);
        p_setu(p, "Height", h);
        p_setu(p, "OutWidth", ow);
        p_setu(p, "OutHeight", oh);
        p_seti(p, "PerfQualityValue", perf_quality(w, ow));
        // IsHDR | MVLowRes | AutoExposure: linear scene light, motion vectors at the render size, no exposure texture
        p_seti(p, "DLSS.Feature.Create.Flags", (1 << 0) | (1 << 1) | (1 << 6));
        p_seti(p, "DLSS.Enable.Output.Subrects", 0);
        // The model per mode (NVSDK_NGX_DLSS_Hint_Render_Preset: 1 A ... 11 K, 12 L, 13 M), as NVIDIA recommends for
        // DLSS 4.5 (310.x): K (transformer) for DLAA, Quality and Balanced, M for Performance, L for Ultra Performance.
        // M and L cost more on cards without FP8 (RTX 20/30). dlss_preset (D.preset, preset_check) takes one model for
        // every mode instead.
        int pk = D.preset ? D.preset : 11, pm = D.preset ? D.preset : 13, pl = D.preset ? D.preset : 12;
        p_setu(p, "DLSS.Hint.Render.Preset.DLAA", (unsigned)pk);
        p_setu(p, "DLSS.Hint.Render.Preset.UltraQuality", (unsigned)pk);
        p_setu(p, "DLSS.Hint.Render.Preset.Quality", (unsigned)pk);
        p_setu(p, "DLSS.Hint.Render.Preset.Balanced", (unsigned)pk);
        p_setu(p, "DLSS.Hint.Render.Preset.Performance", (unsigned)pm);
        p_setu(p, "DLSS.Hint.Render.Preset.UltraPerformance", (unsigned)pl);
        NgxResult r = N.create(cb, NGX_FEATURE_SUPERSAMPLING, p, &e->feat);
        perf_mark_end(PERF_M_NGX, t0, 0);
        if (!NGX_OK(r) || !e->feat) return dl_fail("DLSS feature creation", r);
        D.reset = 1;
        int q = perf_quality(w, ow), pr = q == 0 ? pm : q == 3 ? pl : pk;
        rt_log("dlaa: DLSS feature created at %ux%u -> %ux%u, quality value %d, preset %c (scene format %d, depth format "
               "%d)", w, h, ow, oh, q, 'A' + pr - 1, (int)t->format, (int)z->format);
    }
    if (D.last_color != t->image) target_changed(t, mv == 1);
    NgxResourceVK rc = res_image(cv, t->image, t->format, w, h, VK_IMAGE_ASPECT_COLOR_BIT, 0);
    NgxResourceVK ro = res_image(e->out_view, e->out, VK_FORMAT_R16G16B16A16_SFLOAT, ow, oh, VK_IMAGE_ASPECT_COLOR_BIT, 1);
    NgxResourceVK rz = res_image(zv, z->image, z->format, w, h, VK_IMAGE_ASPECT_DEPTH_BIT, 0);
    NgxResourceVK rm = res_image(e->mv_view, e->mv, VK_FORMAT_R16G16_SFLOAT, w, h, VK_IMAGE_ASPECT_COLOR_BIT, 0);
    p_setp(p, "Color", &rc);
    p_setp(p, "Output", &ro);
    p_setp(p, "Depth", &rz);
    p_setp(p, "MotionVectors", &rm);
    p_setf(p, "Jitter.Offset.X", jx);
    p_setf(p, "Jitter.Offset.Y", jy);
    p_setf(p, "Sharpness", 0.0f);
    p_seti(p, "Reset", D.reset);
    // One bias mask: the reactive one (NX2NAVI_DLSS_REACTIVE, a test) where it ran, else the characters' (mv_pass), else
    // none. The parameter object is reused, so it is cleared explicitly.
    NgxResourceVK rk = mk ? res_image(mk->mask_view, mk->mask, VK_FORMAT_R8_UNORM, w, h, VK_IMAGE_ASPECT_COLOR_BIT, 0)
                          : res_image(e->mask_view, e->mask, VK_FORMAT_R8_UNORM, w, h, VK_IMAGE_ASPECT_COLOR_BIT, 0);
    p_setp(p, "DLSS.Input.Bias.Current.Color.Mask", mk || masked ? (void *)&rk : NULL);
    p_setu(p, "DLSS.Input.Bias.Current.Color.Subrect.Base.X", 0);
    p_setu(p, "DLSS.Input.Bias.Current.Color.Subrect.Base.Y", 0);
    p_setf(p, "MV.Scale.X", 1.0f);
    p_setf(p, "MV.Scale.Y", 1.0f);
    p_setu(p, "DLSS.Render.Subrect.Dimensions.Width", w);
    p_setu(p, "DLSS.Render.Subrect.Dimensions.Height", h);
    p_setf(p, "DLSS.Pre.Exposure", 1.0f);
    p_setf(p, "DLSS.Exposure.Scale", 1.0f);
    // The time since the previous evaluation, which DLSS weighs its history by; 60 fps before the first and after a
    // gap (a pause, a load), where no history carries over anyway.
    uint64_t now = plat_now_ns();
    double dt = D.last_eval_ns ? (now - D.last_eval_ns) / 1e6 : 0;
    D.last_eval_ns = now;
    p_setf(p, "FrameTimeDeltaInMsec", dt > 0 && dt < 100 ? (float)dt : 16.67f);
    c0 = timed ? kern_now_ns() : 0;
    NgxResult r = N.evaluate(cb, e->feat, p, NULL);
    if (timed) T.cpu_ngx += kern_now_ns() - c0;
    if (!NGX_OK(r)) return dl_fail("DLSS evaluation", r);
evaluated:
    if (timed) times_mark(cb, T_DL1);   // through the recorder, after NGX's own commands
    // Frame generation gets the same new start with this frame's inputs (vk_dlaa_fg_inputs): a frame DLSS starts a
    // new history on (a camera cut, another scene size or target) has no motion from the frame before that holds.
    D.fg_reset = D.reset;
    D.reset = 0;
    D.last = e, e->used = D.evals + 1;
    if (D.evals % 600 == 0) query_vram();
    stale_sizes(e);
    full_barrier(cb);
    vk_mvcheck_output(cb, e->out, ow, oh);   // NX2NAVI_MV_CHECK (test)
    // the result into the scene target (DLAA: the bloom, the post passes and the HUD read it and draw over it) or into
    // the scaling pass's target (the HUD draws over it)
    VkImageBlit b = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                     .srcOffsets = {{0, 0, 0}, {(int32_t)ow, (int32_t)oh, 1}},
                     .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, dst->level, dst->base_layer, 1},
                     .dstOffsets = {{0, 0, 0}, {(int32_t)ow, (int32_t)oh, 1}}};
    // NX2NAVI_DLAA_MVDEBUG=1 (test): the motion vectors in place of the picture, x in red and y in green, render pixels
    // as scene light (the post passes still run over them)
    static int mvdebug = -1;
    if (mvdebug < 0) mvdebug = rt_knob("DLAA_MVDEBUG", "0");
    static int maskdebug = -1;
    if (maskdebug < 0) maskdebug = rt_knob("DLAA_MASKDEBUG", "0");
    // (NX2NAVI_DLAA_MASKDEBUG=1 shows whichever bias mask DLSS got: the reactive one, or the characters')
    VkImage shown = mvdebug ? e->mv : maskdebug && mk ? mk->mask : maskdebug && masked ? e->mask : e->out;
    if (shown != e->out) b.srcOffsets[1] = (VkOffset3D){(int32_t)w, (int32_t)h, 1};
    vkCmdBlitImage(cb, shown, VK_IMAGE_LAYOUT_GENERAL, dst->image, VK_IMAGE_LAYOUT_GENERAL, 1, &b, VK_FILTER_NEAREST);
    full_barrier(cb);
    if (dst != t) vk_dlaa_swap_img = t->img, vk_dlaa_swap_view = e->out_view;   // the scaling draw reads DLSS's picture
    if (vk_dlaa_hudless_wanted) hudless(cb, e, ow, oh);   // frame generation keeps the HUD out of its generated frames
    ex->recorded += 4;
    vk_exec_draw_new_cb();   // NGX bound pipelines and descriptor sets of its own: the next draw binds everything again
    D.jdepth = z->image, D.jw = z->width, D.jh = z->height;
    // for frame generation at this frame's present (vk_dlaa_fg_inputs)
    D.fg_depth = (VkFfxImage){z->image, z->format, w, h, 0};
    D.fg_mv = (VkFfxImage){e->mv, VK_FORMAT_R16G16_SFLOAT, w, h, 1};
    D.fg_depth_view = zv, D.fg_mv_view = e->mv_view;
    D.fg_jx = jx, D.fg_jy = jy, D.fg_at = D.presents, D.fg_fresh = 1;
    if (D.jit_used) D.jittered++;
    next_jitter(w, ow);
    if (++D.evals == 1 || D.evals == 300 || D.evals % 3600 == 0)   // 300: one early line with the frames counted
        rt_log("dlaa: %llu frames anti-aliased (scene %ux%u, output %ux%u; %llu frames without the DLAA point, %llu with camera "
               "motion, %llu with the characters' mask, %llu without a camera, %llu jittered (%llu draws moved, %llu not), "
               "%llu jitter holds; resets: %llu size, %llu camera, %llu target; reactive mask %llu, without its copy %llu; "
               "no camera: %llu without a scene-context draw, %llu not in DLSS's submission; %llu kept with zero motion, "
               "%llu runs past %d; %llu cameras read at the draw)",
               (unsigned long long)D.evals, w, h, ow, oh, (unsigned long long)D.frames_missed,
               (unsigned long long)D.mv_frames, (unsigned long long)D.masked, (unsigned long long)D.mv_none,
               (unsigned long long)D.jittered, (unsigned long long)D.jdraws, (unsigned long long)D.jfixed,
               (unsigned long long)D.holds, (unsigned long long)D.why[0], (unsigned long long)D.why[1],
               (unsigned long long)D.why[2], (unsigned long long)R.made, (unsigned long long)R.missed,
               (unsigned long long)D.cam_none_draw, (unsigned long long)D.cam_none_snap, (unsigned long long)D.cam_kept,
               (unsigned long long)D.cam_kept_max, KEEP_MAX, (unsigned long long)D.cam_at_draw);
    window_line();
}

// The HUD-less picture (vk_dlaa_fg_hudless): DLSS's output, which the HUD is then drawn over, through the encode the
// game's last pass gives its picture (linear to sRGB), as RGBA8 UNORM bytes like the picture the present reads. With
// HDR (NX2NAVI_HDR = 1) the game's picture stays float and linear up to the present, which encodes it to PQ
// (vk_hdr_encode): DLSS's output is kept as RGBA16F for the same encode. The game's own HDR (native) encodes inside the
// game, which this cannot follow: no HUD-less picture there.
int vk_dlaa_hudless_wanted;
static void hudless(VkCommandBuffer cb, struct Size *e, uint32_t w, uint32_t h) {
    const VkCtx *c = vk_ctx();
    int hdr = vk_hdr_wanted() && !vk_hdr_native();
    if (vk_hdr_native()) return;
    VkFormat fmt = hdr ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM;
    if (D.hl && (D.hl_w != w || D.hl_h != h || D.hl_fmt != fmt)) {
        vk_queue_lock(), vkDeviceWaitIdle(c->device), vk_queue_unlock();
        hl_destroy();
    }
    if (!D.hl) {
        VkImageView v2 = VK_NULL_HANDLE;
        VkImageUsageFlags u = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        if ((!hdr && !image2d(w, h, VK_FORMAT_R8G8B8A8_SRGB, u, &D.hl_srgb, &D.hl_srgb_mem, &v2)) ||
            !image2d(w, h, fmt, u | VK_IMAGE_USAGE_STORAGE_BIT, &D.hl, &D.hl_mem, &D.hl_view)) {
            if (v2) vkDestroyImageView(c->device, v2, NULL);   // the first image's view, when the second image failed
            hl_destroy();   // and what of the two images was made
            rt_log("dlaa: the HUD-less picture's images could not be made: frame generation sees the HUD");
            vk_dlaa_hudless_wanted = 0;
            return;
        }
        if (v2) vkDestroyImageView(c->device, v2, NULL);
        D.hl_w = w, D.hl_h = h, D.hl_fmt = fmt;
        VkImageMemoryBarrier2 ib[2];
        VkImage im[2] = {D.hl, D.hl_srgb};
        int nim = D.hl_srgb ? 2 : 1;
        for (int i = 0; i < nim; i++)
            ib[i] = (VkImageMemoryBarrier2){.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                            .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                            .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                            .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = im[i],
                                            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = (uint32_t)nim,
                               .pImageMemoryBarriers = ib};
        vkCmdPipelineBarrier2(cb, &di);
    }
    VkImageBlit b = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .srcOffsets = {{0, 0, 0}, {(int32_t)w, (int32_t)h, 1}},
                     .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstOffsets = {{0, 0, 0}, {(int32_t)w, (int32_t)h, 1}}};
    vkCmdBlitImage(cb, e->out, VK_IMAGE_LAYOUT_GENERAL, hdr ? D.hl : D.hl_srgb, VK_IMAGE_LAYOUT_GENERAL, 1, &b,
                   VK_FILTER_NEAREST);
    full_barrier(cb);
    if (hdr) {
        D.hl_at = D.presents, D.hl_fresh = 1;
        return;
    }
    VkImageCopy cp = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                      .extent = {w, h, 1}};
    vkCmdCopyImage(cb, D.hl_srgb, VK_IMAGE_LAYOUT_GENERAL, D.hl, VK_IMAGE_LAYOUT_GENERAL, 1, &cp);
    full_barrier(cb);
    D.hl_at = D.presents, D.hl_fresh = 1;
}

static int scene_color(VkFormat f) {
    return f == VK_FORMAT_B10G11R11_UFLOAT_PACK32 || f == VK_FORMAT_R16G16B16A16_SFLOAT;
}

// Two frame shapes. A deferred frame (the G-buffer, several colour targets at the scene size) arms on the light target
// drawn after it and ends the scene at the first pass a quarter as wide (the bloom), as described above. Ocarina's
// frame has no G-buffer: its light target (B10G11R11 at the scene size, with the D32F scene depth) is drawn into from the
// first opaque pass on, half- and quarter-size effect passes come and go in between. Measured (NX2NAVI_DUMP, Kokiri
// Forest, DLAA in a 2560x1440 picture): the last pass with the scene depth is the bloom composite into a second light
// target (each such pass at the scene's size takes over the arming, with its depth); the game's upscaler (its stand-in
// here, vk_tonemap.c) then fills a light-format target without depth, at the picture's size, which the HUD is drawn
// into; a last draw copies that target into the 8-bit sRGB output target, encoding only, since the game tonemaps before
// its post pass (vk_tonemap.c, vk_hdr.c). The scene ends at the start of the pass the HUD is drawn into: DLSS reads the
// armed pair and its picture goes into that target, which the HUD is drawn over and the last draw reads. With an
// upscaling share that target is larger than the scene; in DLAA it has the scene's size. A frame without it (the
// menus) ends at the last draw, at the 8-bit output target of the scene's size, and DLSS's picture goes back into the
// light target, which that draw reads.
static int srgb8(VkFormat f) { return f == VK_FORMAT_R8G8B8A8_SRGB || f == VK_FORMAT_B8G8R8A8_SRGB; }

// The output target DLAA runs at. With the game's own HDR (vk_hdr.c vk_hdr_native) the same pass, at the same point of
// the frame, tonemaps into its PQ picture (vk_hdr.c vk_hdr_native_picture) instead of the 8-bit sRGB one; matching sRGB
// alone left DLAA off for the whole run there (no frame anti-aliased) and with it the frame generation that takes
// DLAA's depth and motion.
static int output_target(VkFormat f) { return srgb8(f) || vk_hdr_native_picture(f); }

double wide_dlss_share(void);   // widescreen.c: the DLSS mode's share of the picture (1 DLAA), 0 off

// Ocarina's own upscaler is the place for DLSS or FSR: its stand-in (vk_exec_draw.c untranslated_dispatch_fallback) has
// just scaled the scene into the image the rest of the frame shows, and the upscaler's picture goes over it there.
// Until this hook the upscaler ran at the pass after it (vk_dlaa_pass below), which was taken for the pass the HUD is
// drawn into, the stand-in's output image as its target. With the HUD hidden the game draws no such pass: the next pass
// is the frame's last, which encodes the stand-in's image into the 8-bit output. DLSS's or FSR's picture was then
// written into that 8-bit target and the last draw drew the stand-in's picture over it: the screen showed the
// jittered scene at its render size, scaled with a plain filter. On a player's save with the HUD off that was every
// frame: blocks of render pixels, a picture that shook with the jitter by up to one render pixel, the undrawn edge
// row in white, and the upscaler's cost for nothing; with the HUD on (every test save until then) the same settings
// gave the upscaler's picture. Found by dumping the same frame of both saves (NX2NAVI_DUMP): three HUD passes before the
// last one in one, none in the other.
// The image is the stand-in's output when it has the light target's format and the scene's shape and is no smaller
// than the scene (the movie network's and other tensor programs' outputs are RGBA8 or smaller). A frame without the
// dispatch (NX2NAVI_CS_FALLBACK=0, another engine) still ends at vk_dlaa_pass. NX2NAVI_DLAA_AT_UPSCALER=0 (test): as before.
void vk_dlaa_upscaler(VkExec *ex, VkImage out, VkFormat fmt, uint32_t w, uint32_t h) {
    static int on = -1;
    if (on < 0) on = rt_knob("DLAA_AT_UPSCALER", "1");
    if (!on || !dl_active() || !D.armed || D.done || D.gbuf || !out || fmt != VK_FORMAT_B10G11R11_UFLOAT_PACK32) return;
    uint32_t sw = D.color.width, sh = D.color.height;
    if (w < sw || h < sh || out == D.color.image) return;
    if ((double)w * sh > 1.03 * (double)h * sw || (double)h * sw > 1.03 * (double)w * sh) return;   // not the scene's shape
    static int told;
    if (!told++)
        rt_log("dlaa: the upscaler runs where the game's own does: scene %ux%u into its %ux%u output", sw, sh, w, h);
    VkExecTarget dst = {.image = out, .format = fmt, .aspect = VK_IMAGE_ASPECT_COLOR_BIT, .layers = 1, .width = w,
                        .height = h, .samples = 1};
    vk_exec_hook_turn();   // as in vk_dlaa_pass: the replay workers' draws read the jitter and its end
    D.ended = D.done = D.up = 1;
    evaluate(ex, &dst);
}

void vk_dlaa_pass(VkExec *ex) {
    const VkExecTarget *c = &ex->color[0], *z = &ex->depth;
    if (!dl_active()) return;
    if (D.done) {   // diagnostics: another scene drawn into a light target in the same present interval
        if (ex->pass_ncolor == 1 && c->image && scene_color(c->format) && z->image && z->format == VK_FORMAT_D32_SFLOAT &&
            c->width == D.color.width && c->height == D.color.height && z->width == c->width)
            F.second++, F.second_img = c->image;
        return;
    }
    static int deferred;   // a scene-size G-buffer was seen: the deferred shape
    unsigned o[2], s[2];
    res_sizes(o, s);
    // Ocarina's builds carry no scene or output patch sets, so res_sizes reports placeholders there: the sizes come
    // from the frame itself (the light target's, and the output pass's).
    int scene = z->image && z->format == VK_FORMAT_D32_SFLOAT && z->width >= DLAA_SCENE_MIN_W;
    // The G-buffer: several colour targets with the D32F scene depth. Only the light target drawn after it arms the
    // frame; at night the game draws into the light target alone with the scene depth before the G-buffer too, and
    // DLAA then ran at a small pass ahead of the whole scene, which reached the screen jittered.
    // A pass of that shape smaller than the light target already armed is not one: Ocarina draws a half-size effect
    // into three targets with a half-size D32F depth after its scene, and taking that for a G-buffer switched every
    // later frame to the deferred shape, where DLSS's and FSR 3.1's smaller modes never found their output pass.
    if (ex->pass_ncolor >= 3 && scene && !(D.armed && (z->width < D.color.width || z->height < D.color.height))) {
        if (!deferred)
            rt_log("dlaa: a G-buffer pass (%d colour targets, depth %ux%u, target 0 format %d): the deferred frame shape",
                   ex->pass_ncolor, z->width, z->height, c->image ? (int)c->format : 0);
        D.gbuf = 1, D.armed = 0, deferred = 1;
        // target 0 is the object-class buffer (RGBA8, the scene size): the characters' bias mask reads its green. A
        // smaller pass of the same shape (640x360, seen while loading) does not replace the scene's in the same frame.
        if (c->image && c->format == VK_FORMAT_R8G8B8A8_UNORM && c->width == z->width && c->height == z->height &&
            c->samples <= 1 && !c->is3d && (!D.ids_ok || c->width >= D.ids.width))
            D.ids = *c, D.ids_ok = 1;
        return;
    }
    // the scene depth is D32F; a D16 target of the scene size also serves one composite pass before the transparents.
    // A smaller pair never takes over an armed one: the camera rune draws a half-size RGBA16F target with a half-size
    // D32F depth after the lighting (852x480 under a 1704x960 scene), and DLSS ran on that and wrote its picture into
    // the scene's own light target, which left every object black.
    if ((D.gbuf || !deferred) && ex->pass_ncolor == 1 && c->image && scene_color(c->format) && scene &&
        c->width == z->width && c->height == z->height &&
        (!D.armed || (c->width >= D.color.width && c->height >= D.color.height)) && c->samples <= 1 && !c->is3d) {
        int first = !D.armed;
        D.color = *c, D.depth = *z, D.armed = 1;
        static int told;
        if (first && !told++)
            rt_log("dlaa: first scene armed: light target %ux%u format %d, depth %ux%u", c->width, c->height, (int)c->format,
                   z->width, z->height);
        if (first) mask_snapshot(ex);   // the lit scene, before the transparents (reactive mask)
        return;
    }
    if (!D.armed) return;
    if (!deferred) {   // Ocarina's shape: the output pass ends the scene and runs DLSS
        int one = ex->pass_ncolor == 1 && c->image && !z->image && c->image != D.color.image && c->samples <= 1 &&
                  !c->is3d;
        if (one) F.one_n++, F.one_w = c->width, F.one_h = c->height, F.one_fmt = c->format;
        // the output pass is the picture's size, the scene over the mode's share: at Ultra Performance (854 wide) a
        // 960x540 post pass comes first, and DLSS wrote there, which left a 540-line picture stretched to the output
        // A frame whose scene is not at that share (the first presents before the scale patch takes, a dynamic
        // resolution step) has no such pass: after a frame that missed DLSS the first pass wider than the scene is
        // taken for 120 frames (D.share_off, vk_dlaa_frame), so a mismatch costs one frame rather than every frame.
        double share = wide_dlss_share();
        int native = share <= 0 || share >= 1;   // no upscaling share: the scene is as large as the picture
        int big = D.share_off || native || c->width >= 0.95 * D.color.width / share;
        int same = c->width == D.color.width && c->height == D.color.height;
        // The output pass has the scene's shape: the game redraws its 2000x2000 area map into an RGBA8 target of its
        // own on a room change and on the pause menu, one colour target and no depth, larger than the scene both ways.
        // Taken for the output pass, DLSS wrote the game's picture into the map, and the minimap showed that picture
        // behind its room shapes until the pause menu drew the map again. A pass whose width over height is more than
        // 3% from the scene's (sizes round to multiples of 8) is not the output; the ultrawide sets widen the scene
        // with the picture, so their output pass still matches.
        int shape = (double)c->width * D.color.height <= 1.03 * (double)c->height * D.color.width &&
                    (double)c->height * D.color.width <= 1.03 * (double)c->width * D.color.height;
        if (one && !shape && big && c->width > D.color.width && c->height > D.color.height && D.shape_logged < 8)
            D.shape_logged++, rt_log("dlaa: a %ux%u pass is not the output pass: its shape is not the scene's %ux%u",
                                     c->width, c->height, D.color.width, D.color.height);
        if (one && shape && big && c->width > D.color.width && c->height > D.color.height) {
            // DLSS runs: the replay workers' draws read the jitter and its end, and list the per-object motion it
            // takes (vk_exec.c "replay workers"); the G-buffer and the arming above change nothing they read
            vk_exec_hook_turn();
            D.ended = D.done = D.up = 1;
            evaluate(ex, c);
        } else if (one && same && native && scene_color(c->format)) {
            // DLAA in gameplay: the pass the HUD is drawn into is also at the scene's size, in the light format, and
            // the game's upscaler (its stand-in here) has just written the picture there. Waiting for the 8-bit output
            // pass, as in a frame without it, put DLSS's picture back into the armed light target (the bloom
            // composite's), which nothing reads after the upscaler took it: the frame showed no DLAA at all. With an
            // upscaling share the HUD's pass is the larger one above, so a pass of this shape is not it and is ignored
            // there as before.
            static int told;
            if (!told++)
                rt_log("dlaa: the pass the HUD is drawn into is at the scene's size (%ux%u, format %d): DLSS writes there",
                       c->width, c->height, (int)c->format);
            vk_exec_hook_turn();
            D.ended = D.done = D.up = 1;
            evaluate(ex, c);
        } else if (one && same && output_target(c->format)) {
            vk_exec_hook_turn();
            D.ended = D.done = 1;
            evaluate(ex, NULL);
        }
        return;
    }
    if (!D.ended) {
        if (!ex->pass_w || ex->pass_w * 4 > D.color.width) return;
        vk_exec_hook_turn();
        D.ended = 1;
        if (D.color.width >= o[0]) D.done = 1, evaluate(ex, NULL);   // DLAA in place; else wait for the scaling pass
        return;
    }
    // the game's scaling pass: one colour target of the output size, no depth
    if (ex->pass_ncolor == 1 && c->image && !z->image && scene_color(c->format) && c->width > D.color.width &&
        c->width == o[0] && c->height == o[1] && c->samples <= 1 && !c->is3d) {
        vk_exec_hook_turn();
        D.done = D.up = 1;
        evaluate(ex, c);
    }
}

// NX2NAVI_DLSS, read at each present (the menu switches it live): the scene size of the mode (resolution.c) and the
// texture bias that goes with it while DLSS or FSR 3.1 runs. NVIDIA's DLSS guide asks for log2(render / output) - 1 on
// top of the game's own bias: the render scale's part (res_dlss_lod_bias) samples textures for the output's resolution,
// and the -1 sharper still, since the jittered frames DLSS accumulates resolve the detail one level finer: -1 at DLAA,
// -1.58 in Quality (67 %), -2 in Performance. The -1 comes only with the jitter (jitter_on): without it the same pixel
// samples the same spot every frame, and a sharper level would shimmer instead. The sampler clamps the sum to the
// device's limit (vk_texture.c make_sampler).
#define JITTER_LOD_BIAS (-1.0f)
static void dlss_mode(void) {
    static char mode[32];
    static int bias_on = -1;
    const char *m = dynres_override();   // dynamic resolution's step (dynres.c) while it differs from the setting
    if (!m) m = cfg_value("NX2NAVI_DLSS");
    if (strcmp(m, mode)) {
        snprintf(mode, sizeof mode, "%s", m);
        res_dlss_mode(m);
        bias_on = -1;
    }
    int b = dl_active();
    if (b != bias_on)
        bias_on = b, vk_sampler_dlss_bias(b ? res_dlss_lod_bias() + (jitter_on() ? JITTER_LOD_BIAS : 0.0f) : 0.0f);
}


// The preset, read again when a setting changed (the menu); a different one recreates the features at the next DLAA run.
static void preset_check(void) {
    unsigned v = cfg_version();
    if (v == D.cfg_seen) return;
    D.cfg_seen = v;
    int p = preset_value();
    if (p == D.preset) return;
    if (D.evals) {
        if (p) rt_log("dlaa: DLSS preset %c in every mode; the features are created again", 'A' + p - 1);
        else rt_log("dlaa: DLSS presets per mode; the features are created again");
    }
    D.preset = p, D.preset_dirty = 1, D.reset = 1;
}

// The scene's size after a setting changed (a DLSS or FSR 3.1 mode picked in the menu). The game makes its scene targets
// again at the new size and they start empty, so the first frames drawn into them are wrong: the first one is lit
// without what the frame before it left (a dark scene), and two frames later the scene can be black with the interface
// intact (Ultra Performance and DLAA). Measured by capturing every present over six mode changes, 2026-11-10: bad
// frames 3 and 5 presents after the change, with every other setting the same. evaluate notes a scene of another size;
// vk_dlaa_frame counts the presents since a setting last changed, so a game that draws two sizes in turn holds nothing.
static int setting_presents;            // vk_dlaa_frame: presents left in which that counts as a mode change
#define SETTING_PRESENTS 60

// vk_backend.c frame_hold, each present after vk_dlaa_frame: 1 once when the scene took a new size after a setting
// changed. The previous picture then stays on screen over the frames described above.
int vk_dlaa_resized_take(void) {
    if (!__atomic_exchange_n(&size_changed, 0, __ATOMIC_ACQ_REL) || !setting_presents) return 0;
    setting_presents = 0;
    return 1;
}

void vk_dlaa_frame(void) {
    static unsigned cfg_seen;
    static int cfg_known;
    unsigned v = cfg_version();
    D.presents++;
    if (cfg_known && v != cfg_seen) setting_presents = SETTING_PRESENTS;
    else if (setting_presents) setting_presents--;
    cfg_seen = v, cfg_known = 1;
    vk_dlaa_swap_img = NULL, vk_dlaa_swap_view = VK_NULL_HANDLE;
    dlss_mode();
    preset_check();
    if (!N.ngx_up) {   // FSR 3.1 / FSR 4 follows the setting
        fsr_frame = fsr3();
        N.available = fsr_frame && cfg_bool("NX2NAVI_DLAA") &&
                      (taa_is_fsr4() ? vk_ffx_load_fsr4() : vk_ffx_load());
    }
    int on = N.available && cfg_bool("NX2NAVI_DLAA");
    if (on != D.on) {
        rt_log("dlaa: %s", on ? "on" : "off");
        D.on = on;
        // either way the next DLAA frame starts a new history from an unjittered frame
        D.reset = 1, D.cam_prev_ok = 0, D.jit_ok = D.jit_used = D.armed = D.done = D.gbuf = D.ended = D.up = 0;
        D.hold = 0;
        D.fg_fresh = D.hl_fresh = 0;   // frame generation gets nothing made before the switch (fg_current)
        memset(&F, 0, sizeof F);
        return;
    }
    if (!dl_active()) return;
    if (D.share_off) D.share_off--;
    if (!D.done && D.armed) D.share_off = 120;   // vk_dlaa_pass: the output pass without the mode's share for a while
    if (!D.done) {
        D.frames_missed++;
        // the first few after DLSS first ran: what the frame drew, to tell a frame drawn another way from a loading screen
        if (D.evals && D.miss_logged < 24) {
            D.miss_logged++;
            char arm[64] = "never armed";
            if (D.armed) snprintf(arm, sizeof arm, "armed on a %ux%u light target", D.color.width, D.color.height);
            rt_log("dlaa: frame %llu without the DLAA point: %s; %u scene-depth draws (%u into another D32 depth); %u "
                   "later passes with one colour target and no depth (last %ux%u format %d); %llu submissions since the "
                   "last present, %d held submissions waiting for a signal",
                   (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED), arm, F.zdraws, F.zother,
                   F.one_n, F.one_w, F.one_h, (int)F.one_fmt, (unsigned long long)(vk_exec_submit_seq - D.seq_present),
                   vk_exec_held());
        }
    }
    if (D.done && F.second) {
        D.seconds++;
        if (D.second_logged < 24)
            D.second_logged++, rt_log("dlaa: frame %llu: a second scene after DLSS ran (%u passes into light target %p; "
                                      "DLSS ran on %p); %llu submissions since the last present, %d held",
                                      (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED),
                                      F.second, (void *)F.second_img, (void *)D.color.image,
                                      (unsigned long long)(vk_exec_submit_seq - D.seq_present), vk_exec_held());
    }
    D.seq_present = vk_exec_submit_seq;
    if (D.jit_used && !D.done) D.hold = 60, D.holds++;   // jittered but not resolved: that frame shook
    else if (D.hold) D.hold--;
    if (!D.done) D.jit_ok = 0;
    D.armed = D.done = D.jit_used = D.gbuf = D.ended = D.up = D.ids_ok = 0;
    D.cam_addr = 0, D.nviews = 0;
    memset(&F, 0, sizeof F);
}

// ---------------------------------------------------------------- the game's FXAA (NX2NAVI_FXAA, live)
// NX2navi's FXAA is two full-screen passes after the bloom, just before the HUD: one copies the scene target into a
// scratch target, the other draws the anti-aliased result back over the scene (found by diffing frame dumps with the
// game's own switch, ModelSceneExtensionParam EnableFXAA, on and off). With fxaa = 0 the draws of that second pass's
// fragment shader are skipped, so the scene reaches the HUD as it was. Shaders are known by the FNV-1a hash of their
// SPIR-V, as vk_dump.c names them; a translator change gives new hashes and the switch then does nothing.
// With a scene smaller than the output (the game's own 5/6 render scale, or a DLSS mode) the same shader draws the
// scaled picture into the output-sized target, in a pass without the depth attachment the in-place FXAA has: there it
// is always drawn, whatever fxaa says (skipping it would leave the 3D picture out), unless DLSS already wrote that
// target this frame.
// The stock translator's hash, the one with tools/xlate patch 0010 (PCSS, PR #89) applied, the one with patches 0010
// and 0011 (the shadow filter spread) applied, and that one with the zero_locals job flag (Mesa, vk_caps.c).
// A scaling draw this list misses still scales DLSS's picture, not
// the jittered scene: after DLSS ran, passes larger than the light target sample DLSS's output (vk_dlaa_swap_img).
static const uint64_t FXAA_FS[] = {0xf450003ebeb3dc52ull, 0x0a12f4bc2a9a4fbeull, 0x5c43ae18948d52cbull,
                                   0xaad42238801d31d8ull};

static uint64_t spirv_hash(const GfxShader *s) {
    static struct { const GfxShader *s; uint64_t h; } tab[8192];
    uintptr_t k = ((uintptr_t)s >> 4) * 0x9E3779B97F4A7C15ull;
    for (uint32_t i = 0, j = (uint32_t)(k >> 51); i < 8192; i++, j = (j + 1) & 8191) {
        if (tab[j].s == s) return tab[j].h;
        if (!tab[j].s) {
            uint64_t words = 0, h = 0xcbf29ce484222325ull;
            const uint8_t *b = (const uint8_t *)gfx_shader_spirv(s, &words);
            for (uint64_t n = 0; b && n < words * 4; n++) h = (h ^ b[n]) * 0x100000001b3ull;
            tab[j].s = s, tab[j].h = h;
            return h;
        }
    }
    return 0;   // table full: no skip
}

int vk_dlaa_skip_draw(VkExec *ex, const GfxShader *fs) {
    static uint64_t frame_read = ~0ull;
    static int fxaa = 1;
    // fxaa = 0 reports once whether it found the game's FXAA draw: the first skip, or a minute of frames (3600) with
    // none, which is what a translator change that moves FXAA_FS looks like on every GPU, with or without DLSS
    static uint64_t fx_off_frames;
    static int fx_found, fx_reported;
    uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    if (f != frame_read) {   // once a frame: the menu switches it live
        frame_read = f, fxaa = cfg_bool("NX2NAVI_FXAA");
        if (!fxaa && !fx_found && !fx_reported && ++fx_off_frames == 3600) {
            fx_reported = 1;
            rt_log("dlaa: fxaa = 0 found no draw of the game's FXAA in %llu frames (its fragment shader is not in "
                   "FXAA_FS): the FXAA stays on", (unsigned long long)fx_off_frames);
        }
    }
    if (fxaa && !D.up) return 0;   // nothing to skip: no shader lookup per draw at the defaults
    if (!fs) return 0;
    uint64_t h = spirv_hash(fs);
    int known = 0;
    for (size_t i = 0; i < sizeof FXAA_FS / sizeof FXAA_FS[0]; i++) known |= h == FXAA_FS[i];
    if (!known) return 0;
    if (!ex->depth.image) {   // the scaling pass, which DLSS replaced
        static int told;
        if (D.up && !told++) rt_log("dlaa: the game's scaling draw left out, DLSS wrote it (fragment shader %016llx)",
                                    (unsigned long long)h);
        return D.up;
    }
    if (!fxaa && !fx_found) {
        fx_found = 1;
        rt_log("dlaa: fxaa = 0 skips the game's FXAA (fragment shader %016llx)", (unsigned long long)h);
    }
    return !fxaa;
}


// ---------------------------------------------------------------- DLSS Frame Generation through NGX (vk_ngxfg)
// gfx_present.c records this into the present command buffer once the frame's picture, HUD included, is in bb. It
// generates count frames between the previous call's picture and this one into out[0..count-1] (MultiFrameIndex 1 to
// count) and a copy of bb into real; gfx_present.c's presenter shows them in that order, then the rendered one. The
// inputs are the ones the Streamline path gets (vk_dlssg.c): the depth, motion vectors, jitter and camera of the frame
// DLSS last ran on, and the picture without the HUD.
int vk_ngxfg_max(void) { return vk_ngxfg_wanted() && N.available && !FG.failed ? FG.max : 0; }
int vk_ngxfg_on(void) { return vk_ngxfg_max() && !strcmp(cfg_value("NX2NAVI_FRAMEGEN"), "dlss"); }
void vk_ngxfg_reset(void) { FG.reset = 1; }

int vk_ngxfg_evaluate(VkCommandBuffer cb, const VkNgxImage *bb, const VkNgxImage *hudless, const VkNgxImage *out, int count,
                      const VkNgxImage *real) {
    if (!vk_ngxfg_max() || count < 1) return 0;
    if (count > FG.max) count = FG.max;
    VkFfxImage d, m;
    VkImageView dv, mvv;
    VkDlaaCamera c;
    float jx = 0, jy = 0;
    int reset = 0;
    if (!vk_dlaa_fg_inputs(&d, &m, &jx, &jy, &reset) || !vk_dlaa_fg_views(&dv, &mvv) || !vk_dlaa_fg_camera(&c)) {
        FG.reset = 1;   // a frame without DLSS's inputs: the next generation has no valid previous frame
        return 0;
    }
    if (!FG.p && !NGX_OK(N.alloc_params(&FG.p))) {
        FG.failed = 1;
        rt_log("ngxfg: parameter allocation failed: frame generation off");
        return 0;
    }
    NgxParam *p = FG.p;
    if (FG.feat && (FG.w != bb->w || FG.h != bb->h || FG.fmt != bb->fmt)) {
        N.release(FG.feat);
        FG.feat = NULL;
    }
    if (!FG.feat) {
        p_setu(p, "CreationNodeMask", 1);
        p_setu(p, "VisibilityNodeMask", 1);
        p_setu(p, "Width", bb->w);
        p_setu(p, "Height", bb->h);
        p_setu(p, "DLSSG.BackbufferFormat", (unsigned)bb->fmt);
        uint64_t t0 = perf_mark_begin();
        NgxResult r = N.create(cb, NGX_FEATURE_FRAME_GENERATION, p, &FG.feat);
        perf_mark_end(PERF_M_NGX, t0, 0);
        if (!NGX_OK(r) || !FG.feat) {
            FG.failed = 1, FG.feat = NULL;
            rt_log("ngxfg: DLSS Frame Generation feature creation failed (%08x): frame generation off", (unsigned)r);
            return 0;
        }
        FG.w = bb->w, FG.h = bb->h, FG.fmt = bb->fmt, FG.reset = 1;
        rt_log("ngxfg: DLSS Frame Generation created at %ux%u (format %d), depth and motion %ux%u", bb->w, bb->h,
               (int)bb->fmt, d.width, d.height);
    }
    NgxResourceVK rb = res_image(bb->view, bb->img, bb->fmt, bb->w, bb->h, VK_IMAGE_ASPECT_COLOR_BIT, 0);
    NgxResourceVK rz = res_image(dv, d.image, d.format, d.width, d.height, VK_IMAGE_ASPECT_DEPTH_BIT, 0);
    NgxResourceVK rm = res_image(mvv, m.image, m.format, m.width, m.height, VK_IMAGE_ASPECT_COLOR_BIT, 0);
    NgxResourceVK rh = hudless ? res_image(hudless->view, hudless->img, hudless->fmt, hudless->w, hudless->h,
                                           VK_IMAGE_ASPECT_COLOR_BIT, 0) : rb;
    NgxResourceVK rr = res_image(real->view, real->img, real->fmt, real->w, real->h, VK_IMAGE_ASPECT_COLOR_BIT, 1);
    p_setp(p, "DLSSG.Backbuffer", &rb);
    p_setp(p, "DLSSG.Depth", &rz);
    p_setp(p, "DLSSG.MVecs", &rm);
    p_setp(p, "DLSSG.HUDLess", hudless ? &rh : NULL);
    p_setp(p, "DLSSG.UI", NULL);
    p_setp(p, "DLSSG.UIAlpha", NULL);
    p_setp(p, "DLSSG.BidirectionalDistortionField", NULL);
    p_setp(p, "DLSSG.OutputReal", &rr);
    p_setp(p, "DLSSG.OutputDisableInterpolation", NULL);
    p_setp(p, "DLSSG.CameraViewToClip", c.view_to_clip);
    p_setp(p, "DLSSG.ClipToCameraView", c.clip_to_view);
    p_setp(p, "DLSSG.ClipToLensClip", NULL);
    p_setp(p, "DLSSG.ClipToPrevClip", c.clip_to_prev_clip);
    p_setp(p, "DLSSG.PrevClipToClip", c.prev_clip_to_clip);
    p_setf(p, "DLSSG.JitterOffsetX", jx);
    p_setf(p, "DLSSG.JitterOffsetY", jy);
    p_setf(p, "DLSSG.MvecScaleX", 1.0f / (float)m.width);   // the vectors are in render pixels
    p_setf(p, "DLSSG.MvecScaleY", 1.0f / (float)m.height);
    p_setf(p, "DLSSG.CameraPinholeOffsetX", 0.0f);
    p_setf(p, "DLSSG.CameraPinholeOffsetY", 0.0f);
    static const char *const pos[] = {"DLSSG.CameraPosX", "DLSSG.CameraPosY", "DLSSG.CameraPosZ"};
    static const char *const up[] = {"DLSSG.CameraUpX", "DLSSG.CameraUpY", "DLSSG.CameraUpZ"};
    static const char *const rt[] = {"DLSSG.CameraRightX", "DLSSG.CameraRightY", "DLSSG.CameraRightZ"};
    static const char *const fw[] = {"DLSSG.CameraFwdX", "DLSSG.CameraFwdY", "DLSSG.CameraFwdZ"};
    for (int i = 0; i < 3; i++)
        p_setf(p, pos[i], c.pos[i]), p_setf(p, up[i], c.up[i]), p_setf(p, rt[i], c.right[i]), p_setf(p, fw[i], c.fwd[i]);
    p_setf(p, "DLSSG.CameraNear", c.znear);   // the frame's own planes (proj_planes), as vk_dlssg.c gives them
    p_setf(p, "DLSSG.CameraFar", c.zfar);
    p_setf(p, "DLSSG.CameraFOV", c.fov_y);
    p_setf(p, "DLSSG.CameraAspectRatio", c.aspect);
    p_setu(p, "DLSSG.ColorBuffersHDR", 0);
    p_setu(p, "DLSSG.DepthInverted", (unsigned)c.inverted);
    p_setu(p, "DLSSG.CameraMotionIncluded", 1);
    p_setu(p, "DLSSG.Reset", (unsigned)(reset || FG.reset));
    p_setu(p, "DLSSG.AutomodeOverrideReset", 0);
    p_setu(p, "DLSSG.NotRenderingGameFrames", 0);
    p_setu(p, "DLSSG.OrthoProjection", 0);
    p_setu(p, "DLSSG.MvecDilated", 0);
    p_setu(p, "DLSSG.MenuDetectionEnabled", 0);
    struct { const char *base; uint32_t w, h; } rects[] = {
        {"DLSSG.MVecsSubrect", m.width, m.height}, {"DLSSG.DepthSubrect", d.width, d.height},
        {"DLSSG.HUDLessSubrect", bb->w, bb->h}, {"DLSSG.InputBackbufferSubrect", bb->w, bb->h},
        {"DLSSG.OutputInterpolatedSubrect", bb->w, bb->h}, {"DLSSG.OutputRealSubrect", bb->w, bb->h}};
    for (size_t i = 0; i < sizeof rects / sizeof *rects; i++) {
        char k[64];
        snprintf(k, sizeof k, "%sBaseX", rects[i].base), p_setu(p, k, 0);
        snprintf(k, sizeof k, "%sBaseY", rects[i].base), p_setu(p, k, 0);
        snprintf(k, sizeof k, "%sWidth", rects[i].base), p_setu(p, k, rects[i].w);
        snprintf(k, sizeof k, "%sHeight", rects[i].base), p_setu(p, k, rects[i].h);
    }
    p_setu(p, "DLSSG.MultiFrameCount", (unsigned)count);
    int made = 0;
    for (int i = 0; i < count; i++) {
        NgxResourceVK ro = res_image(out[i].view, out[i].img, out[i].fmt, out[i].w, out[i].h, VK_IMAGE_ASPECT_COLOR_BIT, 1);
        p_setp(p, "DLSSG.OutputInterpolated", &ro);
        p_setu(p, "DLSSG.MultiFrameIndex", (unsigned)(i + 1));
        NgxResult r = N.evaluate(cb, FG.feat, p, NULL);
        if (!NGX_OK(r)) {
            static int told;
            if (told++ < 5) rt_log("ngxfg: evaluation %d of %d failed (%08x)", i + 1, count, (unsigned)r);
            break;
        }
        made++;
    }
    full_barrier(cb);
    FG.reset = 0;
    if (made && (++FG.evals == 1 || FG.evals % 3600 == 0))
        rt_log("ngxfg: %d frames with generation (%d generated per rendered frame)", FG.evals, made);
    return made;
}
