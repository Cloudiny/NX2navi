// The in-game menu core: Dear ImGui state, the event queue the window adapter fills, gamepad navigation, the Vulkan back
// end on our device and the menu's pages. Free of any window system (menu.h). ImGui runs on the present path only,
// under gfx_present.c's present lock; the event functions only queue.
//
// The Vulkan set-up: dynamic rendering into the swapchain image, an internal descriptor pool, vertex buffers rotated
// over more frames than are in flight.
//
// The look: colours, rounding, the tab bar with L / R and Q / E, the window over a dimmed game. The pages are rows of
// option_rows.cpp.
//
// Contains code Copyright (c) 2026 Peiikai, Copyright (c) 2026 Revivedx and the Savage Wind Recomp contributors (MIT; see
// THIRD_PARTY_NOTICES.md).
#include <atomic>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "imgui.h"
#include "imgui_internal.h"   // SetFocusID, NavCursorVisible
#include "backends/imgui_impl_vulkan.h"
// A private copy of stb_truetype, the parser Dear ImGui builds its font atlas with, to check a system font before the
// atlas gets it (font_usable). Its asserts only mark the font bad here.
static int g_font_bad;
#define STBTT_STATIC
#define STB_TRUETYPE_IMPLEMENTATION
#define STBTT_assert(x) ((x) ? (void)0 : (void)(g_font_bad = 1))
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#include "imstb_truetype.h"
#include "menu_sound.h"
#include "menu_assets.h"
#include "ui/menu_art.h"
#include "ui/menu_skin.h"
#pragma clang diagnostic pop
#include "ui/amiibo_page.h"
#include "ui/binds_page.h"
#include "ui/coop_page.h"
#include "ui/debug_menu.h"
#include "ui/option_rows.h"
#include "ui/perf_panel.h"

extern "C" {
#include "amiibo.h"
#include "cheats.h"
#include "debug_tools.h"
#include "cfg.h"
#include "folder_pick.h"
#include "fs.h"
#include "input.h"
#include "input_map.h"
#include "mods.h"
#include "perfmon.h"
#include "platform/platform.h"
#include "save_share.h"
#include "sysinfo.h"
#include "ui/menu.h"
#include "ui_text.h"
#include "vk_dlssg.h"
// host.h and vk.h pull in cpu.h, which is C only: the functions the menu needs from them, as declared there.
void rt_log(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
void vk_queue_lock(void);
void vk_queue_unlock(void);
void res_sizes(unsigned out[2], unsigned scene[2]);
int res_adjustable(void);
int wide_available(void);
const char *const *vk_color_preset(const char *name);
int vk_hdr_wanted(void);
int vk_hdr_active(void);
int vk_hdr_needs_restart(void);
int vk_dlaa_needs_restart(void);
int vk_ngxfg_wanted(void);              // vk_dlaa.c: DLSS's frame generation through NGX itself can run in this run
int vk_ngxfg_max(void);                 // ... and how many frames it may generate per rendered one (0: it cannot)
int vk_ffx_available(void);             // vk_ffx.cpp: FSR 3.1 upscaling can run (asked without loading the library)
int vk_ffx_fsr4_available(void);        // vk_ffx.cpp: Linux FSR 4 (amd_fidelityfx_upscaler_vk.so) can run
int vk_ffx_framegen_available(void);    // ... and FSR 3.1 frame generation
int vk_ffx4_available(void);            // vk_ffx4.cpp: AMD's FSR 4 runs on this device (its libraries, card and driver)
int vk_ffx_fsr4(void);                  // vk_ffx.cpp: ... and the setting asks for it (NX2NAVI_FSR4)
void vk_ffx4_retry(void);               // vk_ffx4.cpp: FSR 4 chosen again after failures that turned it off
const char *vk_ffx4_name(void);         // vk_ffx4.cpp: "AMD FSR 4.1.2" once the driver's library was asked
int vk_fg4_available(void);             // vk_fg4.cpp: the frame generation of AMD's FSR 4 library runs on this device
int vk_fg4_failed(void);                // ... it failed in this run, FSR 3.1's generates the frames
int vk_fg4_active(void);                // ... its swapchain is in place
const char *vk_fg4_name(void);          // ... the version the library runs here ("3.1.6", "4.0.2")
int vk_texture_detail_auto(void);       // vk_texture.c: the texture detail auto picks on this card, 0 full to 2 low
double frame_step_limit_hz(void);
int frame_limit_adjustable(void);
int rtss_limit_state(void);
int frame_free_running(void);
extern const int rt_game_version;
extern const char rt_port_version[];
void cheats_frame(uint64_t frame);   // cheats.c
bool tab_cheats(float scale);        // ui/cheats_tab.cpp; true: restart into the rebuild
}

// Roboto Medium (Apache-2.0, ui/fonts/LICENSE-Roboto.txt) with Latin and Cyrillic glyphs. The file comes in with
// #embed where the compiler has it (clang 19 and later define __has_embed). An older one, which a Linux build with the
// distribution's compiler can meet, includes the same bytes as numbers from roboto_ttf.h, which tools/build_shaders.py
// writes beside this file from the font when the game is built (a kit carries it).
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"   // #embed in C++
static const unsigned char roboto_ttf[] = {
#ifdef __has_embed
#embed "fonts/Roboto-Medium.ttf"
#else
#include "roboto_ttf.h"
#endif
};
#pragma clang diagnostic pop

// ---- open state, any thread
static std::atomic<int> g_open;     // the menu is open
static std::atomic<int> g_reset;    // opened or closed since the last laid-out frame: drop stale input

// g_open, the input block and the log line change together under open_lock, so they cannot disagree when the window thread
// (F1, Esc), the polling thread (combo) and the present path (Continue, B) race.
static PlatMutex open_lock = PLAT_MUTEX_INIT;

static PlatMutex ev_lock = PLAT_MUTEX_INIT;   // the event queue below
static int ev_n;

// want 0 or 1 sets the state; -1 flips it. Returns the new state. The window events queued before the change go with it:
// keys pressed just before a close would otherwise reach the menu when it next opens.
static int change_open(int want) {
    plat_mutex_lock(&open_lock);
    int open = want < 0 ? !g_open.load() : want;
    if (g_open.load() != open) {
        g_open.store(open);
        g_reset.store(1);
        plat_mutex_lock(&ev_lock);
        ev_n = 0;
        plat_mutex_unlock(&ev_lock);
        input_menu(open);
        menu_sound_play(open ? MENU_SOUND_OPEN : MENU_SOUND_CLOSE);
        rt_log("menu: %s", open ? "open" : "closed");
    }
    plat_mutex_unlock(&open_lock);
    return open;
}

static void set_open(int open) { change_open(open); }

// The debug menu (ui/debug_menu.cpp; F9 with NX2NAVI_DEBUG_MENU = 1) is laid out in this menu's place: open is the one
// state for both, with its input, its sounds and the game's blocked pad, and g_debug says which window it shows.
static std::atomic<int> g_debug{0};
extern "C" int menu_debug_enabled(void) { return cfg_bool("NX2NAVI_DEBUG_MENU"); }

static int g_failed_flag();
// Until the game's first frame (menu_overlay_frame), boot screen included: the menu does not open, as it would not be
// drawn while its keys queued up and changed settings once the game ran.
static std::atomic<int> g_booting{1};
extern "C" int menu_is_open(void) { return g_open.load(std::memory_order_relaxed); }
extern "C" void menu_toggle(void) {
    if (g_failed_flag() || g_booting.load(std::memory_order_acquire)) return;
    if (!menu_is_open()) g_debug.store(0);   // F1 opens the settings; with either window open it closes that one
    change_open(-1);
}
// F9: opens the debug menu, and closes it again; with the settings menu open it does nothing.
extern "C" void menu_debug_toggle(void) {
    if (!menu_debug_enabled() || g_failed_flag() || g_booting.load(std::memory_order_acquire)) return;
    if (menu_is_open()) {
        if (g_debug.load()) change_open(0);
        return;
    }
    g_debug.store(1);
    change_open(1);
}

// ---- events from the window adapter, queued while the menu is open; the present path hands them to ImGui
enum { EV_KEY, EV_POS, EV_BUTTON, EV_WHEEL, EV_FOCUS, EV_CHAR, EV_MAX = 256 };
struct Ev { int type, a, b; float x, y; };
static Ev ev_q[EV_MAX];

static void push(Ev e) {
    if (!menu_is_open()) return;
    plat_mutex_lock(&ev_lock);
    // checked again under the lock: change_open empties the queue after the state changes, so an event of the closed
    // menu cannot slip in behind that
    if (!menu_is_open()) {
        plat_mutex_unlock(&ev_lock);
        return;
    }
    if (e.type == EV_POS && ev_n && ev_q[ev_n - 1].type == EV_POS) ev_q[ev_n - 1] = e;   // only the newest position
    else if (ev_n < EV_MAX) ev_q[ev_n++] = e;
    plat_mutex_unlock(&ev_lock);
}

extern "C" void menu_event_key(int key, int down) { push({EV_KEY, key, down, 0, 0}); }
extern "C" void menu_event_mouse_pos(float x, float y) { push({EV_POS, 0, 0, x, y}); }
extern "C" void menu_event_mouse_button(int button, int down) { push({EV_BUTTON, button, down, 0, 0}); }
extern "C" void menu_event_wheel(float dx, float dy) { push({EV_WHEEL, 0, 0, dx, dy}); }
extern "C" void menu_event_focus(int focused) { push({EV_FOCUS, focused, 0, 0, 0}); }
// a typed character, for the text fields of the co-op page and the debug menu
extern "C" void menu_event_char(unsigned c) { push({EV_CHAR, (int)c, 0, 0, 0}); }

// The clipboard's text a page asked for (gfx_window_paste): the window thread reads it, the page takes it a frame later.
static PlatMutex paste_lock = PLAT_MUTEX_INIT;
static char paste_text[256];
static int paste_ready;

extern "C" void menu_event_paste(const char *utf8) {
    plat_mutex_lock(&paste_lock);
    snprintf(paste_text, sizeof paste_text, "%s", utf8 ? utf8 : "");
    paste_ready = 1;
    plat_mutex_unlock(&paste_lock);
}

extern "C" int menu_paste_take(char *out, size_t cap) {
    plat_mutex_lock(&paste_lock);
    int ready = paste_ready;
    if (ready && cap) snprintf(out, cap, "%s", paste_text);
    paste_ready = 0;
    plat_mutex_unlock(&paste_lock);
    return ready;
}

static void drain_events(ImGuiIO &io) {
    Ev q[EV_MAX];
    plat_mutex_lock(&ev_lock);
    int n = ev_n;
    for (int i = 0; i < n; i++) q[i] = ev_q[i];
    ev_n = 0;
    plat_mutex_unlock(&ev_lock);
    for (int i = 0; i < n; i++) {
        const Ev &e = q[i];
        switch (e.type) {
        case EV_KEY:
            io.AddKeyEvent((ImGuiKey)e.a, e.b != 0);
            // ImGui takes the modifier state (io.KeyShift, io.KeyCtrl: Shift+Tab, Ctrl+Up) from these keys only
            if (e.a == ImGuiKey_LeftShift) io.AddKeyEvent(ImGuiMod_Shift, e.b != 0);
            if (e.a == ImGuiKey_LeftCtrl) io.AddKeyEvent(ImGuiMod_Ctrl, e.b != 0);
            break;
        case EV_POS: io.AddMousePosEvent(e.x, e.y); break;
        case EV_BUTTON: io.AddMouseButtonEvent(e.a, e.b != 0); break;
        case EV_WHEEL: io.AddMouseWheelEvent(e.x, e.y); break;
        case EV_FOCUS: io.AddFocusEvent(e.a != 0); break;
        case EV_CHAR: io.AddInputCharacter((unsigned)e.a); break;
        }
    }
}

// ---- gamepad navigation: the Switch buttons of our SDL3 gamepad to ImGui's gamepad keys. Nintendo layout: A (east)
// activates, B (south) cancels.
static const struct { int bit; ImGuiKey key; } pad_keys[] = {
    {B_A, ImGuiKey_GamepadFaceDown}, {B_B, ImGuiKey_GamepadFaceRight}, {B_X, ImGuiKey_GamepadFaceUp},
    {B_Y, ImGuiKey_GamepadFaceLeft}, {B_L, ImGuiKey_GamepadL1}, {B_R, ImGuiKey_GamepadR1},
    {B_ZL, ImGuiKey_GamepadL2}, {B_ZR, ImGuiKey_GamepadR2}, {B_LEFT, ImGuiKey_GamepadDpadLeft},
    {B_UP, ImGuiKey_GamepadDpadUp}, {B_RIGHT, ImGuiKey_GamepadDpadRight}, {B_DOWN, ImGuiKey_GamepadDpadDown},
};
static uint64_t pad_prev;   // buttons at the last frame; a button held when the menu opened counts once released
static int stick_dir = -1;  // the D-pad button the left stick holds, -1 none

// The left stick moves the cursor as the D-pad does (ImGui's own stick handling only scrolls the page): past half way
// it holds the D-pad direction of its larger axis, kept until that axis falls under 0.35; ImGui repeats a held one.
static int stick_to_dpad(int32_t lx, int32_t ly) {
    float x = lx / 32767.0f, y = ly / 32767.0f;   // y up
    float along = stick_dir == B_LEFT ? -x : stick_dir == B_RIGHT ? x : stick_dir == B_UP ? y : stick_dir == B_DOWN ? -y : 0;
    if (stick_dir >= 0 && along > 0.35f) return stick_dir;
    if (fmaxf(fabsf(x), fabsf(y)) < 0.5f) return -1;
    return fabsf(x) > fabsf(y) ? (x < 0 ? B_LEFT : B_RIGHT) : (y > 0 ? B_UP : B_DOWN);
}

static void pad_events(ImGuiIO &io, int reset) {
    uint64_t b;
    int32_t lx, ly;
    int have = input_pad_read(&b, &lx, &ly);
    if (have) io.BackendFlags |= ImGuiBackendFlags_HasGamepad;
    else io.BackendFlags &= ~ImGuiBackendFlags_HasGamepad;
    stick_dir = stick_to_dpad(lx, ly);
    if (stick_dir >= 0) b |= 1ull << stick_dir;
    if (reset) {
        pad_prev = b;
        return;
    }
    for (const auto &k : pad_keys) {
        uint64_t m = 1ull << k.bit;
        if ((b ^ pad_prev) & m) io.AddKeyEvent(k.key, (b & m) != 0);
    }
    pad_prev = b;
}

// ---- ImGui and its Vulkan back end (present path)
static int g_ctx_ok;              // context made
static std::atomic<int> g_failed; // set-up failed for good (the menu stays closed)
static int g_failed_flag() { return g_failed.load(std::memory_order_relaxed); }
static int g_vk_ok;               // back end initialised for g_vk_fmt
static VkFormat g_vk_fmt;
static float g_font_scale;        // the scale of the uploaded font atlas; 0 when none is uploaded
static float g_style_scale;       // the scale the style was last sized for
static ImGuiStyle g_base_style;   // the menu's style at 1080 lines
static uint64_t g_last_ns;
static uint64_t g_cost_ns, g_cost_frames;   // CPU time of layout and recording while open, logged per 600 frames

// Sizes are tuned for 1080 lines and follow the window's height, or its width in a window narrower than 1.47:1, where
// the menu's 1400 pixels at 1080 lines would no longer fit 88 % of the width (layout_menu). Under 720 lines at 100%
// display scale they stay at the 720 size; that floor grows with the window's DPI, so a small window on a 150% or 200%
// display (sizes are physical pixels) still gets text as large as the display scale asks for.
static float menu_scale(uint32_t w, uint32_t h) {
    float least = 720.0f * (float)gfx_window_dpi() / 96.0f, lines = fminf((float)h, (float)w * 0.68f);
    return (lines >= least ? lines : least) / 1080.0f;
}

// The gold of the title and its Triforce (the rest of the menu's colours: ui/menu_skin.cpp skin_style).
static const ImVec4 kGold(0.93f, 0.76f, 0.38f, 1);

// The Triforce, its top at (x, y), h high, drawn with the title.
static void triforce(ImDrawList *dl, float x, float y, float h, ImU32 col) {
    float w = h * 1.1547f;   // equilateral: width = height * 2 / sqrt(3)
    ImVec2 t(x, y), l(x - w / 2, y + h), r(x + w / 2, y + h), ml(x - w / 4, y + h / 2), mr(x + w / 4, y + h / 2), mb(x, y + h);
    dl->AddTriangleFilled(t, mr, ml, col);
    dl->AddTriangleFilled(ml, mb, l, col);
    dl->AddTriangleFilled(mr, r, mb, col);
}

static void make_context(void) {
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.LogFilename = nullptr;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavEnableGamepad;
    io.BackendPlatformName = "nx2navi_menu";
    // no window switcher on Ctrl+Tab (the menu is one window; the switcher would list its inner windows by their ids)
    GImGui->ConfigNavWindowingKeyNext = GImGui->ConfigNavWindowingKeyPrev = 0;
    skin_style(g_base_style);
    ImGui::GetStyle() = g_base_style;
    g_ctx_ok = 1;
}

// Dear ImGui's Vulkan back end calls this with each VkResult it gets (it went unchecked before): failures are logged and
// counted, so a texture whose descriptor set was not allocated is not drawn with (ui/menu_art.cpp upload_rgba).
static std::atomic<int> g_vk_errors;
static void menu_vk_check(VkResult r) {
    if (r == VK_SUCCESS) return;
    if (g_vk_errors.fetch_add(1) < 8) rt_log("menu: Dear ImGui's Vulkan back end got VkResult %d", (int)r);
}
extern "C" int menu_vk_errors(void) { return g_vk_errors.load(); }

static void vk_down(void) {
    if (!g_vk_ok) return;
    menu_art_release();
    ImGui_ImplVulkan_Shutdown();
    g_vk_ok = 0;
    g_font_scale = 0;
}

static int vk_up(const MenuTarget *t) {
    if (g_vk_ok && g_vk_fmt == t->format) return 1;
    if (g_vk_ok) {   // the swapchain changed format: pipelines for the new one
        vk_queue_lock();
        vkQueueWaitIdle(t->queue);
        vk_queue_unlock();
        vk_down();
    }
    ImGui_ImplVulkan_InitInfo ii = {};
    ii.ApiVersion = VK_API_VERSION_1_3;
    ii.Instance = t->instance;
    ii.PhysicalDevice = t->phys;
    ii.Device = t->device;
    ii.QueueFamily = t->queue_family;
    ii.Queue = t->queue;
    ii.DescriptorPoolSize = 32;   // the art's sets, and two font atlases while one replaces the other
    ii.CheckVkResultFn = menu_vk_check;
    ii.MinImageCount = 2;
    ii.ImageCount = 4;   // vertex buffers rotate per recorded frame: more than gfx_present.c's 2 frames in flight
    ii.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    ii.UseDynamicRendering = true;
    g_vk_fmt = t->format;
    ii.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    ii.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    ii.PipelineRenderingCreateInfo.pColorAttachmentFormats = &g_vk_fmt;
    if (!ImGui_ImplVulkan_Init(&ii)) return 0;
    g_vk_ok = 1;
    return 1;
}

// Roboto Medium (Latin, Cyrillic, dashes and quotes) at 26 px per 1080 lines, with the glyphs of the Japanese, Korean
// and Chinese language names merged in from a system font when there is one (each language is named in its own script).
// A catalog row also holds menu text (Simplified Chinese), so every string of the row goes into the builder: the merged
// font must cover the whole menu of that language, not only its name. Made again when the window height changes; the old
// texture goes after the queue is idle (ui/menu_art.cpp menu_font_upload).
static const ImWchar kRanges[] = {0x0020, 0x00FF, 0x0400, 0x052F, 0x2010, 0x2027, 0};
static const struct { const char *lang; UiTextId name; } kCjk[] = {
    {"ja", UI_MENU_LANGNAME_JA}, {"ko", UI_MENU_LANGNAME_KO}, {"zh-Hans", UI_MENU_LANGNAME_ZH_HANS},
    {"zh-Hant", UI_MENU_LANGNAME_ZH_HANT},
};

static uint32_t be32(const uint8_t *p) { return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3]; }

// 1 when Dear ImGui can take this font file: more than the 100 bytes AddFontFromMemoryTTF asserts on, every table of its
// directory inside the file (stb_truetype reads tables without bounds checks, so a cut-off file faults when the glyphs
// are made), and stb_truetype parses it. ImGui asserts on a font it cannot parse, and an assert ends the process with
// 0xc0000409 and no line in the log.
static int font_usable(const uint8_t *d, uint64_t n) {
    if (!d || n <= 100 || n > INT_MAX) return 0;
    int off = stbtt_GetFontOffsetForIndex(d, 0);
    if (off < 0 || (uint64_t)off + 12 > n) return 0;
    uint32_t tables = (uint32_t)d[off + 4] << 8 | d[off + 5];
    if ((uint64_t)off + 12 + 16ull * tables > n) return 0;
    for (uint32_t i = 0; i < tables; i++) {
        const uint8_t *r = d + off + 12 + 16 * i;
        if ((uint64_t)be32(r + 8) + be32(r + 12) > n) return 0;
    }
    stbtt_fontinfo fi;
    g_font_bad = 0;
    return stbtt_InitFont(&fi, d, off) && !g_font_bad;
}

extern "C" int menu_font_check(void) {
    uint64_t n = sizeof roboto_ttf;
    uint8_t *bad = (uint8_t *)malloc(n);
    if (!bad) return 0;
    memcpy(bad, roboto_ttf, n);
    uint32_t tables = (uint32_t)bad[4] << 8 | bad[5];
    for (uint32_t i = 0; i < tables; i++)
        if (!memcmp(bad + 12 + 16 * i, "cmap", 4)) memset(bad + 12 + 16 * i, 0, 4);   // no character map
    int ok = font_usable(roboto_ttf, n) && !font_usable(roboto_ttf, 4096) && !font_usable(roboto_ttf, 100) &&
             !font_usable(bad, n);
    memset(bad, 0, 4);             // an unknown font tag
    ok = ok && !font_usable(bad, n);
    free(bad);
    return ok;
}

// The glyphs of every string of a catalog row (ui_text_visit), for the font that draws that language.
static void add_row_glyphs(void *ctx, const char *text) {
    static_cast<ImFontGlyphRangesBuilder *>(ctx)->AddText(text);
}

// The game's own interface fonts from the user's RomFS (menu_assets.c), read once: ClarimoUDPE for the menu's text,
// CetraDisplayPro for its title and tabs. Roboto stays merged behind them for the glyphs they lack; without them the
// menu keeps Roboto alone.
static uint8_t *g_game_font, *g_title_font_data;
static size_t g_game_font_n, g_title_font_n;
static ImFont *g_title_font;

static void game_fonts_read(void) {
    static int done;
    if (done) return;
    done = 1;
    g_game_font = menu_asset_font("Font2/Scalable/ClarimoUDPE-Regular.Nin_Ounce_NVN2.bfotf.zs", &g_game_font_n);
    g_title_font_data = menu_asset_font("Font2/Scalable/CetraDisplayPro-Medium.Nin_Ounce_NVN2.bfotf.zs", &g_title_font_n);
    if (g_game_font && !font_usable(g_game_font, g_game_font_n)) free(g_game_font), g_game_font = NULL;
    if (g_title_font_data && !font_usable(g_title_font_data, g_title_font_n)) free(g_title_font_data), g_title_font_data = NULL;
    rt_log("menu: the game's interface fonts: text %s, titles %s", g_game_font ? "ClarimoUDPE" : "not found (Roboto)",
           g_title_font_data ? "CetraDisplayPro" : "not found");
}

static int fonts_up(float scale) {
    enum { N = sizeof kCjk / sizeof *kCjk };
    ImFontAtlas *fa = ImGui::GetIO().Fonts;
    fa->Clear();
    game_fonts_read();
    g_title_font = nullptr;
    ImFontConfig fc;
    fc.FontDataOwnedByAtlas = false;
    if (g_game_font) {
        fa->AddFontFromMemoryTTF(g_game_font, (int)g_game_font_n, 25.5f * scale, &fc, kRanges);
        ImFontConfig rc;
        rc.MergeMode = true;
        rc.FontDataOwnedByAtlas = false;
        fa->AddFontFromMemoryTTF((void *)roboto_ttf, (int)sizeof roboto_ttf, 26.0f * scale, &rc, kRanges);
    } else {
        fa->AddFontFromMemoryTTF((void *)roboto_ttf, (int)sizeof roboto_ttf, 26.0f * scale, &fc, kRanges);
    }
    uint8_t *data[N] = {};
    uint64_t sizes[N] = {};
    ImVector<ImWchar> ranges[N], title_ranges[N];
    int merged = 0;
    for (int i = 0; i < N; i++) {
        char path[512];
        uint64_t &size = sizes[i];
        if (!plat_font_path(kCjk[i].lang, path, sizeof path) || !(data[i] = plat_file_read(path, &size))) continue;
        if (!font_usable(data[i], size)) {
            rt_log("menu: system font %s left out: the font parser cannot read it (%llu bytes)", path,
                   (unsigned long long)size);
            free(data[i]);
            data[i] = NULL;
            continue;
        }
        merged++;
        ImFontGlyphRangesBuilder b;
        b.AddText(ui_text_in("en", kCjk[i].name));
        ui_text_visit(kCjk[i].lang, add_row_glyphs, &b);
        // Text the menu shows that no catalog row holds: a mod's cheat names and its folder name,
        // both read from disk at runtime. Their glyphs would otherwise fall back to the missing-glyph box.
        for (int k = 0; k < cheats_count(); k++)
            if (const CheatInfo *ci = cheats_info(k); ci && ci->name) b.AddText(ci->name);
        for (int k = 0; k < mods_count(); k++)
            if (const ModInfo *mi = mods_at(k); mi) b.AddText(mi->name);
        b.BuildRanges(&ranges[i]);
        ImFontConfig mc;
        mc.MergeMode = true;
        mc.FontDataOwnedByAtlas = false;
        fa->AddFontFromMemoryTTF(data[i], (int)size, 26.0f * scale, &mc, ranges[i].Data);
    }
    if (g_title_font_data) {   // after the text font's merges, which go to the font added last
        ImFontConfig tc;
        tc.FontDataOwnedByAtlas = false;
        g_title_font = fa->AddFontFromMemoryTTF(g_title_font_data, (int)g_title_font_n, 31.0f * scale, &tc, kRanges);
        ImFontConfig rc;
        rc.MergeMode = true;
        rc.FontDataOwnedByAtlas = false;
        fa->AddFontFromMemoryTTF((void *)roboto_ttf, (int)sizeof roboto_ttf, 30.0f * scale, &rc, kRanges);
        // what the title face shows in the CJK languages: the menu's title and tabs (Simplified Chinese menu text) and
        // the boot screen's title (the game's language)
        static const UiTextId titles[] = {UI_MENU_TITLE, UI_MENU_TAB_SCREEN, UI_MENU_TAB_GAME, UI_MENU_TAB_SOUND,
                                          UI_MENU_TAB_CONTROLS, UI_MENU_TAB_COOP, UI_BOOT_TITLE};
        for (int i = 0; i < N; i++) {
            if (!data[i]) continue;
            ImFontGlyphRangesBuilder b;
            for (UiTextId id : titles) b.AddText(ui_text_in(kCjk[i].lang, id));
            b.BuildRanges(&title_ranges[i]);
            ImFontConfig mc;
            mc.MergeMode = true;
            mc.FontDataOwnedByAtlas = false;
            fa->AddFontFromMemoryTTF(data[i], (int)sizes[i], 31.0f * scale, &mc, title_ranges[i].Data);
        }
    }
    skin_atlas_add(fa);    // the boxes' grain tile (ui/menu_skin.cpp), drawn into the atlas once it is built
    skin_atlas_fill(fa);   // builds the atlas, which the upload below then takes as it is
    unsigned char *px = nullptr;
    int aw = 0, ah = 0;
    fa->GetTexDataAsRGBA32(&px, &aw, &ah);
    ImTextureID tex = 0;
    int ok = menu_font_upload(px, aw, ah, &tex);
    if (ok) fa->SetTexID(tex);
    // ClarimoUDPE's line (ascent to descent, which is all Dear ImGui spaces lines by) leaves wrapped lines touching: the
    // line height grows by a sixth over the glyphs' size, the glyphs stay as rasterized.
    if (ok && g_game_font && fa->Fonts.Size) fa->Fonts[0]->FontSize *= 1.16f;
    fa->ClearInputData();   // the glyphs are in the texture: the font files and the pixels are not needed again
    fa->ClearTexData();
    for (uint8_t *d : data) free(d);
    rt_log("menu: font atlas at %.1f px, language names from %d of %d system fonts", 26.0f * scale, merged, (int)N);
    g_font_scale = ok ? scale : 0;
    return ok;
}

// The atlas for scale. While the scale keeps changing (the window being resized), the atlas is made again only once it
// has held for a quarter of a second, and the old one is drawn scaled meanwhile (io.FontGlobalScale): each atlas reads
// the system fonts from disk and waits for the queue to upload. 0 when no atlas could be made.
static float g_scale_seen;
static uint64_t g_scale_seen_ns;

static int fonts_for(float scale, uint64_t now) {
    ImGuiIO &io = ImGui::GetIO();
    if (g_font_scale == scale) {
        io.FontGlobalScale = 1;
        return 1;
    }
    if (scale != g_scale_seen) g_scale_seen = scale, g_scale_seen_ns = now;
    if (g_font_scale > 0 && now - g_scale_seen_ns < 250000000ull) {
        io.FontGlobalScale = scale / g_font_scale;
        return 1;
    }
    io.FontGlobalScale = 1;
    return fonts_up(scale);
}

extern "C" void menu_vk_release(void) {
    if (g_ctx_ok) vk_down();
}

// ---- test hooks
// NX2NAVI_MENU_TEST=<frame>[,<frame>...][:<frames per page>]: at each such presented frame (up to 8) the menu opens on
// its first tab; each tab stays <frames per page> frames (default 30) and its last frame's swapchain image goes to
// NX2NAVI_MENU_TEST_DIR (default: the working directory) as menu-test-<frame>-<page>.png; after the last tab the menu
// closes.
// NX2NAVI_MENU_TEST_CLICK=<key>:<part>[,...]: mouse clicks on the rows of the page shown (ui/option_rows.cpp click_test).
// NX2NAVI_MENU_TEST_SET=<frame>:<key>=<value>[,<key>=<value>...][;<frame>:...]: cfg_set of each key (NX2NAVI_ prefix
// optional) at that presented frame; changes live settings with no input, menu open or not. nx2navi.ini is written only by
// the pseudo-key SAVE=1 (cfg_save, as a row change does; a failed write shows the menu's save note). FOCUS=<key> (the
// NX2NAVI_ prefix optional) puts the cursor on that setting's row and scrolls it into view when the row is next drawn
// (opt_focus_key), so a page's shot shows a row below the first screenful; it does not switch tabs. COOP_PAGE, COOP_JOIN
// and COOP_ACCEPT_OFFER drive the Co-op tab (ui/coop_page.h coop_test_set).
// Ocarina of Time: the options this port has and co-op (ui/coop_page.cpp)
enum { TAB_SCREEN, TAB_GAME, TAB_SOUND, TAB_CONTROLS, TAB_COOP, TAB_COUNT };
enum { TEST_PAGE_FRAMES = 30, TEST_SETS = 16, TEST_OPENS = 8 };
struct TestSet { uint64_t frame; char key[64], value[64]; };
static struct {
    int parsed;
    uint64_t at[TEST_OPENS];  // NX2NAVI_MENU_TEST frames, in order
    int n_at, next_at;        // how many; the next to open at
    uint64_t per_page;
    uint64_t opened;      // the frame the hook opened the menu; 0 not yet
    int page;             // the tab being shot; -1 none
    TestSet sets[TEST_SETS];
    int n_sets, next_set; // NX2NAVI_MENU_TEST_SET, in frame order
} g_test = {0, {}, 0, 0, 0, 0, -1, {}, 0, 0};

static void test_parse(void) {
    g_test.parsed = 1;
    if (const char *e = cfg_get("NX2NAVI_MENU_TEST")) {
        long long per = 0;
        if (const char *c = strchr(e, ':')) per = atoll(c + 1);
        for (const char *s = e; *s && *s != ':' && g_test.n_at < TEST_OPENS;) {
            char *end;
            unsigned long long at = strtoull(s, &end, 10);
            if (end == s) break;
            if (at > 0 && (!g_test.n_at || at > g_test.at[g_test.n_at - 1])) g_test.at[g_test.n_at++] = at;
            s = *end == ',' ? end + 1 : end;
        }
        g_test.per_page = per > 0 ? (uint64_t)per : TEST_PAGE_FRAMES;
    }
    const char *s = cfg_get("NX2NAVI_MENU_TEST_SET");
    while (s && *s) {   // groups "<frame>:<key>=<value>,..." separated by ';'
        char *end;
        unsigned long long f = strtoull(s, &end, 10);
        if (*end != ':') break;
        s = end + 1;
        while (*s && *s != ';' && g_test.n_sets < TEST_SETS) {
            size_t n = strcspn(s, ",;");
            const char *eq = (const char *)memchr(s, '=', n);
            if (eq) {
                TestSet &t = g_test.sets[g_test.n_sets++];
                t.frame = f;
                int kn = (int)(eq - s), vn = (int)(s + n - eq - 1);
                snprintf(t.key, sizeof t.key, "%s%.*s", strncmp(s, "NX2NAVI_", strlen("NX2NAVI_")) ? "NX2NAVI_" : "", kn, s);
                snprintf(t.value, sizeof t.value, "%.*s", vn, eq + 1);
            }
            s += n;
            if (*s == ',') s++;
        }
        while (*s && *s != ';') s++;
        if (*s == ';') s++;
    }
    if (g_test.n_at || g_test.n_sets)
        rt_log("menu: test hook: open at frame %llu (%d times), %llu frames per page, %d settings changes",
               (unsigned long long)(g_test.n_at ? g_test.at[0] : 0), g_test.n_at, (unsigned long long)g_test.per_page,
               g_test.n_sets);
}

static void test_apply_sets(uint64_t frame) {
    for (; g_test.next_set < g_test.n_sets && g_test.sets[g_test.next_set].frame <= frame; g_test.next_set++) {
        const TestSet &t = g_test.sets[g_test.next_set];
        // a room's secret stays out of the log here too
        const bool secret = !strcmp(t.key, "NX2NAVI_COOP_JOIN") || !strcmp(t.key, "NX2NAVI_COOP_ROOM");
        rt_log("menu: test hook: frame %llu: %s=%s", (unsigned long long)frame, t.key, secret ? "(set, not shown)" : t.value);
        if (!strcmp(t.key, "NX2NAVI_SAVE")) {
            rt_log("menu: test hook: cfg_save %s", cfg_save() ? "wrote" : "failed");
        } else if (!strcmp(t.key, "NX2NAVI_FOCUS")) {
            char row[80];
            snprintf(row, sizeof row, "%s%s", strncmp(t.value, "NX2NAVI_", strlen("NX2NAVI_")) ? "NX2NAVI_" : "", t.value);
            opt_focus_key(row);
        } else if (!strncmp(t.key, "NX2NAVI_COOP_", 13) && !cfg_from_env("NX2NAVI_MENU_TEST_SET")) {
            // From nx2navi.ini the hook sets no co-op key and drives no part of the Co-op tab: a settings file
            // someone hands a player could otherwise turn co-op on, join a room and accept its offer for them
            // (COOP_JOIN, COOP_ACCEPT_OFFER). A test sets it in the environment.
            rt_log("menu: test hook: %s is taken from the environment only, not from nx2navi.ini", t.key);
        } else if (!coop_test_set(t.key, t.value)) {
            cfg_set(t.key, t.value);
        }
    }
}

// ---- the lists the Screen and Game tabs build
enum { MAX_GPUS = 8 };
static char g_gpu_names[MAX_GPUS][VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
static int g_gpu_n = -1;   // -1: not listed yet

// The Vulkan devices in vkEnumeratePhysicalDevices order, which NX2NAVI_GPU_INDEX counts in (vk_device.c pick_device).
static void list_gpus(VkInstance inst) {
    if (g_gpu_n >= 0) return;
    VkPhysicalDevice pd[MAX_GPUS];
    uint32_t n = MAX_GPUS;
    if (vkEnumeratePhysicalDevices(inst, &n, pd) < 0) n = 0;   // VK_INCOMPLETE keeps the first MAX_GPUS
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(pd[i], &p);
        snprintf(g_gpu_names[i], sizeof g_gpu_names[i], "%s", p.deviceName);
    }
    g_gpu_n = (int)n;
}

// Window sizes: the common 16:9 sizes that fit the monitor, the 1920x1080 default and the current size.
static const int kSizes[][2] = {{1280, 720}, {1366, 768}, {1600, 900}, {1920, 1080}, {2560, 1440}, {3200, 1800},
                                {3840, 2160}};
enum { MAX_SIZES = sizeof kSizes / sizeof *kSizes + 1 };

static void row_window_size(void) {
    int sz[MAX_SIZES][2], n = 0, mw = 0, mh = 0, cw = 0, ch = 0;
    int have = gfx_window_monitor_size(&mw, &mh);
    int cur_ok = sscanf(cfg_value("NX2NAVI_WINDOW_SIZE"), "%dx%d", &cw, &ch) == 2;
    bool listed = false;
    for (size_t i = 0; i < sizeof kSizes / sizeof *kSizes; i++) {
        int w = kSizes[i][0], h = kSizes[i][1];
        bool cur = cur_ok && w == cw && h == ch;
        if (have && (w > mw || h > mh) && !cur && !(w == 1920 && h == 1080)) continue;
        listed |= cur;
        sz[n][0] = w, sz[n][1] = h, n++;
    }
    if (cur_ok && !listed) {   // the current size where its area puts it
        int at = n;
        while (at > 0 && (long)sz[at - 1][0] * sz[at - 1][1] > (long)cw * ch) at--;
        memmove(sz[at + 1], sz[at], sizeof sz[0] * (n - at));
        sz[at][0] = cw, sz[at][1] = ch, n++;
    }
    char vals[MAX_SIZES][24], shown[MAX_SIZES][32];
    const char *pv[MAX_SIZES], *pn[MAX_SIZES];
    for (int i = 0; i < n; i++) {
        snprintf(vals[i], sizeof vals[i], "%dx%d", sz[i][0], sz[i][1]);
        snprintf(shown[i], sizeof shown[i], "%d × %d", sz[i][0], sz[i][1]);   // U+00D7, the multiplication sign
        pv[i] = vals[i];
        pn[i] = shown[i];
    }
    opt_choice_list("NX2NAVI_WINDOW_SIZE", UI_MENU_WINDOW_SIZE, pv, pn, n, UI_MENU_WINDOW_SIZE_DESC);
}

// The game resolution (resolution.c): 16:9 sizes from its smallest, 1280x720 (for weak GPUs), up, the common 21:9 and
// 32:9 monitor sizes, the 16:10 screens of the Steam Deck, handhelds and laptops and the 3:2 screens of laptops and
// tablets (Surface, Framework; the camera and the HUD follow
// another shape, resolution.c res_init; the installer lays the HUD out for each, installer/steps.py ASPECT_SIZES), and
// the current value when it is another size.
static void row_output_res(void) {
    static const char *const vals[] = {"1280x720", "1600x900", "1920x1080", "2560x1440", "3200x1800", "3840x2160",
                                       "2560x1080", "3440x1440", "3840x1600", "5120x2160", "3840x1080", "5120x1440",
                                       "1280x800", "1680x1050", "1920x1200", "2560x1600",
                                       "1920x1280", "2160x1440", "2256x1504", "2880x1920", "3000x2000"};
    enum { N = sizeof vals / sizeof *vals };
    char shown[N][32];
    const char *names[N];
    for (int i = 0; i < N; i++) {
        int w = 0, h = 0;
        sscanf(vals[i], "%dx%d", &w, &h);
        snprintf(shown[i], sizeof shown[i], "%d × %d", w, h);   // U+00D7, the multiplication sign
        names[i] = shown[i];
    }
    opt_choice_list("NX2NAVI_OUTPUT_RES", UI_MENU_OUTPUT_RES, vals, names, N, UI_MENU_OUTPUT_RES_DESC);
}

static void row_render_scale(void) {
    static const char *const vals[] = {"game", "50", "67", "75", "100", "125", "150", "200"};
    enum { N = sizeof vals / sizeof *vals };
    char shown[N][32];
    const char *names[N];
    names[0] = menu_text(UI_MENU_RENDER_SCALE_GAME);
    for (int i = 1; i < N; i++) {
        snprintf(shown[i], sizeof shown[i], "%s %%", vals[i]);
        names[i] = shown[i];
    }
    opt_choice_list("NX2NAVI_RENDER_SCALE", UI_MENU_RENDER_SCALE, vals, names, N, UI_MENU_RENDER_SCALE_DESC);
}

static void row_shadow_res(void) {
    static const char *const vals[] = {"game", "256", "512", "768", "1024", "1536", "2048"};
    enum { N = sizeof vals / sizeof *vals };
    const char *names[N];
    names[0] = menu_text(UI_MENU_SHADOW_RES_GAME);
    for (int i = 1; i < N; i++) names[i] = vals[i];
    opt_choice_list("NX2NAVI_SHADOW_RES", UI_MENU_SHADOW_RES, vals, names, N, UI_MENU_SHADOW_RES_DESC);
}

// Texture detail (vk_texture.c "texture detail"): large surface textures made without their largest one or two mip
// levels. Read when the first texture is made, so the row notes the restart. The automatic entry names the level it
// picks on this card.
static void row_texture_detail(void) {
    static const char *const vals[] = {"auto", "full", "reduced", "low"};
    static const UiTextId level[] = {UI_MENU_TEX_DETAIL_FULL, UI_MENU_TEX_DETAIL_REDUCED, UI_MENU_TEX_DETAIL_LOW};
    static char picked[96];   // the row keeps the pointer for the frame
    int a = vk_texture_detail_auto();
    snprintf(picked, sizeof picked, menu_text(UI_MENU_TEX_DETAIL_AUTO), menu_text(level[a < 0 || a > 2 ? 0 : a]));
    const char *names[] = {picked, menu_text(level[0]), menu_text(level[1]), menu_text(level[2])};
    opt_choice_list("NX2NAVI_TEXTURE_DETAIL", UI_MENU_TEX_DETAIL, vals, names, 4, UI_MENU_TEX_DETAIL_DESC);
}

static void row_shadow_scale(void) {
    static const char *const vals[] = {"1", "2", "3", "4"};
    static const char *const names[] = {"960 (1x)", "1920 (2x)", "2880 (3x)", "3840 (4x)"};
    opt_choice_list("NX2NAVI_SHADOW_SCALE", UI_MENU_SHADOW_SCALE, vals, names, 4, UI_MENU_SHADOW_SCALE_DESC);
}

static void row_shadow_range(void) {
    static const char *const vals[] = {"100", "150", "200", "300"};
    static const char *const names[] = {"160 m (1x)", "240 m (1.5x)", "320 m (2x)", "480 m (3x)"};
    opt_choice_list("NX2NAVI_SHADOW_RANGE", UI_MENU_SHADOW_RANGE, vals, names, 4, UI_MENU_SHADOW_RANGE_DESC);
}

static void row_gpu(void) {
    const char *vals[MAX_GPUS + 1], *names[MAX_GPUS + 1];
    char idx[MAX_GPUS][4];
    vals[0] = "";
    names[0] = menu_text(UI_MENU_GPU_AUTO);
    for (int i = 0; i < g_gpu_n; i++) {
        snprintf(idx[i], sizeof idx[i], "%d", i);
        vals[i + 1] = idx[i];
        names[i + 1] = g_gpu_names[i];
    }
    opt_choice_list("NX2NAVI_GPU_INDEX", UI_MENU_GPU, vals, names, g_gpu_n + 1, UI_MENU_GPU_DESC);
}

// The regions of the Mals/<region>.Product.<n>.sarc.zs files in the RomFS index ("USen", "EUpl"), listed once; 0 when
// the index has none (no RomFS yet).
static int mals_regions(const char *out[24]) {
    static char reg[24][5];
    static int n = -1;   // -1: not scanned (the index may load after the menu first draws)
    if (n < 0 && romfs_file_count()) {
        n = 0;
        for (uint64_t i = 0, c = romfs_file_count(); i < c && n < 24; i++) {
            uint64_t off, size;
            const char *p = romfs_entry(i, &off, &size);
            if (!p || strncmp(p, "Mals/", 5) || strlen(p) < 10 || p[9] != '.') continue;
            memcpy(reg[n], p + 5, 4);
            reg[n][4] = 0;
            bool seen = false;
            for (int j = 0; j < n; j++) seen |= !strcmp(reg[j], reg[n]);
            if (!seen) n++;
        }
    }
    for (int i = 0; i < n; i++) out[i] = reg[i];
    return n > 0 ? n : 0;
}

// The game language: one entry per Mals language file the game ships (ui_text_game_langs_in from the RomFS index, else
// ui_text_game_langs by version), each under its own name. A code that names no entry exactly (older "en", "fr-CA" on
// a build without it) selects the first entry of the same catalog language.
static void row_game_lang(void) {
    UiGameLang langs[16];
    const char *regions[24];
    int nr = mals_regions(regions);
    int n = nr ? ui_text_game_langs_in(regions, nr, langs, 16) : ui_text_game_langs(rt_game_version, langs, 16);
    const char *vals[17], *names[17];
    vals[0] = "";
    names[0] = menu_text(UI_MENU_GAME_LANG_AUTO);
    const char *now = cfg_value("NX2NAVI_LANG");
    int cur = *now ? -1 : 0;
    for (int i = 1; i <= n; i++) {
        vals[i] = langs[i - 1].code;
        names[i] = menu_text(langs[i - 1].name);
        if (*now && !strcmp(now, vals[i])) cur = i;
    }
    for (int i = 1; i <= n && *now && cur < 0; i++)
        if (!strcmp(ui_text_lang(now), ui_text_lang(vals[i]))) cur = i;
    opt_choice_list("NX2NAVI_LANG", UI_MENU_GAME_LANG, vals, names, n + 1, UI_MENU_GAME_LANG_DESC, cur);
}

// The frame limit (frame_rate.c): unset, the older fps60 = 0 still means 30. The monitor choice shows the rate it gives.
// The note says when RivaTuner Statistics Server does the limiting (rtss_limit.c), cannot change its profile, or let the
// game run past its limit.
static void row_fps_limit() {
    // 40 and 45: the Steam Deck's screen runs at 40 to 60 Hz (the OLED to 90); 40 divides 40 and 120 Hz evenly
    // 165 and 240: common gaming monitors; unlimited only where the game steps by the clock (frame_rate.c)
    static const char *const vals[] = {"30", "40", "45", "60", "90", "120", "144", "165", "240", "display", "unlimited"};
    static const UiTextId ids[] = {UI_MENU_FPS_30,  UI_MENU_FPS_40,  UI_MENU_FPS_45,  UI_MENU_FPS_60,
                                   UI_MENU_FPS_90,  UI_MENU_FPS_120, UI_MENU_FPS_144, UI_MENU_FPS_144,
                                   UI_MENU_FPS_144, UI_MENU_FPS_DISPLAY, UI_MENU_FPS_UNLIMITED};
    int n = frame_free_running() ? 11 : 10;
    const char *names[11];
    for (int i = 0; i < n; i++) names[i] = menu_text(ids[i]);
    names[7] = "165 FPS", names[8] = "240 FPS";
    static char display[64];
    double hz = gfx_window_refresh_hz();
    if (hz > 0) {
        snprintf(display, sizeof display, menu_text(UI_MENU_FPS_DISPLAY_HZ), hz);
        names[9] = display;
    }
    const char *set = cfg_get("NX2NAVI_FPS_LIMIT"), *old = cfg_get("NX2NAVI_FPS60");
    int cur = (!set || !*set) && old && !strcmp(old, "0") ? 0 : -1;
    int rtss = rtss_limit_state();
    const char *note = rtss == 1   ? menu_text(UI_MENU_FPS_RTSS)
                       : rtss == 2 ? menu_text(UI_MENU_FPS_RTSS_READONLY)
                       : rtss == 3 ? menu_text(UI_MENU_FPS_RTSS_IGNORED)
                                   : nullptr;
    opt_choice_list("NX2NAVI_FPS_LIMIT", UI_MENU_FPS_LIMIT, vals, names, n, UI_MENU_FPS_LIMIT_DESC, cur, note);
}

// "in use: ..." under the VSync row when the swapchain does other than the row says: VSync when it is off and the driver
// cannot present without waiting, Immediate when it is on and the driver has neither mailbox nor relaxed FIFO
// (gfx_present.c pick_present_mode), so the game tears; NULL otherwise, and before the first swapchain.
static const char *vsync_note(void) {
    const char *got = gfx_window_present_mode();
    if (!*got) return nullptr;
    bool off = !strcmp(cfg_value("NX2NAVI_VK_PRESENT"), "immediate"), tears = !strcmp(got, "immediate");
    if (off == tears) return nullptr;
    static char note[96];
    snprintf(note, sizeof note, menu_text(UI_MENU_PRESENT_IN_USE),
             menu_text(tears ? UI_MENU_PRESENT_IMMEDIATE : UI_MENU_PRESENT_VSYNC));
    return note;
}

// ---- the tabs

// The PCI vendor of the GPU the game renders on (menu_overlay_frame reads it from the device it is given; 0 before the
// first frame). DLSS runs on NVIDIA GPUs only, so the rows below leave it out elsewhere (AMD, Intel).
static std::atomic<uint32_t> g_gpu_vendor;
static VkPhysicalDevice g_gpu_vendor_of;
static bool dlss_offered(void) {
    uint32_t v = g_gpu_vendor.load(std::memory_order_relaxed);
    return v == 0 || v == 0x10de;   // unknown (no frame yet): as before, offered
}

// One upscaling choice over three settings: NX2NAVI_UPSCALER (how the picture is fitted to the window), NX2NAVI_DLAA
// (DLSS or FSR 3.1 on) with NX2NAVI_TAA (which of them) and NX2NAVI_DLSS (the mode, the render scale: widescreen.c
// wide_present). A temporal mode fits the picture with bilinear, the upscaler having made it; DLSS's model is NVIDIA's own
// for the mode (vk_dlaa.c evaluate, dlss_preset at its default). The list: bilinear, FSR 1, the DLSS modes where DLSS
// runs (dlss_offered), the FSR 3.1 modes where FSR 3.1 loads (vk_ffx_available); a DLSS or FSR 3.1 mode still set where
// it does not run stays listed last until another choice replaces it, so the row shows what is set.
// FSR 4's name in the Upscaling row and the overlay: the version the driver's library runs to its minor number, as
// AMD names its releases ("AMD FSR 4.1" for 4.1.2); "AMD FSR 4" until the library was asked (vk_ffx4_available).
static const char *fsr4_label(void) {
    static char s[32];
    const char *v = strstr(vk_ffx4_name(), "FSR ");
    int major = 0, minor = 0;
    if (v && sscanf(v + 4, "%d.%d", &major, &minor) == 2 && major >= 4) snprintf(s, sizeof s, "AMD FSR %d.%d", major, minor);
    else snprintf(s, sizeof s, "AMD FSR 4");
    return s;
}

static bool taa_is_fsr(void) {
    const char *t = cfg_value("NX2NAVI_TAA");
    return !strcmp(t, "fsr3") || !strcmp(t, "fsr4");
}
static bool taa_is_fsr4(void) { return !strcmp(cfg_value("NX2NAVI_TAA"), "fsr4"); }
static bool ffx_taa_available(void) { return taa_is_fsr4() ? vk_ffx_fsr4_available() : vk_ffx_available(); }

// The Upscaling row's name of what is set now, for the performance overlay.
static void upscaling_name(char *out, size_t size) {
    static const char *const modes[] = {"dlaa", "quality", "balanced", "performance", "ultra_performance"};
    static const UiTextId dlss[] = {UI_MENU_DLSS_DLAA, UI_MENU_DLSS_QUALITY, UI_MENU_DLSS_BALANCED, UI_MENU_DLSS_PERFORMANCE, UI_MENU_DLSS_ULTRA};
    static const UiTextId fsr3[] = {UI_MENU_FSR3_NATIVE, UI_MENU_DLSS_QUALITY, UI_MENU_DLSS_BALANCED, UI_MENU_DLSS_PERFORMANCE, UI_MENU_DLSS_ULTRA};
    // FSR 3.1 / FSR 4 set where it cannot load runs nothing (vk_dlaa.c vk_dlaa_fsr3): the game scales the picture as the rest says
    if (!cfg_bool("NX2NAVI_DLAA") || (taa_is_fsr() && !ffx_taa_available())) {
        if (strcmp(cfg_value("NX2NAVI_UPSCALER"), "fsr")) snprintf(out, size, "%s", menu_text(UI_MENU_UPSCALER_BILINEAR));
        else snprintf(out, size, menu_text(UI_MENU_OV_FMT_FSR), cfg_int("NX2NAVI_SHARPNESS"));
        return;
    }
    int mode = 0;
    for (int i = 0; i < 5; i++)
        if (!strcmp(cfg_value("NX2NAVI_DLSS"), modes[i])) mode = i;
    if (taa_is_fsr())
        snprintf(out, size, "%s %s", (taa_is_fsr4() || vk_ffx_fsr4()) ? fsr4_label() : "AMD FSR 3.1", menu_text(fsr3[mode]));
    else snprintf(out, size, mode ? "DLSS %s" : "NVIDIA %s", menu_text(dlss[mode]));
}

// The Anti-aliasing row's value for the performance overlay: what row_aa shows, with "TAA + " before it while DLSS's or
// FSR's TAA runs with it (as upscaling_name tells it: FSR 3.1 / FSR 4 set where it cannot load runs nothing), and TAA alone
// for off then.
static void aa_name(char *out, size_t size) {
    static const char *const values[] = {"game", "off", "fxaa", "smaa"};
    static const UiTextId names[] = {UI_MENU_AA_GAME, UI_MENU_FRAMEGEN_OFF, UI_MENU_AA_FXAA, UI_MENU_AA_SMAA};
    int i = 0;
    for (int k = 0; k < 4; k++)
        if (!strcmp(cfg_value("NX2NAVI_AA"), values[k])) i = k;   // taa: the game's SMAA, with the upscaler or alone
    const bool taa = cfg_bool("NX2NAVI_DLAA") && (!taa_is_fsr() || ffx_taa_available());
    if (taa && i == 1) snprintf(out, size, "TAA");
    else if (taa) snprintf(out, size, "TAA + %s", menu_text(names[i]));
    else snprintf(out, size, "%s", menu_text(names[i]));
}

// The Upscaling row's NVIDIA DLSS waits for a restart (vk_dlaa_needs_restart), where DLSS is offered: the row's note and
// the Restart button's count (restart_pending).
static bool upscaling_restart(void) { return dlss_offered() && vk_dlaa_needs_restart(); }

static void row_upscaling(void) {
    static const char *const modes[] = {"dlaa", "quality", "balanced", "performance", "ultra_performance"};
    static const UiTextId mode_names[] = {UI_MENU_DLSS_DLAA, UI_MENU_DLSS_QUALITY, UI_MENU_DLSS_BALANCED,
                                          UI_MENU_DLSS_PERFORMANCE, UI_MENU_DLSS_ULTRA};
    // FSR 3.1's modes take the same render scales as DLSS's (Native AA is FSR's name for DLAA's full size)
    static const UiTextId fsr_names[] = {UI_MENU_FSR3_NATIVE, UI_MENU_DLSS_QUALITY, UI_MENU_DLSS_BALANCED,
                                         UI_MENU_DLSS_PERFORMANCE, UI_MENU_DLSS_ULTRA};
    // FSR 4 (vk_ffx4.cpp) is the same setting as FSR 3.1 with NX2NAVI_FSR4 on, and takes the same modes: listed where
    // it runs on this device, after FSR 3.1's
    enum { BILINEAR, FSR1, DLSS, FSR3, FSR4 };
    struct Entry { int kind, mode; };
    static char text[15][96];
    Entry list[18];
    const char *names[18];
    int n = 0;
    auto add = [&](int kind, int mode) {
        list[n] = {kind, mode};
        if (kind == BILINEAR) names[n] = menu_text(UI_MENU_UPSCALER_BILINEAR);
        else if (kind == FSR1) names[n] = menu_text(UI_MENU_UPSCALER_FSR);
        else {
            char *t = text[(kind == FSR4 ? 10 : kind == FSR3 ? 5 : 0) + mode];
            if (kind == DLSS) snprintf(t, sizeof text[0], mode ? "DLSS %s" : "NVIDIA %s", menu_text(mode_names[mode]));
            else snprintf(t, sizeof text[0], "%s %s", kind == FSR4 ? fsr4_label() : "AMD FSR 3.1", menu_text(fsr_names[mode]));
            names[n] = t;
        }
        n++;
    };
    // what is set now
    Entry now = {strcmp(cfg_value("NX2NAVI_UPSCALER"), "fsr") ? BILINEAR : FSR1, 0};
    if (cfg_bool("NX2NAVI_DLAA")) {
        now = {!taa_is_fsr() ? DLSS : (taa_is_fsr4() || vk_ffx_fsr4()) ? FSR4 : FSR3, 0};   // set where it does
        for (int i = 0; i < 5; i++)                                                          // not run: FSR does
            if (!strcmp(cfg_value("NX2NAVI_DLSS"), modes[i])) now.mode = i;
    }
    const bool dlss = dlss_offered(), fsr = vk_ffx_available();
    const bool fsr4_win = fsr && vk_ffx4_available(), fsr4 = fsr4_win || vk_ffx_fsr4_available();
    add(BILINEAR, 0);
    add(FSR1, 0);
    for (int i = 0; dlss && i < 5; i++) add(DLSS, i);
    for (int i = 0; fsr && i < 5; i++) add(FSR3, i);
    for (int i = 0; fsr4 && i < 5; i++) add(FSR4, i);
    if (!dlss && now.kind == DLSS) add(DLSS, now.mode);
    if (!fsr && now.kind == FSR3) add(FSR3, now.mode);
    if (!fsr4 && now.kind == FSR4) add(FSR4, now.mode);
    int cur = 0;
    for (int i = 0; i < n; i++)
        if (list[i].kind == now.kind && list[i].mode == now.mode) cur = i;
    // FSR 3.1 / FSR 4 still set where it cannot load: nothing upscales, so the row says so under its value
    const char *note = nullptr;
    if ((!fsr && now.kind == FSR3) || (!fsr4 && now.kind == FSR4)) note = menu_text(UI_MENU_FSR3_OFF);
    else if (upscaling_restart()) note = menu_text(UI_MENU_AFTER_RESTART);
    int nv = opt_choice_index("NX2NAVI_UPSCALER", UI_MENU_UPSCALING, names, n, cur, 1,
                              !fsr && !fsr4 ? UI_MENU_UPSCALING_DESC_NO_FSR
                                            : dlss ? UI_MENU_UPSCALING_DESC : UI_MENU_UPSCALING_DESC_FSR,
                              note);
    if (nv < 0 || nv >= n) return;
    const Entry e = list[nv];
    opt_apply("NX2NAVI_UPSCALER", e.kind == FSR1 ? "fsr" : "bilinear");
    opt_apply("NX2NAVI_DLAA", e.kind >= DLSS ? "1" : "0");
    if (e.kind < DLSS) opt_apply("NX2NAVI_FRAMEGEN", "off");   // it needs DLSS's or FSR 3.1's motion vectors (row_framegen)
    if (e.kind >= DLSS) opt_apply("NX2NAVI_DLSS", modes[e.mode]);
    if (e.kind == FSR4 && !fsr4_win) {
        opt_apply("NX2NAVI_TAA", "fsr4");
        opt_apply("NX2NAVI_FSR4", "0");
    } else {
        if (e.kind >= DLSS) opt_apply("NX2NAVI_TAA", e.kind >= FSR3 ? "fsr3" : "dlss");
        if (e.kind >= FSR3) opt_apply("NX2NAVI_FSR4", e.kind == FSR4 ? "1" : "0");
        // FSR 4 that failed in a row stays listed (it runs on this device) while FSR 3.1 serves: the row shows FSR 3.1, and
        // choosing FSR 4 again tries it anew
        if (e.kind == FSR4) vk_ffx4_retry();
    }
    // The Anti-aliasing row keeps its choice, which runs along with DLSS's or FSR's TAA (vk_aa.c vk_aa_mode); its TAA
    // went with the temporal mode, so leaving that leaves the game's SMAA
    if (e.kind < DLSS && !strcmp(cfg_value("NX2NAVI_AA"), "taa")) opt_apply("NX2NAVI_AA", "game");
}

// Anti-aliasing (vk_aa.c NX2NAVI_AA): the game's own SMAA (the default), off, FXAA or SMAA in its place, or TAA, which is
// DLSS or FSR at the scene's size, the Upscaling row's temporal modes. Without a temporal mode set (dlaa), TAA is listed
// where one runs: NVIDIA DLAA where DLSS is offered and taa does not name FSR, else FSR 4 Native AA where it runs, else
// FSR 3.1 Native AA where it loads; choosing it sets that native mode, as the Upscaling row would. While a temporal mode
// is set its TAA always runs, and the row picks what runs with it: the game's SMAA, nothing, FXAA or SMAA, noted as joined
// by the Upscaling row's TAA. A choice then leaves the upscaler and frame generation as they are. taa itself, set before
// or in the ini, shows as the game's SMAA, which is what runs with the upscaler (or without one, where it cannot run).
static void row_aa(void) {
    enum { GAME, OFF, FXAA, SMAA, TAA };
    static const char *const values[] = {"game", "off", "fxaa", "smaa", "taa"};
    const bool dlss = dlss_offered(), fsr = vk_ffx_available();
    const bool fsr4_win = fsr && vk_ffx4_available(), fsr4 = fsr4_win || vk_ffx_fsr4_available();
    const bool temporal = cfg_bool("NX2NAVI_DLAA"), fsr_set = taa_is_fsr();
    const char *names[5] = {menu_text(UI_MENU_AA_GAME), menu_text(UI_MENU_FRAMEGEN_OFF), menu_text(UI_MENU_AA_FXAA),
                            menu_text(UI_MENU_AA_SMAA), nullptr};
    int n = 4;
    enum { NONE, K_DLSS, K_FSR3, K_FSR4 };
    const int kind = dlss && (!fsr_set || !ffx_taa_available()) ? K_DLSS : fsr4 ? K_FSR4 : fsr ? K_FSR3 : NONE;
    static char taa[96];
    if (!temporal && kind) {
        if (kind == K_FSR4) snprintf(taa, sizeof taa, menu_text(UI_MENU_AA_TAA_FSR4), fsr4_label());
        else snprintf(taa, sizeof taa, "%s", menu_text(kind == K_DLSS ? UI_MENU_AA_TAA_DLSS : UI_MENU_AA_TAA_FSR));
        names[n++] = taa;
    }
    int cur = GAME;
    for (int i = 0; i < 4; i++)
        if (!strcmp(cfg_value("NX2NAVI_AA"), values[i])) cur = i;
    // FSR 3.1 / FSR 4 set where it cannot load runs nothing (the Upscaling row says so): no TAA joins the choice then
    const char *note = temporal && (!fsr_set || ffx_taa_available()) ? menu_text(UI_MENU_AA_WITH_TAA) : nullptr;
    int nv = opt_choice_index("NX2NAVI_AA", UI_MENU_AA, names, n, cur, GAME, UI_MENU_AA_DESC, note);
    if (nv < 0 || nv >= n || nv == cur) return;
    if (nv == TAA) {   // listed only without a temporal mode: the native one of that upscaler, as row_upscaling sets it
        opt_apply("NX2NAVI_UPSCALER", "bilinear");
        opt_apply("NX2NAVI_DLAA", "1");
        opt_apply("NX2NAVI_DLSS", "dlaa");
        if (kind == K_FSR4 && !fsr4_win) {
            opt_apply("NX2NAVI_TAA", "fsr4");
            opt_apply("NX2NAVI_FSR4", "0");
        } else {
            opt_apply("NX2NAVI_TAA", kind == K_DLSS ? "dlss" : "fsr3");
            if (kind != K_DLSS) opt_apply("NX2NAVI_FSR4", kind == K_FSR4 ? "1" : "0");
            if (kind == K_FSR4) vk_ffx4_retry();
        }
    }
    opt_apply("NX2NAVI_AA", values[nv]);
}

// Under the HDR value: "after restart" when what is set waits for one (the game's own HDR, to or from it:
// vk_hdr_needs_restart), "in use: off" when HDR is on but the swapchain is not HDR10 (the display or Windows' HDR
// switch is off, vk_hdr_surface_format).
static const char *hdr_note(void) {
    if (vk_hdr_needs_restart()) return menu_text(UI_MENU_AFTER_RESTART);
    if (!vk_hdr_wanted() || vk_hdr_active() || !*gfx_window_present_mode() || !cfg_bool("NX2NAVI_HDR")) return nullptr;
    static char note[96];
    snprintf(note, sizeof note, menu_text(UI_MENU_PRESENT_IN_USE), menu_text(UI_MENU_FRAMEGEN_OFF));
    return note;
}

// HDR (vk_hdr.c), two kinds: the game's picture tone mapped to the display (1, one fixed look), switched on and off at
// once, or the game's own HDR (native), which the game takes up at its start. The three brightness rows follow while
// either is on.
static void row_hdr(void) {
    static const char *const values[] = {"0", "1", "native"};
    const char *names[] = {menu_text(UI_MENU_FRAMEGEN_OFF), menu_text(UI_MENU_HDR_TONEMAPPED), menu_text(UI_MENU_HDR_NATIVE)};
    opt_choice_list("NX2NAVI_HDR", UI_MENU_HDR, values, names, 3, UI_MENU_HDR_DESC, -1, hdr_note());
    if (!cfg_bool("NX2NAVI_HDR")) return;
    opt_slider("NX2NAVI_HDR_PEAK_NITS", UI_MENU_HDR_PEAK, 400, 10000, UI_MENU_HDR_PEAK_DESC, 50, "%.0f nits");
    opt_slider("NX2NAVI_HDR_GAME_NITS", UI_MENU_HDR_GAME, 48, 500, UI_MENU_HDR_GAME_DESC, 1, "%.0f nits");
    opt_slider("NX2NAVI_HDR_UI_NITS", UI_MENU_HDR_UI, 48, 500, UI_MENU_HDR_UI_DESC, 1, "%.0f nits");
}

// DLSS frame generation's multiplier (NX2NAVI_FRAMEGEN_MULTIPLIER, vk_dlssg.c, at once): 2x, 3x or 4x as far as this
// GPU and driver allow (vk_dlssg_multiplier_max, from Streamline). While DLSS's frame generation runs, the row lists only
// those and shows the value in effect, a higher setting clamped to it, with a note saying why the rest are missing;
// before the restart that starts it, or before Streamline reported its limit, it lists all three. Left out while it is
// set but does not run: the frame generation row says why.
static void row_framegen_mult(bool restart) {
    if (!restart && !vk_dlssg_active()) return;
    int old_dll = 0, max = restart ? 0 : vk_dlssg_multiplier_max(&old_dll);
    const char *note = nullptr;
    static char upto[64];
    if (max == 2) note = menu_text(old_dll ? UI_MENU_FRAMEGEN_MULT_OLD_DLL : UI_MENU_FRAMEGEN_MULT_RTX50);
    else if (max == 3) snprintf(upto, sizeof upto, menu_text(UI_MENU_FRAMEGEN_MULT_UPTO), max), note = upto;
    if (!max) max = 4;   // not known yet: every multiplier, the setting as it is
    static const char *const values[] = {"2", "3", "4"};
    static const char *const names[] = {"2x", "3x", "4x"};
    int n = max - 1, cur = vk_dlssg_multiplier_of(cfg_value("NX2NAVI_FRAMEGEN_MULTIPLIER"), max, nullptr) - 2;
    int nv = opt_choice_index("NX2NAVI_FRAMEGEN_MULTIPLIER", UI_MENU_FRAMEGEN_MULT, names, n, cur, 0,
                              UI_MENU_FRAMEGEN_MULT_DESC, note);
    if (nv >= 0 && nv < n) opt_apply("NX2NAVI_FRAMEGEN_MULTIPLIER", values[nv]);
}

// NVIDIA's frame generation runs through NGX itself, not through Streamline (vk_dlaa.c vk_ngxfg_*, gfx_present.c
// ng_*): every build but the Windows one.
static bool framegen_ngx(void) {
#ifdef _WIN32
    return false;
#else
    return true;
#endif
}

// Its multiplier there (cfg.c FG_MULT, gfx_present.c ng_multiplier): 2x up to what the GPU allows, or Dynamic. Left out
// while frame generation is set but cannot run: the frame generation row says why.
static void row_framegen_mult_ngx(void) {
    static const char *const mult[] = {"2", "3", "4", "5", "6"};
    static const char *const mult_names[] = {"2x", "3x", "4x", "5x", "6x"};
    const int max = vk_ngxfg_max();   // generated frames per rendered one
    if (max < 1) return;
    const char *values[6], *names[6];
    int n = 0;
    for (; n < max && n < 5; n++) values[n] = mult[n], names[n] = mult_names[n];
    const int top = n - 1;
    values[n] = "dynamic", names[n++] = menu_text(UI_MENU_FG_MULT_DYNAMIC);
    const char *v = cfg_value("NX2NAVI_FG_MULT");
    int cur = strcmp(v, "dynamic") ? top : n - 1;   // a multiplier above what the GPU allows shows as the highest
    for (int i = 0; i < n; i++)
        if (!strcmp(v, values[i])) cur = i;
    int nv = opt_choice_index("NX2NAVI_FG_MULT", UI_MENU_FRAMEGEN_MULT, names, n, cur, 0, UI_MENU_FG_MULT_DESC, nullptr);
    if (nv >= 0 && nv < n) opt_apply("NX2NAVI_FG_MULT", values[nv]);
}

// Frame generation to or from NVIDIA's waits for a restart: AMD's follows the setting, NVIDIA's is set up with the device
// (vk_dlssg.c). Through NGX it follows the setting too, once NGX is up, which needs DLSS upscaling on when the game
// starts. The row's note and the Restart button's count (restart_pending).
static bool framegen_restart(void) {
    const bool set = !strcmp(cfg_value("NX2NAVI_FRAMEGEN"), "dlss");
    if (!dlss_offered()) return false;
    return framegen_ngx() ? set && !vk_ngxfg_wanted() : set != (vk_dlssg_wanted() != 0);
}

// The frame generation of the library FSR 4 comes with (vk_fg4.cpp), as the row names it: with the version the library
// runs on this card, the machine-learned 4.x on a Radeon RX 9000 and 3.1.x on any other.
static const char *fg4_label(void) {
    static char s[48];
    const char *v = vk_fg4_name();
    snprintf(s, sizeof s, *v ? "AMD FSR 4 library (%s)" : "AMD FSR 4 library", v);
    return s;
}
// It is the one in use: set (NX2NAVI_FSR4_FG) with FSR 4 as the upscaler, where it runs (vk_fg.cpp use4).
static bool fg4_chosen(void) {
    return cfg_bool("NX2NAVI_FSR4_FG") && vk_ffx_fsr4() && vk_fg4_available() && !vk_fg4_failed();
}

// Frame generation (vk_fg.cpp, at once; NVIDIA's after a restart); it needs a temporal upscaler's motion vectors
static void row_framegen(void) {
    // the generated frames are made from DLSS's or FSR 3.1's motion vectors: without one of them it would do nothing,
    // and FSR 3.1 set where it cannot load (vk_ffx_available) is none
    const bool dlss = dlss_offered(), fsr = vk_ffx_framegen_available();
    const bool upscaled = cfg_bool("NX2NAVI_DLAA") && (!taa_is_fsr() || ffx_taa_available());
    const char *v = cfg_value("NX2NAVI_FRAMEGEN");
    int cur = !strcmp(v, "fsr3") ? 1 : !strcmp(v, "dlss") ? 2 : 0;
    // what the texts promise follows what runs here: DLSS's frame generation, AMD's, both or neither
    const UiTextId desc = dlss ? (fsr ? UI_MENU_FRAMEGEN_DESC : UI_MENU_FRAMEGEN_DESC_DLSS)
                               : (fsr ? UI_MENU_FRAMEGEN_DESC_FSR : UI_MENU_FRAMEGEN_DESC_NONE);
    UiTextId note = dlss ? (fsr ? UI_MENU_FRAMEGEN_NOTE : UI_MENU_FRAMEGEN_NOTE_DLSS)
                         : (fsr ? UI_MENU_FRAMEGEN_NOTE_FSR : UI_MENU_FRAMEGEN_NOTE);   // neither: the general note
    if (!dlss && !fsr && cur == 0) opt_disable_next(UI_MENU_FRAMEGEN_NEEDS_NONE);
    else if (!upscaled) opt_disable_next(dlss ? (fsr ? UI_MENU_FRAMEGEN_NEEDS : UI_MENU_FRAMEGEN_NEEDS_DLSS)
                                              : (fsr ? UI_MENU_FRAMEGEN_NEEDS_FSR : UI_MENU_FRAMEGEN_NEEDS_NONE));
    // Off, AMD's where FSR 3.1 loads and NVIDIA DLSS (vk_dlssg.c, through Streamline and nx2navi_dlssg.dll) on NVIDIA GPUs,
    // now that Reflex paces it from the frame's start and generated frames are presented; elsewhere one stays listed
    // while it is set
    // AMD's has two libraries where FSR 4 runs: the FSR 3.1 library's, and the one of the library FSR 4 comes with,
    // listed after it as a choice of its own. That one generates only beside FSR 4 upscaling, so choosing it sets the
    // upscaler to FSR 4 as well; both are NX2NAVI_FRAMEGEN = fsr3, told apart by NX2NAVI_FSR4_FG.
    enum { OFF, FSR3, FSR4LIB, DLSS };
    const bool four = cur == 1 && fg4_chosen();
    const bool four_offered = fsr && vk_ffx4_available() && vk_fg4_available() && !vk_fg4_failed();
    const int now = cur == 2 ? DLSS : cur == 1 ? (four ? FSR4LIB : FSR3) : OFF;
    const char *all_names[] = {menu_text(UI_MENU_FRAMEGEN_OFF), "AMD FSR 3.1", fg4_label(), "NVIDIA DLSS"};
    int kinds[4];
    const char *names[4];
    int n = 0, at = 0;
    for (int i = 0; i < 4; i++) {
        if ((i == FSR3 && !fsr && cur != 1) || (i == FSR4LIB && !four_offered && !four) || (i == DLSS && !dlss && cur != 2)) continue;
        if (i == now) at = n;
        kinds[n] = i, names[n++] = all_names[i];
    }
    // AMD's follows the setting; NVIDIA's is set up with the device, so only a change to or from it waits for a restart.
    // Through NGX it follows the setting too, once NGX is up, which needs DLSS upscaling on when the game starts.
    const bool ngx = framegen_ngx();
    const bool restart = framegen_restart();
    if (restart) note = UI_MENU_AFTER_RESTART;
    else if (cur == 1 && !fsr) note = UI_MENU_FSR3_OFF;   // set where it cannot load: "in use: off" and why
    // NVIDIA's set since the start but not running: "in use: off" and why (vk_dlssg_status; the log has the details).
    // Through NGX that is NVIDIA's frame generation library missing beside the game, or a GPU it does not run on.
    const int st = cur != 2 || restart ? DLSSG_RUNS : ngx ? (vk_ngxfg_max() ? DLSSG_RUNS : DLSSG_NO_FILES) : vk_dlssg_status();
    if (st == DLSSG_UNAVAILABLE) note = UI_MENU_FRAMEGEN_UNAVAILABLE;
    else if (st == DLSSG_NO_FILES) note = UI_MENU_FRAMEGEN_OFF_FILES;
    // the note under AMD's names FSR 3.1 as what it needs; the FSR 4 library's runs beside FSR 4, so it has none
    int nv = opt_choice_index("NX2NAVI_FRAMEGEN", UI_MENU_FRAMEGEN, names, n, at, 0, desc, four && !restart ? NULL : menu_text(note));
    if (nv >= 0 && nv < n) {
        const int k = kinds[nv];
        opt_apply("NX2NAVI_FRAMEGEN", k == OFF ? "off" : k == DLSS ? "dlss" : "fsr3");
        if (k == FSR4LIB) {
            opt_apply("NX2NAVI_FSR4_FG", "1");
            if (!vk_ffx_fsr4()) opt_apply("NX2NAVI_TAA", "fsr3"), opt_apply("NX2NAVI_FSR4", "1"), vk_ffx4_retry();
        } else if (k == FSR3 && cfg_bool("NX2NAVI_FSR4_FG")) {
            opt_apply("NX2NAVI_FSR4_FG", "0");
        }
    }
    else if (cur == 2 && upscaled && dlss && ngx) row_framegen_mult_ngx();
    else if (cur == 2 && upscaled && dlss) row_framegen_mult(restart);
}

// One Resolution row for the picture's height (NX2NAVI_GAME_RES, widescreen.c picture_height: 1080, the game's 1440 or
// more) and the window outside fullscreen (NX2NAVI_WINDOW_SIZE): each height is shown as the size of the game's shape at
// it (16:9 2560 x 1440, 21:9 3440 x 1440, 32:9 5120 x 1440, the window's own shape where the picture follows the window),
// and choosing one sets both. The window takes it at once (gfx_present.c applies a changed size at the next present),
// within the monitor, and the picture after a restart. Native draws at the window's size and leaves the window as it is.
// As two rows of their own, a window of one size took a picture of another and scaled it for nothing.
static void row_resolution(void) {
    static const int heights[] = {1080, 1440, 1800, 2160, 2880};
    static const int wide169[] = {1920, 2560, 3200, 3840, 5120}, wide219[] = {2560, 3440, 4300, 5120, 6880},
                     wide329[] = {3840, 5120, 6400, 7680, 10240};
    enum { N = sizeof heights / sizeof *heights };
    const char *aspect = cfg_value("NX2NAVI_GAME_ASPECT");
    int mw = 0, mh = 0, ww = 0, wh = 0;
    int have_mon = gfx_window_monitor_size(&mw, &mh) && mw > 0 && mh > 0;
    int have_win = sscanf(cfg_value("NX2NAVI_WINDOW_SIZE"), "%dx%d", &ww, &wh) == 2 && ww > 0 && wh > 0;
    // the window's shape where the picture follows it: the monitor's in fullscreen, else the window's own
    double shape = cfg_bool("NX2NAVI_FULLSCREEN") && have_mon ? (double)mw / mh : have_win ? (double)ww / wh : 16.0 / 9.0;
    static char sizes[N][24], names[N + 1][48];
    const char *pn[N + 1];
    for (int i = 0; i < N; i++) {
        int w = !strcmp(aspect, "21:9") ? wide219[i] : !strcmp(aspect, "32:9") ? wide329[i] :
                !strcmp(aspect, "window") ? ((int)(heights[i] * shape + 0.5) & ~1) : wide169[i];
        int h = heights[i], fw = w, fh = h;
        // the window within the monitor, in the same shape (the picture keeps its height and is scaled into it)
        if (have_mon && (fw > mw || fh > mh)) {
            double k = (double)mw / fw < (double)mh / fh ? (double)mw / fw : (double)mh / fh;
            fw = (int)(fw * k) & ~1, fh = (int)(fh * k) & ~1;
        }
        snprintf(sizes[i], sizeof sizes[i], "%dx%d", fw, fh);
        snprintf(names[i], sizeof names[i], "%d × %d", w, h);   // U+00D7, the multiplication sign
        pn[i] = names[i];
    }
    pn[N] = menu_text(UI_MENU_GAME_RES_NATIVE);
    const char *res = cfg_value("NX2NAVI_GAME_RES");
    int cur = -1;
    for (int i = 0; i < N; i++)
        if (atoi(res) == heights[i]) cur = i;
    if (!strcmp(res, "window")) cur = N;
    int nv = opt_choice_index("NX2NAVI_GAME_RES", UI_MENU_GAME_RES, pn, N + 1, cur, 1, UI_MENU_GAME_RES_DESC);
    if (nv < 0 || nv > N) return;
    if (nv == N) {
        opt_apply("NX2NAVI_GAME_RES", "window");
        return;
    }
    char h[16];
    snprintf(h, sizeof h, "%d", heights[nv]);
    opt_apply("NX2NAVI_GAME_RES", h);
    opt_apply("NX2NAVI_WINDOW_SIZE", sizes[nv]);
}

static void tab_screen(float scale) {
    opt_begin("##screen", scale);
    opt_toggle("NX2NAVI_FULLSCREEN", UI_MENU_FULLSCREEN, UI_MENU_FULLSCREEN_DESC);
    // the window's size is the Resolution row's below where the picture's size can be set (row_resolution); a build
    // without the widescreen patch sets draws the game's own size and keeps the window row
    if (!wide_available()) row_window_size();
    // On is the default mode, mailbox with its fallbacks (gfx_present.c pick_present_mode): plain fifo would drop the
    // game to 30 at every slow frame on a 60 Hz display. A fifo already saved also reads as on.
    opt_toggle_value("NX2NAVI_VK_PRESENT", UI_MENU_PRESENT_VSYNC, UI_MENU_PRESENT_DESC, "mailbox", "immediate", vsync_note());
    row_gpu();
    static const UiTextId perf[] = {UI_MENU_PERF_OFF, UI_MENU_PERF_FPS, UI_MENU_PERF_FULL};
    opt_choice("NX2NAVI_PERF_OVERLAY", UI_MENU_PERF_OVERLAY, perf, 3, UI_MENU_PERF_OVERLAY_DESC);
    opt_group(UI_MENU_GRAPHICS);
    // The game resolution needs the output and scene patch sets (resolution.c res_adjustable): Ocarina's build has
    // none and always draws its own size, so the row would change nothing
    if (res_adjustable()) row_output_res();
    // Widescreen needs the patch sets of recomp/patches-v111.txt (widescreen.c wide_available)
    static const UiTextId wide[] = {UI_MENU_GAME_ASPECT_16_9, UI_MENU_GAME_ASPECT_WINDOW, UI_MENU_GAME_ASPECT_21_9,
                                    UI_MENU_GAME_ASPECT_32_9};
    if (wide_available()) opt_choice("NX2NAVI_GAME_ASPECT", UI_MENU_GAME_ASPECT, wide, 4, UI_MENU_GAME_ASPECT_DESC);
    if (wide_available()) row_resolution();
    row_upscaling();
    row_aa();
    row_framegen();
    row_hdr();
    // The sharpening runs under FSR 1 (vk_fsr.c vk_fsr_record, RCAS) and FSR 3.1 (vk_ffx.cpp, its own RCAS); with
    // bilinear or DLSS the slider would do nothing
    int dl = cfg_bool("NX2NAVI_DLAA"), fsr3 = dl && taa_is_fsr() && ffx_taa_available();
    if (!fsr3 && (strcmp(cfg_value("NX2NAVI_UPSCALER"), "fsr") || dl)) opt_disable_next(UI_MENU_SHARPNESS_NEEDS_FSR);
    opt_slider("NX2NAVI_SHARPNESS", UI_MENU_SHARPNESS, 0, 100, UI_MENU_SHARPNESS_DESC, 5, "%.0f %%");
    // Anisotropic filtering of the surface textures that tile (vk_texture.c aniso_raised), at once (vk_sampler_aniso_live)
    static const char *const aniso[] = {"game", "2", "4", "8", "16"};
    const char *aniso_names[] = {menu_text(UI_MENU_ANISO_GAME), "2x", "4x", "8x", "16x"};
    opt_choice_list("NX2NAVI_ANISO", UI_MENU_ANISO, aniso, aniso_names, 5, UI_MENU_ANISO_DESC);
    row_texture_detail();
    opt_end();
}

// ---- the Game tab's Saves group (save_share.c). Export writes the save the game would load next into a zip in the
// player's Downloads folder, for another player or, with this run's log and nx2navi.ini, a bug report. Import takes a
// zip the player picks in the host's file dialog and leaves its save for the next start (fs_host.c import_pending): the
// game holds its progress in memory and would write it back over a save replaced under it. The work runs off the
// present thread (a thread of its own, or the dialog's); the rows wait greyed meanwhile, and the line under them says
// what the last one did, else what an import did at this start.
static PlatMutex g_share_lock = PLAT_MUTEX_INIT;
static std::atomic<int> g_share_busy{0};
static int g_share_text = -1;   // the last result's UiTextId, -1 none; a format that takes g_share_arg
static char g_share_arg[PLAT_PATH_MAX + 64];
static bool g_share_warn;

static void share_done(UiTextId text, const char *arg, bool warn) {
    plat_mutex_lock(&g_share_lock);
    g_share_text = text;
    snprintf(g_share_arg, sizeof g_share_arg, "%s", arg ? arg : "");
    g_share_warn = warn;
    plat_mutex_unlock(&g_share_lock);
    g_share_busy.store(0);
}

static void share_export_main(void *arg) {
    const int bug_report = (int)(intptr_t)arg;
    char dir[PLAT_PATH_MAX], path[PLAT_PATH_MAX + 64];
    SaveShareResult r = save_share_default_dir(dir, sizeof dir) ? save_share_export(dir, bug_report, path, sizeof path)
                                                                : SAVE_SHARE_WRITE_FAILED;
    if (r == SAVE_SHARE_OK) share_done(UI_MENU_SAVE_EXPORTED, path, false);
    else share_done(r == SAVE_SHARE_NO_SAVE ? UI_MENU_SAVE_E_NO_SAVE : r == SAVE_SHARE_READ_FAILED ? UI_MENU_SAVE_E_READ_SAVE
                                                                                                   : UI_MENU_SAVE_E_WRITE,
                    dir, true);
}

// The file dialog's answer, on its thread: the import's checks and the copy into the profile run here.
static void share_zip_picked(const char *path, void *ctx) {
    (void)ctx;
    if (!path) {   // cancelled: the last line stays
        g_share_busy.store(0);
        return;
    }
    char detail[160];
    SaveShareResult r = save_share_import(path, detail, sizeof detail);
    UiTextId text;
    switch (r) {
    case SAVE_SHARE_OK: text = UI_MENU_SAVE_IMPORT_READY; break;
    case SAVE_SHARE_READ_FAILED: text = UI_MENU_SAVE_E_READ; break;
    case SAVE_SHARE_WRITE_FAILED: text = UI_MENU_SAVE_E_STAGE; break;
    case SAVE_SHARE_OTHER_GAME: text = UI_MENU_SAVE_E_OTHER_GAME; break;
    case SAVE_SHARE_NEWER: text = UI_MENU_SAVE_E_NEWER; break;
    case SAVE_SHARE_DAMAGED: text = UI_MENU_SAVE_E_DAMAGED; break;
    case SAVE_SHARE_UNSAFE: text = UI_MENU_SAVE_E_UNSAFE; break;
    default: text = UI_MENU_SAVE_E_NOT_SAVE; break;
    }
    share_done(text, r == SAVE_SHARE_OK ? detail : path, r != SAVE_SHARE_OK);
}

static void rows_saves(void) {
    opt_group(UI_MENU_SAVES);
    const bool busy = g_share_busy.load() != 0;
    int start = 0;   // 1 export, 2 export for a bug report, 3 import
    if (busy) opt_disable_next(UI_MENU_SAVE_BUSY);
    if (opt_button("##save_export", UI_MENU_SAVE_EXPORT, UI_MENU_SAVE_EXPORT_DESC)) start = 1;
    if (busy) opt_disable_next(UI_MENU_SAVE_BUSY);
    if (opt_button("##save_export_bug", UI_MENU_SAVE_EXPORT_BUG, UI_MENU_SAVE_EXPORT_BUG_DESC)) start = 2;
    // a stat per frame while the Game tab shows; the launcher's own import drops a waiting one
    const bool waiting = save_share_pending() != 0;
    if (busy) opt_disable_next(UI_MENU_SAVE_BUSY);
    if (opt_button("##save_import", UI_MENU_SAVE_IMPORT, UI_MENU_SAVE_IMPORT_DESC,
                   waiting ? menu_text(UI_MENU_SAVE_IMPORT_WAITING) : ""))
        start = 3;
    if (waiting) {
        if (busy) opt_disable_next(UI_MENU_SAVE_BUSY);
        if (opt_button("##save_import_cancel", UI_MENU_SAVE_IMPORT_CANCEL, UI_MENU_SAVE_IMPORT_CANCEL_DESC)) {
            if (save_share_cancel()) share_done(UI_MENU_SAVE_E_STAGE, "", true);
            else share_done(UI_MENU_SAVE_IMPORT_CANCELLED, "", false);
        }
    }
    if (start && !busy) {
        g_share_busy.store(1);
        if (start == 3) {
            char dir[PLAT_PATH_MAX];
            file_pick(save_share_default_dir(dir, sizeof dir) ? dir : "", "zip", share_zip_picked, NULL);
        } else if (!plat_thread_spawn(share_export_main, (void *)(intptr_t)(start == 2), "save export", 0)) {
            share_done(UI_MENU_SAVE_E_READ_SAVE, "", true);
        }
    }
    char line[PLAT_PATH_MAX * 2] = "";
    plat_mutex_lock(&g_share_lock);
    bool warn = g_share_warn;
    if (g_share_text >= 0) snprintf(line, sizeof line, menu_text((UiTextId)g_share_text), g_share_arg);
    plat_mutex_unlock(&g_share_lock);
    const char *done = fs_save_import_done();
    if (!line[0] && *done) {
        if (!strcmp(done, "-")) snprintf(line, sizeof line, "%s", menu_text(UI_MENU_SAVE_IMPORTED_NEW));
        else snprintf(line, sizeof line, menu_text(UI_MENU_SAVE_IMPORTED), done);
        warn = false;
    }
    if (line[0]) opt_text_str(line, warn);
}

static void tab_game(float scale) {
    opt_begin("##game", scale);
    row_fps_limit();   // frame_rate.c: Ocarina steps by the clock, so the pacer alone sets its rate
    row_game_lang();
    static const UiTextId mode[] = {UI_MENU_MODE_DOCKED, UI_MENU_MODE_HANDHELD};
    opt_choice("NX2NAVI_OPERATION_MODE", UI_MENU_OPERATION_MODE, mode, 2, UI_MENU_OPERATION_MODE_DESC);
    opt_toggle("NX2NAVI_ASSUME_FOCUS", UI_MENU_ASSUME_FOCUS, UI_MENU_ASSUME_FOCUS_DESC);
    // discord_rpc.c reads the setting on its own thread and connects or clears the status within half a second
    opt_toggle("NX2NAVI_DISCORD_RPC", UI_MENU_DISCORD_RPC, UI_MENU_DISCORD_RPC_DESC);
    rows_saves();
    // The debug menu, a tool for testing with English text of its own: off unless switched on here
    if (dbg_supported()) {
        opt_group_text("Testing");
        const bool on = cfg_bool("NX2NAVI_DEBUG_MENU");
        if (opt_toggle_text("debug_menu", "Debug menu (F9)",
                            "F9 opens a window for testing: the game's flags by name, a teleport to every area's start "
                            "positions, its cutscenes to start, a free camera, the game's speed, and a search of its "
                            "memory with values to set and hold. All of these can break a save, and the game saves by itself after a change "
                            "of area: keep a copy of the save first.", on))
            opt_apply("NX2NAVI_DEBUG_MENU", on ? "0" : "1");
    }
    opt_end();
}

static void tab_sound(float scale) {
    opt_begin("##sound", scale);
    opt_slider("NX2NAVI_VOLUME", UI_MENU_VOLUME, 0, 100, UI_MENU_VOLUME_DESC, 5, "%.0f %%");
    opt_slider("NX2NAVI_MENU_SOUND_VOLUME", UI_MENU_MENU_SOUNDS, 0, 100, UI_MENU_MENU_SOUNDS_DESC, 5, "%.0f %%");
    static const UiTextId channels[] = {UI_MENU_CHANNELS_AUTO, UI_MENU_CHANNELS_STEREO, UI_MENU_CHANNELS_51};
    opt_choice("NX2NAVI_AUDIO_CHANNELS", UI_MENU_AUDIO_CHANNELS, channels, 3, UI_MENU_AUDIO_CHANNELS_DESC);
    opt_end();
}

// A confirmation is open (Controls: restart, Advanced: reset): B and Esc close it, not the menu; L / R stay on the page.
static bool g_modal;

// The note under the layout: Automatic with a gamepad connected, the layout it picked ("in use: Xbox"), named
// PlayStation for a pad whose bottom button is labelled cross. NULL otherwise.
static const char *pad_layout_note(void) {
    int got = input_pad_layout();
    if (got < 0 || strcmp(cfg_value("NX2NAVI_PAD_LAYOUT"), "auto")) return nullptr;
    static char note[96];
    UiTextId name = got != PAD_LAYOUT_XBOX                     ? UI_MENU_PAD_LAYOUT_NINTENDO
                    : input_pad_label(PAD_SOUTH) == PAD_LABEL_CROSS ? UI_MENU_PAD_LAYOUT_PLAYSTATION
                                                                  : UI_MENU_PAD_LAYOUT_XBOX;
    snprintf(note, sizeof note, menu_text(UI_MENU_PAD_LAYOUT_IN_USE), menu_text(name));
    return note;
}

// The action a key presses, or -1: the walk key is a key of its own, but Shift is A (dash) in the default bindings, and a
// walk key that also presses something walks and does that something at the same time. The bindings come from a
// snapshot, and every key of an action is looked at: binds_refresh rebuilds the live table on the polling thread while
// the menu draws, and an action may hold more than the two keys its cells show.
static int key_action(const InputMap *m, int key) {
    for (int a = 0; a < ACTION_COUNT; a++) {
        uint8_t keys[INPUT_BINDING_MAX];
        int n = input_map_keys_of(m, a, keys, INPUT_BINDING_MAX);
        for (int i = 0; i < n; i++)
            if (keys[i] == key) return a;
    }
    return -1;
}

// The walk key as the handful of keys a player is likely to pick. "none" rather than an empty value closes it: an
// unset key falls back to the default in the settings table, the same as a fresh installation. nx2navi.ini takes any name
// input_map.c knows (walks of the keyboard sticks, not of the gamepad). A key the current bindings also use says which
// action it presses.
static void row_walk_key(void) {
    static const char *const keys[] = {"none", "lshift", "lalt", "lctrl", "tab", "capslock"};
    const char *names[sizeof keys / sizeof *keys] = {menu_text(UI_MENU_OFF), "Shift", "Alt", "Ctrl", "Tab", "Caps Lock"};
    const char *cur = cfg_value("NX2NAVI_WALK_KEY");
    static char note[64];
    const char *n = nullptr;
    InputMap m;
    input_map_snapshot(&m);
    for (int k = KEY_A; k < KEY_COUNT && !n; k++) {
        if (!input_key_name(k) || strcasecmp(input_key_name(k), cur)) continue;
        int a = key_action(&m, k);
        if (a >= 0 && input_action_name(a)) snprintf(note, sizeof note, menu_text(UI_MENU_WALK_KEY_TAKEN), input_action_name(a)), n = note;
    }
    opt_choice_list("NX2NAVI_WALK_KEY", UI_MENU_WALK_KEY, keys, names, sizeof keys / sizeof *keys, UI_MENU_WALK_KEY_DESC,
                    -1, n);
}

// The note under the prompts. The settings name a folder of prompts (input_prompts_folder: the prompts and the pad
// layout, as a layout that puts A on the right button needs the face buttons named the other way round), but the game
// reads its fonts and UI archive once at start (romfs.c overlay_init). While the folder in use differs: "files
// missing" when the setup did not write the one asked for, why it was left out at start (damaged, a game resolution
// that is not 16:9, a RomFS mod with the same files, NX2NAVI_ROMFS_PATCHES), else "prompts change after restart" with
// *restart set: a restart would bring the prompts asked for. The action prompt's diamond of face buttons is laid out
// for the pad layout at start too (romfs.c action_prompt_overlay), whatever the prompts: while the layout asks for the
// other places, a restart would lay it out again, so *restart is set whatever the note, and with the folder in use the
// note is "prompts change after restart". NULL when both match.
static const char *prompts_note(bool *restart) {
    const char *layout = cfg_value("NX2NAVI_PAD_LAYOUT"), *used = romfs_prompts();
    const char *want = input_prompts_folder(input_prompts_pick(cfg_get("NX2NAVI_PROMPTS"), layout), layout);
    int diamond = romfs_action_prompt();   // -1 when it could not be laid out: a restart would not change it
    *restart = diamond >= 0 && diamond != input_pad_layout_pick(layout, 0);
    if (want == used || (want && used && !strcmp(want, used))) return *restart ? menu_text(UI_MENU_PROMPTS_RESTART) : nullptr;
    // A game resolution of another shape needs a folder of its own for the PC prompts (romfs-patches-pc-<w>x<h>), which
    // the setup does not write: the shape is the reason then, not the setup.
    int off = romfs_prompts_off(want);
    if (want && !romfs_prompts_ready(want) && off != ROMFS_PROMPTS_SHAPE) return menu_text(UI_MENU_PROMPTS_MISSING);
    switch (off) {
    case ROMFS_PROMPTS_FAILED: return menu_text(UI_MENU_PROMPTS_FAILED);
    case ROMFS_PROMPTS_SHAPE: return menu_text(UI_MENU_PROMPTS_OFF_SHAPE);
    case ROMFS_PROMPTS_MODS: return menu_text(UI_MENU_PROMPTS_OFF_MODS);
    case ROMFS_PROMPTS_PATCHES: return menu_text(UI_MENU_PROMPTS_OFF_PATCHES);
    default: *restart = true; return menu_text(UI_MENU_PROMPTS_RESTART);
    }
}

// How many settings wait for a restart, on the Restart button: the restart-only ones whose value differs from the one the
// game started with (cfg.c cfg_pending_count), NVIDIA's upscaling, frame generation and HDR while their rows say "after
// restart" (their settings are live ones, so the rows decide), and the prompts when a changed pad layout asks for
// others (the prompts setting itself, when changed, is among the first).
static int restart_pending(void) {
    bool prompts;
    prompts_note(&prompts);
    return cfg_pending_count() + upscaling_restart() + framegen_restart() + (vk_hdr_needs_restart() != 0)
           + (prompts && !cfg_pending("NX2NAVI_PROMPTS"));
}

// The confirmation of the menu's Restart button, over the menu; Cancel has the cursor first. Restart writes nx2navi.ini,
// then closes the game as Quit does and starts it again (gfx_window_request_restart). It waits while the game writes a
// save, and the game stays when nx2navi.ini cannot be written, as the changed settings would not come back. From the
// Cheats tab (restart_rebuild) Restart also lets the next start rebuild for cheats whose last rebuild failed
// (cheats_rebuild_allow). The Quit button's confirmation is the same box (g_confirm_quit), and it waits too: the game
// runs on while the menu is open, and closing the window finishes only a Commit that has begun (gfx_window_closed,
// fs_save_flush), so a save between writing its files and committing them would be dropped.
static bool restart_rebuild, g_confirm_quit;
static void restart_confirm(float scale) {
    g_modal = false;
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::PushStyleVar(ImGuiStyleVar_PopupBorderSize, 0);
    const bool open = ImGui::BeginPopupModal("##confirm", nullptr,
                                             ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::PopStyleVar();
    if (!open) return;
    g_modal = true;
    const ImVec2 box = ImGui::GetWindowPos(), box_size = ImGui::GetWindowSize();
    skin_panel(ImGui::GetWindowDrawList(), box, ImVec2(box.x + box_size.x, box.y + box_size.y), scale);
    ImGui::TextUnformatted(menu_text(g_confirm_quit ? UI_MENU_QUIT_CONFIRM : UI_MENU_RESTART_CONFIRM));
    bool saving = fs_save_busy() != 0;
    if (saving || (!g_confirm_quit && !cfg_last_save_ok()))
        ImGui::TextColored(ImVec4(0.95f, 0.45f, 0.35f, 1), "%s",
                           menu_text(!saving ? UI_MENU_SAVE_FAILED : g_confirm_quit ? UI_MENU_QUIT_SAVING
                                                                                    : UI_MENU_RESTART_SAVING));
    ImGui::Spacing();
    const char *yes = menu_text(g_confirm_quit ? UI_MENU_QUIT : UI_MENU_RESTART_YES), *no = menu_text(UI_MENU_RESET_NO);
    const ImGuiStyle &st = ImGui::GetStyle();
    float bw = fmaxf(ImGui::CalcTextSize(yes).x, ImGui::CalcTextSize(no).x) + st.FramePadding.x * 4;
    bool close = skin_widget_button(no, ImVec2(bw, 0), scale);
    if (ImGui::IsWindowAppearing()) ImGui::SetItemDefaultFocus();
    ImGui::SameLine(0, st.ItemSpacing.x + skin_cursor_reach(scale));
    ImGui::BeginDisabled(saving);
    if (skin_widget_button(yes, ImVec2(bw, 0), scale)) {
        if (g_confirm_quit) {
            gfx_window_request_close();
        } else if (cfg_save()) {
            if (restart_rebuild) cheats_rebuild_allow();
            gfx_window_request_restart();
        }
    }
    ImGui::EndDisabled();
    if (ImGui::IsKeyPressed(ImGuiKey_Escape, false) || ImGui::IsKeyPressed(ImGuiKey_GamepadFaceRight, false))
        close = true, menu_sound_play(MENU_SOUND_CANCEL);
    if (close) {
        ImGui::CloseCurrentPopup();
        restart_rebuild = false;
    }
    ImGui::EndPopup();
}

static void tab_controls(float scale) {
    opt_begin("##controls", scale);
    // The mouse camera first, where a player on keyboard and mouse looks for it; both rows are live. How the mouse
    // reaches the game, where the game version has the direct mode's patch set (mouse_direct.c; the values in their
    // cfg.c order, stick then direct), and its sensitivity. The slider stops short of 0, which nx2navi.ini takes to
    // switch the mouse camera and the cursor capture off.
    if (input_direct_mouse_available()) {
        static const UiTextId mouse_mode[] = {UI_MENU_MOUSE_STICK, UI_MENU_MOUSE_DIRECT};
        opt_choice("NX2NAVI_MOUSE_MODE", UI_MENU_MOUSE_MODE, mouse_mode, 2, UI_MENU_MOUSE_MODE_DESC);
    }
    opt_slider("NX2NAVI_MOUSE_SENS", UI_MENU_MOUSE_SENS, 0.05f, 10, UI_MENU_MOUSE_SENS_DESC, 0.05f, "%.2f");
    static const UiTextId layout[] = {UI_MENU_PAD_LAYOUT_AUTO, UI_MENU_PAD_LAYOUT_NINTENDO, UI_MENU_PAD_LAYOUT_XBOX,
                                      UI_MENU_PAD_LAYOUT_PLAYSTATION};
    opt_choice("NX2NAVI_PAD_LAYOUT", UI_MENU_PAD_LAYOUT, layout, 4, UI_MENU_PAD_LAYOUT_DESC, pad_layout_note());
    // The setting's values in their cfg.c order (nintendo, xbox, playstation, steamdeck, pc). The game reads its prompts
    // at start: the row's note says when a restart would bring others, and the bar's Restart button does it.
    static const UiTextId prompts[] = {UI_MENU_PROMPTS_NINTENDO, UI_MENU_PROMPTS_XBOX, UI_MENU_PROMPTS_PLAYSTATION,
                                       UI_MENU_PROMPTS_STEAMDECK, UI_MENU_PROMPTS_PC};
    bool restart;
    opt_choice("NX2NAVI_PROMPTS", UI_MENU_PROMPTS, prompts, 5, UI_MENU_PROMPTS_DESC, prompts_note(&restart));
    opt_toggle("NX2NAVI_MOTION", UI_MENU_MOTION, UI_MENU_MOTION_DESC);
    opt_slider("NX2NAVI_RUMBLE_STRENGTH", UI_MENU_RUMBLE_STRENGTH, 0, 300, UI_MENU_RUMBLE_STRENGTH_DESC, 10, "%.0f %%");
    opt_slider("NX2NAVI_RUMBLE_CURVE", UI_MENU_RUMBLE_CURVE, 0.2f, 1.5f, UI_MENU_RUMBLE_CURVE_DESC, 0.05f, "%.2f");
    opt_toggle("NX2NAVI_RUMBLE_TRIGGERS", UI_MENU_RUMBLE_TRIGGERS, UI_MENU_RUMBLE_TRIGGERS_DESC);
    opt_group(UI_MENU_GROUP_STICKS);
    double now[2];
    bool pad = input_pad_stick_percent(now) != 0;
    char stick_note[2][64];
    for (int s = 0; s < 2; s++)
        snprintf(stick_note[s], sizeof stick_note[s], menu_text(UI_MENU_STICK_DEADZONE_NOW), now[s]);
    opt_slider("NX2NAVI_LSTICK_DEADZONE", UI_MENU_LSTICK_DEADZONE, 0, 50, UI_MENU_STICK_DEADZONE_DESC, 1, "%.0f %%",
               pad ? stick_note[0] : nullptr, "NX2NAVI_STICK_DEADZONE");
    opt_slider("NX2NAVI_RSTICK_DEADZONE", UI_MENU_RSTICK_DEADZONE, 0, 50, UI_MENU_STICK_DEADZONE_DESC, 1, "%.0f %%",
               pad ? stick_note[1] : nullptr, "NX2NAVI_STICK_DEADZONE");
    opt_slider("NX2NAVI_STICK_SENS", UI_MENU_STICK_SENS, 0.1f, 3, UI_MENU_STICK_SENS_DESC, 0.05f, "%.2f");
    // The keyboard's sticks are all or nothing: the walk key scales them to the walk scale while it is held
    // (input.c walk_get)
    row_walk_key();
    opt_slider("NX2NAVI_WALK_SCALE", UI_MENU_WALK_SCALE, 0.2f, 0.9f, UI_MENU_WALK_SCALE_DESC, 0.01f, "%.2f");
    binds_rows(scale);
    opt_end();
    binds_after_page();
}

// The Co-op tab (ui/coop_page.cpp): the rows, then the window a row opens.
static void tab_coop(float scale) {
    opt_begin("##coop", scale);
    coop_rows(scale);
    opt_end();
    coop_after_page(scale);
}

static const UiTextId kTabNames[TAB_COUNT] = {UI_MENU_TAB_SCREEN, UI_MENU_TAB_GAME, UI_MENU_TAB_SOUND,
                                              UI_MENU_TAB_CONTROLS, UI_MENU_TAB_COOP};
static int g_tab;               // the tab shown

// The tabs the menu shows: the Co-op tab is a beta, shown only with coop_beta = 1 in nx2navi.ini (NX2NAVI_COOP_BETA),
// which no menu row sets. Co-op itself is untouched by it: coop = 1 still runs a session (docs/coop.md).
static int tabs_shown(void) { return cfg_bool("NX2NAVI_COOP_BETA") ? TAB_COUNT : TAB_COOP; }
static int g_tab_request = -1;  // select this tab at the next layout; -1 none
static int g_focus_page = -1;   // this tab's first row takes the cursor once it is shown; -1 none

// A row of another tab: that tab opens with the row under the cursor.
static void go_to(int tab, const char *key) {
    g_tab_request = tab;
    opt_focus_key(key);
}

// The game's interface sounds (menu_sound.c) for what the frame did: a press or click on an item, the cursor on another
// item, another tab by a click (L, R, Q and E play theirs where they are read: g_tab_sounded). The first frame after
// opening only records: the open sound has played, and the cursor going to the first row is part of it. g_sound_fresh
// marks that frame, as the frame count alone does not when nothing was laid out while the menu was closed.
static bool g_sound_fresh;
static int g_tab_sounded = -1;   // the tab L / R / Q / E asked for, its sound played

static void menu_sounds_frame(void) {
    ImGuiContext &g = *GImGui;
    static int last_frame = -2, last_tab = -1;
    static ImGuiID last_nav;
    bool fresh = g_sound_fresh || g.FrameCount != last_frame + 1;
    g_sound_fresh = false;
    last_frame = g.FrameCount;
    if (!fresh) {
        if (g.ActiveId && g.ActiveIdIsJustActivated) menu_sound_play(MENU_SOUND_DECIDE);
        else if (g.NavId && g.NavId != last_nav) menu_sound_play(MENU_SOUND_CURSOR);
        if (g_tab != last_tab && last_tab >= 0 && g_tab != g_tab_sounded)
            menu_sound_play(g_tab > last_tab ? MENU_SOUND_TAB_RIGHT : MENU_SOUND_TAB_LEFT);
    }
    if (g_tab != last_tab) g_tab_sounded = -1;
    last_nav = g.NavId, last_tab = g_tab;
}

static void layout_menu(uint32_t w, uint32_t h, float scale) {
    ImGui::GetBackgroundDrawList()->AddRectFilled(ImVec2(0, 0), ImVec2((float)w, (float)h), IM_COL32(10, 6, 2, 150));
    float ww = fminf(w * 0.88f, 1400 * scale), wh = fminf(h * 0.86f, 820 * scale);
    ImGui::SetNextWindowPos(ImVec2(w * 0.5f, h * 0.5f), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(ww, wh), ImGuiCond_Always);
    ImTextureID frame_tex;
    const bool framed = menu_art(MENU_ART_FRAME_DECO, &frame_tex, nullptr, nullptr);
    if (framed) ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);   // the game's frame is the border
    ImGui::Begin("##menu", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                                        ImGuiWindowFlags_NoSavedSettings);
    if (framed) ImGui::PopStyleVar();
    const ImVec2 win_p = ImGui::GetWindowPos(), win_s = ImGui::GetWindowSize();
    // The game's ornamented frame (ui/menu_art.cpp; nothing without the RomFS's), in gold. Its lines lie inside the
    // texture (about 4 % of the width, 7 % of the height in), so the frame is drawn that far outside the window and its
    // lines run along the window's edge, clear of the title and the rows; unclipped, it may reach past the window.
    if (framed) {
        ImDrawList *fdl = ImGui::GetWindowDrawList();
        const float ox = 15 * scale, oy = 30 * scale;
        fdl->PushClipRectFullScreen();
        // the art holds the top and left edges with their corners; turned half a turn it gives the bottom and right ones
        for (int turned = 0; turned < 2; turned++)
            menu_art_draw9_turned(fdl, MENU_ART_FRAME_DECO, ImVec2(win_p.x - ox, win_p.y - oy),
                                  ImVec2(win_p.x + win_s.x + ox, win_p.y + win_s.y + oy), 96 * scale, 0.2f,
                                  IM_COL32(236, 196, 112, 235), 0, turned);
        fdl->PopClipRect();
    }
    if (g_title_font) ImGui::PushFont(g_title_font);
    {   // the Triforce, then "Ocarina of Time" and the page title
        ImVec2 at = ImGui::GetCursorScreenPos();
        float th = ImGui::GetTextLineHeight();
        triforce(ImGui::GetWindowDrawList(), at.x + th * 0.6f, at.y + th * 0.02f, th * 0.96f, ImGui::GetColorU32(kGold));
        ImGui::SetCursorScreenPos(ImVec2(at.x + th * 1.5f, at.y));
        ImGui::TextColored(kGold, "Ocarina of Time");
        ImGui::SameLine();
        ImGui::TextDisabled("|");
        ImGui::SameLine();
        ImGui::TextUnformatted(menu_text(UI_MENU_TITLE));
    }
    const float title_end = ImGui::GetItemRectMax().x;
    if (g_title_font) ImGui::PopFont();
    // the port version and the game update, right-aligned on the title line: what a bug report needs first; left out
    // where it would run into the title (a narrow window, a long title)
    char game[16], version[96];
    snprintf(game, sizeof game, "%d.%d.%d", rt_game_version / 100, rt_game_version / 10 % 10, rt_game_version % 10);
    snprintf(version, sizeof version, "NX2navi %s  -  game %s", rt_port_version, game);
    float version_x = win_s.x - ImGui::GetStyle().WindowPadding.x - ImGui::CalcTextSize(version).x;
    if (win_p.x + version_x >= title_end + ImGui::GetStyle().ItemSpacing.x * 2) {
        ImGui::SameLine(version_x);
        ImGui::TextDisabled("%s", version);
    }

    // a text field takes Q and E as letters, and a window of a page keeps the tab keys
    const bool keeps_keys = g_modal || binds_modal() || coop_modal() || ImGui::GetIO().WantTextInput;
    int step = keeps_keys                                                                                ? 0
             : ImGui::IsKeyPressed(ImGuiKey_GamepadR1, false) || ImGui::IsKeyPressed(ImGuiKey_E, false) ? 1
             : ImGui::IsKeyPressed(ImGuiKey_GamepadL1, false) || ImGui::IsKeyPressed(ImGuiKey_Q, false) ? -1
                                                                                                         : 0;
    const int ntabs = tabs_shown();
    if (g_tab >= ntabs) g_tab_request = 0;   // the Co-op tab was shown and coop_beta went off
    if (step) g_tab_request = g_tab_sounded = (g_tab + step + ntabs) % ntabs;
    if (step) menu_sound_play(step > 0 ? MENU_SOUND_TAB_RIGHT : MENU_SOUND_TAB_LEFT);
    if (g_tab_request >= 0) g_focus_page = g_tab_request;   // the new page's first row takes the cursor

    const ImGuiStyle &st = ImGui::GetStyle();
    float bottom = ImGui::GetFrameHeight() + st.ItemSpacing.y * 2;
    if (ImGui::BeginTabBar("##tabs", ImGuiTabBarFlags_NoTabListScrollingButtons)) {
        for (int k = 0; k < ntabs; k++) {
            char label[128];
            snprintf(label, sizeof label, "  %s  ###tab%d", menu_text(kTabNames[k]), k);   // the id stays per language
            // the tab labels in the title face, on the rows' button (ui/menu_skin.cpp)
            bool shown = skin_tab_item(label, menu_text(kTabNames[k]), g_tab_request == k ? ImGuiTabItemFlags_SetSelected : 0, scale, g_title_font);
            if (!shown) continue;
            g_tab = k;
            if (g_focus_page == k) {
                opt_focus_first();
                g_focus_page = -1;
            }
            ImGui::Spacing();
            // the page's rows are a NavFlattened child of this window itself: ImGui clips the items of a direct child
            // only, so behind a second child level the rows scrolled out above the tabs won D-pad moves from the
            // buttons and the tabs (up from "Continue" went to the tabs, down from the tabs to "Continue")
            opt_reserve_bottom(bottom);
            switch (k) {
            case TAB_SCREEN: tab_screen(scale); break;
            case TAB_GAME: tab_game(scale); break;
            case TAB_SOUND: tab_sound(scale); break;
            case TAB_CONTROLS: tab_controls(scale); break;
            case TAB_COOP: tab_coop(scale); break;
            }
            ImGui::EndTabItem();
        }
        g_tab_request = -1;
        ImGui::EndTabBar();
    }

    // bottom row: the save status on the left, the two buttons on the right, on the game's control-guide bar
    ImGui::Spacing();
    float y = ImGui::GetCursorPosY();
    {
        float sy = win_p.y + y, fh = ImGui::GetFrameHeight();
        // within the bottom row's own height: above it the page's last row may still show
        menu_art_draw(ImGui::GetWindowDrawList(), MENU_ART_GUIDE_BAR, ImVec2(win_p.x + st.WindowPadding.x, sy),
                      ImVec2(win_p.x + win_s.x - st.WindowPadding.x, sy + fh), IM_COL32(96, 70, 36, 170));
    }
    if (!cfg_last_save_ok()) {   // stays until a save succeeds; the settings hold for the session
        ImGui::SetCursorPosY(y + st.FramePadding.y);
        ImGui::TextColored(ImVec4(0.95f, 0.45f, 0.35f, 1), "%s", menu_text(UI_MENU_SAVE_FAILED));
    }
    // Restart says how many settings wait for it (restart_pending); "###restart" keeps the button's id, and so the cursor
    // on it, when the count changes
    char shown[80], restart[96];
    const int pending = restart_pending();
    if (pending) snprintf(shown, sizeof shown, menu_text(UI_MENU_RESTART_COUNT), pending);
    else snprintf(shown, sizeof shown, "%s", menu_text(UI_MENU_RESTART_YES));
    snprintf(restart, sizeof restart, "%s###restart", shown);
    const char *cont = menu_text(UI_MENU_CONTINUE), *quit = menu_text(UI_MENU_QUIT);
    float bw = fmaxf(fmaxf(ImGui::CalcTextSize(cont).x, ImGui::CalcTextSize(shown).x), ImGui::CalcTextSize(quit).x) + st.FramePadding.x * 4;
    // the save menu's brown buttons, with the game's cursor on the selected one (ui/menu_skin.cpp)
    float gap = st.ItemSpacing.x + skin_cursor_reach(scale);
    // right-aligned; in a window too narrow for all three, from the left edge
    const float buttons_x = fmaxf(st.WindowPadding.x, ImGui::GetWindowWidth() - st.WindowPadding.x - bw * 3 - gap * 2);
    // Lines of nx2navi.ini that are no settings of this version do nothing (cfg.c cfg_unknown_keys). Said here, in the
    // save status's place while saving works, so that a line added by hand is not taken for the cause of what the game
    // does: a player took such a line for the fix of a crash. The list is cut to the room left of the buttons.
    const char *unknown = cfg_unknown_keys(NULL);
    if (cfg_last_save_ok() && *unknown) {
        char note[512];
        snprintf(note, sizeof note, menu_text(UI_MENU_INI_UNKNOWN), unknown);
        size_t keep = strlen(note), head = keep > strlen(unknown) ? keep - strlen(unknown) : keep;
        const float room = buttons_x - st.WindowPadding.x - gap;
        while (keep > head && ImGui::CalcTextSize(note).x > room) {
            keep--;
            snprintf(note + keep, sizeof note - keep, "...");
        }
        if (keep > head) {
            ImGui::SetCursorPos(ImVec2(st.WindowPadding.x, y + st.FramePadding.y));
            ImGui::TextColored(ImVec4(0.90f, 0.80f, 0.50f, 1), "%s", note);
        }
    }
    ImGui::SetCursorPos(ImVec2(buttons_x, y));
    if (skin_widget_button(cont, ImVec2(bw, 0), scale)) set_open(0);
    ImGui::SameLine(0, gap);
    if (skin_widget_button(restart, ImVec2(bw, 0), scale)) g_confirm_quit = false, ImGui::OpenPopup("##confirm");
    ImGui::SameLine(0, gap);
    if (skin_widget_button(quit, ImVec2(bw, 0), scale)) g_confirm_quit = true, ImGui::OpenPopup("##confirm");
    restart_confirm(scale);
    menu_sounds_frame();
    bool was_modal = g_modal || binds_modal() || coop_modal() || ImGui::GetIO().WantTextInput;
    if (!was_modal && (ImGui::IsKeyPressed(ImGuiKey_Escape, false) || ImGui::IsKeyPressed(ImGuiKey_GamepadFaceRight, false)))
        set_open(0);
    ImGui::End();
}

static int g_save_pending;   // a changed setting still waits for its write to nx2navi.ini (opt_save_tick)

// ---- the performance overlay (NX2NAVI_PERF_OVERLAY), in the top right corner. "fps": the frame rate over the presents of
// the last half second. "full": a panel of perfmon.c's numbers:
// the machine's load and temperatures, the frame rate and frame times with a graph, the render thread's waits.
// Off, it costs a cfg_version load per present.
static unsigned g_perf_ver = ~0u;
static int g_perf_mode;                  // 0 off, 1 fps, 2 full
static std::atomic<int> g_perf_save;     // F10 changed the mode: nx2navi.ini is written on the present thread
static uint64_t g_perf_last_ns, g_perf_win_ns, g_perf_max_ns;
static int g_perf_frames;
static char g_perf_text[96];

static int perf_mode_of(const char *s) { return !strcmp(s, "full") ? 2 : !strcmp(s, "fps") ? 1 : 0; }

extern "C" void menu_perf_cycle(void) {
    static const char *const modes[] = {"off", "fps", "full"};
    cfg_set("NX2NAVI_PERF_OVERLAY", modes[(perf_mode_of(cfg_value("NX2NAVI_PERF_OVERLAY")) + 1) % 3]);
    g_perf_save.store(1);
}

// The device the game renders on, for perfmon's GPU counters (by LUID), NVML (by name) and ADL (by PCI bus, else
// name); once, before full mode.
static void perf_gpu_once(const MenuTarget *t) {
    static bool done;
    if (done) return;
    done = true;
    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(t->phys, NULL, &ne, NULL);
    VkExtensionProperties *ep = (VkExtensionProperties *)calloc(ne ? ne : 1, sizeof *ep);
    bool has_bus = false;
    if (ep && vkEnumerateDeviceExtensionProperties(t->phys, NULL, &ne, ep) >= 0)
        for (uint32_t i = 0; i < ne && !has_bus; i++) has_bus = !strcmp(ep[i].extensionName, VK_EXT_PCI_BUS_INFO_EXTENSION_NAME);
    free(ep);
    VkPhysicalDevicePCIBusInfoPropertiesEXT bus = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT};
    VkPhysicalDeviceIDProperties id = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES, has_bus ? &bus : nullptr};
    VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id};
    vkGetPhysicalDeviceProperties2(t->phys, &p2);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(t->phys, &mp);
    uint64_t vram = 0;
    for (uint32_t h = 0; h < mp.memoryHeapCount; h++)
        if (mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) vram += mp.memoryHeaps[h].size;
    perfmon_set_gpu(p2.properties.deviceName, id.deviceLUIDValid ? id.deviceLUID : NULL, has_bus ? (int)bus.pciBus : -1,
                    (float)(vram >> 20));
}

static void perf_tick(uint64_t now, const MenuTarget *t) {
    if (cfg_version() != g_perf_ver) {
        g_perf_ver = cfg_version();
        int m = perf_mode_of(cfg_value("NX2NAVI_PERF_OVERLAY"));
        if (m != g_perf_mode) {
            g_perf_last_ns = 0, g_perf_text[0] = 0;
            if (m == 2) perf_gpu_once(t);
            perfmon_mode(m);
        }
        g_perf_mode = m;
    }
    if (g_perf_mode != 1) return;
    uint64_t d = now - g_perf_last_ns;
    if (!g_perf_last_ns || d > 2000000000ull) {   // first frame, or a stall the window should not average in
        g_perf_win_ns = g_perf_max_ns = 0;
        g_perf_frames = 0;
    } else {
        g_perf_win_ns += d;
        g_perf_frames++;
        if (d > g_perf_max_ns) g_perf_max_ns = d;
        if (g_perf_win_ns >= 500000000ull) {
            double out = gfx_output_fps();   // with frame generation: the display's rate too
            if (out > 0)
                snprintf(g_perf_text, sizeof g_perf_text, menu_text(UI_PERF_LINE_FPS_FG), g_perf_frames * 1e9 / g_perf_win_ns, out);
            else
                snprintf(g_perf_text, sizeof g_perf_text, menu_text(UI_PERF_LINE_FPS), g_perf_frames * 1e9 / g_perf_win_ns);
            g_perf_win_ns = g_perf_max_ns = 0;
            g_perf_frames = 0;
        }
    }
    g_perf_last_ns = now;
}

// The picture lines: the window, the game image the present shows and the 3D scene it holds (resolution.c), then the
// settings as the menu's rows name them, so the overlay and the menu agree.
static void perf_picture(const MenuTarget *t, PerfPanel *p) {
    unsigned out[2], scene[2];
    res_sizes(out, scene);
    char (*v)[64] = p->picture;
    snprintf(v[0], sizeof v[0], menu_text(UI_MENU_OV_FMT_SIZE), t->width, t->height);
    snprintf(v[1], sizeof v[1], menu_text(UI_MENU_OV_FMT_SIZE), (unsigned)t->image_w, (unsigned)t->image_h);
    snprintf(v[2], sizeof v[2], menu_text(UI_MENU_OV_FMT_SCENE), scene[0], scene[1], scene[0] * 100.0 / out[0]);
    upscaling_name(v[3], sizeof v[3]);
    aa_name(v[4], sizeof v[4]);
    double hz = frame_step_limit_hz();
    snprintf(v[5], sizeof v[5], "%s", menu_text(UI_MENU_OV_NONE));
    if (hz > 0) snprintf(v[5], sizeof v[5], menu_text(UI_MENU_OV_FPS), hz);
    snprintf(v[6], sizeof v[6], "%s", menu_text(strcmp(cfg_value("NX2NAVI_VK_PRESENT"), "immediate") ? UI_MENU_ON : UI_MENU_OFF));
    v[7][0] = v[8][0] = 0;
    if (wide_available()) {
        const char *a = cfg_value("NX2NAVI_GAME_ASPECT");
        snprintf(v[7], sizeof v[7], "%s", menu_text(!strcmp(a, "window") ? UI_MENU_GAME_ASPECT_WINDOW : !strcmp(a, "21:9") ? UI_MENU_GAME_ASPECT_21_9
                                                    : !strcmp(a, "32:9") ? UI_MENU_GAME_ASPECT_32_9 : UI_MENU_GAME_ASPECT_16_9));
        const char *r = cfg_value("NX2NAVI_GAME_RES");
        if (!strcmp(r, "window")) snprintf(v[8], sizeof v[8], "%s", menu_text(UI_MENU_GAME_RES_NATIVE));
        else snprintf(v[8], sizeof v[8], "%sp", r);
    }
    const char *hdr = cfg_value("NX2NAVI_HDR");
    snprintf(v[9], sizeof v[9], "%s", menu_text(!strcmp(hdr, "native") ? UI_MENU_HDR_NATIVE : cfg_bool("NX2NAVI_HDR") ? UI_MENU_HDR_TONEMAPPED : UI_MENU_FRAMEGEN_OFF));
    const char *fg = cfg_value("NX2NAVI_FRAMEGEN");
    const bool fsr_fg = !strcmp(fg, "fsr3") && vk_ffx_framegen_available();   // set but not loadable: off
    snprintf(v[10], sizeof v[10], "%s", fsr_fg ? (vk_fg4_active() ? fg4_label() : "AMD FSR 3.1")
                                      : !strcmp(fg, "dlss") ? "NVIDIA DLSS" : menu_text(UI_MENU_FRAMEGEN_OFF));
    if (!strcmp(fg, "dlss") && vk_dlssg_active())   // its multiplier as set, clamped to what this GPU allows
        snprintf(v[10], sizeof v[10], "NVIDIA DLSS %dx",
                 vk_dlssg_multiplier_of(cfg_value("NX2NAVI_FRAMEGEN_MULTIPLIER"), vk_dlssg_multiplier_max(nullptr), nullptr));
    // The 3D scene's size is known only where resolution.c sets it (the builds with its patch sets); Ocarina sizes its
    // own scene dynamically and res_sizes would report a fixed 1600x900, so the line is left out
    p->scene_known = res_adjustable() != 0;
}

// The full panel (ui/perf_panel.cpp draws it in the menu's look): the numbers of this frame.
static void perf_full_draw(const MenuTarget *t, float scale) {
    static PerfPanel p;
    perfmon_snapshot(&p.s);
    // the port version and the game update, formatted once per menu language
    static char ver[96];
    static const char *ver_fmt;
    if (ver_fmt != menu_text(UI_MENU_VERSION)) {
        ver_fmt = menu_text(UI_MENU_VERSION);
        char game[16];
        snprintf(game, sizeof game, "%d.%d.%d", rt_game_version / 100, rt_game_version / 10 % 10, rt_game_version % 10);
        snprintf(ver, sizeof ver, ver_fmt, rt_port_version, game);
    }
    p.version = ver;
    perf_picture(t, &p);
    p.fg_fps = gfx_output_fps() > 0 ? (float)gfx_output_fps() : 0;
    p.under_menu = g_open.load(std::memory_order_relaxed) != 0;
    perf_panel_draw(&p, (float)t->width, scale);
}

static void perf_draw(const MenuTarget *t, float scale) {
    if (g_perf_mode == 2) return perf_full_draw(t, scale);
    float w = (float)t->width;
    if (!g_perf_mode || !g_perf_text[0]) return;
    // with the menu open: under its window, over the dimming layout_menu drew before
    ImDrawList *dl = g_open.load(std::memory_order_relaxed) ? ImGui::GetBackgroundDrawList() : ImGui::GetForegroundDrawList();
    ImVec2 sz = ImGui::CalcTextSize(g_perf_text);
    float padx = 14 * scale, pady = 9 * scale;
    ImVec2 p(w - 12 * scale - sz.x - padx * 2, 12 * scale);   // the top right corner: the game's HUD leaves it empty
    // a small panel in the menu's look (ui/menu_skin.cpp), its frame scaled to the line
    skin_panel(dl, p, ImVec2(p.x + sz.x + padx * 2, p.y + sz.y + pady * 2), scale * 0.6f, 0.9f);
    dl->AddText(ImVec2(p.x + padx, p.y + pady), SKIN_TEXT, g_perf_text);
}

static int overlay_frame(uint64_t frame, const MenuTarget *t, const char **shot) {
    *shot = NULL;
    if (t->phys && t->phys != g_gpu_vendor_of) {   // the GPU's vendor for the rows that depend on it (dlss_offered)
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(t->phys, &props);
        g_gpu_vendor.store(props.vendorID, std::memory_order_relaxed);
        g_gpu_vendor_of = t->phys;
    }
    menu_art_start();      // the game's interface textures decode on a worker from the first present on
    cheats_frame(frame);   // the data cheats and the buttons of the cheats that need them held
    if (!g_test.parsed) test_parse();
    if (g_test.next_set < g_test.n_sets && frame >= g_test.sets[g_test.next_set].frame) test_apply_sets(frame);
    if (g_perf_save.exchange(0)) cfg_save();
    perf_tick(plat_now_ns(), t);
    amiibo_tick();
    const bool toast = coop_toast_tick();   // a co-op player came or went: a toast over the game, the menu closed
    if (g_booting.load(std::memory_order_relaxed)) g_booting.store(0, std::memory_order_release);   // the game runs
    bool open = g_open.load(std::memory_order_relaxed);
    // the frame the menu is first laid out after opening: also when its g_reset went to an earlier frame of the overlay
    // alone (the menu opened between the two loads), the pad's buttons held now must not count as presses
    static bool was_open;
    bool opened = open && !was_open;
    was_open = open;
    if (!open) {
        if (g_save_pending) g_save_pending = opt_save_tick(1);   // closed: the last change goes to nx2navi.ini now
        if (g_test.next_at < g_test.n_at && !g_test.opened && frame >= g_test.at[g_test.next_at]) {
            g_test.opened = frame;
            g_test.page = 0;
            g_tab_request = 0;
            g_debug.store(cfg_get("NX2NAVI_MENU_TEST_DEBUG") != NULL);   // the test shoots the debug menu's tabs instead
            if (g_debug.load()) debug_menu_tab(0);
            set_open(1);
            open = true;
        } else if (!g_perf_mode && !toast) {
            return 0;
        }
    }
    if (g_failed_flag()) {
        set_open(0);
        return 0;
    }
    uint64_t t0 = plat_now_ns();
    if (!g_ctx_ok) make_context();
    float scale = menu_scale(t->width, t->height);
    if (!vk_up(t) || !fonts_for(scale, t0)) {
        rt_log("menu: Dear ImGui Vulkan back end set-up failed; the menu stays closed");
        g_failed.store(1);
        set_open(0);
        return 0;
    }
    if (g_style_scale != scale) {
        ImGui::GetStyle() = g_base_style;
        ImGui::GetStyle().ScaleAllSizes(scale);
        g_style_scale = scale;
    }
    list_gpus(t->instance);
    ImGuiIO &io = ImGui::GetIO();
    int reset = g_reset.exchange(0);
    // Read again after taking the reset: the menu may have opened since the read above (F1 on the window thread while
    // the overlay alone was drawn), and that frame's reset would be spent on an overlay frame, so the first open frame
    // ran without its focus row and with the opening key still down.
    open = g_open.load(std::memory_order_acquire);
    if (reset) {
        io.ClearEventsQueue();   // events handed over before the close and not yet taken (ImGui spreads them over frames)
        io.ClearInputKeys();
        io.ClearInputMouse();
        binds_cancel();          // a cell waiting for a press when the menu closed: no press binds on the next opening
        coop_cancel();           // the Co-op tab's window closes and its secret hides
        g_sound_fresh = true;
        g_last_ns = 0;
        g_focus_page = g_tab_request >= 0 ? g_tab_request : g_tab;   // opening: the first row has the cursor
        if (GImGui->OpenPopupStack.Size) ImGui::ClosePopupToLevel(0, false);   // a confirmation left open on closing
        g_modal = false;
    }
    io.DisplaySize = ImVec2((float)t->width, (float)t->height);
    double dt = g_last_ns ? (t0 - g_last_ns) / 1e9 : 1 / 60.0;
    io.DeltaTime = (float)(dt < 1e-4 ? 1e-4 : dt > 0.1 ? 0.1 : dt);
    g_last_ns = t0;
    if (!open) {   // the overlay alone: no input reaches ImGui
    } else if (g_test.page < 0) {
        drain_events(io);
        pad_events(io, reset || opened);
    } else {   // the test hook's pages: the user's input in a background test window must not change settings
        plat_mutex_lock(&ev_lock);
        ev_n = 0;
        plat_mutex_unlock(&ev_lock);
    }
    // no ImGui_ImplVulkan_NewFrame: all it does is upload Dear ImGui's own font atlas when the back end holds none,
    // which it never does here (fonts_up uploads it, ui/menu_art.cpp menu_font_upload)
    ImGui::NewFrame();
    if (open && g_debug.load(std::memory_order_relaxed)) {
        debug_menu_layout(t->width, t->height, scale);
        // Esc, or B on the pad when no field is being typed in, closes it
        if (ImGui::IsKeyPressed(ImGuiKey_Escape, false) || (!io.WantTextInput && ImGui::IsKeyPressed(ImGuiKey_GamepadFaceRight, false)))
            set_open(0);
    } else if (open) {
        layout_menu(t->width, t->height, scale);
    }
    perf_draw(t, scale);
    if (!open && toast) coop_toast_draw((float)t->width, (float)t->height, scale);
    ImGui::Render();
    if (open) g_save_pending = opt_save_tick(0);
    g_cost_ns += plat_now_ns() - t0;
    if (g_test.page >= 0 && frame >= g_test.opened + g_test.per_page * (uint64_t)(g_test.page + 1)) {
        static char path[512];   // this frame's image goes to a file; then the next tab, or the menu closes
        const char *dir = cfg_get("NX2NAVI_MENU_TEST_DIR");
        snprintf(path, sizeof path, "%s/menu-test-%llu-%d.png", dir ? dir : ".", (unsigned long long)frame, g_test.page);
        *shot = path;
        if (++g_test.page == (g_debug.load() ? DEBUG_MENU_TABS : tabs_shown())) {
            g_test.page = -1;
            g_test.opened = 0, g_test.next_at++;   // the next NX2NAVI_MENU_TEST frame opens it again
            set_open(0);   // the next frame lays out nothing
        } else {
            g_tab_request = g_test.page;
            if (g_debug.load()) debug_menu_tab(g_test.page);
        }
    }
    return ImGui::GetDrawData()->TotalVtxCount > 0;
}

// A text field takes typing: the window delivers characters (and a handheld shows its keyboard) while one is active.
// Every overlay frame lets the window follow, also those overlay_frame leaves early because the menu is closed: when
// only the frames the menu laid out did, closing it with a field active (F1, Minus and Plus) left SDL's text input on,
// and on Windows the input method attached to the game window, until the menu opened again.
extern "C" int menu_overlay_frame(uint64_t frame, const MenuTarget *t, const char **shot) {
    int r = overlay_frame(frame, t, shot);
    menu_text_input_follow(g_ctx_ok && g_open.load(std::memory_order_relaxed) && ImGui::GetIO().WantTextInput);
    return r;
}

// ---- the boot screen (boot.c's wait for the prewarm; gfx_present_boot presents it on a black frame)

static std::atomic<int> g_boot_phase;
static std::atomic<long long> g_boot_done, g_boot_total;
static uint64_t g_boot_t0;

extern "C" void menu_boot_set(int phase, long long done, long long total) {
    g_boot_done.store(done, std::memory_order_relaxed);
    g_boot_total.store(total, std::memory_order_relaxed);
    g_boot_phase.store(phase, std::memory_order_release);
}

// A box in the middle of the screen in the menu's look (ui/menu_skin.cpp skin_boot_panel): the title and the step with
// its count and a bar. The input queue is left alone: keys pressed now reach the menu once the game runs.
extern "C" int menu_boot_frame(const MenuTarget *t) {
    if (g_failed_flag() || g_boot_phase.load(std::memory_order_acquire) < 0) return 0;   // -1: done, a black frame
    if (!g_ctx_ok) make_context();
    float scale = menu_scale(t->width, t->height);
    if (!vk_up(t) || !fonts_for(scale, plat_now_ns())) {
        rt_log("menu: Dear ImGui Vulkan back end set-up failed; no boot screen, and the menu stays closed");
        g_failed.store(1);
        return 0;
    }
    menu_art_start();   // the box takes the game's textures as soon as they are decoded, plain shapes until then
    if (g_style_scale != scale) {
        ImGui::GetStyle() = g_base_style;
        ImGui::GetStyle().ScaleAllSizes(scale);
        g_style_scale = scale;
    }
    uint64_t now = plat_now_ns();
    if (!g_boot_t0) g_boot_t0 = now;
    ImGuiIO &io = ImGui::GetIO();
    io.DisplaySize = ImVec2((float)t->width, (float)t->height);
    double dt = g_last_ns ? (now - g_last_ns) / 1e9 : 1 / 60.0;
    io.DeltaTime = (float)(dt < 1e-4 ? 1e-4 : dt > 0.1 ? 0.1 : dt);
    g_last_ns = now;
    // no ImGui_ImplVulkan_NewFrame: all it does is upload Dear ImGui's own font atlas when the back end holds none,
    // which it never does here (fonts_up uploads it, ui/menu_art.cpp menu_font_upload)
    ImGui::NewFrame();

    int phase = g_boot_phase.load(std::memory_order_acquire);
    long long done = g_boot_done.load(std::memory_order_relaxed), total = g_boot_total.load(std::memory_order_relaxed);
    char step[160];
    if (phase == 1 || phase == 2)
        snprintf(step, sizeof step, ui_text(phase == 1 ? UI_BOOT_SHADERS : UI_BOOT_PIPELINES), done, total);
    else
        snprintf(step, sizeof step, "%s", ui_text(UI_BOOT_STARTING));
    const char *title = ui_text(UI_BOOT_TITLE);

    // the box in the menu's look (ui/menu_skin.cpp): a filled bar while there is a count, else a sweeping segment
    ImFont *body = ImGui::GetFont();
    float body_px = ImGui::GetFontSize();
    ImFont *head = g_title_font ? g_title_font : body;
    float head_px = g_title_font ? g_title_font->FontSize * io.FontGlobalScale : body_px * 1.2f;   // as GetFontSize scales
    float frac = -1;
    if ((phase == 1 || phase == 2) && total > 0) frac = done <= 0 ? 0 : done >= total ? 1 : (float)done / (float)total;
    float s = (float)fmod((now - g_boot_t0) / 1e9 * 0.8, 2.0);
    skin_boot_panel(ImGui::GetForegroundDrawList(), head, head_px, body, body_px, title, step, frac, s < 1 ? s : 2 - s,
                    (float)t->width, (float)t->height, scale);
    ImGui::Render();
    return ImGui::GetDrawData()->TotalVtxCount > 0;
}

extern "C" void menu_overlay_draw(VkCommandBuffer cb) {
    uint64_t t0 = plat_now_ns();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cb);
    g_cost_ns += plat_now_ns() - t0;
    if (++g_cost_frames == 600) {
        rt_log("menu: %.3f ms CPU per open frame (layout and recording, mean of 600)", g_cost_ns / 600 / 1e6);
        g_cost_ns = g_cost_frames = 0;
    }
}
