// The SDL3 back end of the window (gfx_wsys.h), for hosts without Win32: the window on a thread of its own that also
// runs SDL's event loop, F11 fullscreen (borderless, the desktop's mode), the settings NX2NAVI_WINDOW_SIZE,
// NX2NAVI_FULLSCREEN and NX2NAVI_MONITOR, the display's refresh rate, focus, the Vulkan surface. The swapchain and the
// present are gfx_present.c's. Keyboard and mouse events go to input_sdl.c and to the menu (ui/menu_sdl.cpp).
//
// SDL wants its video calls on the thread that initialised video, so this thread does them all: the others hand it
// work through run_on_window (a user event and a wait) or post a user event. Video is initialised before the window
// exists: the Vulkan instance asks for SDL's surface extensions first (gfx_window_vk_extensions), the window comes with
// gfx_window_start (wsys_open).
//
// NX2NAVI_WINDOW_SIZE is in pixels. On a display that scales windows (Wayland, HiDPI) the window asks for its pixels in
// the display's units (SDL_GetWindowDisplayScale), and SDL reports mouse positions in those units: input_sdl.c and the
// menu scale them back (gfx_window_sdl_density).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include "cfg.h"
#include "gfx.h"
#include "gfx_wsys.h"
#include "input.h"
#include "input_sdl.h"
#include "ui/menu.h"
#include "vk.h"
#include "platform/platform.h"

static SDL_Window *g_win;
static uint32_t g_ev_base;               // SDL_RegisterEvents: the user events below
enum { EV_RUN, EV_SETTINGS, EV_CLOSE, EV_TEXT_ON, EV_TEXT_OFF, EV_PASTE, EV_COUNT };
static PlatEvent g_video_ready;          // manual-reset: SDL video is up (or failed)
static int g_video_ok;
static PlatOnce g_thread_once = PLAT_ONCE_INIT;
static uint32_t g_window_tid;            // the window thread, which runs the video calls
static volatile int32_t g_density_milli = 1000;   // pixels per window unit, x1000

// ---- work on the window thread
typedef struct {
    void (*fn)(void *arg);
    void *arg;
    PlatEvent done;
} RunReq;

static void run_on_window(void (*fn)(void *arg), void *arg) {
    if (plat_thread_id() == g_window_tid) {
        fn(arg);
        return;
    }
    RunReq r = {fn, arg};
    plat_event_init(&r.done, 1, 0);
    SDL_Event e = {.user = {.type = g_ev_base + EV_RUN, .data1 = &r}};
    if (!SDL_PushEvent(&e)) rt_fatal("gfx: cannot reach the window thread (%s)", SDL_GetError());
    plat_event_wait(&r.done);
}

static void post(int ev) {
    SDL_Event e = {.user = {.type = g_ev_base + (uint32_t)ev}};
    SDL_PushEvent(&e);
}

double gfx_window_sdl_density(void) { return __atomic_load_n(&g_density_milli, __ATOMIC_RELAXED) / 1000.0; }

SDL_Window *gfx_window_sdl(void) { return g_win; }

void gfx_window_sdl_run(void (*fn)(void *arg), void *arg) { run_on_window(fn, arg); }

// ---- settings: size and fullscreen
static char g_applied_size[32];
static int g_applied_fs;

static void note_applied(void) {
    snprintf(g_applied_size, sizeof g_applied_size, "%s", cfg_value("NX2NAVI_WINDOW_SIZE"));
    g_applied_fs = gfx_cfg_fullscreen();
}

static int is_fullscreen(void) { return (SDL_GetWindowFlags(g_win) & SDL_WINDOW_FULLSCREEN) != 0; }

// The display scale as a DPI, 96 at 100% (the menu's text size floor); 96 before the window exists.
unsigned gfx_window_dpi(void) {
    float scale = g_win ? SDL_GetWindowDisplayScale(g_win) : 0;
    return scale > 0 ? (unsigned)(scale * 96 + 0.5f) : 96;
}

// The window's size in its units for w x h pixels.
static void set_pixel_size(int w, int h) {
    float scale = SDL_GetWindowDisplayScale(g_win);
    if (!(scale > 0)) scale = 1;
    SDL_SetWindowSize(g_win, (int)(w / scale + 0.5f), (int)(h / scale + 0.5f));
}

static void set_fullscreen(int on) {
    input_cursor_suspend(1);
    SDL_SetWindowFullscreen(g_win, on);   // the desktop's mode: borderless over the display
    SDL_SyncWindow(g_win);
    input_cursor_suspend(0);
}

// Window thread: the window takes the size and fullscreen state of the settings when they changed. A window resized by
// hand keeps its size when another setting changes.
static void apply_settings(void) {
    int size_changed = strcmp(cfg_value("NX2NAVI_WINDOW_SIZE"), g_applied_size) != 0, fs = gfx_cfg_fullscreen();
    int fs_changed = fs != g_applied_fs;
    note_applied();
    if (size_changed) {
        int w, h;
        gfx_cfg_window_size(&w, &h);
        if (is_fullscreen()) SDL_SetWindowFullscreen(g_win, false);
        set_pixel_size(w, h);
        if (is_fullscreen() || (fs && !fs_changed)) SDL_SetWindowFullscreen(g_win, true);
    }
    if (fs_changed && fs != is_fullscreen()) set_fullscreen(fs);
}

void gfx_window_apply_settings(void) {
    if (g_win) post(EV_SETTINGS);
}

void gfx_window_request_close(void) {
    if (g_win) post(EV_CLOSE);
}

// SDL sends SDL_EVENT_TEXT_INPUT between SDL_StartTextInput and SDL_StopTextInput only (and shows an on-screen keyboard
// there where the system has one, as Steam's on a handheld); the window thread makes the calls.
void gfx_window_text_input(int on) {
    if (g_win) post(on ? EV_TEXT_ON : EV_TEXT_OFF);
}

// SDL's clipboard belongs to the window thread: the text is read there and handed to the menu.
void gfx_window_paste(void) {
    if (g_win) post(EV_PASTE);
}

// ---- the display's refresh rate
static void refresh_rate_update(void) {
    const SDL_DisplayMode *m = SDL_GetCurrentDisplayMode(SDL_GetDisplayForWindow(g_win));
    double hz = 0;
    if (m && m->refresh_rate_denominator) hz = (double)m->refresh_rate_numerator / m->refresh_rate_denominator;
    else if (m) hz = m->refresh_rate;
    gfx_window_set_refresh(hz);
}

static void density_update(void) {
    float d = SDL_GetWindowPixelDensity(g_win);
    __atomic_store_n(&g_density_milli, (int32_t)((d > 0 ? d : 1) * 1000 + 0.5f), __ATOMIC_RELAXED);
}

// ---- the window thread
// NX2NAVI_MONITOR's display while a Wayland compositor has yet to place the fullscreen window (open_window): SDL learns
// the output a Wayland window is on later, with a display, scale or mode change (Hyprland sends only the latter two).
static SDL_DisplayID g_place_pending;

static void report_placement(SDL_DisplayID want) {
    SDL_DisplayID at = SDL_GetDisplayForWindow(g_win);
    const char *name = SDL_GetDisplayName(at);
    if (at == want) rt_log("gfx: the window is on the chosen monitor \"%s\"", name ? name : "an unknown monitor");
    else rt_log("gfx: the window is not on the chosen monitor; it is on \"%s\"", name ? name : "an unknown monitor");
}

static void handle(const SDL_Event *e) {
    if (e->type == g_ev_base + EV_RUN) {
        RunReq *r = e->user.data1;
        r->fn(r->arg);
        plat_event_set(&r->done);
        return;
    }
    if (e->type == g_ev_base + EV_SETTINGS) {
        if (g_win) apply_settings();
        return;
    }
    if (e->type == g_ev_base + EV_TEXT_ON || e->type == g_ev_base + EV_TEXT_OFF) {
        if (g_win && e->type == g_ev_base + EV_TEXT_ON) SDL_StartTextInput(g_win);
        else if (g_win) SDL_StopTextInput(g_win);
        return;
    }
    if (e->type == g_ev_base + EV_PASTE) {
        char text[256];
        plat_clipboard_get(text, sizeof text);
        menu_event_paste(text);
        return;
    }
    if (e->type == g_ev_base + EV_CLOSE) gfx_window_closed();
    if (!g_win) return;
    input_sdl_event(e);
    menu_sdl_event(e);
    switch (e->type) {
    case SDL_EVENT_KEY_DOWN:
        // not with a modifier held: overlays and capture tools take Ctrl, Shift or Alt+F11 for their own
        if (e->key.scancode == SDL_SCANCODE_F11 && !e->key.repeat &&
            !(e->key.mod & (SDL_KMOD_CTRL | SDL_KMOD_SHIFT | SDL_KMOD_ALT | SDL_KMOD_GUI))) {
            set_fullscreen(!is_fullscreen());
            cfg_set("NX2NAVI_FULLSCREEN", is_fullscreen() ? "1" : "0");   // the menu shows it; the next start keeps it
            cfg_save();
            note_applied();
        }
        break;
    case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
    case SDL_EVENT_WINDOW_RESIZED:
    case SDL_EVENT_WINDOW_RESTORED:
        density_update();
        gfx_window_mark_dirty();
        break;
    case SDL_EVENT_WINDOW_DISPLAY_CHANGED:
    case SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED:
    case SDL_EVENT_DISPLAY_CURRENT_MODE_CHANGED:
        density_update();
        refresh_rate_update();
        gfx_window_mark_dirty();
        if (g_place_pending && SDL_GetDisplayForWindow(g_win) == g_place_pending) {
            report_placement(g_place_pending);
            g_place_pending = 0;
        }
        break;
    case SDL_EVENT_WINDOW_FOCUS_GAINED:
    case SDL_EVENT_WINDOW_FOCUS_LOST:
        gfx_window_focus_changed(e->type == SDL_EVENT_WINDOW_FOCUS_GAINED);
        break;
    case SDL_EVENT_QUIT:
    case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
        gfx_window_closed();
        break;
    }
}

static void window_main(void *arg) {
    (void)arg;
    g_window_tid = plat_thread_id();
    SDL_SetHint(SDL_HINT_APP_ID, "nx2navi");   // Wayland's app id: the desktop file and the taskbar match it
    SDL_SetHint(SDL_HINT_VIDEO_ALLOW_SCREENSAVER, "0");
    g_video_ok = SDL_Init(SDL_INIT_VIDEO);
    if (!g_video_ok) rt_log("gfx: SDL video init failed (%s)", SDL_GetError());
    // SDL's Vulkan queries need its Vulkan loader, which a window would load at creation: the instance and the device
    // are made before the window.
    else if (!SDL_Vulkan_LoadLibrary(NULL)) {
        rt_log("gfx: SDL cannot load Vulkan (%s)", SDL_GetError());
        g_video_ok = 0;
    } else {
        g_ev_base = SDL_RegisterEvents(EV_COUNT);
        rt_log("gfx: SDL %d.%d.%d, video driver %s", SDL_VERSIONNUM_MAJOR(SDL_GetVersion()),
               SDL_VERSIONNUM_MINOR(SDL_GetVersion()), SDL_VERSIONNUM_MICRO(SDL_GetVersion()), SDL_GetCurrentVideoDriver());
    }
    plat_event_set(&g_video_ready);
    if (!g_video_ok) return;
    for (;;) {
        SDL_Event e;
        if (SDL_WaitEventTimeout(&e, 100)) handle(&e);
        else input_sdl_idle();   // a cursor change requested from another thread meanwhile
    }
}

static void video_start(void *arg) {
    (void)arg;
    plat_event_init(&g_video_ready, 1, 0);
    if (!plat_thread_spawn(window_main, NULL, "window", 0)) rt_fatal("gfx: cannot start the window thread");
    plat_event_wait(&g_video_ready);
}

static int video_up(void) {
    plat_once(&g_thread_once, video_start, NULL);
    return g_video_ok;
}

int gfx_window_vk_extensions(const char **out, int max) {
    if (!video_up()) return 0;
    Uint32 n = 0;
    const char *const *ext = SDL_Vulkan_GetInstanceExtensions(&n);
    if (!ext) {
        rt_log("gfx: SDL_Vulkan_GetInstanceExtensions failed (%s)", SDL_GetError());
        return 0;
    }
    int k = 0;
    for (Uint32 i = 0; i < n && k < max; i++) out[k++] = ext[i];
    return k;
}

int gfx_window_vk_presentable(VkInstance inst, VkPhysicalDevice pd, uint32_t family) {
    return video_up() && SDL_Vulkan_GetPresentationSupport(inst, pd, family);
}

// ---- monitor choice (NX2NAVI_MONITOR, --monitor, --list-monitors)
// The monitors in SDL_GetDisplays order, which is the index the setting counts in (1-based) and --list-monitors prints.
// The video driver sets that order, which need not match the desktop's own list of monitors: it holds while the same
// monitors stay connected, but a monitor that appears or goes away can move the ones after it. SDL has one name per
// monitor: make and model, with the connector (such as DP-2) on Wayland and X11. The name pins one monitor more firmly
// than its index. These calls belong to the thread that initialised video: the window thread, or the main thread in
// --list-monitors.
#define MON_MAX 32

typedef struct {
    SDL_DisplayID id;
    SDL_Rect rc;        // the monitor's pixels, at its position in the desktop's coordinates
    int primary;
    unsigned dpi;       // the display scale as a DPI, 96 at 100%
    char name[128];     // SDL_GetDisplayName; empty when SDL has none
} MonInfo;

typedef struct { MonInfo m[MON_MAX]; int n; } MonList;

static int enum_monitors(MonList *l) {
    *l = (MonList){0};
    int n = 0;
    SDL_DisplayID *ids = SDL_GetDisplays(&n);
    if (!ids) return 0;
    SDL_DisplayID primary = SDL_GetPrimaryDisplay();
    for (int i = 0; i < n && l->n < MON_MAX; i++) {
        SDL_Rect rc;
        if (!SDL_GetDisplayBounds(ids[i], &rc)) continue;
        // A scaled desktop (Wayland at 125%) counts its coordinates in scaled units: the pixel density turns the size
        // back into the monitor's pixels, and the content scale and the density together are the display scale
        // (SDL_GetWindowDisplayScale gives a window the same product). The position stays in the desktop's units.
        const SDL_DisplayMode *mode = SDL_GetCurrentDisplayMode(ids[i]);
        float density = mode && mode->pixel_density > 0 ? mode->pixel_density : 1, scale = SDL_GetDisplayContentScale(ids[i]);
        if (!(scale > 0)) scale = 1;
        const char *name = SDL_GetDisplayName(ids[i]);
        MonInfo *m = &l->m[l->n++];
        *m = (MonInfo){.id = ids[i], .rc = {rc.x, rc.y, (int)(rc.w * density + 0.5f), (int)(rc.h * density + 0.5f)},
                       .primary = ids[i] == primary, .dpi = (unsigned)(96 * scale * density + 0.5f)};
        snprintf(m->name, sizeof m->name, "%s", name ? name : "");
    }
    SDL_free(ids);
    return l->n;
}

// `2  "HP Inc. HP E243 CNK0161ZY2 (DP-3)"  1920x1080 at 3456,756  dpi 120 (125%)`: one monitor, for the log and
// --list-monitors, with "  primary" after the rectangle on the primary one.
static void monitor_line(const MonList *l, int i, char *out, size_t cap) {
    const MonInfo *m = &l->m[i];
    snprintf(out, cap, "%d  \"%s\"  %dx%d at %d,%d%s  dpi %u (%u%%)", i + 1, m->name, m->rc.w, m->rc.h, m->rc.x, m->rc.y,
             m->primary ? "  primary" : "", m->dpi, m->dpi * 100 / 96);
}

static int monitor_unset(const char *v) { return !v || !*v || !SDL_strcasecmp(v, "auto"); }

// Which monitor of the list a NX2NAVI_MONITOR value (not empty, not auto) names, as a 0-based index; -1 when none.
// Rules, in this order: "primary"; all digits, a 1-based index; any other text, a case-insensitive substring of a
// monitor's name (the first monitor in list order when several match). why gets the rule that matched, or the reason
// nothing did. The same rules as gfx_window_win.c resolve_monitor, which also matches the GDI device name.
static int resolve_monitor(const MonList *l, const char *want, char *why, size_t cap) {
    if (!SDL_strcasecmp(want, "primary")) {
        for (int i = 0; i < l->n; i++)
            if (l->m[i].primary) {
                snprintf(why, cap, "the primary monitor");
                return i;
            }
        snprintf(why, cap, "no monitor is marked primary");
        return -1;
    }
    if (want[strspn(want, "0123456789")] == 0) {
        long idx = strtol(want, NULL, 10);
        if (idx >= 1 && idx <= l->n) {
            snprintf(why, cap, "index %ld of %d", idx, l->n);
            return (int)idx - 1;
        }
        snprintf(why, cap, "index %s is outside 1..%d", want, l->n);
        return -1;
    }
    int first = -1, hits = 0;
    for (int i = 0; i < l->n; i++)
        if (SDL_strcasestr(l->m[i].name, want)) {
            if (first < 0) first = i;
            hits++;
        }
    if (first < 0) {
        snprintf(why, cap, "no monitor name contains it");
        return -1;
    }
    if (hits > 1) snprintf(why, cap, "monitor name contains it; the first of %d matches", hits);
    else snprintf(why, cap, "monitor name contains it");
    return first;
}

// The monitor NX2NAVI_MONITOR picks, copied to *out; 0 when it is unset or auto (the system places the window), or when
// it matches nothing now (the monitors present are logged, and the default placement applies). log: say what was decided
// and why. The window thread asks with log set; the size query below, which widescreen.c makes before the window
// exists, stays quiet. Window thread.
static int chosen_monitor(MonInfo *out, int log) {
    const char *want = cfg_get("NX2NAVI_MONITOR");
    if (monitor_unset(want)) return 0;
    MonList l;
    char why[256], line[512];
    int n = enum_monitors(&l), pick = -1;
    if (n) pick = resolve_monitor(&l, want, why, sizeof why);
    else snprintf(why, sizeof why, "no monitor was found");
    if (pick < 0) {
        if (log) {
            rt_log("gfx: NX2NAVI_MONITOR=%s picks no monitor (%s); the window gets the default placement. Monitors present:",
                   want, why);
            for (int i = 0; i < n; i++) {
                monitor_line(&l, i, line, sizeof line);
                rt_log("gfx:   %s", line);
            }
        }
        return 0;
    }
    if (log) {
        monitor_line(&l, pick, line, sizeof line);
        rt_log("gfx: NX2NAVI_MONITOR=%s picks monitor %s (%s)", want, line, why);
    }
    *out = l.m[pick];
    return 1;
}

// --list-monitors (main.c): one line per monitor, then what the current NX2NAVI_MONITOR would pick. No game files. It
// runs on the main thread before any window thread exists, so that thread initialises video here and shuts it down again.
int list_monitors(void) {
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        fprintf(stderr, "list-monitors: SDL video init failed (%s)\n", SDL_GetError());
        return 1;
    }
    MonList l;
    char line[512], why[256];
    int n = enum_monitors(&l);
    if (!n) {
        fprintf(stderr, "list-monitors: no monitor found\n");
        SDL_Quit();
        return 1;
    }
    printf("index  name  desktop rectangle  primary  dpi\n");
    for (int i = 0; i < n; i++) {
        monitor_line(&l, i, line, sizeof line);
        printf("%s\n", line);
    }
    const char *want = cfg_get("NX2NAVI_MONITOR");
    if (monitor_unset(want)) {
        printf("NX2NAVI_MONITOR is not set (auto): the system places the window\n");
    } else {
        int pick = resolve_monitor(&l, want, why, sizeof why);
        if (pick >= 0) printf("NX2NAVI_MONITOR=%s picks monitor %d (%s)\n", want, pick + 1, why);
        else printf("NX2NAVI_MONITOR=%s picks no monitor (%s)\n", want, why);
    }
    SDL_Quit();
    return 0;
}

typedef struct { int w, h, ok; } SizeReq;

static void chosen_size(void *arg) {
    SizeReq *r = arg;
    MonInfo mon;
    if (chosen_monitor(&mon, 0)) r->w = mon.rc.w, r->h = mon.rc.h, r->ok = 1;
}

// Pixels of the monitor the window is on; 0 when unknown. Before the window exists (widescreen.c wide_init at start) a
// fullscreen window opens on the monitor NX2NAVI_MONITOR picks, so that one's size; without a choice the system decides
// where the window goes, which is not known yet.
int gfx_window_monitor_size(int *w, int *h) {
    if (g_win) {
        const SDL_DisplayMode *m = SDL_GetCurrentDisplayMode(SDL_GetDisplayForWindow(g_win));
        if (!m) return 0;
        *w = (int)(m->w * m->pixel_density + 0.5f);
        *h = (int)(m->h * m->pixel_density + 0.5f);
        return 1;
    }
    SizeReq r = {0};
    if (monitor_unset(cfg_get("NX2NAVI_MONITOR")) || !video_up()) return 0;
    run_on_window(chosen_size, &r);
    *w = r.w, *h = r.h;
    return r.ok;
}

// ---- gfx_wsys.h
static int open_ok;

static void open_window(void *arg) {
    (void)arg;
    int w, h;
    gfx_cfg_window_size(&w, &h);
    // NX2NAVI_MONITOR: the window is created centred on the chosen monitor; without a choice the system places it.
    MonInfo mon;
    int on_mon = chosen_monitor(&mon, 1);
    SDL_PropertiesID props = SDL_CreateProperties();
    const char *user_title = cfg_get("NX2NAVI_WINDOW_TITLE");
    if (user_title && *user_title) {
        SDL_SetHint(SDL_HINT_APP_NAME, user_title);
    } else {
        user_title = "NX2navi";
    }
#if defined(__linux__)
    const char *wm_class = cfg_get("NX2NAVI_WINDOW_CLASS");
    if (wm_class && *wm_class) {
        SDL_SetHint("SDL_VIDEO_WAYLAND_WMCLASS", wm_class);
        SDL_SetHint("SDL_VIDEO_X11_WMCLASS", wm_class);
    }
#endif
    SDL_SetStringProperty(props, SDL_PROP_WINDOW_CREATE_TITLE_STRING, user_title);
    SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_WIDTH_NUMBER, w);
    SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_HEIGHT_NUMBER, h);
    SDL_SetBooleanProperty(props, SDL_PROP_WINDOW_CREATE_VULKAN_BOOLEAN, true);
    SDL_SetBooleanProperty(props, SDL_PROP_WINDOW_CREATE_RESIZABLE_BOOLEAN, true);
    SDL_SetBooleanProperty(props, SDL_PROP_WINDOW_CREATE_HIGH_PIXEL_DENSITY_BOOLEAN, true);
    if (on_mon) {
        SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_X_NUMBER, SDL_WINDOWPOS_CENTERED_DISPLAY(mon.id));
        SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_Y_NUMBER, SDL_WINDOWPOS_CENTERED_DISPLAY(mon.id));
    }
    g_win = SDL_CreateWindowWithProperties(props);
    SDL_DestroyProperties(props);
    if (!g_win) {
        rt_log("gfx: SDL_CreateWindow failed (%s)", SDL_GetError());
        return;
    }
    SDL_Surface *icon = NULL;
    const char *icon_env = cfg_get("NX2NAVI_WINDOW_ICON");
    if (icon_env && *icon_env) icon = SDL_LoadBMP(icon_env);
    if (!icon) {
        const char *base = SDL_GetBasePath();
        if (base) {
            char pbuf[1024];
            snprintf(pbuf, sizeof pbuf, "%snx2navi.bmp", base);
            icon = SDL_LoadBMP(pbuf);
        }
    }
    if (!icon) {
        const char *appdir = getenv("APPDIR");
        if (appdir && *appdir) {
            char pbuf[1024];
            snprintf(pbuf, sizeof pbuf, "%s/nx2navi.bmp", appdir);
            icon = SDL_LoadBMP(pbuf);
        }
    }
    if (icon) {
        SDL_SetWindowIcon(g_win, icon);
        SDL_DestroySurface(icon);
    }
    set_pixel_size(w, h);   // w x h were window units; the setting is in pixels
    // Centred again at the size it ended up with. SDL also takes this position as the display the fullscreen below
    // covers: a Wayland window cannot be moved, and this is how its fullscreen still opens on the chosen output.
    if (on_mon) SDL_SetWindowPosition(g_win, SDL_WINDOWPOS_CENTERED_DISPLAY(mon.id), SDL_WINDOWPOS_CENTERED_DISPLAY(mon.id));
    density_update();
    note_applied();
    refresh_rate_update();
    if (cfg_get("NX2NAVI_BACKGROUND_WINDOW")) rt_log("gfx: NX2NAVI_BACKGROUND_WINDOW has no effect with SDL windows");
    int wayland = !SDL_strcmp(SDL_GetCurrentVideoDriver(), "wayland");
    // open_window runs once, so this is logged once
    if (on_mon && wayland)
        rt_log("gfx: Wayland leaves the placement of a windowed window to the compositor, so NX2NAVI_MONITOR only decides "
               "where the game opens in fullscreen");
    if (gfx_cfg_fullscreen()) SDL_SetWindowFullscreen(g_win, true);
    SDL_SyncWindow(g_win);
    // A Wayland compositor places the fullscreen window after SDL_SyncWindow returns; handle reports it once the window
    // is on the chosen monitor.
    if (on_mon && (gfx_cfg_fullscreen() || !wayland)) {
        if (wayland && SDL_GetDisplayForWindow(g_win) != mon.id) g_place_pending = mon.id;
        else report_placement(mon.id);
    }
    input_sdl_window(g_win);
    open_ok = 1;
}

int wsys_open(void) {
    if (!video_up()) return 0;
    if (!g_win) run_on_window(open_window, NULL);
    return open_ok;
}

typedef struct { VkInstance inst; VkSurfaceKHR surface; } SurfaceReq;

static void make_surface(void *arg) {
    SurfaceReq *r = arg;
    if (!SDL_Vulkan_CreateSurface(g_win, r->inst, NULL, &r->surface)) {
        rt_log("gfx: SDL_Vulkan_CreateSurface failed (%s)", SDL_GetError());
        r->surface = VK_NULL_HANDLE;
    }
}

VkSurfaceKHR wsys_surface(VkInstance inst) {
    SurfaceReq r = {inst, VK_NULL_HANDLE};
    run_on_window(make_surface, &r);
    return r.surface;
}

void wsys_client_size(int *w, int *h) {
    *w = *h = 0;
    if (g_win) SDL_GetWindowSizeInPixels(g_win, w, h);
}

int wsys_minimized(void) { return g_win && (SDL_GetWindowFlags(g_win) & (SDL_WINDOW_MINIMIZED | SDL_WINDOW_HIDDEN)); }

static void resize_window(void *arg) {
    const int *wh = arg;
    set_pixel_size(wh[0], wh[1]);
    SDL_SyncWindow(g_win);
}

void wsys_resize(int w, int h) {
    int wh[2] = {w, h};
    run_on_window(resize_window, wh);
}

// ---- --probe-window: synthetic events through the event handler, on the window thread
static void probe_event(void *arg) { handle(arg); }

void wsys_probe_focus(int in) {
    SDL_Event e = {.window = {.type = in ? SDL_EVENT_WINDOW_FOCUS_GAINED : SDL_EVENT_WINDOW_FOCUS_LOST,
                              .windowID = SDL_GetWindowID(g_win)}};
    run_on_window(probe_event, &e);
}

void wsys_probe_escape(void) {
    SDL_Event e = {.key = {.type = SDL_EVENT_KEY_DOWN, .scancode = SDL_SCANCODE_ESCAPE, .key = SDLK_ESCAPE, .down = true,
                           .windowID = SDL_GetWindowID(g_win)}};
    run_on_window(probe_event, &e);
    e.key.type = SDL_EVENT_KEY_UP, e.key.down = false;
    run_on_window(probe_event, &e);
}

// SDL reports no activation by a title bar click: only the client click exists.
int wsys_probe_click(int where) {
    if (where == 2) return 0;
    SDL_Event e = {.button = {.type = SDL_EVENT_MOUSE_BUTTON_DOWN, .button = SDL_BUTTON_LEFT, .down = true,
                              .windowID = SDL_GetWindowID(g_win)}};
    run_on_window(probe_event, &e);
    e.button.type = SDL_EVENT_MOUSE_BUTTON_UP, e.button.down = false;
    run_on_window(probe_event, &e);
    return 1;
}
