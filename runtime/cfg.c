// nx2navi.ini: file-backed defaults for the NX2NAVI_* environment variables (docs/controls.md, nx2navi.ini.example).
// Format: "key = value" lines, '#' or ';' at the start of a line begins a comment, [sections] only group. A key is a
// variable name without "NX2NAVI_" (an optional NX2NAVI_ prefix is accepted), case-insensitive. A value is the rest of the
// line, trimmed, optionally in double quotes; an empty value counts as unset. The environment wins over the file.
//
// Live settings: cfg_set changes a value in memory (readers see the new pointer; the old string is kept, so a pointer
// cfg_get handed out stays valid) and cfg_save writes only the changed keys back, keeping the rest of the file.
// Readers take no lock: an entry is complete before n_entries counts it, and value pointers are swapped atomically.
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "cfg_private.h"
#include "coop_app.h"
#include "discord_app.h"
#include "host.h"
#include "platform/platform.h"

#define PREFIX "NX2NAVI_"
#define PREFIX_LEN (sizeof PREFIX - 1)
#define CFG_MAX_KEYS 512

typedef struct { char *key, *value; int dirty, removed; } CfgEntry;   // key upper-case, without the prefix; value NULL after cfg_reset

static CfgEntry entries[CFG_MAX_KEYS];
static int n_entries;
static unsigned version;
static PlatMutex set_lock = PLAT_MUTEX_INIT;   // cfg_set, cfg_reset, cfg_save
static uint64_t kept_n, kept_bytes;            // set_lock: values cfg_set replaced and kept for readers (cfg_sizes)
static int save_ok = 1;                         // the last cfg_save_to wrote (or had nothing to write)

// Every name the runtime reads (and the patch sets in generated/), without the prefix. A file key outside this list is
// logged as unknown. A new variable goes here too, or its ini key logs a harmless "unknown key" line.
static const char *const known[] = {
    "CORES", "TM_DISPATCH", "NVN2_CS_TABLE", "SKIP_SHADERS", "MOVIE_DEBAND", "MOVIE_FPS", "MOVIE_SKIP", "MOVIE_OVERLAY", "MOVIES", "NPLN", "NVN2_PROGRAMS", "NVN_TRAP", "OCARINA_REALTIME", "ONESHOT_BUDGET", "ONESHOT_WAIT_MS", "OCARINA_STEP", "PHYSICS_STEPS", "PICKUP_COUNT", "STEP_STEADY", "VI_HZ", "VI_VRR", "TIME_DUMP", "SHADER_DUMP", "SHOT_AFTER", "SHOT_DIR", "SHOT_EVERY",   // Switch 2 port: movie skip (romfs.c), NVN2 proc traps (nvn_procs.c)
    "AA", "AA_LOG", "AB", "AMIIBO_DIR", "ANISO", "APP_POOL", "ASPECT", "ASPECT_PATCH", "ASSUME_FOCUS", "ASTC", "ASTC_BC", "ASTC_TIME", "ASYNC_PREEMPT", "AUDIO", "AUDIO_CHANNELS", "AUDIO_DEVICE", "AUDIO_GAME_PRIO", "AUDIO_OFF", "AUDIO_PREFILL_MS", "AUDIO_STATS",
    "BARRIER_GAME", "MESH_AHEAD", "MESH_FALLBACK", "MESH_FALLBACK_MB", "MESH_FALLBACK_WORKERS", "PROBE_MESHFB_DIR",
    "BACKGROUND_WINDOW", "BIND_A", "BIND_B", "BIND_DOWN", "BIND_L", "BIND_LCLICK", "BIND_LEFT", "BIND_LSTICK_DOWN", "BIND_LSTICK_LEFT",
    "BIND_LSTICK_RIGHT", "BIND_LSTICK_UP", "BIND_MINUS",
    "BIND_PLUS", "BIND_R", "BIND_RCLICK", "BIND_RIGHT", "BIND_RSTICK_DOWN", "BIND_RSTICK_LEFT", "BIND_RSTICK_RIGHT", "BIND_RSTICK_UP",
    "BIND_SHAKE",
    "BIND_SKIP", "BIND_STATS", "BIND_UP", "BIND_X", "BIND_Y",
    "BIND_ZL", "BIND_ZR", "BLOCK_VIEWS_3D", "CACHE_DIE", "CLOTH_REALTIME", "COLOR_BRIGHTNESS", "COMPUTE_KEEP", "COLOR_CONTRAST", "COLOR_GAMMA",
    "COLOR_PRESET", "COLOR_SATURATION", "CONTACT_DEBUG", "CONTACT_SHADOWS", "CORE_DIAG", "CRIT", "CRIT_OUT",
    "DESC_TEMPLATE", "DEVICE_PROFILE", "DISCORD_APP_ID", "DISCORD_IMAGE", "DISCORD_RPC", "DLAA", "DLAA_CAM_KEEP", "DLAA_CUT", "DLSS_IDLE_FREE", "DLAA_EDGE_FILL", "DLAA_AT_UPSCALER", "FFX_DEBUG", "DLAA_JITTER", "DLAA_MV", "DLAA_MASKDEBUG", "DLAA_MVDEBUG", "DLAA_OBJMV", "DLAA_SURVEY", "DLAA_TARGET_KEEP", "DLAA_TIMES", "DLSS", "DLSS_CHAR_MASK", "DLSS_PRESET", "DLSS_REACTIVE", "DUMP", "DUMP_BUF", "DUMP_DIR",
    "DUMP_LOG", "DUMP_RAW", "DUMP_SHADERS", "DUMP_FILE", "DS3", "LOG_CS_BINDS", "LOG_TEX_FLAGS", "LOG_MESH", "MESH", "POOL", "NATIVE_TENSOR", "LOG_RCOND", "LOG_RT_SIZES", "DUMP_WHITE", "DUMP_AFTER", "DUMP_MOVIE", "SUBMIT_FINISH", "ACQUIRE_WAIT", "CRASH_FILTER_GUARD", "CS_FALLBACK", "EXEFS", "FEEDBACK", "FIX_C0", "FIX_ZERO", "FLIGHT", "FLIGHT_AT", "FLIGHT_DIR", "FLIGHT_FRAMES", "GPU_TIMELINE",
    "FFX_PIPE_CACHE", "FLIGHT_SPIKE_MS", "FLIGHT_THUMBS", "FLIGHT_EVERY", "FLUSH_SPAN", "FLUSH_UPLOAD", "FOG_CULL", "FOLIAGE_SHADOW_CULL", "FOLIAGE_SHADOW_VRS", "FOV", "FOV_PATCH", "FG_HUDLESS_SHOW", "FG_MULT", "FPS60", "FPS_LIMIT", "FRAMEGEN", "FRAMEGEN_MULTIPLIER", "FREEZE_IDLE", "FREEZE_ASYNC", "FREEZE_BUDGET_US", "FREE_CORES",
    "FULLSCREEN", "FXAA", "GAME_ASPECT", "GAME_RES", "GPA", "GPA_EMPTY", "GPA_TWIN", "GPA_TWIN_MAX_KB", "MIRROR_CAPTURE_KB", "WB_SKIP_MB", "GPL", "GPL_DRAW_BUILD", "GPL_FEEDBACK", "PIPE_AHEAD", "PIPE_FAST_WORKERS", "PIPE_FRAME_WAIT_MS", "PIPE_LINK_DRAW_MS", "PIPE_SKIP_STEADY", "PROG_LEAD", "GPU", "GPU_INDEX", "GPU_PASSES", "GPU_NAME", "HDR",
    "HDR_GAME_NITS", "HDR_LIVE", "HEAP_LAZY", "HDR_PEAK_NITS", "HDR_UI_NITS",
    // retired: the HDR look's settings of earlier versions, now fixed (vk_hdr.c, shaders/hdr_tonemap.frag); read by
    // nothing, listed so that a file which still sets them logs no unknown key
    "HDR_BLOWOUT", "HDR_CONTRAST", "HDR_EXPOSURE", "HDR_FLARE", "HDR_GAMMA", "HDR_GRADE", "HDR_HIGHLIGHT_SAT",
    "HDR_HIGHLIGHTS", "HDR_HUE", "HDR_HUE_CLIP", "HDR_MATCH_SLOPE", "HDR_MIDGRAY", "HDR_SATURATION", "HDR_SHADOWS",
    "HDR_TONEMAP", "HDR_TONE_SLOPE", "HIGHFPS_FIX", "INPUT", "INPUT_RECORD", "INPUT_SCRIPT", "LANG", "LIB_PREBUILD",
    "BIG_BUILD_FREE_MB", "BIG_BUILD_MAX_MB", "BUILD_FREE_MB", "BUILD_RESERVE_MB", "LIB_PREBUILD_BIG", "LIB_PREBUILD_BIG_BG", "LIB_PREBUILD_KEEP", "LIB_PREBUILD_THREADS", "LIB_WARM", "PIPE_LINK_MS", "FRAME_HOLD_MS", "PACE_SMOOTH", "OQ_BATCH", "GATE_WAIT_MS", "LOCALE", "LOD_GUARD", "LSTICK_DEADZONE", "MEMORY", "MEM_CENSUS", "MEM_NEED_MB", "MENU_LANG", "MENU_SOUND_VOLUME", "MENU_TEST",
    "MENU_CURSOR", "MENU_CURSOR_LOG", "MENU_TEST_CLICK", "MENU_TEST_DIR", "MENU_TEST_SET", "MINIMAL_LAYOUT", "MICROPHONE", "MIRROR_GAP", "MODS_DIR", "MODS_DISABLED", "MODS_ORDER", "MODS_REBUILD", "MONITOR", "MOTION", "MOUSE_MODE", "MOUSE_DIRECT_PATCH", "MOUSE_SENS", "MOUSE_TRACE", "MOVIE_HOST", "MOVIE_PACE", "MOVIE_CLOCK", "TEST_HITCH",
    "MVOBJ_ALL", "MVOBJ_BANKS", "MVOBJ_EQUAL", "MVOBJ_EXACT", "MVOBJ_EXEC", "MVOBJ_NOC4", "MVOBJ_ONLY", "MVOBJ_TIMES", "MVPREV_CHECK", "MVPREV_GROW", "MVPREV_SKIP", "MVPREV_TRUST", "MV_CHECK",
    "MV_SURVEY",
    "ONE_PER_CORE", "OPERATION_MODE", "OUTPUT_RES", "PAD_LAYOUT", "PACE_HZ", "PASS_CRUMBS", "PATCH_OUTPUT", "PATCH_RES_FIXED", "PATCH_SCENE", "PATCH_WIDE_ASPECT", "PATCH_WIDE_HUD", "PATCH_WIDE_MOVIE", "PATCH_WIDE_OUT", "PATCH_FREECAM", "PATCH_WIDE_SCENE", "PCSS", "PCSS_K", "PERF_OVERLAY", "PERF_SLOW_MS", "PIPELINE_CACHE", "PIPELINE_CACHE_MAX_MB", "PIPELINE_CACHE_PLAY", "PIPELINE_CACHE_PLAY_MB", "PIPE_BINARIES", "PIPE_BINARIES_MAX_MB", "PIPE_FEEDBACK", "PIPE_MEMO", "PIPE_PREBUILD", "PIPE_SEED", "PIPE_SKIP", "PIPE_SYNC", "PIPE_WAIT", "POWER_FULL", "TEX_SAME", "TEX_FLAT3D", "TEX_KEEP_TAKEN", "TEX_STALE_KEPT", "PRESENT_HELD", "HELD_SYNC",
    "PREPARE_PIPELINES", "PREWARM_FULL", "PREWARM_WAIT_MS", "PREWARM_XLATE", "PROBE_MOVIE", "PROBE_MOVIE_SHOT", "PROFILE", "PROFILE_CHAINS", "PROFILE_DIR", "PROFILE_FRAMES", "PROFILE_REPLAY", "PROFILE_SLOW",
    "POOL_FAST", "POOL_GROUP", "POOL_GROUP_CHECK", "POOL_PRESNAP",
    "PROFILE_MAIN", "PROFILE_RENDER", "PROMPTS", "PUSH_SKIP", "PUSH_STAGE", "RDOC", "RDOC_DIR", "RDOC_DLL", "REALTIME", "REC_END", "REEXEC",
    "REFLEX", "RENDER_SCALE", "REPLAY_ASYNC", "REPLAY_PIPE", "REPLAY_RUNS", "REPLAY_WORKERS", "RTSS", "RTSS_ENUM", "SCENE_SHARE",
    "PRESNAP_PLACE",
    "REPLAY_CHECK", "RESOLVE_CHECK", "RESTART_WAIT", "ROMFS", "ROMFS_INDEX", "ROMFS_MMAP", "ROMFS_MODS", "ROMFS_PATCHES",
    "ROMFS_PROOF_DIR", "ROMFS_VERIFY", "RSTICK_DEADZONE", "RUMBLE", "RUMBLE_CURVE", "RUMBLE_STRENGTH", "RUMBLE_TRIGGERS", "SHADER_CACHE",
    "SHADER_CACHE_FLUSH", "SHADER_CARRY", "SHADER_MEMO", "SHADOW_BULK_WRITE", "SHADOW_CHECK", "SHADOW_COMPARE", "SHADOW_RANGE", "SHADOW_RES", "SHADOW_SCALE", "SHADOW_TABLE_STORES", "SHARED_FONTS", "SHARPNESS", "SHOTS", "SHOTS_DIR",
    "SVC_SLOW_MS",
    "SHOTS_FULL", "SPEED_PROBE", "SPIN", "SPIN_HANDOFF_US", "SPLIT_DRAWS", "SPLIT_IDLE", "SPLIT_IN_PASS", "SPLIT_US", "SPV_MARK_TEST", "SPV_OVERRIDE", "STICK_DEADZONE", "STICK_SENS",
    "SWKBD_ANSWER",
    "SET_AT", "SWKBD_TEST", "TAA", "DBG_TEST", "DEBUG_MENU", "MENU_TEST_DEBUG", "FSR4", "FSR4_FG", "FG4_VERSION", "FG4_SWAPCHAIN", "TEST_PAD", "TEST_PAD_AT", "TEST_PREPARE_SHADERS", "TEX_ALIAS", "DEPTH_INIT_INLINE", "DEPTH_ALIAS_LOG", "TEX_DISUSE", "TEX_DISUSE_EVICT", "TEX_GEN_UNDO", "TEX_MEMO", "TEX_VRAM_PRESSURE", "TEX_RELEASED_MB", "TEX_RELEASED_MAX", "TEX_REUSE", "TEX_SPLIT_FILL", "TEX_COMMIT_FREE_MB", "TEX_PROC_COMMIT_MB", "TEX_EVICT", "TEX_COMMIT_EVICT", "MIRROR_EVICT", "VRAM_BUDGET_MB", "TIMER_1MS", "TRACE", "TXD_TRANSPOSE", "UBO_SNAPSHOT", "EVO_TRACE", "EVENT_AT_SYNC", "UPSCALER", "VK_DYN_A2C", "VK_DYN_VERTEX", "VK_NAN_PRESERVE", "VK_NOOPT", "VK_NO_FLOAT64",
    "VK_PRESENT", "VK_RECONVERGE", "VK_NAN_BITS", "VK_RECORDER", "VK_SSBO_ALIGN", "VK_SUBGROUP", "VK_SUBGROUP_PIN", "VK_VALIDATION", "VOLUME", "VRAM_HEADROOM", "VS_OUTS", "WATER_PROJECTIVE",
    "WAIT_WATCH_S", "WALK_KEY", "WALK_SCALE", "WB_EXACT", "WB_FENCE_LIMIT", "WB_MERGE", "WB_NO_REUP", "WB_MERGE_MAX", "WB_PROBE", "WB_SPLIT", "WB_LAZY", "WB_LAZY_CHUNK", "WB_LAZY_TOUCH", "WINDOW_SIZE", "XLATE", "XLATE_ID", "XLATE_KEEP", "XLATE_PROCS",
    "ZSTD_CHECK", "ZSTD_NATIVE",
    "BIND_REUSE", "DYN_DELTA", "STG_ARENA", "SUBMIT_BATCH", "ASTC_GPU_DESWIZZLE", "ASTC_SCRATCH_KEEP", "ASTC_ARENA", "ASTC_BC1",
    "TEXUP_LOG_MS", "DYN_RES", "DYN_RES_TARGET", "DYN_RES_STEP", "TEXTURE_DETAIL",
    "RECORDER_CORE", "RECORDER_PRIO", "REPLAY_TIME", "LOCK_TIME", "POOL_CREATE_BUDGET", "POOL_CREATE_MS", "POOL_FINI_IDLE", "POOL_LOAD_BACKLOG", "POOL_LOAD_MS",
    "VMA_BLOCK_MB", "SYNC_WAKE_ALL", "LOG_ASYNC", "DEVICE_LOST_AT", "STAGING_FAIL_AT", "VRAM_BALLAST_MB",
    "COOP", "COOP_BETA", "COOP_NAME", "COOP_SERVER", "COOP_PORT", "COOP_ROOM", "COOP_AUTO", "COOP_DROP",   // co-op (coop.c); the last three have no menu row
    "COOP_PROGRESS", "COOP_ITEMS", "COOP_SONGS", "COOP_QUEST", "COOP_KEYS", "COOP_RUPEES", "COOP_AMMO",   // the room's rules (coop_game.c)
    "COOP_TEST_SET",   // test: sets flags by hash at given frames (coop_game.c); no menu row
    "COOP_ACCEPT",     // accepts a co-op room's members and rules before anything of it is applied (coop_game.c)
    "COOP_REMOVE",     // the room's maker removes a member: its id (coop.c), or a key's fingerprint (coop_game.c); no row (the Co-op tab's Remove)
    "COOP_RESET_POOLS",   // the room's maker resets the pools when it changes to a value other than 0 (coop_game.c); no row (the tab's Reset the pools)
};

// The nx2navi.ini section a key cfg_set may write goes to when the file lacks it (nx2navi.ini.example's sections).
static const struct { const char *key, *section; } sections[] = {
    {"FPS_LIMIT", "frame rate"}, {"FPS60", "frame rate"}, {"RTSS", "frame rate"},
    {"REALTIME", "frame rate"}, {"CLOTH_REALTIME", "frame rate"}, {"HIGHFPS_FIX", "frame rate"},
    {"PHYSICS_STEPS", "frame rate"}, {"PICKUP_COUNT", "frame rate"}, {"STEP_STEADY", "frame rate"},
    {"FULLSCREEN", "display"}, {"WINDOW_SIZE", "display"}, {"PERF_OVERLAY", "display"}, {"VK_PRESENT", "display"},
    {"GPU_INDEX", "display"}, {"MONITOR", "display"}, {"OUTPUT_RES", "display"}, {"GAME_ASPECT", "display"}, {"GAME_RES", "display"}, {"RENDER_SCALE", "display"}, {"UPSCALER", "display"},
    {"SHARPNESS", "display"}, {"ASPECT", "display"}, {"FOV", "display"}, {"ANISO", "display"},
    {"TEXTURE_DETAIL", "display"},
    {"COLOR_PRESET", "display"},
    {"COLOR_BRIGHTNESS", "display"}, {"COLOR_CONTRAST", "display"}, {"COLOR_SATURATION", "display"},
    {"COLOR_GAMMA", "display"}, {"HDR", "display"}, {"HDR_LIVE", "display"}, {"HDR_PEAK_NITS", "display"}, {"HDR_GAME_NITS", "display"},
    {"HDR_UI_NITS", "display"}, {"SHADOW_RES", "display"},
    {"DLAA", "display"}, {"FSR4", "display"}, {"FSR4_FG", "display"}, {"DLSS", "display"}, {"DLSS_PRESET", "display"}, {"DLSS_IDLE_FREE", "display"}, {"DLSS_CHAR_MASK", "display"}, {"FXAA", "display"}, {"AA", "display"},
    {"FRAMEGEN", "display"}, {"FRAMEGEN_MULTIPLIER", "display"}, {"FG_MULT", "display"},
    {"DEVICE_PROFILE", "display"}, {"ASSUME_FOCUS", "display"},
    {"DISCORD_RPC", "discord"}, {"DISCORD_APP_ID", "discord"}, {"DISCORD_IMAGE", "discord"},
    {"COOP", "coop"}, {"COOP_BETA", "coop"}, {"COOP_NAME", "coop"}, {"COOP_SERVER", "coop"}, {"COOP_PORT", "coop"},
    {"COOP_PROGRESS", "coop"}, {"COOP_ITEMS", "coop"}, {"COOP_SONGS", "coop"}, {"COOP_QUEST", "coop"}, {"COOP_KEYS", "coop"},
    {"COOP_RUPEES", "coop"}, {"COOP_AMMO", "coop"}, {"COOP_ACCEPT", "coop"},
    {"SHADOW_SCALE", "display"}, {"SHADOW_RANGE", "display"}, {"CONTACT_SHADOWS", "display"},
    {"PCSS", "display"},
    {"MODS_DIR", "mods"}, {"MODS_ORDER", "mods"}, {"MODS_DISABLED", "mods"}, {"MODS_REBUILD", "mods"},
    {"MENU_LANG", "language"}, {"LANG", "language"}, {"OPERATION_MODE", "language"},
    {"VOLUME", "audio"}, {"AUDIO_CHANNELS", "audio"}, {"AUDIO_DEVICE", "audio"}, {"MICROPHONE", "audio"}, {"MENU_SOUND_VOLUME", "audio"},
    {"MOTION", "input"}, {"RUMBLE_STRENGTH", "input"}, {"RUMBLE_CURVE", "input"}, {"RUMBLE_TRIGGERS", "input"}, {"DS3", "input"},
    {"PAD_LAYOUT", "input"}, {"PROMPTS", "input"}, {"STICK_DEADZONE", "input"}, {"LSTICK_DEADZONE", "input"}, {"RSTICK_DEADZONE", "input"},
    {"STICK_SENS", "input"}, {"DEBUG_MENU", "input"},
    {"MENU_CURSOR", "input"}, {"MOUSE_MODE", "input"}, {"MOUSE_SENS", "input"}, {"WALK_KEY", "input"}, {"WALK_SCALE", "input"}, {"INPUT_RECORD", "input"},
    {"BIND_A", "input"}, {"BIND_B", "input"}, {"BIND_X", "input"}, {"BIND_Y", "input"}, {"BIND_LCLICK", "input"},
    {"BIND_RCLICK", "input"}, {"BIND_L", "input"}, {"BIND_R", "input"}, {"BIND_ZL", "input"}, {"BIND_ZR", "input"},
    {"BIND_PLUS", "input"}, {"BIND_MINUS", "input"}, {"BIND_LEFT", "input"}, {"BIND_UP", "input"},
    {"BIND_RIGHT", "input"}, {"BIND_DOWN", "input"}, {"BIND_LSTICK_LEFT", "input"}, {"BIND_LSTICK_UP", "input"},
    {"BIND_LSTICK_RIGHT", "input"}, {"BIND_LSTICK_DOWN", "input"}, {"BIND_RSTICK_LEFT", "input"},
    {"BIND_RSTICK_UP", "input"}, {"BIND_RSTICK_RIGHT", "input"}, {"BIND_RSTICK_DOWN", "input"}, {"BIND_SHAKE", "input"},
    {"ASYNC_PREEMPT", "performance"}, {"REPLAY_ASYNC", "performance"}, {"REPLAY_WORKERS", "performance"}, {"REPLAY_PIPE", "performance"},
    {"FREE_CORES", "performance"},
    {"DYN_RES", "performance"}, {"DYN_RES_TARGET", "performance"}, {"DYN_RES_STEP", "performance"},
    {"RECORDER_CORE", "performance"}, {"RECORDER_PRIO", "performance"}, {"BIND_REUSE", "performance"},
    {"DYN_DELTA", "performance"}, {"STG_ARENA", "performance"}, {"SUBMIT_BATCH", "performance"},
    {"BUILD_FREE_MB", "performance"}, {"BUILD_RESERVE_MB", "performance"}, {"MEM_NEED_MB", "diagnostics"},
    {"ASTC_GPU_DESWIZZLE", "performance"}, {"ASTC_SCRATCH_KEEP", "performance"}, {"ASTC_ARENA", "performance"}, {"ASTC_BC1", "performance"}, {"POOL_CREATE_BUDGET", "performance"},
    {"POOL_CREATE_MS", "performance"}, {"POOL_FINI_IDLE", "performance"}, {"FREEZE_BUDGET_US", "performance"}, {"POOL_LOAD_BACKLOG", "performance"}, {"POOL_LOAD_MS", "performance"},
    {"TIMER_1MS", "performance"}, {"PIPE_WAIT", "performance"}, {"PIPE_SKIP", "performance"}, {"ONESHOT_BUDGET", "performance"},
    {"PIPE_FRAME_WAIT_MS", "performance"}, {"PIPE_SKIP_STEADY", "performance"},
    {"CACHE_DIE", "performance"}, {"ONE_PER_CORE", "performance"},
    {"LIB_PREBUILD", "performance"}, {"LIB_PREBUILD_THREADS", "performance"}, {"LIB_WARM", "performance"}, {"PIPE_LINK_MS", "performance"}, {"PIPE_LINK_DRAW_MS", "performance"}, {"PIPE_AHEAD", "performance"}, {"PIPE_FAST_WORKERS", "performance"}, {"GPL_FEEDBACK", "performance"}, {"FRAME_HOLD_MS", "performance"}, {"PACE_SMOOTH", "performance"}, {"OQ_BATCH", "performance"}, {"GATE_WAIT_MS", "performance"}, {"PIPE_PREBUILD", "performance"}, {"PIPE_SEED", "performance"},
    {"COMPUTE_KEEP", "performance"},
    {"PREWARM_XLATE", "performance"}, {"GPL", "performance"}, {"ASTC", "performance"}, {"MEMORY", "performance"},
    {"TEX_ALIAS", "performance"}, {"DEPTH_INIT_INLINE", "performance"}, {"DEPTH_ALIAS_LOG", "diagnostics"}, {"TEX_DISUSE", "performance"}, {"TEX_DISUSE_EVICT", "performance"}, {"TEX_GEN_UNDO", "performance"}, {"TEX_REUSE", "performance"}, {"TEX_SPLIT_FILL", "performance"}, {"TEX_VRAM_PRESSURE", "performance"}, {"TEX_PROC_COMMIT_MB", "performance"}, {"ASTC_BC", "performance"}, {"LOD_GUARD", "performance"}, {"FIX_C0", "performance"}, {"FIX_ZERO", "performance"}, {"WATER_PROJECTIVE", "performance"},
    {"BIND_SKIP", "performance"}, {"SHADER_CACHE_FLUSH", "performance"}, {"SHADER_CARRY", "performance"}, {"PIPELINE_CACHE_PLAY", "performance"}, {"PIPELINE_CACHE_PLAY_MB", "performance"}, {"FFX_PIPE_CACHE", "performance"}, {"SHADER_MEMO", "performance"}, {"PIPE_MEMO", "performance"}, {"TEX_MEMO", "performance"}, {"DESC_TEMPLATE", "performance"},
    {"PUSH_SKIP", "performance"}, {"SPLIT_DRAWS", "performance"}, {"SPLIT_IDLE", "performance"}, {"SPLIT_IN_PASS", "performance"}, {"SPLIT_US", "performance"}, {"VK_VALIDATION", "performance"},
    {"VK_DYN_VERTEX", "performance"}, {"VK_DYN_A2C", "performance"}, {"VK_SUBGROUP_PIN", "performance"}, {"VK_RECONVERGE", "performance"},
    {"VK_NAN_PRESERVE", "performance"}, {"VK_NAN_BITS", "performance"},
    {"TRACE", "diagnostics"}, {"FLIGHT", "diagnostics"}, {"PASS_CRUMBS", "diagnostics"}, {"FLIGHT_SPIKE_MS", "diagnostics"},
    {"WB_FENCE_LIMIT", "diagnostics"}, {"WB_MERGE", "diagnostics"}, {"WB_EXACT", "diagnostics"}, {"MEM_CENSUS", "diagnostics"}, {"WB_NO_REUP", "diagnostics"}, {"WB_PROBE", "diagnostics"}, {"WB_SPLIT", "performance"}, {"SHADOW_BULK_WRITE", "performance"}, {"WB_LAZY", "performance"}, {"HEAP_LAZY", "performance"}, {"BARRIER_GAME", "performance"}, {"MESH_AHEAD", "performance"}, {"MESH_FALLBACK", "performance"}, {"MESH_FALLBACK_MB", "performance"}, {"MESH_FALLBACK_WORKERS", "performance"}, {"WB_LAZY_TOUCH", "diagnostics"}, {"WB_LAZY_CHUNK", "performance"}, {"FLUSH_UPLOAD", "diagnostics"}, {"GPA_TWIN_MAX_KB", "diagnostics"}, {"MIRROR_CAPTURE_KB", "diagnostics"}, {"WB_SKIP_MB", "diagnostics"},
    {"WB_MERGE_MAX", "diagnostics"}, {"MV_SURVEY", "diagnostics"}, {"TEXUP_LOG_MS", "diagnostics"},
    {"WAIT_WATCH_S", "diagnostics"},
    {"VMA_BLOCK_MB", "diagnostics"}, {"DEVICE_LOST_AT", "diagnostics"}, {"STAGING_FAIL_AT", "diagnostics"}, {"VRAM_BALLAST_MB", "diagnostics"},
};

// The settings the menu shows: the value in effect when the key is unset, and what the key accepts. The only place a
// default lives: consumers read cfg_value / cfg_int / cfg_double / cfg_bool. A row with an empty default has no fixed
// value (it depends on the machine); its consumer reads cfg_get and keeps an "unset" branch.
#define LIVE 1
#define RESTART 0
// The lazy write-back (memory.c "Lazy write-back") is on where it has run: its fetch inside the access-fault handler
// has only ever run on Windows. This is the platform's part of the default, which memory.c mem_init goes by to take
// the access faults; the back end then turns it off for a driver it is not trusted on while the key is unset
// (wb_lazy_choice.h: AMD's own), and with shadow storage whatever is set (vk_memory.c vk_mem_lazy_decide).
#ifdef _WIN32
#define WB_LAZY_DEFAULT "1"
#else
#define WB_LAZY_DEFAULT "0"
#endif

static const CfgMeta metas[] = {
    // Screen
    {"FULLSCREEN", "0", "0|1", LIVE}, {"WINDOW_SIZE", "1920x1080", NULL, LIVE},
    {"VK_PRESENT", "mailbox", "mailbox|fifo|immediate", LIVE}, {"GPU_INDEX", "", NULL, RESTART},
    {"MONITOR", "auto", NULL, RESTART},   // auto|primary|<1-based index>|<part of a monitor name> (gfx_window_win.c, gfx_window_sdl.c chosen_monitor)
    {"PERF_OVERLAY", "off", "off|fps|full", LIVE},
    {"OUTPUT_RES", "1920x1080", NULL, RESTART}, {"RENDER_SCALE", "game", "game|50|67|75|100|125|150|200", RESTART},
    {"UPSCALER", "fsr", "fsr|bilinear", LIVE}, {"SHARPNESS", "80", "0..100", LIVE},
    {"DLAA", "0", "0|1", LIVE}, {"DLSS", "dlaa", "dlaa|quality|balanced|performance|ultra_performance", LIVE},
    {"DLSS_PRESET", "default", "default|k|j", LIVE}, {"DLSS_IDLE_FREE", "1", "0|1|2", RESTART}, {"TAA", "dlss", "dlss|fsr3|fsr4", LIVE},   // vk_dlaa.c fsr3
    {"FSR4", "0", "0|1", LIVE},   // vk_ffx.cpp vk_ffx_fsr4
    {"FSR4_FG", "0", "0|1", LIVE},   // vk_fg.cpp use4
    {"FRAMEGEN", "off", "off|fsr3|dlss", LIVE},   // vk_fg.cpp, vk_dlssg.c
    {"FRAMEGEN_MULTIPLIER", "2", "2|3|4", LIVE},   // vk_dlssg.c: frames on screen per rendered frame with DLSS
    // the same where DLSS's frame generation runs through NGX itself (gfx_present.c ng_multiplier): 2x to 6x or dynamic
    {"FG_MULT", "2", "2|3|4|5|6|dynamic", LIVE},
    {"DLSS_CHAR_MASK", "65", "0|35|65|100", LIVE}, {"FXAA", "1", "0|1", LIVE},
    {"AA", "game", "game|off|fxaa|smaa|taa", LIVE},   // vk_aa.c: what runs in place of the game's SMAA
    {"GAME_ASPECT", "16:9", "16:9|window|21:9|32:9", RESTART},   // widescreen.c
    {"GAME_RES", "1440", "1080|1440|1800|2160|2880|window", RESTART},   // widescreen.c: the picture's height
    {"ASPECT", "fit", "fit|stretch", LIVE}, {"FOV", "100", "70..140", LIVE},
    {"ANISO", "16", "game|2|4|8|16", LIVE},   // vk_texture.c sampler_features, vk_sampler_aniso_live; 16: the game asks 1x or 2x
    {"TEXTURE_DETAIL", "auto", "auto|full|reduced|low", RESTART},   // vk_texture.c "texture detail", read at the first texture
    // colour adjustment (vk_color.c); the preset is the menu's, the four values are what the present uses
    {"COLOR_PRESET", "default", "default|vivid|clarity|custom", LIVE}, {"COLOR_BRIGHTNESS", "0", "-30..30", LIVE},
    {"COLOR_CONTRAST", "100", "70..150", LIVE}, {"COLOR_SATURATION", "100", "50..200", LIVE},
    {"COLOR_GAMMA", "1", "0.8..1.2", LIVE},
    // HDR output (vk_hdr.c): on and off apply at once (the HDR tonemap is spliced into the game's as shaders load
    // whatever is set, vk_hdr_armed); native is the game's own and is asked for once, at start
    // (vk_hdr_needs_restart). HDR_LIVE = 0 leaves the HDR path out of a run that starts with HDR off. The look is
    // fixed: only the three whites are settings.
    {"HDR", "0", "0|1|native", LIVE}, {"HDR_LIVE", "1", "0|1", RESTART},
    {"HDR_PEAK_NITS", "1000", "400..10000", LIVE}, {"HDR_GAME_NITS", "203", "48..500", LIVE},
    {"HDR_UI_NITS", "203", "48..500", LIVE},
    {"DEVICE_PROFILE", "auto", "auto|desktop|deck", RESTART},
    {"SHADOW_RES", "game", "game|256|512|768|1024|1536|2048", RESTART},
    {"SHADOW_SCALE", "1", "1..4", RESTART}, {"SHADOW_COMPARE", "1", "0|1", RESTART},   // vk_texture.c
    {"SHADOW_BULK_WRITE", "0", "0|1", RESTART},   // vk_memory.c shadow_write_back
    // vk_memory.c "Pages a shader stores into through the device address table"
    {"SHADOW_TABLE_STORES", "0", "0|1", RESTART}, {"SHADOW_CHECK", "0", "0|1", RESTART},
    {"SVC_SLOW_MS", "0", "0..60000", RESTART},   // kernel.c kern_svc: log services slower than this
    {"SHADOW_RANGE", "100", "50..400", RESTART}, {"CONTACT_SHADOWS", "0", "0..100", RESTART},
    {"PCSS", "0", "0..200", RESTART},
    // Game
    {"FPS_LIMIT", "60", "30|40|45|60|90|120|144|165|240|display|unlimited", LIVE}, {"FPS60", "1", "0|1", LIVE},
    {"RTSS", "1", "0|1", RESTART},   // rtss_limit.c: RivaTuner Statistics Server takes the limit where it runs
    {"REALTIME", "1", "0|1", RESTART}, {"CLOTH_REALTIME", "1", "0|1", RESTART},
    {"MODS_REBUILD", "auto", "auto|off", RESTART},
    {"HIGHFPS_FIX", "1", "0|1", RESTART},
    {"LANG", "", "ja|en-US|fr|de|it|es|zh-CN|ko|nl|pt|ru|zh-TW|en-GB|fr-CA|es-419|zh-Hans|zh-Hant|pt-BR|pl", RESTART},
    {"OPERATION_MODE", "docked", "docked|handheld", LIVE},
    {"ASSUME_FOCUS", "1", "0|1", LIVE},   // read on every focus change (ov_system.c)
    // Sound, Controls, Menu
    {"VOLUME", "100", "0..100", LIVE}, {"MENU_SOUND_VOLUME", "80", "0..100", LIVE}, {"AUDIO_CHANNELS", "stereo", "auto|stereo|5.1", LIVE},   // audio_mixer.c audio_channels_live
    {"AUDIO_DEVICE", "default", NULL, RESTART},   // default|<1-based index>|<part of a device name> (audio_mixer.c resolve_audio_device)
    {"MICROPHONE", "1", "0|1", RESTART},   // ov_audio_input.c mic_on: 0 refuses the game's input streams
    {"MOTION", "1", "0|1", LIVE}, {"RUMBLE_STRENGTH", "130", "0..300", LIVE}, {"RUMBLE_CURVE", "0.6", "0.2..1.5", LIVE},
    {"RUMBLE_TRIGGERS", "0", "0|1", LIVE}, {"PAD_LAYOUT", "nintendo", "auto|nintendo|xbox|playstation", LIVE},
    {"DS3", "0", "0|1", LIVE},   // input_ds3_win.c ds3_read: read a DualShock 3 directly; off: no HID scan
    {"PROMPTS", "nintendo", "nintendo|xbox|playstation|steamdeck|pc", RESTART},
    {"STICK_DEADZONE", "8", "0..50", LIVE}, {"STICK_SENS", "1", "0.1..3", LIVE},   // camera_speed.c cam_speed_tick
    {"DEBUG_MENU", "0", "0|1", LIVE},   // ui/menu.cpp menu_debug_toggle
    {"LSTICK_DEADZONE", "", "0..50", LIVE}, {"RSTICK_DEADZONE", "", "0..50", LIVE},
    {"MOUSE_MODE", "direct", "stick|direct", LIVE},
    {"MOUSE_SENS", "1", "0..10", LIVE}, {"MENU_CURSOR", "1", "0|1", LIVE},
    {"WALK_KEY", "none", NULL, LIVE}, {"WALK_SCALE", "0.51", "0.2..0.9", LIVE},
    {"MENU_LANG", "", "en|ru|pt-BR|zh-Hans", LIVE},
    {"AMIIBO_DIR", "", NULL, LIVE},
    // Discord Rich Presence (discord_rpc.c): on by default (it only talks to a Discord client running on this PC, and
    // does nothing without one); the Game tab of the settings menu and the launcher's Settings switch it. The application's id and art key default to discord_app.h's NX2NAVI_DISCORD_*_DEFAULT.
    {"DISCORD_RPC", "1", "0|1", LIVE},
    {"DISCORD_APP_ID", NX2NAVI_DISCORD_APP_ID_DEFAULT, NULL, LIVE},
    {"DISCORD_IMAGE", NX2NAVI_DISCORD_IMAGE_DEFAULT, NULL, LIVE},
    // Co-op (coop.c): off until the player turns it on. The server is host[:port] of a relay (tools/coop_relay.py), and has no
    // default until coop_app.h names one; the name is at most 24 bytes, empty for Player-xxxx; the port is the local UDP port
    // (0: any). COOP_ROOM, COOP_AUTO, COOP_DROP and COOP_TEST_SET are read with cfg_get, no row.
    {"COOP", "0", "0|1", LIVE}, {"COOP_BETA", "0", "0|1", LIVE}, {"COOP_NAME", "", NULL, LIVE},
    {"COOP_SERVER", NX2NAVI_COOP_SERVER_DEFAULT, NULL, LIVE}, {"COOP_PORT", "0", "0..65535", RESTART},
    // The rules a room made by this player syncs by (coop_state.h, docs/coop.md); a joiner's own are not used in the room.
    {"COOP_PROGRESS", "shared", "shared|own", LIVE}, {"COOP_ITEMS", "shared", "shared|own", LIVE},
    {"COOP_SONGS", "shared", "shared|own", LIVE}, {"COOP_QUEST", "shared", "shared|own", LIVE},
    {"COOP_KEYS", "pool", "pool|own", LIVE}, {"COOP_RUPEES", "own", "own|pool", LIVE}, {"COOP_AMMO", "own", "own|pool", LIVE},
    // Nothing from a room is applied before its members and rules are accepted (coop_game.c): ask (the log names the
    // offer's number, which this setting or the menu accepts), always (every room, the save still copied first), or a number.
    {"COOP_ACCEPT", "ask", NULL, LIVE},
    // Advanced: scheduling
    {"ASYNC_PREEMPT", "1", "0|1|2", RESTART}, {"REPLAY_ASYNC", "0", "0|1", RESTART},
    {"REPLAY_WORKERS", "3", "0..16", RESTART}, {"REPLAY_PIPE", "0", "0..4", RESTART},
    {"FREE_CORES", "0", "0|1", RESTART}, {"TIMER_1MS", "1", "0|1", RESTART},
    // dynamic resolution (dynres.c): the scene follows the GPU time toward DYN_RES_TARGET milliseconds
    {"DYN_RES", "0", "0|1", LIVE}, {"DYN_RES_TARGET", "16.7", "1..100", LIVE}, {"DYN_RES_STEP", "45", "8..600", LIVE},
    // the recording thread: a core of its own (auto: with 12 or more logical processors) and the highest priority
    // (vk_exec.c rec_claim)
    {"RECORDER_CORE", "0", "auto|0|1", RESTART}, {"RECORDER_PRIO", "1", "0|1", RESTART}, {"REPLAY_TIME", "0", "0|1", RESTART}, {"LOCK_TIME", "0", "0|1", RESTART}, {"POOL_CREATE_BUDGET", "16", "1..4096", RESTART}, {"POOL_CREATE_MS", "6", "0..100", RESTART}, {"POOL_FINI_IDLE", "0", "0|1", RESTART}, {"FREEZE_BUDGET_US", "1000", "0..100000", RESTART}, {"POOL_LOAD_BACKLOG", "512", "0..65536", RESTART}, {"POOL_LOAD_MS", "50", "1..1000", RESTART},
    // held submissions handed to the queue together (vk_sync.c, submit batching)
    {"SUBMIT_BATCH", "4", "1..16", RESTART},
    {"CACHE_DIE", "1", "0|1|ab", RESTART}, {"ONE_PER_CORE", "0", "0|1|ab", RESTART},   // perf.c cpu_placement
    // shaders and pipelines
    {"PIPE_WAIT", "default", "default|draw", RESTART}, {"PIPE_SKIP", "any", "strict|any|fullscreen", RESTART},
    {"PIPE_LINK_MS", "6", "0..100", RESTART},   // vk_exec_draw.c link_budget
    {"PIPE_LINK_DRAW_MS", "3", "1..50", RESTART},   // vk_exec_draw.c link_draw_ms
    {"PIPE_FRAME_WAIT_MS", "50", "0..500", RESTART},   // vk_exec_draw.c frame_wait
    {"PIPE_SKIP_STEADY", "1", "0|1", RESTART},  // vk_exec.c steady_on
    {"PREPARE_PIPELINES", "all", "0|mesh|all", RESTART},   // vk_pipeline.c seed_pipelines
    // vk_device.c: task and mesh programs through compute passes (vk_exec_draw.c "mesh fallback"); auto: on a device
    // without mesh shaders. The megabytes are the buffers the passes write (vk_mesh_fb.c).
    {"MESH_FALLBACK", "auto", "auto|0|1", RESTART}, {"MESH_FALLBACK_MB", "128", "16..4096", RESTART},
    {"MESH_FALLBACK_WORKERS", "auto", "auto|1|2|3|4", RESTART},   // vk_exec_draw.c mfb_worker_count
    {"PIPE_AHEAD", "1", "0|1", RESTART},        // vk_exec.c vk_exec_ahead, vk_pipeline.c vk_pipeline_hint
    {"GPL_FEEDBACK", "1", "0|1", RESTART},      // vk_pipeline.c gpl_feedback
    {"PIPE_FAST_WORKERS", "1", "0|1", RESTART}, // vk_pipeline.c lib_worker, fast_take
    {"FRAME_HOLD_MS", "120", "0..500", RESTART},   // vk_backend.c frame_hold
    {"PACE_SMOOTH", "1", "0|1", RESTART},   // vk_backend.c pace_smooth
    {"OQ_BATCH", "1", "0|1", RESTART},   // vk_exec_xfer.c oq_take
    {"GATE_WAIT_MS", "10", "0..100", RESTART},   // vk_exec.c vk_exec_gate_wait
    {"LIB_PREBUILD", "1", "0|1", RESTART}, {"LIB_PREBUILD_THREADS", "", "1..64", RESTART},
    {"LIB_WARM", "1", "0|1|force", RESTART},   // vk_pipeline.c lw_start: the driver's cache of stage libraries, warmed when cold
    {"PIPE_PREBUILD", "1", "0|1", RESTART}, {"PIPE_SEED", "1", "0|1", RESTART},   // vk_pipeline.c keys_open_seed
    {"PREWARM_XLATE", "", "1..64", RESTART}, {"GPL", "", "0|1", RESTART},   // vk_caps.c vk_caps_gpl: by the driver, off with AMD's for RDNA 2 and older
    {"LIB_PREBUILD_BIG_BG", "1", "0|1", RESTART}, {"PIPE_FEEDBACK", "1", "0|1|2", RESTART},
    {"PIPELINE_CACHE_MAX_MB", "512", "0..65536", RESTART},
    {"FFX_PIPE_CACHE", "1", "0|1", RESTART},   // vk_ffx.cpp vk_ffx_proc_addr: FSR 3.1's pipelines through the pipeline caches
    {"PIPELINE_CACHE_PLAY", "1", "0|1", RESTART}, {"PIPELINE_CACHE_PLAY_MB", "64", "1..65536", RESTART},   // vk_pipeline.c "the play cache": what a run compiles goes into a file of its own
    {"PIPE_BINARIES", "0", "0|1", RESTART}, {"ONESHOT_WAIT_MS", "30000", "0..60000", RESTART}, {"ONESHOT_BUDGET", "1", "0|1", RESTART}, {"PIPE_BINARIES_MAX_MB", "1024", "0..65536", RESTART}, {"BIG_BUILD_FREE_MB", "8192", "0..262144", RESTART},
    {"BIG_BUILD_MAX_MB", "4096", "0..262144", RESTART},
    {"BUILD_FREE_MB", "4096", "0..1048576", RESTART}, {"BUILD_RESERVE_MB", "64", "0..65536", RESTART},   // vk_pipeline.c gate_enter
    {"MEM_NEED_MB", "24576", "0..1048576", RESTART},   // memory.c mem_start_note; the default is mem_need.h MEM_NEED_DEFAULT_MB
    // all|<count>: the compute pipelines the boot keeps, those dispatched most recently (vk_pipeline.c, compute-used.bin)
    {"COMPUTE_KEEP", "512", NULL, RESTART},
    {"SHADER_CARRY", "1", "0|1", RESTART},   // gfx_shader.c "Carry": a previous translator's translations until translated again
    // textures and memory
    {"ASTC", "auto", "auto|gpu|native|cpu", RESTART}, {"MEMORY", "auto", "auto|import|mirror|verify", RESTART},
    {"TEX_ALIAS", "1", "0|1", RESTART}, {"DEPTH_INIT_INLINE", "1", "0|1", RESTART}, {"DEPTH_ALIAS_LOG", "0", "0|1", RESTART}, {"TEX_DISUSE", "1", "0|1|2", RESTART}, {"TEX_DISUSE_EVICT", "0", "0|1", RESTART}, {"TEX_GEN_UNDO", "1", "0|1", RESTART}, {"TEX_REUSE", "1", "0|1", RESTART}, {"TEX_SPLIT_FILL", "1", "0|1|2", RESTART}, {"TEX_VRAM_PRESSURE", "80", "0..100", RESTART}, {"ASTC_BC", "", "0|1", RESTART}, {"ZSTD_NATIVE", "1", "0|1", RESTART}, {"ZSTD_CHECK", "0", "0|1", RESTART},
    {"STG_ARENA", "1", "0|1", RESTART}, {"ASTC_GPU_DESWIZZLE", "1", "0|1", RESTART},
    {"ASTC_SCRATCH_KEEP", "1", "0|1", RESTART}, {"ASTC_ARENA", "1", "0|1", RESTART},   // vk_astc.c scratch_get, vk_texture.c astc_arena
    // render fixes
    {"LOD_GUARD", "1", "0|1", RESTART}, {"FIX_C0", "1", "0|1", RESTART}, {"FIX_ZERO", "1", "0|1", RESTART},
    {"WATER_PROJECTIVE", "1", "0|1", RESTART},   // vk_pipeline.c water_projective_pair
    {"BIND_SKIP", "1", "0|1", RESTART}, {"SHADER_CACHE_FLUSH", "0", "0|1", RESTART}, {"SHADER_MEMO", "1", "0|1", RESTART}, {"PIPE_MEMO", "1", "0|1", RESTART}, {"TEX_MEMO", "1", "0|1", RESTART},
    {"BIND_REUSE", "1", "0|1", RESTART}, {"DYN_DELTA", "1", "0|1", RESTART},
    {"DESC_TEMPLATE", "1", "0|1", RESTART},
    {"PUSH_SKIP", "1", "0|1", RESTART}, {"SPLIT_DRAWS", "64", "0..1000", RESTART},
    {"SPLIT_US", "1000", "0..20000", RESTART}, {"SPLIT_IDLE", "0", "0|1", RESTART}, {"SPLIT_IN_PASS", "1", "0|1", RESTART},
    // Vulkan
    {"VK_VALIDATION", "0", "0|1", RESTART}, {"VK_DYN_VERTEX", "", "0|1", RESTART},
    {"VK_DYN_A2C", "", "0|1", RESTART}, {"VK_SUBGROUP_PIN", "1", "0|1", RESTART},
    {"VK_RECONVERGE", "1", "0|1", RESTART},   // vk_pipeline.c spirv_reconverge
    {"VK_NAN_PRESERVE", "", "0|1", RESTART},   // vk_pipeline.c spirv_nan_preserve; unset: every driver but NVIDIA's
    {"VK_NAN_BITS", "", "0|1", RESTART},   // vk_pipeline.c spirv_nan_bits; unset: AMD's driver for RDNA 1 and 2
    {"VK_NOOPT", "0", "0|1|mesh", RESTART},   // vk_pipeline.c noopt_flag_for
    // diagnostics
    {"TRACE", "", NULL, RESTART}, {"FLIGHT", "0", "0|1", RESTART}, {"FLIGHT_SPIKE_MS", "100", "0..60000", RESTART},
    {"WB_FENCE_LIMIT", "65536", "0..1073741824", RESTART}, {"WB_MERGE", "1", "0|1", RESTART}, {"WB_EXACT", "1", "0|1", RESTART}, {"MEM_CENSUS", "0", "0|1", RESTART}, {"WB_NO_REUP", "1", "0|1", RESTART}, {"WB_PROBE", "0", "0|1", RESTART}, {"WB_SPLIT", "1", "0|1|2|3", RESTART}, {"WB_LAZY", WB_LAZY_DEFAULT, "0|1", RESTART}, {"HEAP_LAZY", "0", "0|1", RESTART}, {"WB_LAZY_TOUCH", "0", "0..16", RESTART}, {"WB_LAZY_CHUNK", "0", "0|1", RESTART}, {"FLUSH_UPLOAD", "0", "0|1", RESTART}, {"GPA_TWIN_MAX_KB", "0", "0..1048576", RESTART},
    {"MIRROR_CAPTURE_KB", "0", "0..1048576", RESTART}, {"WB_SKIP_MB", "0", "0..4096", RESTART},   // vk_memory.c capture_cap, wb_skip
    {"WB_MERGE_MAX", "32", "1..256", RESTART}, {"TEXUP_LOG_MS", "0", "0..60000", RESTART},   // vk_texture.c vk_texstat_frame
    {"WAIT_WATCH_S", "0", "0..3600", RESTART},   // kernel.c wait_watch: a lock or thread-end wait this long is logged; 0 off
    {"VMA_BLOCK_MB", "64", "16..256", RESTART},   // vk_headroom.c vk_vma_block_size
    {"ASTC_BC1", "1", "0|1", RESTART},   // vk_astc.c bc1_on
    {"PASS_CRUMBS", "", "0|1", RESTART},   // vk_crumbs.c pass marks; unset: VK_AMD_buffer_marker only
    {"STAGING_FAIL_AT", "0", "0..1000000000", RESTART},   // vk_memory.c staging_fail_now: a test of give_back
    {"VRAM_BALLAST_MB", "0", "0..131072", RESTART},   // vk_memory.c vk_mem_ballast: a test of a smaller card
    {"DEVICE_LOST_AT", "0", "0..1000000000", RESTART},   // vk_sync.c lost_at_due: a test of the lost device's end
    {"INPUT_RECORD", "", NULL, RESTART},
};

static int is_known(const char *key) {
    for (size_t i = 0; i < sizeof known / sizeof *known; i++)
        if (!strcmp(known[i], key)) return 1;
    return !strncmp(key, "CHEAT_", 6);   // cheats.c and the cheat patch sets (tools/cheats.py)
}

// The keys of the file's lines that are not settings of this version (add), lower-case, ", " between them, each once;
// what did not fit is counted and left out. Written by cfg_load, before any other thread reads settings.
static char unknown_keys[384];
static int n_unknown, unknown_cut;   // how many; whether the list left some out

static void note_unknown(const char *key) {
    char low[64];
    size_t n = 0;
    for (; key[n] && n < sizeof low - 1; n++) {
        // a key of other characters (a line of text with an '=' in it) is counted, not listed: the menu cuts the list
        // by bytes to fit its row
        if (!isalnum((unsigned char)key[n]) && key[n] != '_') break;
        low[n] = (char)tolower((unsigned char)key[n]);
    }
    if (key[n]) {
        n_unknown++, unknown_cut = 1;
        return;
    }
    low[n] = 0;
    size_t have = strlen(unknown_keys);
    for (const char *p = unknown_keys; (p = strstr(p, low)) != NULL; p += n)   // a repeated line counts once
        if ((p == unknown_keys || p[-1] == ' ') && (p[n] == ',' || !p[n])) return;
    n_unknown++;
    if (have + n + 3 > sizeof unknown_keys) {
        unknown_cut = 1;
        return;
    }
    snprintf(unknown_keys + have, sizeof unknown_keys - have, "%s%s", have ? ", " : "", low);
}

const char *cfg_unknown_keys(int *count) {
    if (count) *count = n_unknown;
    return unknown_keys;
}

static void clear(void) {
    for (int i = 0; i < n_entries; i++) free(entries[i].key), free(entries[i].value);
    n_entries = 0;
    unknown_keys[0] = 0, n_unknown = unknown_cut = 0;
}

static char *trim(char *s) {
    while (*s == ' ' || *s == '\t') s++;
    char *e = s + strlen(s);
    while (e > s && (e[-1] == ' ' || e[-1] == '\t' || e[-1] == '\r')) *--e = 0;
    return s;
}

// The upper-case key without the prefix, in place.
static char *norm_key(char *key) {
    for (char *p = key; *p; p++) *p = (char)toupper((unsigned char)*p);
    return strncmp(key, PREFIX, PREFIX_LEN) ? key : key + PREFIX_LEN;
}

static void add(char *key, char *value, int line) {
    key = norm_key(key);
    size_t vl = strlen(value);
    if (vl >= 2 && value[0] == '"' && value[vl - 1] == '"') value[vl - 1] = 0, value++;
    if (!*key) return;
    if (!is_known(key)) {
        rt_log("cfg: nx2navi.ini line %d: unknown key '%s'", line, key);
        note_unknown(key);
    }
    if (!*value) return;   // "key =" leaves the setting at its default
    for (int i = 0; i < n_entries; i++)
        if (!strcmp(entries[i].key, key)) {   // a repeated key: the later line wins
            free(entries[i].value);
            entries[i].value = strdup(value);
            return;
        }
    if (n_entries == CFG_MAX_KEYS) return;
    entries[n_entries++] = (CfgEntry){strdup(key), strdup(value), 0, 0};
}

// 0.1.9 had the Xbox prompts as a pad layout: pad_layout = xbox_prompts is pad_layout = nintendo with prompts = xbox.
// Read as those two, which the menu shows and the next cfg_save writes; a prompts line of the file's own wins. The
// value still works from the environment (input_pad_layout_pick, input_prompts_pick).
static void migrate(void) {
    CfgEntry *layout = NULL, *prompts = NULL;
    for (int i = 0; i < n_entries; i++) {
        if (!strcmp(entries[i].key, "PAD_LAYOUT")) layout = &entries[i];
        if (!strcmp(entries[i].key, "PROMPTS")) prompts = &entries[i];
    }
    if (!layout || strcasecmp(layout->value, "xbox_prompts")) return;
    free(layout->value);
    layout->value = strdup("nintendo");
    layout->dirty = 1;
    if (!prompts && n_entries < CFG_MAX_KEYS) entries[n_entries++] = (CfgEntry){strdup("PROMPTS"), strdup("xbox"), 1, 0};
    rt_log("cfg: nx2navi.ini pad_layout = xbox_prompts is read as pad_layout = nintendo and prompts = %s",
           prompts ? prompts->value : "xbox");
}

// A file saved as UTF-16 (Notepad's "Unicode": the byte order mark FF FE or FE FF, then two bytes per character) as
// UTF-8 in a new buffer; NULL for any other file. Read as bytes such a file ends at the zero after its first character:
// no line was taken, and every setting kept its default without a word in the log.
static uint8_t *utf16_text(const uint8_t *d, uint64_t size) {
    if (size < 2 || !((d[0] == 0xFF && d[1] == 0xFE) || (d[0] == 0xFE && d[1] == 0xFF))) return NULL;
    int be = d[0] == 0xFE;
    uint8_t *out = malloc((size_t)(size / 2) * 3 + 1), *o = out;
    if (!out) return NULL;
    for (uint64_t i = 2; i + 1 < size; i += 2) {
        uint32_t c = be ? (uint32_t)d[i] << 8 | d[i + 1] : (uint32_t)d[i + 1] << 8 | d[i];
        if (!c) break;
        if (c >= 0xD800 && c < 0xDC00 && i + 3 < size) {   // a pair: one character above U+FFFF
            uint32_t lo = be ? (uint32_t)d[i + 2] << 8 | d[i + 3] : (uint32_t)d[i + 3] << 8 | d[i + 2];
            if (lo >= 0xDC00 && lo < 0xE000) c = 0x10000 + ((c - 0xD800) << 10) + (lo - 0xDC00), i += 2;
        }
        if (c < 0x80) *o++ = (uint8_t)c;
        else if (c < 0x800) *o++ = (uint8_t)(0xC0 | c >> 6), *o++ = (uint8_t)(0x80 | (c & 0x3F));
        else if (c < 0x10000)
            *o++ = (uint8_t)(0xE0 | c >> 12), *o++ = (uint8_t)(0x80 | (c >> 6 & 0x3F)), *o++ = (uint8_t)(0x80 | (c & 0x3F));
        else
            *o++ = (uint8_t)(0xF0 | c >> 18), *o++ = (uint8_t)(0x80 | (c >> 12 & 0x3F)),
            *o++ = (uint8_t)(0x80 | (c >> 6 & 0x3F)), *o++ = (uint8_t)(0x80 | (c & 0x3F));
    }
    *o = 0;
    return out;
}

int cfg_load(const char *path) {
    clear();
    __atomic_add_fetch(&version, 1, __ATOMIC_RELEASE);   // the lookup memo (cfg_memo) forgets the freed strings
    uint64_t size = 0;
    uint8_t *data = plat_file_read(path, &size);
    if (!data) return 0;
    uint8_t *wide = utf16_text(data, size);
    if (wide) {
        free(data);
        data = wide, size = strlen((const char *)wide);
        rt_log("cfg: %s is saved as UTF-16; read as that (the menu saves it as UTF-8)", path);
    }
    char *p = (char *)data;
    if (size >= 3 && !memcmp(p, "\xEF\xBB\xBF", 3)) p += 3;   // editors add a BOM
    for (int line = 1; *p; line++) {
        char *eol = strchr(p, '\n');
        if (eol) *eol = 0;
        char *s = trim(p);
        char *eq = strchr(s, '=');
        if (*s && *s != '#' && *s != ';' && *s != '[') {
            if (!eq) rt_log("cfg: nx2navi.ini line %d: no '=' in '%s'", line, s);
            else {
                *eq = 0;
                add(trim(s), trim(eq + 1), line);
            }
        }
        if (!eol) break;
        p = eol + 1;
    }
    free(data);
    migrate();
    return 1;
}

// ---- device profiles: defaults for a known machine, below the environment and nx2navi.ini and above the table, so any
// key the player sets wins and resetting a key in the menu returns to the profile's value. NX2NAVI_DEVICE_PROFILE = auto
// (default: the Steam Deck when the firmware says so), desktop (none) or deck.
//
// The Steam Deck (LCD Jupiter, OLED Galileo; 1280x800, 16:10, 4 Zen 2 cores, RDNA 2 with 8 CUs sharing 16 GB): the game
// at the screen's 1280x800 with its own 5/6 scene (the aspect patch gives the camera its 16:10, resolution.c, and the
// romfs-patches-1280x800 overlay the HUD's);
// fullscreen and vsync, as gamescope composes Game Mode; 40 fps, which divides 40 Hz (and 120 Hz on the OLED); the
// game's six virtual cores folded onto four, one per physical core (kernel.c cores_enabled); ASTC textures stored as
// BC3, as the automatic choice gives with 16 GB shared between the CPU and the GPU; and a spinning guest thread handed
// off after 30 us instead of 50 (kernel.c), with four cores to share.
// Starting points to replace with measured ones (docs/tester-guide.md, Steam Deck).
static const struct { const char *key, *value; } DECK[] = {
    {"FULLSCREEN", "1"}, {"WINDOW_SIZE", "1280x800"}, {"OUTPUT_RES", "1280x800"}, {"RENDER_SCALE", "game"},
    {"VK_PRESENT", "fifo"}, {"FPS_LIMIT", "40"}, {"CORES", "4"}, {"ASTC_BC", "1"}, {"SPIN_HANDOFF_US", "30"},
};
static int profile = -1;   // 0 none, 1 Steam Deck

static int read_line(const char *path, char *out, size_t n) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    int ok = fgets(out, (int)n, f) != NULL;
    fclose(f);
    out[ok ? strcspn(out, "\r\n") : 0] = 0;
    return ok;
}

// The Steam Deck: Valve's board, product Jupiter or Galileo (/sys/class/dmi/id); SteamDeck=1, which Steam sets on it.
static int on_steam_deck(void) {
    const char *env = getenv("SteamDeck");
    if (env && !strcmp(env, "1")) return 1;
    char vendor[64], product[64];
    return read_line("/sys/class/dmi/id/board_vendor", vendor, sizeof vendor) && !strcmp(vendor, "Valve") &&
           read_line("/sys/class/dmi/id/product_name", product, sizeof product) &&
           (!strcmp(product, "Jupiter") || !strcmp(product, "Galileo"));
}

static const char *raw_get(const char *name);

static int profile_of(void) {
    int p = __atomic_load_n(&profile, __ATOMIC_ACQUIRE);
    if (p >= 0) return p;
    const char *set = raw_get(PREFIX "DEVICE_PROFILE");
    if (set && !strcmp(set, "deck")) p = 1;
    else if (set && *set && strcmp(set, "auto")) p = 0;
    else p = on_steam_deck();
    __atomic_store_n(&profile, p, __ATOMIC_RELEASE);
    return p;
}

const char *cfg_device_profile(void) { return profile_of() ? "deck" : "desktop"; }

static const char *raw_get(const char *name) {
    const char *e = getenv(name);
    if (e || strncmp(name, PREFIX, PREFIX_LEN)) return e;
    int n = __atomic_load_n(&n_entries, __ATOMIC_ACQUIRE);
    for (int i = 0; i < n; i++)
        if (!strcasecmp(entries[i].key, name + PREFIX_LEN)) return __atomic_load_n(&entries[i].value, __ATOMIC_ACQUIRE);
    return NULL;
}

static const char *cfg_get_now(const char *name) {
    const char *v = raw_get(name);
    if (v || strncmp(name, PREFIX, PREFIX_LEN) || !profile_of()) return v;
    for (size_t i = 0; i < sizeof DECK / sizeof *DECK; i++)
        if (!strcasecmp(DECK[i].key, name + PREFIX_LEN)) return DECK[i].value;
    return NULL;
}

// Whether name's value comes from the player (the environment or nx2navi.ini), not a profile or the table.
int cfg_is_set(const char *name) { return raw_get(name) != NULL; }

unsigned cfg_version(void) { return __atomic_load_n(&version, __ATOMIC_ACQUIRE); }

int cfg_setenv(const char *name, const char *value) {
    int ok = plat_setenv(name, value);
    __atomic_add_fetch(&version, 1, __ATOMIC_RELEASE);
    return ok;
}

int cfg_from_env(const char *name) { return getenv(name) != NULL; }

void cfg_set(const char *name, const char *value) {
    if (strncmp(name, PREFIX, PREFIX_LEN) || !value) return;
    char key[64];
    snprintf(key, sizeof key, "%s", name + PREFIX_LEN);
    for (char *p = key; *p; p++) *p = (char)toupper((unsigned char)*p);
    plat_mutex_lock(&set_lock);
    int i = 0, n = n_entries;
    while (i < n && strcmp(entries[i].key, key)) i++;
    if (i < n) {
        if (!entries[i].value || strcmp(entries[i].value, value)) {
            if (entries[i].value) kept_n++, kept_bytes += strlen(entries[i].value) + 1;   // cfg_sizes
            __atomic_store_n(&entries[i].value, strdup(value), __ATOMIC_RELEASE);   // the old string stays for readers
            entries[i].dirty = 1;
        }
        entries[i].removed = 0;   // a value set after cfg_reset replaces the line instead of deleting it
    } else if (n < CFG_MAX_KEYS) {
        entries[n] = (CfgEntry){strdup(key), strdup(value), 1, 0};
        __atomic_store_n(&n_entries, n + 1, __ATOMIC_RELEASE);
    }
    __atomic_add_fetch(&version, 1, __ATOMIC_RELEASE);
    plat_mutex_unlock(&set_lock);
}

int cfg_reset(const char *const *keys, int n) {
    int done = 0;
    plat_mutex_lock(&set_lock);
    for (int k = 0; k < n; k++) {
        char key[64], var[80];
        snprintf(key, sizeof key, "%s", keys[k]);
        const char *name = norm_key(key);
        snprintf(var, sizeof var, PREFIX "%s", name);
        if (getenv(var)) continue;   // the environment wins; the file line stays
        int i = 0;
        while (i < n_entries && strcmp(entries[i].key, name)) i++;
        if (i == n_entries) {   // no value in memory ("key =" in the file): the line still has to go
            if (i == CFG_MAX_KEYS) continue;
            entries[i] = (CfgEntry){strdup(name), NULL, 0, 1};
            __atomic_store_n(&n_entries, i + 1, __ATOMIC_RELEASE);
        } else if (entries[i].value) {
            __atomic_store_n(&entries[i].value, NULL, __ATOMIC_RELEASE);   // the old string stays for readers
            entries[i].dirty = 0;
            entries[i].removed = 1;   // cfg_save_to deletes the key's lines
            done++;
        } else {
            entries[i].removed = 1;
        }
    }
    if (done) __atomic_add_fetch(&version, 1, __ATOMIC_RELEASE);
    plat_mutex_unlock(&set_lock);
    return done;
}

int cfg_last_save_ok(void) { return __atomic_load_n(&save_ok, __ATOMIC_ACQUIRE); }

void cfg_each_active(void (*cb)(const char *key, const char *value, int from_env, void *ctx), void *ctx) {
    int n = __atomic_load_n(&n_entries, __ATOMIC_ACQUIRE);
    for (int i = 0; i < n; i++) {
        const char *v = __atomic_load_n(&entries[i].value, __ATOMIC_ACQUIRE);
        if (v) cb(entries[i].key, v, 0, ctx);
    }
    for (size_t i = 0; i < sizeof known / sizeof *known; i++) {
        char var[80];
        snprintf(var, sizeof var, PREFIX "%s", known[i]);
        const char *e = getenv(var);
        if (e) cb(known[i], e, 1, ctx);
    }
}

const CfgMeta *cfg_meta(const char *key) {
    char k[64];
    snprintf(k, sizeof k, "%s", key);
    const char *name = norm_key(k);
    for (size_t i = 0; i < sizeof metas / sizeof *metas; i++)
        if (!strcmp(metas[i].key, name)) return &metas[i];
    return NULL;
}

const char *cfg_default(const char *key) {
    char k[64];
    snprintf(k, sizeof k, "%s", key);
    const char *name = norm_key(k);
    if (profile_of())
        for (size_t i = 0; i < sizeof DECK / sizeof *DECK; i++)
            if (!strcasecmp(DECK[i].key, name)) return DECK[i].value;
    const CfgMeta *m = cfg_meta(key);
    return m ? m->def : NULL;
}

const CfgMeta *cfg_meta_at(int i) { return i >= 0 && (size_t)i < sizeof metas / sizeof *metas ? &metas[i] : NULL; }

static const char *cfg_value_now(const char *name) {
    const char *e = cfg_get_now(name);
    if (e && *e) return e;
    const CfgMeta *m = cfg_meta(name);
    if (!m) rt_fatal("cfg: %s has no row in the settings table", name);
    return m->def;
}

// Lookup memo. cfg_get reads the environment (getenv walks the whole block comparing names) and then the file's
// entries and the settings table by name, and several per-draw paths ask for live settings: on the replay thread those
// lookups were about a sixth of the samples in a play profile (ucrtbase's string routines). Each thread keeps the
// answers it got per name until the settings version changes (cfg_set, cfg_reset, cfg_load); values stay valid that
// long because a set keeps the old string for readers and a load bumps the version. NX2NAVI_CFG_MEMO=0 (environment
// only, read once) asks every time.
#define CFG_MEMO_N 256
typedef struct { unsigned ver; uint32_t h; uint8_t kind; char name[55]; const char *v; } CfgMemo;
static _Thread_local CfgMemo cfg_memo[CFG_MEMO_N];
static int cfg_memo_on = -1;

static const char *cfg_memo_get(const char *name, int kind) {
    if (cfg_memo_on < 0) {
        const char *e = getenv(PREFIX "CFG_MEMO");
        cfg_memo_on = !(e && !strcmp(e, "0"));
    }
    size_t len = strlen(name);
    if (!cfg_memo_on || len >= sizeof cfg_memo[0].name) return kind ? cfg_value_now(name) : cfg_get_now(name);
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < len; i++) h = (h ^ (uint8_t)name[i]) * 16777619u;
    h = h * 2 + (uint32_t)kind;
    CfgMemo *m = &cfg_memo[h % CFG_MEMO_N];
    unsigned ver = __atomic_load_n(&version, __ATOMIC_ACQUIRE) + 1;   // 0 marks an empty slot
    if (m->ver == ver && m->h == h && m->kind == kind && !memcmp(m->name, name, len + 1)) return m->v;
    const char *v = kind ? cfg_value_now(name) : cfg_get_now(name);   // read after ver: a change meanwhile misses next time
    m->ver = ver, m->h = h, m->kind = (uint8_t)kind, m->v = v;
    memcpy(m->name, name, len + 1);
    return v;
}

const char *cfg_get(const char *name) { return cfg_memo_get(name, 0); }
const char *cfg_value(const char *name) { return cfg_memo_get(name, 1); }

// The row's default as a number, for a set value that does not parse.
static double default_number(const char *name) {
    return strtod(cfg_meta(name)->def, NULL);
}

double cfg_double(const char *name) {
    const char *v = cfg_value(name);
    char *end;
    double d = strtod(v, &end);
    return end == v || *end || d != d ? default_number(name) : d;
}

int cfg_int(const char *name) {
    const char *v = cfg_value(name);
    char *end;
    long l = strtol(v, &end, 10);
    return end == v || *end ? (int)default_number(name) : (int)l;
}

int cfg_bool(const char *name) { return strcmp(cfg_value(name), "0") != 0; }

// ---- the restart-only settings' values at start (cfg_init), which the running game keeps using whatever the menu sets
// later: cfg_pending compares against them, so the menu says "after restart" only while a restart would change
// something. One per row of metas, NULL for a live one.
static const char *at_start[sizeof metas / sizeof *metas];

static const char *meta_value(size_t i) {
    char name[80];
    snprintf(name, sizeof name, PREFIX "%s", metas[i].key);
    return cfg_value(name);
}

static void start_values(void) {
    for (size_t i = 0; i < sizeof metas / sizeof *metas; i++)
        if (!metas[i].live) at_start[i] = strdup(meta_value(i));   // a copy: cfg_load (the probes) frees what it read
}

// A value the environment sets never waits: the menu cannot change it and a restart keeps the environment. The command
// line's --monitor and --audio-device arrive that way after the start values are taken (main.c).
static int pending_at(size_t i) {
    char name[80];
    if (!at_start[i]) return 0;
    snprintf(name, sizeof name, PREFIX "%s", metas[i].key);
    return !cfg_from_env(name) && strcmp(at_start[i], cfg_value(name)) != 0;
}

int cfg_pending(const char *key) {
    const CfgMeta *m = cfg_meta(key);
    return m ? pending_at((size_t)(m - metas)) : 0;
}

int cfg_pending_count(void) {
    int n = 0;
    for (size_t i = 0; i < sizeof metas / sizeof *metas; i++) n += pending_at(i);
    return n;
}

// ---- cfg_save: the file as lines, each with its line ending
typedef struct { const char *s; size_t n; } Line;   // n includes the '\n' (and '\r') when present

static int line_count(const char *p, size_t size) {
    int n = 0;
    for (size_t i = 0; i < size; i++) n += p[i] == '\n';
    return n + 1;
}

// The line's text without its ending, BOM and surrounding blanks, into out.
static void line_text(const Line *l, int first, char *out, size_t cap) {
    size_t n = l->n;
    const char *s = l->s;
    if (first && n >= 3 && !memcmp(s, "\xEF\xBB\xBF", 3)) s += 3, n -= 3;
    while (n && (s[n - 1] == '\n' || s[n - 1] == '\r')) n--;
    if (n >= cap) n = cap - 1;
    memcpy(out, s, n);
    out[n] = 0;
    char *t = trim(out);
    memmove(out, t, strlen(t) + 1);
}

// The key of a "key = value" line (comment = 0) or of a commented "# key = value" line (comment = 1), normalised,
// into key; 0 when the line is not one.
static int line_key(const char *text, int comment, char *key, size_t cap) {
    if (comment) {
        if (*text != '#' && *text != ';') return 0;
        text++;
        while (*text == ' ' || *text == '\t') text++;
    }
    if (!*text || *text == '#' || *text == ';' || *text == '[') return 0;
    const char *eq = strchr(text, '=');
    if (!eq) return 0;
    size_t n = (size_t)(eq - text);
    while (n && (text[n - 1] == ' ' || text[n - 1] == '\t')) n--;
    if (!n || n >= cap) return 0;
    for (size_t i = 0; i < n; i++)
        if (!isalnum((unsigned char)text[i]) && text[i] != '_') return 0;
    char tmp[128];
    memcpy(tmp, text, n);
    tmp[n] = 0;
    snprintf(key, cap, "%s", norm_key(tmp));
    return 1;
}

// The section name of a "[name]" line, lower-case, into out; 0 when the line is not one.
static int line_section(const char *text, char *out, size_t cap) {
    if (*text != '[') return 0;
    const char *e = strchr(text, ']');
    if (!e) return 0;
    char tmp[128];
    size_t n = (size_t)(e - text - 1);
    if (n >= sizeof tmp) n = sizeof tmp - 1;
    memcpy(tmp, text + 1, n);
    tmp[n] = 0;
    snprintf(out, cap, "%s", trim(tmp));
    for (char *p = out; *p; p++) *p = (char)tolower((unsigned char)*p);
    return 1;
}

typedef struct { char *buf; size_t n, cap; } Out;
static void put(Out *o, const char *s, size_t n) {
    if (o->n + n + 1 > o->cap) {
        o->cap = (o->n + n + 1) * 2;
        o->buf = realloc(o->buf, o->cap);
        if (!o->buf) rt_fatal("cfg: out of memory writing nx2navi.ini");
    }
    memcpy(o->buf + o->n, s, n);
    o->n += n;
}
static void puts_(Out *o, const char *s) { put(o, s, strlen(s)); }

static const char *section_of(const char *key) {
    for (size_t i = 0; i < sizeof sections / sizeof *sections; i++)
        if (!strcmp(sections[i].key, key)) return sections[i].section;
    if (!strncmp(key, "CHEAT_", 6)) return "cheats";   // cheats.c: one key per cheat of the cheat file
    return NULL;
}

// A value as cfg_load reads it back: quoted when it starts or ends with a blank (cfg_load trims those) or is itself
// quoted (cfg_load takes one pair of quotes off). A mod's folder name in mods_order can start with a space.
static void put_value(Out *o, const char *v) {
    size_t n = strlen(v);
    int quote = n && (v[0] == ' ' || v[0] == '\t' || v[n - 1] == ' ' || v[n - 1] == '\t' ||
                      (n >= 2 && v[0] == '"' && v[n - 1] == '"'));
    if (quote) puts_(o, "\"");
    puts_(o, v);
    if (quote) puts_(o, "\"");
}

// "key = value" + eol for entry e (key lower-case)
static void put_entry(Out *o, const CfgEntry *e, const char *eol) {
    char k[64];
    snprintf(k, sizeof k, "%s", e->key);
    for (char *p = k; *p; p++) *p = (char)tolower((unsigned char)*p);
    puts_(o, k);
    puts_(o, " = ");
    put_value(o, e->value);
    puts_(o, eol);
}

int cfg_save_to(const char *path) {
    plat_mutex_lock(&set_lock);
    int n = n_entries, any = 0;
    for (int i = 0; i < n; i++) any |= entries[i].dirty | entries[i].removed;
    if (!any) {
        __atomic_store_n(&save_ok, 1, __ATOMIC_RELEASE);
        plat_mutex_unlock(&set_lock);
        return 1;
    }
    uint64_t size = 0;
    uint8_t *data = plat_file_read(path, &size);
    if (!data && plat_file_exists(path) && plat_file_size(path) != 0) {
        // the file is there but unreadable (locked, transient error): rebuilding it from nothing would lose its lines
        rt_log("cfg: cannot read %s, nothing written", path);
        __atomic_store_n(&save_ok, 0, __ATOMIC_RELEASE);
        plat_mutex_unlock(&set_lock);
        return 0;
    }
    const char *text = data ? (const char *)data : "";
    int nl = data ? line_count(text, size) : 0;
    Line *lines = calloc((size_t)nl + 1, sizeof *lines);
    // after[i]: entries to write after line i (-1 = none); replace[i]: the entry whose value line i now carries
    int *replace = malloc(sizeof(int) * ((size_t)nl + 1)), *after_key = malloc(sizeof(int) * (size_t)(n + 1));
    char *drop = calloc((size_t)nl + 1, 1);   // lines of a reset key
    if (!lines || !replace || !after_key || !drop) rt_fatal("cfg: out of memory writing nx2navi.ini");
    for (int i = 0, at = 0; i < nl; i++) {
        size_t start = (size_t)at;
        while ((uint64_t)at < size && text[at] != '\n') at++;
        if ((uint64_t)at < size) at++;
        lines[i] = (Line){text + start, (size_t)at - start};
        replace[i] = -1;
    }
    if (nl && !lines[nl - 1].n) nl--;   // the text ended with a newline: no last empty line
    const char *eol = data && strstr(text, "\r\n") ? "\r\n" : "\n";
    char t[1024], k[128], sec[128];
    for (int e = 0; e < n; e++) {
        after_key[e] = -2;   // -2 clean, -1 append in a new section at the end, >= 0 insert after that line
        if (entries[e].removed)
            for (int i = 0; i < nl; i++) {
                line_text(&lines[i], i == 0, t, sizeof t);
                if (line_key(t, 0, k, sizeof k) && !strcmp(k, entries[e].key)) drop[i] = 1;
            }
        if (!entries[e].dirty) continue;
        int last = -1;
        for (int i = 0; i < nl; i++) {
            line_text(&lines[i], i == 0, t, sizeof t);
            if (line_key(t, 0, k, sizeof k) && !strcmp(k, entries[e].key)) last = i;
        }
        if (last >= 0) {
            replace[last] = e;
            continue;
        }
        const char *want = section_of(entries[e].key);
        int in = 0, example = -1, tail = -1;
        for (int i = 0; i < nl && want; i++) {
            line_text(&lines[i], i == 0, t, sizeof t);
            if (line_section(t, sec, sizeof sec)) {
                in = !strcmp(sec, want);
                if (in) tail = i;
                continue;
            }
            if (!in) continue;
            if (*t) tail = i;
            if (line_key(t, 1, k, sizeof k) && !strcmp(k, entries[e].key)) example = i;
        }
        after_key[e] = example >= 0 ? example : tail;
    }
    Out o = {0};
    for (int i = 0; i < nl; i++) {
        if (drop[i]) {
            // the line leaves the file; keys inserted after it still need a line ending before them
        } else if (replace[i] >= 0) {
            // keep the line's indentation and key spelling, put the new value after " = "
            const char *s = lines[i].s;
            size_t len = lines[i].n;
            if (i == 0 && len >= 3 && !memcmp(s, "\xEF\xBB\xBF", 3)) put(&o, s, 3), s += 3, len -= 3;
            const char *eq = memchr(s, '=', len);
            size_t kn = (size_t)(eq - s);
            while (kn && (s[kn - 1] == ' ' || s[kn - 1] == '\t')) kn--;
            put(&o, s, kn);
            puts_(&o, " = ");
            put_value(&o, entries[replace[i]].value);
            int had_eol = len && s[len - 1] == '\n';
            if (had_eol) puts_(&o, eol);
        } else {
            put(&o, lines[i].s, lines[i].n);
        }
        for (int e = 0; e < n; e++)
            if (after_key[e] == i) {
                if (o.n && o.buf[o.n - 1] != '\n') puts_(&o, eol);
                put_entry(&o, &entries[e], eol);
            }
    }
    // keys whose section the file lacks: one new section each, at the end
    for (int e = 0; e < n; e++) {
        if (after_key[e] != -1) continue;
        const char *want = section_of(entries[e].key);
        if (o.n && o.buf[o.n - 1] != '\n') puts_(&o, eol);
        if (want) {
            if (o.n) puts_(&o, eol);
            puts_(&o, "[");
            puts_(&o, want);
            puts_(&o, "]");
            puts_(&o, eol);
        }
        put_entry(&o, &entries[e], eol);
        // the following keys of the same new section go right after this one
        for (int f = e + 1; f < n && want; f++)
            if (after_key[f] == -1 && section_of(entries[f].key) && !strcmp(section_of(entries[f].key), want)) {
                put_entry(&o, &entries[f], eol);
                after_key[f] = -3;
            }
    }
    int ok = plat_file_write_atomic(path, o.buf ? o.buf : "", o.n);
    if (ok)
        for (int e = 0; e < n; e++) entries[e].dirty = entries[e].removed = 0;
    else
        rt_log("cfg: cannot write %s", path);
    __atomic_store_n(&save_ok, ok ? 1 : 0, __ATOMIC_RELEASE);
    free(o.buf);
    free(drop);
    free(after_key);
    free(replace);
    free(lines);
    free(data);
    plat_mutex_unlock(&set_lock);
    return ok;
}

static char ini_path[1024];

static int find_ini(void) {
    if (ini_path[0]) return 1;
    char path[1024];
    if (!plat_exe_path(path, sizeof path)) return 0;
    char *cut = plat_last_sep(path);
    if (!cut) cut = strrchr(path, '/');
    if (!cut || cut + 12 >= path + sizeof path) return 0;   // room for "nx2navi.ini" and its NUL after the separator
    strcpy(cut + 1, "nx2navi.ini");
    snprintf(ini_path, sizeof ini_path, "%s", path);
    return 1;
}

int cfg_save(void) {
    if (find_ini()) return cfg_save_to(ini_path);
    __atomic_store_n(&save_ok, 0, __ATOMIC_RELEASE);
    return 0;
}

// What the file sets, in the log once at start: the settings of this version with their values, in lines of up to
// about 900 characters, and then the keys that are no settings of this version, which do nothing. A report's log then
// shows which settings its run had and which of the file's lines had no effect; before, it held only the count of
// keys, and a player who had added a line this version does not read could take it for the cause of what changed.
static void log_file_settings(void) {
    char line[1024];
    size_t k = 0;
    for (int i = 0; i < n_entries; i++) {
        if (!entries[i].value || !is_known(entries[i].key)) continue;
        char key[64];
        snprintf(key, sizeof key, "%s", entries[i].key);
        for (char *p = key; *p; p++) *p = (char)tolower((unsigned char)*p);
        // a co-op room's secret, the relay, the player's name: only that they are set (cfg_private.h)
        const char *value = cfg_private_key(entries[i].key) && entries[i].value[0] ? CFG_PRIVATE_SHOWN : entries[i].value;
        size_t vl = strlen(value);
        if (vl > 200) vl = 200;   // a long value (a list of mods) is cut: the line shows that the key is set
        if (k && k + strlen(key) + vl + 5 > 900) {
            rt_log("cfg: nx2navi.ini sets: %s", line);
            k = 0;
        }
        k += (size_t)snprintf(line + k, sizeof line - k, "%s%s = %.*s", k ? ", " : "", key, (int)vl, value);
    }
    if (k) rt_log("cfg: nx2navi.ini sets: %s", line);
    if (n_unknown)
        rt_log("cfg: nx2navi.ini has %d key%s that %s no setting of this version and %s nothing: %s%s", n_unknown,
               n_unknown == 1 ? "" : "s", n_unknown == 1 ? "is" : "are", n_unknown == 1 ? "does" : "do", unknown_keys,
               unknown_cut ? ", ..." : "");
}

static void load_ini(void) {
    if (!find_ini()) return;
    if (!cfg_load(ini_path)) {
        rt_log("cfg: no settings file at %s", ini_path);
        return;
    }
    rt_log("cfg: read %s: %d keys", ini_path, n_entries);
    log_file_settings();
    for (int i = 0; i < n_entries; i++) {
        char var[128];
        snprintf(var, sizeof var, PREFIX "%s", entries[i].key);
        if (getenv(var)) rt_log("cfg: %s is set in the environment and overrides nx2navi.ini", var);
    }
}

int cfg_ini_read(const char *path, const char *key, char *out, size_t cap) {
    if (cap) out[0] = 0;
    uint64_t size = 0;
    uint8_t *data = plat_file_read(path, &size);
    if (!data || !size) {   // gone or empty for a moment (an editor saving it), or locked: no reading, not "no value"
        free(data);
        return -1;
    }
    int found = 0;
    char *p = (char *)data;
    if (size >= 3 && !memcmp(p, "\xEF\xBB\xBF", 3)) p += 3;   // as cfg_load: editors add a BOM
    for (;;) {
        char *eol = strchr(p, '\n');
        if (eol) *eol = 0;
        char *s = trim(p), *eq = strchr(s, '=');
        if (*s && *s != '#' && *s != ';' && *s != '[' && eq) {
            *eq = 0;
            const char *k = trim(s);
            size_t i = 0;
            while (k[i] && key[i] && tolower((unsigned char)k[i]) == tolower((unsigned char)key[i])) i++;
            if (!k[i] && !key[i]) {   // as cfg_load: a repeated key's later line wins, quotes go, "key =" sets nothing
                char *v = trim(eq + 1);
                size_t vl = strlen(v);
                if (vl >= 2 && v[0] == '"' && v[vl - 1] == '"') v[vl - 1] = 0, v++;
                snprintf(out, cap, "%s", v);
                found = *v != 0;
            }
        }
        if (!eol) break;
        p = eol + 1;
    }
    free(data);
    return found;
}

int cfg_ini_now(const char *key, char *out, size_t cap) { return find_ini() ? cfg_ini_read(ini_path, key, out, cap) : 0; }

void cfg_init(void) {
    load_ini();
    if (profile_of())
        rt_log("cfg: device profile Steam Deck: defaults for the keys nx2navi.ini does not set (NX2NAVI_DEVICE_PROFILE=desktop "
               "turns it off)");
    start_values();
}

// NX2NAVI_TRACE=perf "mem: sizes" (perf.c): settings entries, and the replaced values cfg_set keeps for readers.
void cfg_sizes(PerfSizes *s) {
    plat_mutex_lock(&set_lock);
    uint64_t n = (uint64_t)n_entries, kn = kept_n, kb = kept_bytes;
    plat_mutex_unlock(&set_lock);
    perf_size(s, "settings", n, CFG_MAX_KEYS, PERF_SIZE_NONE);
    perf_size(s, "replaced setting values kept", kn, 0, kb);
}
