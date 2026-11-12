"""Build Ocarina of Time (Switch 2, game 1.1.1) natively for Windows or Linux from the player's own game files.

    python -I tools/make_game.py --nup "C:\\path\\game.nup" --out "C:\\Games\\Ocarina"
    python -I tools/make_game.py --exefs "C:\\path\\exefs" --romfs "C:\\path\\romfs" --out "C:\\Games\\Ocarina"
    python -I tools/make_game.py --nup "C:\\path\\game.nup" --check          # only checks the game's version
    python -I tools/make_game.py --rebuild --out "C:\\Games\\Ocarina"         # a newer kit, the files of the last build

Steps, each skipped when a run before finished it for the same inputs (state in <out>/build-cache/state.json):
  1. check    the title (0400486024DEC000) and main's build id (game 1.1.1)
  2. exefs    the five ExeFS files, every module's segments unpacked (ZBIC through the player's own `zbic` package)
              and stored plainly, as the runtime's loader reads them -> <out>/exefs
  3. romfs    every RomFS file, checked against the container's sha256, into one image -> <out>/romfs/romfs.bin and
              index.json (16-byte aligned, in the byte order of the paths). When the lift and the compile follow, it
              runs on a thread of its own beside them: only the RomFS index and romfs.c wait for the image
  4. lift     recomp/gen.py with recomp/patches-v111.txt -> <out>/build-cache/lift, plus the two SDK bring-up fixes.
              The code cheats and ExeFS patch files of the mods in <out>/mods (tools/mod_files.py) become patch sets
              first (tools/cheats.py, tools/exefs_patches.py), which the game switches on with their mod; the step's
              key names those files, and build-cache/mod-code.json records the ones it took up (the launcher
              rebuilds the game before it starts when a mod brings others, launcher/mods.py)
  5. compile  the lifted C and runtime/ with MSYS2's clang, linked into <out>/ocarina.exe; the MSYS2 DLLs it loads,
              the shader translator and romfs-index.bin beside it
  6. shaders  every shader in the game files (tools/prepare_shaders.py, default bind state) translated, and each one's
              GPU driver build done once into the driver's own cache, so play stops for them far less often
              (`ocarina.exe --prepare-shaders`); variants met in play with another bind state are still translated
              then, and the driver cache is per executable and is lost when the driver clears it
  7. prompts  only with button art in the kit (<kit>/prompt-art, tools/make_launcher.py and make_kit.py
              --prompt-art): tools/pad_prompts.py draws the Xbox, PlayStation, Steam Deck and keyboard and mouse button prompts
              from it into <out>/romfs-patches-* (the Controls menu's Button prompts loads them), the keys as the
              bind_ lines of <out>/nx2navi.ini give them. Its key names everything it reads (the RomFS, the art, the
              tools, those lines), so it runs again only when one of them changed or a folder is gone, and running
              it makes no other step run again. A failure here leaves the game and its shaders complete.
  8. scan     tools/romfs_check.py decompresses every file of the game data once, writes <out>/romfs-check.json and
              makes stand-ins for damaged textures in <out>/mods. Its key names the game files and the check's
              version (SCAN_TOOL, changed by hand: an edit of the tool alone does not run it again), and it
              is last: nothing of it runs while the game does.
              Without art the step is skipped and the game shows its own prompts, as before

The exefs and romfs steps depend only on the game files; lift and compile depend on this kit's code too. A newer kit
therefore lifts and compiles again but keeps the unpacked game files, and --rebuild needs no game files at all. The
compile step keeps every object with the dependency list clang wrote for it (build-cache/obj/objects.json) and
compiles a file again only when it, a header it includes, the flags or the compiler changed; the kit's own folder
is not part of that key, so a kit in another folder (the launcher keeps each version in its own) reuses the objects.
The list goes to disk every 25 objects while the compile runs, so a compile that is cut short keeps what it finished.
A kit with lib/libnx2navirt.a built by --make-runtime-lib for the same runtime/ and the same compiler links the
runtime from it and compiles only the runtime files that include headers written per install. The new ocarina.exe
replaces the old one only once it links (with lld when the toolchain has it), so a failed build leaves the last good
one in place. Compiles run on every logical processor, or on as many as the free memory allows (default_jobs; --jobs
sets the number), the largest files first. The lifted code is built for this
PC's x86-64 level (cpu_level), with MSYS2 CLANG64's clang when it sits beside the UCRT64 folder and reports the same
version (fast_clang: the same objects, about a fifth faster).

    python tools/make_game.py --make-runtime-lib <kit>/lib --msys <ucrt64>   # the release's prebuilt runtime

On Linux the same steps make <out>/ocarina with the clang, clang++ and lld on PATH and the system's libraries
(Vulkan, SDL3, Opus, libvpx, dav1d, zstd; libstdc++ static, rpath $ORIGIN), as tools/build_game.py's Linux
build does: the runtime's *_sdl.* and platform/linux.c take the place of its *_win.*, platform/win32.c and
nx2navi.rc, and --msys is not used. VMA_INCLUDE can name the directory holding vk_mem_alloc.h; otherwise the
system header or the pinned download is used. README-BuildKit.md, "Linux", lists the packages.

Game content is written only to --out; a downloaded VMA header is cached in toolchain/vma, and --make-runtime-lib
writes DIR. The game folder holds only what this PC built from the player's files: never
share it. Needs Python 3.10+ with numpy, lz4, zstandard and zbic (pip), with fonttools and pillow too when the kit has
button art, and MSYS2 UCRT64's clang and libraries (README-BuildKit.md lists the commands). Delete <out>/build-cache
afterwards to free about 5 GB; the next build then compiles everything again.

--events adds one machine-readable line per step and progress change to stdout, each starting with "@@" and followed
by a JSON object (launcher/build.py reads them): {"ev": "step", "step": ..., "state": "start" | "done" | "skip"},
{"ev": "progress", "step": ..., "done": n, "total": n, "unit": ...}, {"ev": "phase", ...}, {"ev": "error", "message":
...} and {"ev": "end", ...}.
"""
import argparse
import concurrent.futures
import errno
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT / 'tools'))
import nup  # noqa: E402
import mod_files  # noqa: E402
import prompt_art  # noqa: E402

TITLE_ID = 0x0400486024DEC000
MAIN_BUILD_ID = 'e0f3ccfaf50065ae4ea01eb5aeb13b1a'   # main of game 1.1.1
EXEFS_FILES = ('main', 'main.npdm', 'sdk', 'subsdk0', 'rtld')
MODULES = ('main', 'sdk', 'subsdk0', 'rtld')
ALIGN = 16
STATE_VERSION = 2
STEPS = ('exefs', 'romfs', 'lift', 'compile', 'shaders', 'prompts', 'scan')
# Steps whose key names every input they read rather than the step before them: a step before them that runs does
# not make them run again, and they make no step after them run again.
OWN_KEY_STEPS = ('prompts', 'scan')
# (a step that runs, a later step that reads nothing it writes): the later step is not made to run again. The lift
# reads only the unpacked ExeFS (step_key), so the RomFS step leaves it alone: an install that was cut short while the
# image was still being packed, the lift beside it already done, packs the image again but does not lift again.
# The shaders step's key names the lift and the compile it depends on without the mods' code (step_key): a lift and a
# compile that run for a mod's code cheats or ExeFS patches leave the shaders prepared, as they leave the game's
# shaders as they were, and one that runs for anything else (another kit) changes that key and prepares them again.
UNAFFECTED = {('romfs', 'lift'), ('lift', 'shaders'), ('compile', 'shaders')}
# The Python packages tools/pad_prompts.py imports (pip name: import name), checked before a build with button art.
PROMPT_PACKAGES = {'zstandard': 'zstandard', 'fonttools': 'fontTools', 'pillow': 'PIL'}
# The line tools/pad_prompts.py prints for each overlay folder it wrote.
PROMPTS_WRITTEN = re.compile(r'^.+: \d+ files, the font grew by -?\d+ bytes$')
SCAN_CHECKED = re.compile(r'^checked (\d+) of (\d+)$')   # tools/romfs_check.py's progress lines
# What stands for tools/romfs_check.py in the scan step's key (step_key). The key held the sha256 of the script's
# bytes, so any edit of the script, a comment's too, made every install check its 26 GB of game files again, four
# minutes. It is this constant now, changed by hand when a change to the script can change what the check finds
# or writes, and only then. Its value is the sha256 the script had in 0.7.0 and 0.7.1, the releases that brought the
# check, so the key of every install that passed it stays what it is. tests/launcher_test.py
# test_the_file_checks_logic_is_the_recorded_one fails when the script's logic changes, to ask this question.
SCAN_TOOL = '55ea0c150aaf9dac93a2e6fd3b4ca11e1c572b5eee8b6bd74960535964384634'
# The same for the tools that draw the button prompts (tools/pad_prompts.py, prompt_art.py, button_icons.py) in the
# prompts step's key, which held the digest of the three scripts' bytes: two of four releases changed a script, and
# each time every install drew its prompts again, a minute. Changed by hand when a change to the scripts can change
# what they write. Its value is their digest in 0.7.0 and 0.7.1, so the key of an install made by those stays what it
# is. tests/launcher_test.py test_the_prompt_tools_logic_is_the_recorded_one asks the question when the logic changes.
PROMPT_TOOLS = 'ad59f2c4533f6ea2630b916ef5f3695f72daffe9b2b7663eea4168a4f08e4249'

# Bring-up fixes in the lifted SDK, each anchored on exact generated text that must occur once.
LIFT_FIXES = [
    # nn::oe::SetupGpuErrorHandler (sdk+0x59a430) registers a system worker the host does not provide: GPU error
    # reporting is not needed, so it returns at once.
    ('sdk', 'void f_700579a430(Cpu *c) {\n    FN_ENTER(0x700579a430ull);\n',
     'void f_700579a430(Cpu *c) {\n    c->x[0] = 0; return;   /* bring-up: GpuErrorHandler (make_game.py) */\n'
     '    FN_ENTER(0x700579a430ull);\n'),
    # The thread-start path loads TPIDR_EL0 from TLR+0x1F0 before the SDK stored its TLS there for a host-created
    # thread: a zero slot keeps the TLS pointer the host seeded.
    ('sdk', '    { uint64_t ea = c->x[19] + 0x1b8ull; ST64(ea, x8); }\n    x8 = (uint64_t)(c->tpidrro);\n'
     '    { uint64_t ea = x8 + 0x1f0ull; uint64_t v = LD64(ea); x8 = (uint64_t)(v); }\n    c->tpidr = x8;\n',
     '    { uint64_t ea = c->x[19] + 0x1b8ull; ST64(ea, x8); }\n    x8 = (uint64_t)(c->tpidrro);\n'
     '    { uint64_t ea = x8 + 0x1f0ull; uint64_t v = LD64(ea); x8 = (uint64_t)(v); }\n'
     '    if (x8) c->tpidr = x8;   /* zero slot: keep the host\'s TLS (make_game.py) */\n'),
]
WIN = os.name == 'nt'
if WIN:
    LINK_LIBS = ['-lvulkan-1', '-lSDL3', '-lopus', '-lvpx', '-l:libdav1d.a', '-lzstd', '-l:libatomic.a', '-lbcrypt', '-lws2_32',
                 '-limm32', '-luuid', '-lole32', '-luser32', '-lshell32', '-lgdi32', '-ladvapi32', '-lwinmm']
else:   # tools/build_game.py's Linux libraries, plus dav1d (the AV1 movies) and libatomic
    LINK_LIBS = ['-lvulkan', '-lSDL3', '-lopus', '-lvpx', '-ldav1d', '-lzstd', '-l:libstdc++.a', '-latomic',
                 '-lm', '-lpthread', '-ldl']
SYSTEM_DLLS_SKIP = {'vulkan-1.dll'}   # the graphics driver's loader, never a copy
EXE_NAME = 'ocarina.exe' if WIN else 'ocarina'
XLATE_NAME = 'nx2navi_xlate.exe' if WIN else 'nx2navi_xlate'
# Headers this build writes per install (gen.py, romfs_index.py): a runtime source that includes one, directly or
# through a runtime header, is compiled on the player's PC; every other runtime source can come prebuilt.
GENERATED_HEADERS = ('layout.h', 'patches.h', 'romfs_config.h')
RUNTIME_LIB = 'libnx2navirt.a'
FFX4_DLLS = ('amd_fidelityfx_loader_vk.dll', 'amd_fidelityfx_upscaler_vk.dll')   # AMD's FSR SDK 2.4, for FSR 4
FFX4_FG_DLL = 'amd_fidelityfx_framegeneration_vk.dll'   # ... and its frame generation, with the two above
RUNTIME_PATTERNS = ('*.c', '*.cpp', 'platform/*.c', 'ui/*.cpp', 'third_party/imgui/*.cpp',
                    'third_party/imgui/backends/*.cpp', 'third_party/monocypher/*.c')
INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]+"([^"]+)"', re.M)

# Runs a script the way `python script.py` does, with the script's folder first on sys.path. The launcher's
# embeddable Python leaves the script folder off sys.path (its ._pth file), so every script this build starts goes
# through this.
RUN_SCRIPT = ('import os, runpy, sys; s = sys.argv[1]; sys.argv = sys.argv[1:]; '
              'sys.path.insert(0, os.path.dirname(os.path.abspath(s))); runpy.run_path(s, run_name="__main__")')

EVENTS = False
OUT_LOCK = threading.Lock()   # the RomFS step prints from its own thread (main)


def say(msg):
    with OUT_LOCK:
        print(msg, flush=True)


def emit(ev, **fields):
    """One "@@{json}" line for the launcher (--events); nothing otherwise."""
    if EVENTS:
        line = '@@' + json.dumps({'ev': ev, **fields}, ensure_ascii=False)
        with OUT_LOCK:
            print(line, flush=True)


def fail(msg):
    emit('error', message=msg)
    with OUT_LOCK:
        print(f'\nerror: {msg}', file=sys.stderr, flush=True)
    sys.exit(1)


def error_text(e):
    """An exception in a line a player can read: what the system said and about which file for a file error, the
    full disk named as such, else the exception's kind and message."""
    if isinstance(e, OSError) and e.strerror:
        if e.errno == errno.ENOSPC:
            return f'the disk is full ({e.filename})' if e.filename else 'the disk is full'
        return e.strerror + (f': {e.filename}' if e.filename else '')
    return f'{type(e).__name__}: {e}' if str(e) else type(e).__name__


class Background:
    """One step on a thread of its own. join() waits for it and raises its failure (fail's SystemExit included) in
    the caller; failed() tells the caller early, so it can stop work that would be thrown away."""

    def __init__(self, fn):
        self.exc = None
        self.thread = threading.Thread(target=self._run, args=(fn,), name='romfs', daemon=True)
        self.thread.start()

    def _run(self, fn):
        try:
            fn()
        except BaseException as e:   # noqa: B036 - handed to join()
            self.exc = e

    def failed(self):
        return self.exc is not None

    def join(self):
        self.thread.join()
        if self.exc is not None:
            raise self.exc


def py_cmd(script, *args):
    """The command line that runs a Python script of this kit with this interpreter (see RUN_SCRIPT)."""
    flags = ['-I'] if sys.flags.isolated else []
    return [sys.executable, *flags, '-c', RUN_SCRIPT, str(script), *map(str, args)]


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while b := f.read(8 << 20):
            h.update(b)
    return h.hexdigest()


def tree_digest(root, patterns):
    """sha256 over the relative names and contents of the files under root matching the glob patterns."""
    h = hashlib.sha256()
    files = sorted({p for pat in patterns for p in Path(root).glob(pat) if p.is_file() and '__pycache__' not in p.parts})
    for p in files:
        h.update(p.relative_to(root).as_posix().encode() + b'\0')
        h.update(p.read_bytes())
        h.update(b'\0')
    return h.hexdigest()


# ---------------------------------------------------------------- input

class Source:
    """The game files, from a .nup container or from extracted exefs/romfs folders."""

    def __init__(self, a):
        self.n = None
        if a.nup:
            try:
                self.n = nup.Nup(a.nup)
            except (OSError, nup.NupError) as e:
                fail(f'cannot read {a.nup}: {e}')
            # The base title and its update (title id + 0x800), the update first and the newest of each kind first.
            # A module comes from the first of them that holds it whole: an update's ExeFS may leave a module out or
            # hold something else under its name.
            pkgs = [p for p in self.n.packages if p['title_id'] in (TITLE_ID, TITLE_ID | 0x800)]
            if not pkgs:
                ids = ', '.join(f"{p['title_id']:016X}" for p in self.n.packages)
                fail(f'this .nup holds {ids}, not Ocarina of Time (0400486024DEC000)')
            pkgs.sort(key=lambda p: (p['kind'] != 1, -p['version']))
            self.exefs = {}
            for name in EXEFS_FILES:
                for p in pkgs:
                    fi = dict(self.n.tree(p['exefs'])).get(name) if p['exefs'] < self.n.ndir else None
                    magic = b'META' if name.endswith('.npdm') else b'NSO0'
                    if fi is not None and self.n.size(fi) >= 0x100 and next(self.n.chunks(fi))[:4] == magic:
                        self.exefs[name] = fi
                        break
            self.romfs = []
            for p in pkgs:
                if p['romfs'] < self.n.ndir:
                    self.romfs = list(self.n.tree(p['romfs']))
                if self.romfs:
                    break
            st = os.stat(a.nup)
            self.identity = f'nup {os.path.abspath(a.nup)} {st.st_size} {int(st.st_mtime)}'
        else:
            self.exefs_dir, self.romfs_dir = Path(a.exefs), Path(a.romfs)
            for d in (self.exefs_dir, self.romfs_dir):
                if not d.is_dir():
                    fail(f'{d} is not a folder')
            self.exefs = {name: self.exefs_dir / name for name in EXEFS_FILES if (self.exefs_dir / name).is_file()}
            self.romfs = [(p.relative_to(self.romfs_dir).as_posix(), p) for p in self.romfs_dir.rglob('*') if p.is_file()]
            # every file's size and modification time, not only their count: a RomFS extracted again (or edited)
            # with the same number of files, or a replaced sdk, kept the old unpacked files ("done before")
            h = hashlib.sha256()
            for rel, f in sorted([(n, self.exefs[n]) for n in self.exefs] + self.romfs):
                st = f.stat()
                h.update(f'{rel}\0{st.st_size}\0{st.st_mtime_ns}\n'.encode('utf-8', 'surrogateescape'))
            self.identity = (f'folders {self.exefs_dir.resolve()} {self.romfs_dir.resolve()} {len(self.romfs)} '
                             f'{h.hexdigest()[:32]}')
        missing = [x for x in EXEFS_FILES if x not in self.exefs]
        if missing:
            fail(f'the ExeFS lacks {", ".join(missing)} (or holds something else under that name)')
        if not self.romfs:
            fail('the RomFS is empty')

    def read_exefs(self, name):
        return self.n.read(self.exefs[name]) if self.n else self.exefs[name].read_bytes()

    def romfs_size(self, ref):
        return self.n.size(ref) if self.n else ref.stat().st_size

    def romfs_chunks(self, ref):
        if self.n:
            yield from self.n.chunks(ref)
            return
        with open(ref, 'rb') as f:
            while b := f.read(nup.CHUNK):
                yield b


class BuiltSource:
    """--rebuild: the ExeFS and RomFS an earlier run unpacked into the game folder, under that run's input identity.
    The ExeFS modules are stored plainly there but keep their headers, so check() reads the same build id."""

    def __init__(self, out, state):
        self.out = out
        if state.get('version') != STATE_VERSION or not state.get('input'):
            fail(f'{out} holds no finished build of this kit to rebuild from: give --nup, or --exefs and --romfs')
        for step in ('exefs', 'romfs'):
            if step not in state.get('steps', {}):
                fail(f'the last build in {out} did not finish its {step} step: give the game files again')
        missing = [n for n in EXEFS_FILES if not (out / 'exefs' / n).is_file()]
        missing += [f'romfs/{n}' for n in ('romfs.bin', 'index.json') if not (out / 'romfs' / n).is_file()]
        if missing:
            fail(f'{out} lacks {", ".join(missing)}: give the game files again')
        self.identity = state['input']

    def read_exefs(self, name):
        return (self.out / 'exefs' / name).read_bytes()


def check(src):
    main = src.read_exefs('main')
    if main[:4] != b'NSO0':
        fail('main is not an NSO module')
    build = main[0x40:0x50].hex()
    if build != MAIN_BUILD_ID:
        fail(f'this is not game version 1.1.1: main has build {build[:8]}, the recomp supports build {MAIN_BUILD_ID[:8]}. '
             'Install the 1.1.1 update and dump the game again.')
    return main


# ---------------------------------------------------------------- exefs

def flat_nso(raw, name):
    """An NSO with its segments stored uncompressed (the runtime's loader reads only those); segment hashes kept."""
    flags = struct.unpack_from('<I', raw, 0xC)[0]
    out = bytearray(raw[:0x100])
    for i, seg in enumerate(('text', 'ro', 'data')):
        file_off, mem_off, size = struct.unpack_from('<III', raw, 0x10 + i * 0x10)
        stored = struct.unpack_from('<I', raw, 0x60 + i * 4)[0]
        data = raw[file_off:file_off + (stored if flags >> i & 1 else size)]
        if flags >> i & 1:
            if flags & 0x80:
                try:
                    import zbic
                except ImportError:
                    fail('the game\'s modules need the zbic package: pip install zbic')
                data = zbic.decompress(data)
            else:
                import lz4.block
                data = lz4.block.decompress(data, uncompressed_size=size)
        if len(data) != size:
            fail(f'{name}: {seg} unpacks to {len(data):#x} bytes, its header says {size:#x}')
        if flags >> (3 + i) & 1 and hashlib.sha256(data).digest() != raw[0xA0 + 0x20 * i:0xC0 + 0x20 * i]:
            fail(f'{name}: the {seg} segment is damaged (sha256 mismatch)')
        struct.pack_into('<III', out, 0x10 + i * 0x10, len(out), mem_off, size)
        struct.pack_into('<I', out, 0x60 + i * 4, size)
        out += data
    struct.pack_into('<I', out, 0xC, flags & ~0x87)   # segments stored plainly, no ZBIC
    return bytes(out)


def step_exefs(src, out):
    dst = out / 'exefs'
    dst.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(EXEFS_FILES):
        raw = src.read_exefs(name)
        data = flat_nso(raw, name) if name in MODULES else raw
        (dst / name).write_bytes(data)
        say(f'  {name:9} {len(raw) / 1e6:6.1f} MB -> {len(data) / 1e6:6.1f} MB')
        emit('progress', step='exefs', done=i + 1, total=len(EXEFS_FILES), unit='files')


def exefs_digest(out):
    return tree_digest(out / 'exefs', EXEFS_FILES)


# ---------------------------------------------------------------- romfs

ROMFS_READ_AHEAD = 64 << 20   # total file bytes requested from workers in one group


def unreadable(rel, err):
    """Ends the build on a game file that could not be read while the RomFS image was packed, naming the file. A
    file that does not match the dump's own checksum (nup.NupError) means the dump is damaged: a copy that went
    wrong or a dump that was cut short, which only a new copy or a new dump mends."""
    if isinstance(err, nup.NupError):
        why = str(err).split(': ', 1)[-1]   # without nup.py's "file <number>: ", which says nothing to a player
        if why.startswith('damaged (') and why.endswith(')'):
            why = why[len('damaged ('):-1]
        fail(f'the dump is damaged: {rel}: {why}. Copy the dump to this PC again, or dump the game again, and then '
             'try again')
    fail(f'cannot read {rel} from the game files: {error_text(err)}')


def romfs_read_groups(files):
    group, buffered = [], 0
    for entry in files:
        size = entry[2]
        ahead = size if size <= ROMFS_READ_AHEAD else 0   # the writer streams larger files
        if buffered + ahead > ROMFS_READ_AHEAD or len(group) == 256:
            yield group
            group, buffered = [], 0
        group.append(entry)
        buffered += ahead
    if group:
        yield group


def step_romfs(src, out, jobs):
    dst = out / 'romfs'
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(((rel, ref, src.romfs_size(ref)) for rel, ref in src.romfs), key=lambda e: e[0].encode())
    total = sum(size for _, _, size in files)
    # What a run that was cut short (Cancel, a full disk, a power cut) left of the image goes first: this run
    # writes the image anew, and counted as used space its up to 26 GB made a retry on a nearly full disk fail the
    # check below although the retry needs no more room than the first try.
    part = dst / 'romfs.bin.part'
    try:
        part.unlink(missing_ok=True)
    except OSError as e:
        fail(f'cannot remove {part}, left by an earlier run ({e.strerror or e}): close whatever has it open')
    free = shutil.disk_usage(dst).free
    if free < total + (1 << 30):
        fail(f'{dst} needs {total / 1e9:.1f} GB free for the RomFS image, it has {free / 1e9:.1f} GB')
    say(f'  {len(files)} files, {total / 1e9:.1f} GB')
    entries, h, pos, done, t0 = [], hashlib.sha256(), 0, 0, time.time()
    emit('progress', step='romfs', done=0, total=total, unit='bytes')

    def load(e):   # small files are read whole on a worker; big ones are streamed by the writer
        _, ref, size = e
        if size > ROMFS_READ_AHEAD:
            return None
        try:
            return b''.join(src.romfs_chunks(ref))
        except (nup.NupError, OSError) as err:   # handed to the writer, which knows the file's name
            return err

    def pieces(rel, ref, data):
        """The file's bytes; a file of the game files that cannot be read, or does not match its checksum in the
        dump, ends the step with its name (unreadable)."""
        if isinstance(data, Exception):
            unreadable(rel, data)
        if data is not None:
            yield data
            return
        try:
            yield from src.romfs_chunks(ref)
        except (nup.NupError, OSError) as err:
            unreadable(rel, err)

    try:
        with open(part, 'wb', buffering=16 << 20) as img, concurrent.futures.ThreadPoolExecutor(jobs) as ex:
            for batch, group in enumerate(romfs_read_groups(files)):
                for (rel, ref, size), data in zip(group, ex.map(load, group)):
                    pad = -pos % ALIGN
                    if pad:
                        img.write(b'\0' * pad)
                        h.update(b'\0' * pad)
                        pos += pad
                    offset = pos
                    for chunk in pieces(rel, ref, data):
                        img.write(chunk)
                        h.update(chunk)
                        pos += len(chunk)
                    if pos - offset != size:
                        fail(f'{rel} changed size while packing the RomFS: expected {size} bytes, read {pos - offset}')
                    entries.append({'path': rel, 'offset': offset, 'size': size})
                    done += size
                emit('progress', step='romfs', done=done, total=total, unit='bytes')
                if batch % 40 == 0:
                    say(f'  RomFS {done / 1e9:5.1f} of {total / 1e9:.1f} GB, {time.time() - t0:.0f} s')
    except BaseException:
        # a failed image is of no use, the next run writes it anew: up to 26 GB of it would only fill the disk
        try:
            part.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    part.replace(dst / 'romfs.bin')
    index = {'status': 'pass', 'romfs_size': pos, 'romfs_sha256': h.hexdigest(), 'files': entries}
    (dst / 'index.json').write_text(json.dumps(index), encoding='utf-8')
    say(f'  RomFS image {pos / 1e9:.2f} GB in {time.time() - t0:.0f} s')


# ---------------------------------------------------------------- lift

EMITTED = re.compile(r'emitted (\d+)/(\d+) chunks')


def lift_digest():
    """The lifter's code, the patch table and the bring-up fixes: what the lifted C depends on besides the ExeFS."""
    h = hashlib.sha256(tree_digest(KIT / 'recomp', ('*.py', 'patches-v111.txt')).encode())
    h.update(repr(LIFT_FIXES).encode())
    return h.hexdigest()


# The mods' code (tools/mod_files.py): the tools that make patch sets of it, the patch file each writes, the status
# file the runtime reads in build-cache next to the exe (runtime/cheats.c, runtime/mods.c) and what the log calls it
MOD_STAGES = (('cheats.py', 'patches-cheats.txt', 'cheats.status', 'code cheats'),
              ('exefs_patches.py', 'patches-exefs.txt', 'exefs.status', 'ExeFS patches'))
# build-cache/: {"<mod>/<path>": sha256} of the files the last lift took up, built in or refused (the status files say
# which): the launcher builds the game again before Play only for files no build has seen (launcher/mods.py)
MOD_RECORD = 'mod-code.json'


def mod_code(out):
    """{"<mod>/<path in the mod>": sha256} of the code cheats and ExeFS patch files of <out>/mods for main's build."""
    return mod_files.code_record(out / 'mods', mod_files.main_build_id(out / 'exefs' / 'main'))


def mod_digest(out):
    """The mods' part of the lift's key: '' without mod code (the key of a game without mods stays what it was), else
    the files by name and content and the tools that turn them into patch sets."""
    code = mod_code(out)
    if not code:
        return ''
    h = hashlib.sha256(json.dumps(code, sort_keys=True).encode())
    for tool in ('cheats.py', 'exefs_patches.py', 'mod_files.py'):
        h.update(sha256_file(KIT / 'tools' / tool).encode() if (KIT / 'tools' / tool).is_file() else b'no tool')
    return h.hexdigest()


def run_lifter(out, lift, patches):
    """gen.py with the patch file into lift; its exit code."""
    cmd = py_cmd(KIT / 'recomp' / 'gen.py', '--exefs', out / 'exefs', '--out', lift, '--patches', patches)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                         errors='replace', bufsize=1)
    for line in p.stdout:
        line = line.rstrip('\n')
        say(f'  {line}')
        m = EMITTED.search(line)
        if m:
            emit('progress', step='lift', done=int(m.group(1)), total=int(m.group(2)), unit='chunks')
    return p.wait()


def mod_patches(out, cache, plain):
    """The patch file with the mods' code cheats and ExeFS patches added (plain when there are none), the status files
    written, and the mods' code. Each status file goes first: without it the runtime takes every such file as not
    built. The files are the player's downloads, so one the tools refuse costs its stage, never the build."""
    patches, done = plain, []
    code = mod_code(out)
    for tool, written, status, what in MOD_STAGES:
        (cache / status).unlink(missing_ok=True)
        if not code:
            continue
        cmd = py_cmd(KIT / 'tools' / tool, '--exefs', out / 'exefs', '--mods', out / 'mods', '--patches', patches,
                     '--out', cache / written, '--status', cache / status)
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                           errors='replace')
        for line in r.stdout.splitlines():
            say(f'  {line}')
        if r.returncode:
            (cache / status).unlink(missing_ok=True)
            say(f'  the mods\' {what} could not be built in (see above): the game is built without them')
        else:
            patches = cache / written
            done.append(status)
    return patches, done, code


def step_lift(out, cache):
    """gen.py into build-cache/lift. The folder is kept: gen.py rewrites only the chunks whose text changed and
    removes the ones it no longer writes. The mods' code cheats and ExeFS patches go in as patch sets (mod_patches);
    should the lifter refuse those, the game is lifted without them."""
    lift = cache / 'lift'
    plain = KIT / 'recomp' / 'patches-v111.txt'
    (cache / MOD_RECORD).unlink(missing_ok=True)
    patches, done, code = mod_patches(out, cache, plain)
    if run_lifter(out, lift, patches):
        if patches == plain:
            fail('the code generator failed (see above)')
        say('  the code generator refused the mods\' code: the game is lifted without it')
        for status in done:
            (cache / status).unlink(missing_ok=True)
        if run_lifter(out, lift, plain):
            fail('the code generator failed (see above)')
    if code:
        (cache / MOD_RECORD).write_text(json.dumps(code, indent=1, sort_keys=True), encoding='utf-8')
    for mod, old, new in LIFT_FIXES:
        hits = [p for p in sorted((lift / mod).glob('*.c')) if old in p.read_text(encoding='utf-8')]
        if len(hits) != 1 or hits[0].read_text(encoding='utf-8').count(old) != 1:
            fail(f'a bring-up fix found its anchor {len(hits)} times in the lifted {mod}, not once')
        p = hits[0]
        p.write_text(p.read_text(encoding='utf-8').replace(old, new), encoding='utf-8', newline='\n')


# ---------------------------------------------------------------- compile

def c_string(text):
    """text as a C string literal for a source file in UTF-8, which modules.c is: JSON's escapes for the quote, the
    backslash and the control characters are C's too, and every other character stands as itself. With JSON's
    default, which writes each character outside ASCII as \\uXXXX, a character outside the Basic Multilingual Plane
    came out as two surrogate halves (\\ud83e\\uddda), which C refuses ("invalid universal character"): a game
    folder with an emoji or a rare Han character in its path failed at the end of the compile."""
    return json.dumps(text, ensure_ascii=False)


def modules_c(lift, out, gen):
    layout = json.loads((lift / 'layout.json').read_text(encoding='utf-8'))
    L = ['#include "dispatch.h"', '#include "host.h"', '#include "layout.h"', '']
    for m in layout:
        n = m['name']
        L += [f'extern const uint64_t {n}_fn_addrs[];', f'extern const GuestFn {n}_fn_ptrs[];',
              f'extern const uint64_t {n}_fn_count;']
    L += ['', 'const RtLayoutEntry rt_layout[] = {']
    L += [f'    {{"{m["name"]}", MOD_{m["name"].upper()}_BASE, MOD_{m["name"].upper()}_SIZE}},' for m in layout]
    L += ['};', f'const int rt_layout_count = {len(layout)};', '', 'int rt_linked_modules(RtModule *out, int max) {',
          '    int n = 0;']
    for m in layout:
        n, u = m['name'], m['name'].upper()
        L.append(f'    if (n < max) out[n++] = (RtModule){{"{n}", MOD_{u}_BASE, MOD_{u}_SIZE, {n}_fn_addrs, {n}_fn_ptrs, '
                 f'{n}_fn_count}};')
    L += ['    return n;', '}', '', 'const RtFileHash rt_exefs_sha256[] = {']
    L += [f'    {{"{name}", "{sha256_file(out / "exefs" / name)}"}},' for name in EXEFS_FILES]
    L += ['};', f'const int rt_exefs_sha256_count = {len(EXEFS_FILES)};', '',
          f'const char rt_default_exefs[] = {c_string(str(out / "exefs"))};',
          f'const char rt_default_romfs[] = {c_string(str(out / "romfs"))};',
          f'const char rt_default_profile[] = {c_string(str(out / "profile"))};',
          'const int rt_game_version = 111;',
          'const unsigned long long rt_app_pool_bytes = 9809ull << 20;   /* the Switch 2 application pool */',
          'const int rt_nvn_api_version[2] = {231, 40};   /* NVN2: main asks for major 231, minor 40 or later */',
          f'const char rt_port_version[] = {c_string(kit_version())};']
    p = gen / 'modules.c'
    write_if_changed(p, '\n'.join(L) + '\n')
    if WIN:   # on Linux ui/menu_sdl.cpp defines it
        write_if_changed(gen / 'shim.c', '/* Defined by the SDL window path only; this build uses the Win32 one. */\n'
                                         'void menu_sdl_event(void *e) { (void)e; }\n')
    return p


def write_if_changed(path, text):
    if not path.is_file() or path.read_text(encoding='utf-8') != text:
        path.write_text(text, encoding='utf-8')


def kit_version():
    v = KIT / 'VERSION'
    return v.read_text(encoding='utf-8').strip() if v.is_file() else '0.0.1-ocarina'


def pe_imports(path):
    """The DLL names a PE file imports (its import directory), lower case."""
    d = Path(path).read_bytes()
    pe = struct.unpack_from('<I', d, 0x3C)[0]
    nsec, opt = struct.unpack_from('<H', d, pe + 6)[0], struct.unpack_from('<H', d, pe + 20)[0]
    sec = pe + 24 + opt
    rva = struct.unpack_from('<I', d, pe + 24 + 120)[0]   # PE32+ data directory 1: imports

    def off(r):
        for i in range(nsec):
            va, vs, raw = (struct.unpack_from('<I', d, sec + 40 * i + k)[0] for k in (12, 8, 20))
            if va <= r < va + max(vs, 1):
                return r - va + raw
        return None

    names, at = [], off(rva) if rva else None
    while at is not None:
        name_rva = struct.unpack_from('<I', d, at + 12)[0]
        if not name_rva:
            break
        o = off(name_rva)
        names.append(d[o:d.index(b'\0', o)].decode('ascii').lower())
        at += 20
    return names


def copy_if_changed(src, dst):
    """Atomically replace dst unless its contents match, even when source modification times are normalized.
    A same-directory temporary copy preserves modes, replaces read-only files on POSIX and leaves dst intact on copy
    failure. The copy is a temporary file, not a file in a temporary folder: such a folder is made for its owner alone,
    and some network shares refuse that while they take ordinary files."""
    dst = Path(dst)
    s = os.stat(src)
    try:
        d = os.stat(dst)
        if d.st_size == s.st_size and sha256_file(src) == sha256_file(dst):
            return False
    except FileNotFoundError:
        pass
    replacement = None
    try:
        fd, name = tempfile.mkstemp(dir=dst.parent, prefix='.nx2navi-copy-')
        os.close(fd)
        replacement = Path(name)
        shutil.copy2(src, replacement)
        os.replace(replacement, dst)
    except PermissionError:
        fail(f'cannot replace {dst}: close the game and check output-directory permissions')
    finally:
        if replacement is not None:   # still there only when the copy or the replace failed
            try:
                replacement.unlink(missing_ok=True)
            except OSError:
                pass
    return True


DLSS_DLL, DLSS_NOTICE = 'nvngx_dlss.dll', 'nvngx_dlss-NOTICE.txt'   # tools/dlss_pin.py, in the kit's bin/
# The Linux library's name ends in the DLSS version (tools/dlss_pin.py): it is found by its prefix, as the runtime
# finds it beside the game (runtime/resolution.c dlss_dll).
DLSS_SO_PREFIX, DLSS_SO_NOTICE = 'libnvidia-ngx-dlss.so.', 'libnvidia-ngx-dlss-NOTICE.txt'
# DLSS Frame Generation (runtime/vk_dlssg.c) loads these from beside the game: nx2navi_dlssg.dll (runtime/dlssg/,
# tools/build_dlssg.py), NVIDIA Streamline's interposer and the plugins it uses, and NGX's frame generation library.
# A release carries them with their licence texts in bin/dlssg/ (tools/make_launcher.py --dlssg); without the whole set
# the game leaves DLSS Frame Generation off.
DLSSG_DIR = 'dlssg'
DLSSG_FILES = ('nx2navi_dlssg.dll', 'sl.interposer.dll', 'sl.common.dll', 'sl.dlss_g.dll', 'sl.pcl.dll', 'sl.reflex.dll',
               'nvngx_dlssg.dll')


def notice_sha256(path):
    """The sha256 line of a DLSS notice: the library a kit placed beside the game."""
    try:
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.startswith('sha256 '):
                return line.split()[1].lower()
    except (OSError, UnicodeDecodeError):
        pass
    return None


def dlss_files(folder, linux):
    """The DLSS libraries in folder, by name: nvngx_dlss.dll, or on Linux every libnvidia-ngx-dlss.so.<version>."""
    return sorted(p for p in folder.glob(DLSS_SO_PREFIX + '*' if linux else DLSS_DLL) if p.is_file())


def place_dlss(out, linux=False):
    """NVIDIA's DLSS library from the kit's bin/ beside the game, where NGX loads it (runtime/vk_dlaa.c), with its
    notice. A different library the player put there (another DLSS version) is kept; the one an earlier build placed,
    named by the notice beside it, is replaced. linux: the Linux library, whose name carries the DLSS version, so a
    kit with another version replaces the earlier build's file under its old name. Returns a line for the build's
    log when it kept the player's."""
    note_name = DLSS_SO_NOTICE if linux else DLSS_NOTICE
    srcs, note = dlss_files(KIT / 'bin', linux), KIT / 'bin' / note_name
    if not srcs:
        return None
    src = srcs[-1]
    beside = dlss_files(out, linux)
    for dst in beside:
        s, d = os.stat(src), os.stat(dst)
        if dst.name != src.name or d.st_size != s.st_size or int(d.st_mtime) != int(s.st_mtime):
            have = sha256_file(dst)
            if have != notice_sha256(out / note_name) and have != sha256_file(src):
                return f"{dst.name} kept: the one already beside the game is not the kit's (another DLSS version)"
    copy_if_changed(src, out / src.name)
    for dst in beside:
        if not dst.samefile(out / src.name):   # another version's library, placed by an earlier build: replaced
            dst.unlink()
    if note.is_file() and (not (out / note_name).is_file() or (out / note_name).read_bytes() != note.read_bytes()):
        shutil.copy2(note, out / note_name)   # by content: two versions' notices can share size and second
    return None


def place_dlssg(out):
    """DLSS Frame Generation's libraries and their licence texts, the kit's bin/dlssg/, beside the game. Returns how
    many files it holds, 0 when the kit has no complete set (nothing is placed then)."""
    src = KIT / 'bin' / DLSSG_DIR
    if not all((src / n).is_file() for n in DLSSG_FILES):
        return 0
    files = sorted(p for p in src.iterdir() if p.is_file())
    for p in files:
        copy_if_changed(p, out / p.name)
    return len(files)


def copy_dlls(exe, msys_bin, out):
    seen, todo, copied = set(), [exe], []
    while todo:
        for name in pe_imports(todo.pop()):
            src = msys_bin / name
            if name in seen or name in SYSTEM_DLLS_SKIP or not src.is_file():
                continue
            seen.add(name)
            copy_if_changed(src, out / src.name)
            copied.append(src.name)
            todo.append(src)
    return copied


def parse_deps(text):
    """The prerequisites of a make-style dependency file as clang -MMD writes it (the target dropped)."""
    text = text.replace('\\\r\n', ' ').replace('\\\n', ' ')
    tokens, cur, i = [], '', 0
    while i < len(text):
        c = text[i]
        if c == '\\' and i + 1 < len(text) and text[i + 1] in ' #':
            cur += text[i + 1]
            i += 2
            continue
        if c == '$' and text[i + 1:i + 2] == '$':
            cur += '$'
            i += 2
            continue
        if c.isspace():
            if cur:
                tokens.append(cur)
            cur = ''
        else:
            cur += c
        i += 1
    if cur:
        tokens.append(cur)
    for k, t in enumerate(tokens):   # the target ends with ':' (a drive letter's colon is followed by a separator)
        if t.endswith(':'):
            return tokens[k + 1:]
    return tokens


class ObjectCache:
    """Which objects are current: per object the key of its command and source, and the sha256 (with size and
    modification time, to skip hashing an untouched file) of every file its dependency list names
    (build-cache/obj/objects.json). Paths under a root are kept relative to it ({'<kit>': KIT}), so the same files
    in a kit at another place still count as the same."""

    def __init__(self, path, roots=None):
        self.path = path
        self.roots = {k: os.path.normcase(os.path.abspath(v)) for k, v in (roots or {}).items()}
        try:
            self.entries = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
        except (OSError, ValueError):
            self.entries = {}
        self._sha = {}

    def norm(self, p):
        a = os.path.normcase(os.path.abspath(p))
        for name, root in self.roots.items():
            if a == root or a.startswith(root + os.sep):
                return name + a[len(root):].replace(os.sep, '/')
        return a

    def real(self, p):
        for name, root in self.roots.items():
            if p.startswith(name):
                return root + p[len(name):].replace('/', os.sep)
        return p

    def file_sha(self, p):
        p = os.path.normcase(os.path.abspath(p))
        if p not in self._sha:
            try:
                self._sha[p] = sha256_file(p)
            except OSError:
                self._sha[p] = None
        return self._sha[p]

    def _same(self, p, rec):
        path = self.real(p)
        if isinstance(rec, str):   # a cache without sizes and times
            return self.file_sha(path) == rec
        digest, size, mtime = rec
        try:
            st = os.stat(path)
        except OSError:
            return False
        if st.st_size == size and st.st_mtime_ns == mtime:
            return True
        return st.st_size == size and self.file_sha(path) == digest

    def current(self, obj, key):
        e = self.entries.get(obj.name)
        return (e is not None and e.get('key') == key and obj.is_file()
                and all(self._same(p, rec) for p, rec in e.get('deps', {}).items()))

    def record(self, obj, key, dep_file, src):
        try:
            deps = parse_deps(Path(dep_file).read_text(encoding='utf-8', errors='replace'))
        except OSError:
            deps = []
        out = {}
        for p in deps or [str(src)]:
            try:
                st = os.stat(p)
                out[self.norm(p)] = [self.file_sha(p), st.st_size, st.st_mtime_ns]
            except OSError:
                out[self.norm(p)] = [None, -1, -1]
        self.entries[obj.name] = {'key': key, 'deps': out}

    def forget(self, obj):
        self.entries.pop(obj.name, None)

    def keep_only(self, names):
        """Drops the entries of objects this build does not make (the runtime objects an earlier version compiled
        here and a later one links prebuilt): they would stay as that version wrote them for good. The number dropped."""
        gone = [n for n in self.entries if n not in names]
        for n in gone:
            del self.entries[n]
        return len(gone)

    def adopt(self, obj, key, old_keys, old_kits=()):
        """Records obj under key when its entry has one of old_keys, the keys an earlier version gave the same compile,
        its object is there and every file it was compiled from reads as it did. A file in an earlier kit's folder
        (old_kits, app/<version>) is read at the same place in this kit, as the object must match this kit's headers:
        that folder may be gone. 1 when taken over."""
        e = self.entries.get(obj.name)
        if not e or e.get('key') not in old_keys or not obj.is_file():
            return 0
        kit = self.roots.get('<kit>')
        kits = [os.path.normcase(os.path.abspath(k)) for k in old_kits]
        deps = {}
        for p, rec in e.get('deps', {}).items():
            path = self.real(p)
            a = os.path.normcase(os.path.abspath(path))
            for k in kits:
                if kit and a.startswith(k + os.sep):
                    path = kit + a[len(k):]
                    break
            digest = rec if isinstance(rec, str) else rec[0]
            if not digest or self.file_sha(path) != digest:
                return 0
            st = os.stat(path)
            deps[self.norm(path)] = [digest, st.st_size, st.st_mtime_ns]
        self.entries[obj.name] = {'key': key, 'deps': deps}
        return 1

    def save(self):
        """Whole or not at all: written beside the file and moved over it, so a process that ends in the middle
        (Cancel) leaves the last complete objects.json."""
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.entries), encoding='utf-8')
        tmp.replace(self.path)


def compiler_ids(clang, env):
    """(id, id up to 0.6.0): what names the compiler in the object cache's keys. The id is its --version text and its
    size. The keys up to 0.6.0 held the file's modification time too, so a tool update that put the same compiler back
    made every object stale, a full compile of the game for nothing; that form is still made, to take those objects
    over (object_alts)."""
    p = subprocess.run([str(clang), '--version'], capture_output=True, text=True, env=env, errors='replace')
    st = os.stat(clang)
    return (sha256_bytes(f'{p.stdout}\n{st.st_size}'.encode()),
            sha256_bytes(f'{p.stdout}\n{st.st_size}\n{int(st.st_mtime)}'.encode()))


def compiler_id(clang, env):
    return compiler_ids(clang, env)[0]


def object_key(cmd, src, dst, cache, salt, sysroot=False):
    """The cache key of compiling src into dst with cmd: the compiler's id (salt), the command's flags with the cache's
    roots standing for their paths, and the source's sha256. --sysroot is left out: only fast_clang's compiler gets it,
    and its objects are the UCRT64 compiler's byte for byte, so the object of either serves (sysroot=True keeps it: the
    keys up to 0.6.0)."""
    flags = [c for c in cmd[1:] if c not in (str(src), str(dst)) and (sysroot or not c.startswith('--sysroot='))]
    for name, root in cache.roots.items():
        flags = [name + os.path.normcase(f)[len(root):].replace(os.sep, '/') if os.path.normcase(f).startswith(root)
                 else f for f in flags]
    return sha256_bytes(json.dumps([salt, [os.path.basename(cmd[0])] + flags, cache.file_sha(src)]).encode())


def legacy_key(src, dst, cache, salt, clang, clangxx, kit, lift, gen, vma):
    """The key 0.5.0 and earlier gave compiling src into dst: their whole command with its paths as they were, the
    kit's own folder (app/<version>) among them, and the compiler's id up to 0.6.0 (salt)."""
    rt = Path(kit) / 'runtime'
    cpp = src.suffix in ('.cpp', '.cxx')
    guest = dst.name.startswith(('main_', 'sdk_', 'subsdk0_'))
    cmd = [str(clangxx if cpp else clang), '-c', '-std=gnu++17' if cpp else '-std=gnu11', '-O1' if guest else '-O2',
           '-fno-strict-aliasing', '-w', '-I', str(rt), '-I', str(lift), '-I', str(gen),
           '-I', str(rt / 'third_party' / 'imgui'), '-I', str(vma), '-o', str(dst), str(src)]
    flags = [c for c in cmd if c not in (str(src), str(dst))]
    return sha256_bytes(json.dumps([salt, flags, cache.file_sha(src)]).encode())


def earlier_kits():
    """The folders of the launcher's earlier versions beside this kit's (app/<version>), by path, whether there or not:
    the keys of their objects name them, and the launcher keeps only the previous one (app/previous.txt)."""
    app = KIT.parent
    if app.name != 'app':
        return []
    kits = {p for p in app.iterdir() if p.is_dir() and p != KIT and (p / 'runtime').is_dir()}
    try:
        prev = (app / 'previous.txt').read_text(encoding='utf-8').strip()
        if prev and app / prev != KIT:
            kits.add(app / prev)
    except OSError:
        pass
    return sorted(kits)


# Peak memory of one compile (tools/build_game.py JOB_MEM: 120 to 180 MiB for a chunk of lifted code, 568 MiB for the
# largest file), and the share of the free memory the compiles may take in all; the rest stays with the system, the
# RomFS step beside them and whatever else the player runs.
JOB_MEM = 768 << 20
JOB_MEM_SHARE = 0.8


def default_jobs(cpus=None, free=None):
    """(compiles at a time, why) without --jobs: one per logical processor, but no more than JOB_MEM_SHARE of the
    free memory holds at JOB_MEM each, and at least one. One per processor alone took no account of memory: on a
    PC with many threads and little free memory the compiles filled it, the largest files first as they start
    together. cpus, free: the processor count and the free bytes instead of this machine's (tests); without a
    reading of the free memory (tools/hostmem.py) the processor count alone decides, as before."""
    cpus = cpus or os.cpu_count() or 4
    if free is None:
        try:
            import hostmem
            free = hostmem.available_ram()
        except (ImportError, OSError, AttributeError):
            free = None
    if free is None:
        return cpus, f'{cpus} processors, free memory unknown'
    by_mem = max(1, int(free * JOB_MEM_SHARE) // JOB_MEM)
    return min(cpus, by_mem), f'{cpus} processors, {free / 2 ** 30:.1f} GiB of free memory allow {by_mem}'


SOURCE_WAIT, SOURCE_PAUSE = 60, 2   # seconds source_size waits for a drive in all, and between two tries


def source_size(src):
    """The size of a source file about to be compiled. The sources sit in the build cache beside the game, which a
    player may keep on a network share, and a share that loses its connection under the build's load answers 'not there'
    until it is back: while the source's folder does not answer either, the question is asked again for up to
    SOURCE_WAIT seconds. A folder that answers without the file means the file is gone, which no wait brings back."""
    src = Path(src)
    end = time.monotonic() + SOURCE_WAIT
    told = False
    while True:
        try:
            return src.stat().st_size
        except OSError as e:
            away = not src.parent.is_dir()
            if not away or time.monotonic() >= end:
                fail(f'cannot read {src} ({e.strerror or e}): '
                     + ('the drive or network share with the game folder stopped answering; check it and run the '
                        'build again' if away else
                        'the file is gone from the build cache; delete the lift folder in the build cache and run '
                        'the build again, which makes it anew'))
            if not told:
                say(f'  {src.parent} does not answer; waiting up to {SOURCE_WAIT} s for the drive to come back')
                told = True
            time.sleep(SOURCE_PAUSE)


# compile_all writes the object cache to disk after this many more objects, and at most this often (seconds): the
# cache of a whole game is some megabytes, and writing it after every object would take longer than the compiles.
CACHE_SAVE_EVERY, CACHE_SAVE_PAUSE = 25, 2.0


def compile_all(srcs, cmd_for, cache, jobs, env, progress=None, salt='', stop=None, alts=None):
    """Compile each (source, object) whose object is not current in cache. cmd_for(src, dst) -> argv without the
    dependency-file options; salt identifies the compiler. Returns (compiled, reused, failures) with failures as
    (src, stderr tail). The cache's roots stand for their paths in the flags too. The largest sources start first:
    compile time grows with size, and a big file that started last would run on alone after the others finished.
    stop() true makes the compiles not yet started end at once, unrecorded (a step beside this one failed).
    alts(src, dst) -> (keys, kits): the keys earlier versions gave the same compile, and the folders of their kits; an
    object recorded under one of them is taken over (ObjectCache.adopt) instead of compiled again.
    Each object is recorded when its compile ends, whichever started first, and the cache goes to disk every
    CACHE_SAVE_EVERY objects: a compile that is cut short (Cancel ends the process, a crash, a power cut) keeps the
    objects finished until then, where it used to keep none and the next run compiled the whole game again."""
    todo, keys = [], {}
    adopted = 0
    for src, dst in srcs:
        keys[dst] = object_key(cmd_for(src, dst), src, dst, cache, salt)
        if cache.current(dst, keys[dst]):
            continue
        if alts and cache.adopt(dst, keys[dst], *alts(src, dst)):
            adopted += 1
            continue
        todo.append((src, dst))
    reused = len(srcs) - len(todo)
    if adopted:
        say(f'  {adopted} objects an earlier version compiled are kept: the same compile of the same sources')
    todo.sort(key=lambda job: source_size(job[0]), reverse=True)

    def cc(job):
        src, dst = job
        dep = dst.with_suffix('.d')
        if stop and stop():
            return src, dst, dep, None, ''
        cmd = cmd_for(src, dst)
        cmd[1:1] = ['-MMD', '-MF', str(dep)]
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, errors='replace')
        return src, dst, dep, p.returncode, p.stderr[-1500:]

    done, bad = reused, []
    unsaved, saved_at = 0, time.monotonic()
    if progress:
        progress(done, len(srcs))
    with concurrent.futures.ThreadPoolExecutor(max(1, jobs)) as ex:
        # submitted largest first, taken as they end: in submission order every result would wait for the largest
        # file's, and nothing would be recorded during the minutes that one compiles
        for result in concurrent.futures.as_completed([ex.submit(cc, job) for job in todo]):
            src, dst, dep, rc, err = result.result()
            done += 1
            if rc is None:   # not started (stop)
                continue
            if rc:
                bad.append((src, err))
                cache.forget(dst)
            else:
                cache.record(dst, keys[dst], dep, src)
                unsaved += 1
            if unsaved >= CACHE_SAVE_EVERY and time.monotonic() - saved_at >= CACHE_SAVE_PAUSE:
                try:
                    cache.save()
                except OSError:   # the save at the end still reports a disk that cannot be written
                    pass
                unsaved, saved_at = 0, time.monotonic()
            if progress:
                progress(done, len(srcs))
    cache.save()
    return len(todo), reused, bad


def report_failures(bad):
    """The files that did not compile, to stderr: the compiler's errors of the first three, and the names of all of
    them, so that a paste of the build's end says which they were."""
    for src, err in bad[:3]:
        print(f'--- {src}\n{err}', file=sys.stderr)
    print('did not compile: ' + ', '.join(Path(src).name for src, _ in bad), file=sys.stderr, flush=True)


def runtime_headers(rt):
    """The headers tools/build_shaders.py builds into runtime/ (the runtime's own shaders, the menu font's bytes): in
    the repository as source only, so from a source checkout they are not there until they are built, and every
    runtime file that includes one would fail to compile. A kit carries them built; a release's kit carries neither
    tools/build_shaders.py nor the compilers it calls, and skips this."""
    try:
        import build_shaders
    except ModuleNotFoundError:
        return
    stale = build_shaders.stale(rt)
    if stale:
        say(f'  building {len(stale)} of the runtime\'s own headers (runtime/*_spv.h, the menu font)')
        build_shaders.build(rt, stale)


def preflight(msys, jobs, sources=None):
    """Linux, before the long steps of a build whose compile step will run: this machine's compiler, headers and
    libraries are tried on the runtime itself. Every runtime source that needs no header of this install is checked
    with the compile step's own command (clang -fsyntax-only), and an empty program is linked against the libraries
    the game links. The build takes the compiler, the linker and the libraries from the system (only the Vulkan
    headers and VMA come with the kit), and a distribution's older compiler, older headers or a missing development
    package stopped the compile step only after the RomFS image and the lift, ten minutes and more into the build.
    sources: the files to check instead (tests)."""
    clang, clangxx, env, vma = toolchain(msys)
    version = clang_version(clang, env)
    say(f'check: the compiler is {version}')
    rt = KIT / 'runtime'
    runtime_headers(rt)
    inc = ['-I', str(rt), '-I', str(rt / 'third_party' / 'imgui'), '-I', str(vma)]
    vulkan = vulkan_headers(clang, env)
    cmd_for = runtime_cmd(clang, clangxx, inc, first=vulkan)
    if sources is None:
        tainted = per_install_names()
        sources = [p for p, _ in runtime_sources() if not per_install(p, tainted)]

    def check(src):
        cmd = [c for c in cmd_for(src, Path('unused.o')) if c not in ('-c', '-o', 'unused.o')]
        p = subprocess.run([cmd[0], '-fsyntax-only', *cmd[1:]], capture_output=True, text=True, env=env, errors='replace')
        return src, p.returncode, p.stderr[-1500:]

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max(1, jobs)) as ex:
        bad = [(src, err) for src, rc, err in ex.map(check, sources) if rc]
    if bad:
        report_failures(bad)
        lacking = sorted(set(re.findall(r"fatal error: '([^']+)' file not found", ''.join(e for _, e in bad))))
        fail(f'{len(bad)} of the runtime\'s {len(sources)} files do not compile on this machine, so the build would stop '
             f'in its compile step. ' +
             (f'These headers are missing: {", ".join(lacking)}. Install the development packages '
              f'README-BuildKit.md lists under "Linux".' if lacking else
              f'The compiler ({version}) or a library\'s headers are older than the runtime needs: the releases are '
              f'built with clang 23, and the Vulkan headers come ' +
              (f'from {vulkan[1]}' if vulkan else 'from the system') + '. The errors above name what is missing.'))
    with tempfile.TemporaryDirectory(prefix='nx2navi-check-') as tmp:
        probe = Path(tmp) / 'probe.c'
        probe.write_text('int main(void) { return 0; }\n', encoding='ascii')
        ld = (['-fuse-ld=lld'] if shutil.which('ld.lld') else []) + ['-no-pie']
        p = subprocess.run([str(clangxx), *ld, '-o', str(Path(tmp) / 'probe'), '-x', 'c', str(probe), '-x', 'none',
                            *LINK_LIBS], capture_output=True, text=True, env=env, errors='replace')
    if p.returncode:
        print(p.stderr[-2000:], file=sys.stderr)
        fail('the libraries the game links are not all there (the linker\'s lines above name the missing ones): '
             'install the development packages README-BuildKit.md lists under "Linux". '
             'On Fedora the static C++ library is its own package, libstdc++-static.')
    say(f'check: {len(sources)} runtime files and the libraries pass, {time.time() - t0:.0f} s')


def steps_to_run(state, identity, out, until=None):
    """The steps from exefs to shaders this build will run, as main's loop decides it: a step whose key is not the
    recorded one, and every step after one that runs but those that read nothing of it (UNAFFECTED). until: the
    step the build stops after (--until). For the checks made before the long steps (space_needs, probe_toolchain)."""
    runs = []
    for name in STEPS[:STEPS.index('shaders') + 1]:
        if (any((earlier, name) not in UNAFFECTED for earlier in runs)
                or state['steps'].get(name) != step_key(name, identity, out)):
            runs.append(name)
        if name == until:
            break
    return runs


GiB = 1 << 30
# Disk space the steps after the RomFS image take in the game folder, the least each is known to need: a build that
# has less is sure to stop, half an hour in when it is the shaders. Measured on a 0.7.1 game folder: the lifted C
# 1.16 GiB, its objects 0.34 GiB and the new exe beside the old one 0.3 GiB; with the lifted C already there only
# the changed files and the exe. A full shader preparation writes every translation as a loose file (a player's
# pack of two translators' translations held 8,290 MiB, 327,560 files) and the game's shader programs and pipeline
# cache, 0.84 GiB. Folding the translations into their pack takes as much again, but a pack that cannot be written
# does not stop the build (pack_shader_cache), so it is not asked for here.
SPACE_BUILD_CACHE, SPACE_REBUILD, SPACE_SHADERS = 2 * GiB, 1 * GiB, 4 * GiB
SPACE_IMAGE_MARGIN = 1 * GiB   # beside the RomFS image, as step_romfs asks


def space_needs(runs, out, image_bytes=0, carried=False):
    """[(what, bytes)] this build will write into the game folder, for the steps it runs (steps_to_run).
    image_bytes: the size of the RomFS image when that step runs; carried: the shaders step keeps the shaders
    prepared before (step_shaders update) and writes none. A game folder that already holds translations (a
    preparation that was cut short, or one repeated with the same translator) is not asked for the shaders' space: the
    preparation then writes only what is missing, which may be nothing, and asking for all of it would stop a build
    that has the room it needs."""
    needs = []
    if 'romfs' in runs:
        needs.append(('the RomFS image', image_bytes + SPACE_IMAGE_MARGIN))
    if 'compile' in runs:
        lifted = (out / 'build-cache' / 'lift' / 'layout.json').is_file()
        needs.append(('the build cache (the translated code and its compiled files)',
                      SPACE_REBUILD if lifted else SPACE_BUILD_CACHE))
    if 'shaders' in runs and not carried and not shaders_begun(out):
        needs.append(('the prepared shaders', SPACE_SHADERS))
    return needs


def shaders_begun(out):
    """True when the game's shader cache holds a translation already: its pack, or a loose file."""
    cache = out / 'shader-cache'
    if (cache / 'shader-cache.pack').is_file():
        return True
    try:
        with os.scandir(cache) as it:
            return any(e.name.endswith('.spv') for e in it)
    except OSError:
        return False


def check_space(out, needs):
    """Stops the build before its first long step when the game folder's disk cannot hold what the build will
    write (space_needs), naming each part and how much is missing. Only the RomFS image's space was looked at,
    and only when that step began: a disk that held the image but not the build cache or the shaders stopped the
    build in the compile or, after half an hour, in the shader preparation. What an earlier run left of the image
    counts as free: the RomFS step removes it (step_romfs)."""
    need = sum(n for _, n in needs)
    if not need:
        return
    free = shutil.disk_usage(out).free
    try:
        free += (out / 'romfs' / 'romfs.bin.part').stat().st_size
    except OSError:
        pass
    if free >= need:
        return
    parts = ', '.join(f'{n / 1e9:.1f} GB for {what}' for what, n in needs)
    fail(f'{out} has {free / 1e9:.1f} GB free and this build needs {need / 1e9:.1f} GB there: {parts}. Free '
         f'{(need - free) / 1e9:.1f} GB more on that disk, or build into a folder on another one')


def probe_toolchain(msys, work):
    """Windows, before the long steps of a build whose compile step will run: an empty program is compiled and
    linked with the installed toolchain, against the libraries the game links and with the linker the compile step
    uses, in the folder work. Build tools that are incomplete, or that security software keeps from starting or
    from writing an exe, stopped the build in its compile step, after the RomFS image and the lift, ten minutes and
    more in; this takes a second. (Linux tries its compiler on the runtime's own sources: preflight.)"""
    clang, clangxx, env, _ = toolchain(msys)
    repair = (f'If the launcher installed these tools, delete the folder {msys.parent} and try again: it unpacks '
              f'them anew. If you installed MSYS2 yourself, install the packages README-BuildKit.md lists')
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        src, obj, exe = work / 'probe.c', work / 'probe.o', work / ('probe.exe' if WIN else 'probe')
        src.write_text('int main(void) { return 0; }\n', encoding='ascii')
        ld = ['-fuse-ld=lld'] if (msys / 'bin' / 'ld.lld.exe').is_file() else []
        steps = (('compile', clang, [str(clang), '-c', '-std=gnu11', '-O1', '-o', str(obj), str(src)]),
                 ('link', clangxx, [str(clangxx), *ld, '-o', str(exe), str(obj), '-L', str(msys / 'lib'), *LINK_LIBS]))
        for what, tool, cmd in steps:
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, env=env, errors='replace')
            except OSError as e:
                fail(f'{tool} does not start ({error_text(e)}): the build tools are damaged, or security software '
                     f'blocks them. {repair}')
            if not p.returncode:
                continue
            said = (p.stderr.strip().splitlines() or [f'exit code {p.returncode}'])[-1][-300:]
            if what == 'compile':
                fail(f'{tool} cannot compile a test program ({said}): the build tools are incomplete, or security '
                     f'software blocks them. {repair}')
            lacking = sorted(set(re.findall(r'unable to find library (-l\S+)', p.stderr)))
            fail('the build tools cannot link the game: '
                 + (f'{msys / "lib"} lacks {", ".join(lacking)}' if lacking else f'the linker says: {said}')
                 + f'. {repair}')
    finally:
        shutil.rmtree(work, ignore_errors=True)
    say('check: the build tools compile and link a test program')


def runtime_sources():
    """(source, object name) of every runtime file this OS's build compiles: the Win32 files on Windows, the SDL
    window's (*_sdl.*) and platform/linux.c elsewhere (tools/build_game.py's foreign_os)."""
    rt = KIT / 'runtime'
    out = []
    for pat in RUNTIME_PATTERNS:
        for p in sorted(rt.glob(pat)):
            if (p.stem.endswith('_sdl') or p.name == 'linux.c') if WIN else (p.stem.endswith('_win') or p.name == 'win32.c'):
                continue
            out.append((p, 'rt_' + p.relative_to(rt).as_posix().replace('/', '_').rsplit('.', 1)[0] + '.o'))
    return out


def per_install_names(generated=GENERATED_HEADERS):
    """The basenames of the generated headers and of every runtime header that includes one, at any depth."""
    rt = KIT / 'runtime'
    headers = {}
    for p in list(rt.rglob('*.h')) + list(rt.rglob('*.inc')):
        headers[p.name] = {os.path.basename(i) for i in INCLUDE.findall(p.read_text(encoding='utf-8', errors='replace'))}
    tainted = set(generated)
    grew = True
    while grew:
        grew = False
        for name, incs in headers.items():
            if name not in tainted and incs & tainted:
                tainted.add(name)
                grew = True
    return tainted


def per_install(src, tainted):
    incs = {os.path.basename(i) for i in INCLUDE.findall(Path(src).read_text(encoding='utf-8', errors='replace'))}
    return bool(incs & tainted)


def clang_version(clang, env):
    p = subprocess.run([str(clang), '--version'], capture_output=True, text=True, env=env, errors='replace')
    return (p.stdout.splitlines() or [''])[0].strip()


def runtime_digest():
    return tree_digest(KIT / 'runtime', ('**/*',))


# The x86-64 levels the lifted code is built and tested for (docs/building.md, "Checks"), as the macros clang defines
# for a CPU that has them.
CPU_V2 = {'__SSE3__', '__SSSE3__', '__SSE4_1__', '__SSE4_2__', '__POPCNT__', '__LAHF_SAHF__',
          '__GCC_HAVE_SYNC_COMPARE_AND_SWAP_16'}
CPU_V3 = CPU_V2 | {'__AVX__', '__AVX2__', '__BMI__', '__BMI2__', '__F16C__', '__FMA__', '__LZCNT__', '__MOVBE__',
                   '__XSAVE__'}


def cpu_level(clang, env):
    """This PC's x86-64 level for the lifted code: 3 (AVX2, FMA, BMI2), 2 (SSE4.2, POPCNT) or 0 when clang cannot
    tell. The game runs on the PC that builds it, so it takes this CPU's level, as tools/build_game.py --cpu does by
    hand. Without one the lifted code would be generic x86-64: every fused multiply-add of the game a call to
    runtime/fma.c's software fma, and a level the differential tests do not cover."""
    # Compiler wrappers can discard -march=native, even for this read-only host probe. Only the probe opts out:
    # the actual compiles keep their environment and use the explicit, supported x86-64 level selected below.
    probe_env = dict(env, NIX_ENFORCE_NO_NATIVE='0')
    for name in env:
        if name.startswith('NIX_ENFORCE_NO_NATIVE_'):
            probe_env[name] = '0'
    p = subprocess.run([str(clang), '-march=native', '-E', '-dM', '-x', 'c', os.devnull], capture_output=True,
                       text=True, env=probe_env, errors='replace')
    have = set(re.findall(r'^#define (__\w+) ', p.stdout, re.M)) if p.returncode == 0 else set()
    level = 3 if CPU_V3 <= have else 2 if CPU_V2 <= have else 0
    if level:
        return level
    # Where clang's own detection finds no level (a sandbox or a wrapper that hides the processor from it), Linux
    # still names the processor's features in /proc/cpuinfo (cpu_flags). Only a level every listed feature proves is
    # returned, so the lifted code never uses an instruction this processor lacks.
    flags = cpu_flags()
    have = {macro for flag, macro in CPUINFO_MACROS.items() if flags and flag in flags}
    return 3 if CPU_V3 <= have else 2 if CPU_V2 <= have else 0


def clang_ident(clang, env):
    """(version, target) a clang reports, or None when it does not run."""
    try:
        p = subprocess.run([str(clang), '--version'], capture_output=True, text=True, env=env, errors='replace')
    except OSError:
        return None
    v, t = re.search(r'clang version (\S+)', p.stdout), re.search(r'^Target: (\S+)', p.stdout, re.M)
    return (v.group(1), t.group(1)) if p.returncode == 0 and v and t else None


def fast_clang(msys, clang, env):
    """MSYS2's CLANG64 clang beside the UCRT64 one (msys64/clang64, which the launcher installs): the same LLVM release
    built with profile-guided optimisation, about a fifth faster on the lifted code. With UCRT64's headers
    (--sysroot) it writes the same objects as UCRT64's clang, byte for byte but the COFF timestamp; it is used only
    when it reports the same version and target. None otherwise."""
    if not WIN:
        return None
    fast = Path(msys).parent / 'clang64' / 'bin' / 'clang.exe'
    if not fast.is_file():
        return None
    mine = clang_ident(clang, env)
    return fast if mine and clang_ident(fast, env) == mine else None


def runtime_cmd(clang, clangxx, inc, level=0, fast=None, msys=None, first=(),
                native_cpu=False, native_lto=False):
    """fast: fast_clang's compiler for the lifted code, with msys as its sysroot. first: include flags that go
    ahead of inc for the runtime's sources only (vulkan_headers); the lifted code includes none of those headers,
    and its command, which is part of its objects' keys, stays as it was."""
    def cmd_for(src, dst):
        cpp = src.suffix in ('.cpp', '.cxx')
        # the lifted code at -O1 (its size makes -O2 slow to build), the runtime at -O2 by default; neither may assume
        # strict aliasing: the lifted code reaches guest memory through casts of one address to several types
        # (tools/build_game.py's flags). The lifted code is built for this PC's level (cpu_level); the runtime stays
        # generic unless NX2NAVI_NATIVE_CPU=1 opts into host tuning on Linux (-march=native -O3, plus ThinLTO with lld).
        guest = dst.name.startswith(('main_', 'sdk_', 'subsdk0_'))
        if guest:
            march = [f'-march=x86-64-v{level}'] if level else []
            opt = ['-O1']
        elif native_cpu and not WIN:
            march = ['-march=native', '-mtune=native']
            opt = ['-O3', '-flto=thin'] if native_lto else ['-O3']
        else:
            march = []
            opt = ['-O2']
        cc = [str(fast), f'--sysroot={msys}'] if guest and fast else [str(clangxx if cpp else clang)]
        # -fdeclspec: third_party/ffx_api/ffx_api.h declares its entry points __declspec(dllexport) unconditionally
        declspec = ['-fdeclspec'] if cpp and not WIN else []
        return [*cc, '-c', '-std=gnu++17' if cpp else '-std=gnu11', *opt,
                *march, '-fno-strict-aliasing', '-w', *declspec, *([] if guest else first), *inc, '-o', str(dst), str(src)]
    return cmd_for


def make_runtime_lib(dest, msys, jobs, obj=None):
    """--make-runtime-lib: compile every runtime source that does not include a per-install header into
    <dest>/libnx2navirt.a, with runtime-lib.json naming the runtime/ tree and the compiler it was built from. The
    objects stay in obj (default <dest>/obj), which a release keeps out of what it ships."""
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    obj = Path(obj).resolve() if obj else dest / 'obj'
    obj.mkdir(parents=True, exist_ok=True)
    clang, clangxx, env, vma = toolchain(msys)
    rt = KIT / 'runtime'
    tainted = per_install_names()
    srcs = [(p, obj / name) for p, name in runtime_sources() if not per_install(p, tainted)]
    inc = ['-I', str(rt), '-I', str(rt / 'third_party' / 'imgui'), '-I', str(vma)]
    cache = ObjectCache(obj / 'objects.json', {'<kit>': KIT})
    t0 = time.time()
    compiled, reused, bad = compile_all(srcs, runtime_cmd(clang, clangxx, inc, first=vulkan_headers(clang, env)),
                                        cache, jobs, env,
                                        lambda d, t: (d % 20 == 0 or d == t) and say(f'  {d} of {t}'),
                                        salt=compiler_id(clang, env))
    if bad:
        report_failures(bad)
        fail(f'{len(bad)} runtime files did not compile')
    lib = dest / RUNTIME_LIB
    lib.unlink(missing_ok=True)
    rsp = dest / 'ar.rsp'
    rsp.write_text('\n'.join('"%s"' % o.as_posix() for _, o in srcs), encoding='utf-8')
    ar = os.environ.get('NX2NAVI_AR') or (str(msys / 'bin' / 'ar.exe') if WIN else shutil.which('ar'))
    if not ar:
        fail('ar is not on PATH: install binutils or set NX2NAVI_AR')
    if subprocess.run([ar, 'rcs', str(lib), '@' + str(rsp)], env=env).returncode:
        fail('the runtime library could not be archived')
    meta = {'runtime': runtime_digest(), 'compiler': clang_version(clang, env),
            'objects': sorted(o.name for _, o in srcs)}
    (dest / 'runtime-lib.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    rsp.unlink(missing_ok=True)
    say(f'{lib}: {len(srcs)} runtime files ({compiled} compiled, {reused} kept), {lib.stat().st_size / 1e6:.1f} MB, '
        f'{time.time() - t0:.0f} s')


def prebuilt_runtime(clang, env):
    """(library, object names it holds) when the kit's lib/ matches this runtime/ and this compiler, else None."""
    lib, meta = KIT / 'lib' / RUNTIME_LIB, KIT / 'lib' / 'runtime-lib.json'
    try:
        m = json.loads(meta.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if not lib.is_file() or m.get('runtime') != runtime_digest() or m.get('compiler') != clang_version(clang, env):
        say('  the prebuilt runtime does not match this kit or compiler: compiling runtime/')
        return None
    return lib, set(m.get('objects', []))


def romfs_index(out, gen):
    if subprocess.run(py_cmd(KIT / 'tools' / 'romfs_index.py', '--romfs', out / 'romfs', '--out', gen)).returncode:
        fail('the RomFS index could not be written')


def windres_command(msys, rco):
    """MSYS2's windres on nx2navi.rc, run with runtime/ as the working directory (compile_resources says why)."""
    pp = ('-E', '-xc', '-DRC_INVOKED')   # windres's own preprocessor flags, which --preprocessor replaces
    return [str(msys / 'bin' / 'windres.exe'), '--use-temp-file', f'--preprocessor={msys / "bin" / "gcc.exe"}',
            *(f'--preprocessor-arg={a}' for a in pp), '-O', 'coff', '-i', 'nx2navi.rc', '-o', str(rco)]


RC_LINE = re.compile(r'(\d+)\s+(24|ICON)\s+"([^"]+)"')


def resource_file(rt):
    """runtime/nx2navi.rc as a .res file (the format Windows' resource compiler writes): the manifest as resource 24,
    the icon's images as RT_ICON 1..n and their directory as RT_GROUP_ICON, US English as windres gives them. lld
    takes a .res among its inputs and builds the exe's resource section from it.

    Python writes it, so no tool opens a path by the system's ANSI code page: windres does (binutils), and a build
    folder or a user folder whose name holds a character outside that code page (C:\\Users\\Víctor on a Cyrillic or a
    Japanese Windows, D:\\Games\\Zelda：Ocarina) failed with "Invalid argument" or "Cannot create temporary file"."""
    def resource(typ, name, data, flags):
        head = struct.pack('<HHHHIHHII', 0xffff, typ, 0xffff, name, 0, flags, 0x0409, 0, 0)
        return struct.pack('<II', len(data), 8 + len(head)) + head + data + b'\0' * (-len(data) % 4)

    out = bytearray(struct.pack('<II', 0, 32) + struct.pack('<HHHHIHHII', 0xffff, 0, 0xffff, 0, 0, 0, 0, 0, 0))
    for line in (rt / 'nx2navi.rc').read_text(encoding='utf-8').splitlines():
        m = RC_LINE.fullmatch(line.strip())
        if not m:
            continue
        name, kind, data = int(m.group(1)), m.group(2), (rt / m.group(3)).read_bytes()
        if kind == '24':
            out += resource(24, name, data, 0x1030)
            continue
        reserved, ico_type, n = struct.unpack_from('<HHH', data, 0)
        if reserved or ico_type != 1 or not n:
            fail(f'runtime/{m.group(3)} is not an icon file')
        group = struct.pack('<HHH', 0, 1, n)
        for i in range(n):
            w, h, colors, res0, planes, bits, size, off = struct.unpack_from('<BBBBHHII', data, 6 + 16 * i)
            out += resource(3, i + 1, data[off:off + size], 0x1010)
            group += struct.pack('<BBBBHHIH', w, h, colors, res0, planes, bits, size, i + 1)
        out += resource(14, name, group, 0x1030)
    return bytes(out)


def compile_resources(msys, rt, obj, env, lld=False):
    """With lld (lld=True) obj/nx2navi.res written by resource_file, which lld links in itself; else windres below.

    runtime/nx2navi.rc into obj/nx2navi_rc.o: the manifest of the game's exe (per-monitor DPI awareness, so the
    picture keeps the monitor's real size rather than one Windows stretches) and its icon.

    windres preprocesses the .rc with gcc through one command line it writes itself, quoting each path that holds a
    space. By default it hands that line to cmd.exe /c, which drops the first and the last double quote of a line
    holding more than two: gcc's quoted path loses its opening quote, and under a user folder with a space in its
    name (C:\\Users\\Ali Sayed) cmd.exe stops at that space ("'C:\\Users\\Ali' is not recognized").

    With --use-temp-file windres starts gcc itself instead, no shell involved: it splits the line into arguments and
    passes each one to gcc intact. Its splitter reads a quoted first word right, but it cuts a quoted -I path at its
    spaces and reads past the end of the line after a quoted last word, so the line holds no path but gcc's, its
    first word: the .rc goes by its bare name from runtime/ as the working directory, where windres also finds the
    manifest and the icon, and there is no -I. windres's temporary files go to a folder of the build cache, deleted
    afterwards (it leaves two of them behind on every run).

    The object is written under another name and replaces the old one only when its bytes differ, so an unchanged
    resource keeps its time and does not make the link key change (a rebuild with nothing new does not link again)."""
    if lld:
        res = obj / 'nx2navi.res'
        data = resource_file(rt)
        if not res.is_file() or res.read_bytes() != data:   # same bytes: keep the time, so the link key holds
            res.write_bytes(data)
        return res
    rco, new = obj / 'nx2navi_rc.o', obj / 'nx2navi_rc.o.new'
    tmp = obj / 'windres-tmp'
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    try:
        r = subprocess.run(windres_command(msys, new), cwd=rt, env=dict(env, TMP=str(tmp), TEMP=str(tmp)),
                           capture_output=True, text=True, errors='replace')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if r.returncode:
        new.unlink(missing_ok=True)
        fail(f'windres failed on runtime/nx2navi.rc: {r.stderr.strip()[:400]}')
    if rco.is_file() and rco.read_bytes() == new.read_bytes():
        new.unlink()
    else:
        os.replace(new, rco)
    return rco


def toolchain(msys):
    """(clang, clang++, environment, the folder holding vk_mem_alloc.h): MSYS2 UCRT64's on Windows, the ones on PATH
    elsewhere. VMA_INCLUDE overrides tools/build_game.py's VMA lookup (the system's header, else the pinned one
    fetched into toolchain/vma)."""
    if WIN:
        return (msys / 'bin' / 'clang.exe', msys / 'bin' / 'clang++.exe',
                dict(os.environ, PATH=str(msys / 'bin') + os.pathsep + os.environ.get('PATH', '')), msys / 'include' / 'vma')
    for tool in ('clang', 'clang++', 'ld.lld'):
        if not shutil.which(tool):
            fail(f'{tool} is not on PATH: install the packages README-BuildKit.md lists under "Linux"')
    if os.environ.get('VMA_INCLUDE'):
        vma = Path(os.environ['VMA_INCLUDE'])
        if not (vma / 'vk_mem_alloc.h').is_file():
            fail(f'{vma / "vk_mem_alloc.h"} is missing: correct VMA_INCLUDE')
    else:
        from build_game import vma_include
        vma = vma_include()
    return Path(shutil.which('clang')), Path(shutil.which('clang++')), dict(os.environ), vma


def vulkan_headers(clang, env):
    """The include flags that put the kit's Vulkan headers ahead of the system's for the runtime's sources: the
    pinned release in <kit>/toolchain/vulkan-headers (tools/vulkan_pin.py), which the AppImage's kit carries and a
    source checkout fetches when the system's headers are older (tools/build_game.py vulkan_include). None on
    Windows, where MSYS2 supplies the headers, and none where the system's headers are new enough."""
    if WIN:
        return []
    try:
        from build_game import vulkan_include
        folder = vulkan_include(clang, env)
    except ImportError:   # a kit without tools/vulkan_pin.py: the system's headers, as before it existed
        return []
    return ['-I', str(folder)] if folder else []


def xlate_source():
    """The folder of the shader translator to copy beside the game: the kit's bin/, else a source checkout's
    build/xlate/ (tools/build_xlate.py)."""
    for d in (KIT / 'bin', KIT / 'build' / 'xlate'):
        if (d / XLATE_NAME).is_file():
            return d
    fail(f'{XLATE_NAME} is missing: build it with tools/build_xlate.py')


def xlate_id():
    """The shader translator's id file's sha256 for the shaders step's key, 'missing' without a translator: the key is
    worked out before the steps run (steps_to_run), and the shaders step itself says what is missing when it runs."""
    for d in (KIT / 'bin', KIT / 'build' / 'xlate'):
        if (d / XLATE_NAME).is_file():
            return sha256_file(d / 'nx2navi_xlate.id')
    return 'missing'


# The x86-64 level the lifted code was compiled for, beside the objects (lifted_level).
LIFTED_BUILD = 'lifted-build.json'
LIFTED_OBJECTS = ('main_', 'sdk_', 'subsdk0_')   # the objects of the lifted code, by the module their names start with


def lifted_level(obj, now, update):
    """The x86-64 level to compile the lifted code for. A build from the game files takes this processor's (now). An
    update keeps the level the objects were compiled for, so that it compiles only what changed: the one recorded
    (LIFTED_BUILD), else 0 (generic x86-64) when the lifted objects are those of 0.5.0 or earlier, which compiled for no
    level and whose entries name their files by sha256 alone. Only the lifted objects' entries say that: 0.5.0 also
    compiled the runtime here, and those entries outlive it in a cache whose lifted code a later version compiled for
    this processor. Never above this processor's: a game folder moved to an older PC is compiled again for it."""
    if not update:
        return now
    try:
        rec = json.loads((obj / LIFTED_BUILD).read_text(encoding='utf-8'))
        return min(int(rec['level']), now)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    try:
        entries = json.loads((obj / 'objects.json').read_text(encoding='utf-8'))
        old = any(isinstance(rec, str) for name, e in entries.items() if name.startswith(LIFTED_OBJECTS)
                  for rec in e.get('deps', {}).values())
    except (OSError, ValueError, AttributeError):
        old = False
    return 0 if old else now


def retune_due(obj, msys):
    """Rebuild the Game (--rebuild --retune): whether the lifted code is recorded as compiled for another level than
    this processor's, or for none (LIFTED_BUILD). The compile step then runs although its inputs are unchanged."""
    clang, _, env, _ = toolchain(msys)
    try:
        return int(json.loads((obj / LIFTED_BUILD).read_text(encoding='utf-8'))['level']) != cpu_level(clang, env)
    except (OSError, ValueError, KeyError, TypeError):
        return True


def step_compile(out, cache_dir, msys, jobs, romfs_job=None, update=False):
    """romfs_job: the RomFS step still running beside this one (a first build). Everything that does not include
    romfs_config.h compiles meanwhile; the RomFS index and the few files that include it follow once it is done.
    update: a rebuild after a kit update (lifted_level)."""
    lift, gen, obj = cache_dir / 'lift', cache_dir / 'gen', cache_dir / 'obj'
    gen.mkdir(parents=True, exist_ok=True)
    obj.mkdir(parents=True, exist_ok=True)
    rt = KIT / 'runtime'
    runtime_headers(rt)
    clang, clangxx, env, vma = toolchain(msys)
    if not romfs_job:
        romfs_index(out, gen)
    srcs = [(p, obj / f'{mod}_{p.stem}.o') for mod in ('main', 'sdk', 'subsdk0') for p in sorted((lift / mod).glob('*.c'))]
    prebuilt = prebuilt_runtime(clang, env)
    runtime = [(p, obj / name) for p, name in runtime_sources()]
    if prebuilt:
        runtime = [(p, o) for p, o in runtime if o.name not in prebuilt[1]]
        say(f'  the runtime comes prebuilt; {len(runtime)} runtime files are compiled here')
    srcs += runtime
    srcs.append((modules_c(lift, out, gen), obj / 'modules.o'))
    if WIN:
        srcs.append((gen / 'shim.c', obj / 'shim.o'))
    inc = ['-I', str(rt), '-I', str(lift), '-I', str(gen), '-I', str(rt / 'third_party' / 'imgui'), '-I', str(vma)]
    cid, cid_old = compiler_ids(clang, env)   # fast_clang's compiler writes the same objects: not part of the key
    now = cpu_level(clang, env)
    level = lifted_level(obj, now, update)
    if level == now:
        say(f'  the lifted code is built for x86-64-v{level}, this processor\'s level' if level else
            '  the lifted code is built for generic x86-64: the processor\'s level could not be read')
    else:
        say(f'  the lifted code is built for {f"x86-64-v{level}" if level else "generic x86-64"}, as the objects kept '
            f'from before this update were (Rebuild the Game compiles it for this processor\'s level)')
    fast = fast_clang(msys, clang, env)
    if fast:
        say(f'  the lifted code compiles with {fast}')
    vulkan = vulkan_headers(clang, env)
    if vulkan:
        say(f'  the Vulkan headers come from {vulkan[1]}')
    has_lld = bool(shutil.which('ld.lld'))
    native_cpu = not WIN and os.environ.get('NX2NAVI_NATIVE_CPU') == '1'
    native_lto = native_cpu and not prebuilt and has_lld
    cmd_for = runtime_cmd(clang, clangxx, inc, level, fast, msys, first=vulkan,
                          native_cpu=native_cpu, native_lto=native_lto)
    # The keys earlier versions gave the same compiles (compile_all alts): 0.6.0's, with the compiler's file time and,
    # with the fast compiler, its id and --sysroot; 0.5.0's and earlier's, with their kit's folder in the paths; and
    # up to 0.7.1 the runtime's without the Vulkan headers' folder ahead of the system's (an object that compiled
    # against the system's headers is as good).
    plain = runtime_cmd(clang, clangxx, inc, level, None, msys,
                        native_cpu=native_cpu, native_lto=native_lto)
    no_vulkan = runtime_cmd(clang, clangxx, inc, level, fast, msys,
                            native_cpu=native_cpu, native_lto=native_lto)
    fast_old = (cid_old + compiler_ids(fast, env)[1]) if fast else None
    kits = earlier_kits()

    def alts(src, dst):
        keys = [object_key(plain(src, dst), src, dst, cache, cid_old, sysroot=True)]
        if vulkan:
            keys.append(object_key(no_vulkan(src, dst), src, dst, cache, cid))
        if fast:
            keys.append(object_key(cmd_for(src, dst), src, dst, cache, fast_old, sysroot=True))
        keys += [legacy_key(src, dst, cache, cid_old, clang, clangxx, k, lift, gen, vma) for k in kits]
        return keys, kits

    t0 = time.time()
    last = [0.0]

    def progress(done, total):
        now = time.time()
        if done == total or now - last[0] >= 0.25:
            last[0] = now
            emit('progress', step='compile', done=done, total=total, unit='files')
        if done % 50 == 0 or done == total:
            say(f'  compiled {done} of {total}, {now - t0:.0f} s')

    say(f'  {len(srcs)} files on {jobs} threads')
    cache = ObjectCache(obj / 'objects.json', {'<kit>': KIT})
    late = []
    if romfs_job:   # the lifted chunks include only cpu.h and simd.h: only the runtime and generated files are read
        tainted = per_install_names(('romfs_config.h',))
        late = [(s, o) for s, o in runtime + srcs[-(2 if WIN else 1):] if per_install(s, tainted)]
        say(f'  {len(late)} files wait for the RomFS index: {", ".join(s.name for s, _ in late)}')
    early = [j for j in srcs if j not in late]
    compiled, reused, bad = compile_all(early, cmd_for, cache, jobs, env, lambda d, t: progress(d, len(srcs)),
                                        salt=cid, stop=romfs_job.failed if romfs_job else None, alts=alts)
    if romfs_job:
        if romfs_job.thread.is_alive():
            say('  waiting for the RomFS image')
        romfs_job.join()   # raises the RomFS step's failure; a failed compile still keeps the finished image
        if not bad:
            romfs_index(out, gen)
            c2, r2, bad = compile_all(late, cmd_for, cache, jobs, env,
                                      lambda d, t: progress(len(early) + d, len(srcs)), salt=cid, alts=alts)
            compiled, reused = compiled + c2, reused + r2
    say(f'  {compiled} compiled, {reused} unchanged since the last build')
    if bad:
        report_failures(bad)
        fail(f'{len(bad)} files did not compile')
    if cache.keep_only({o.name for _, o in srcs}):
        cache.save()
    (obj / LIFTED_BUILD).write_text(json.dumps({'level': level}), encoding='utf-8')   # lifted_level reads it next update

    objs = [o for _, o in srcs] + ([compile_resources(msys, rt, obj, env, lld=(msys / 'bin' / 'ld.lld.exe').is_file())]
                                   if WIN else [])
    exe = out / EXE_NAME
    rsp = cache_dir / 'link.rsp'
    rsp.write_text('\n'.join('"%s"' % o.as_posix() for o in objs), encoding='utf-8')
    # lld links several times faster than GNU ld; the stack matches tools/build_game.py's build
    # 16 MiB for the main thread with either linker (GNU ld takes the same flag): the default 2 MiB is too little
    if WIN:
        ld = (['-fuse-ld=lld'] if (msys / 'bin' / 'ld.lld.exe').is_file() else []) + ['-Wl,--stack,16777216']
    else:   # tools/build_game.py's Linux flags; the main thread's stack is the process's (ulimit -s), not the linker's
        ld = (['-fuse-ld=lld'] if has_lld else []) + (['-flto=thin', '-Wl,--lto-O3', '-Wl,--icf=safe'] if native_lto else []) + ['-no-pie', '-Wl,-z,noexecstack', '-Wl,-rpath,$ORIGIN']
    # the link map feeds tools/prof_report.py (symbols for an NX2NAVI_PROFILE sampling profile); both linkers write one.
    # -Xlinker passes the path as one argument; -Wl, would cut it at every comma of the game folder's path (issue #470)
    ld += ['-Xlinker', '--Map=' + (cache_dir / 'ocarina.map').as_posix()]
    lib = ['-Wl,--whole-archive', str(prebuilt[0]), '-Wl,--no-whole-archive'] if prebuilt else []
    stamp = [[o.name, o.stat().st_size, o.stat().st_mtime_ns] for o in objs]
    if prebuilt:
        stamp.append([prebuilt[0].name, prebuilt[0].stat().st_size, sha256_file(prebuilt[0])])
    link_key = sha256_bytes(json.dumps([LINK_LIBS, ld, cid] + stamp).encode())
    state_link = cache_dir / 'link.key'
    if exe.is_file() and state_link.is_file() and state_link.read_text(encoding='utf-8') == link_key:
        say(f'  {EXE_NAME} is current, not linked again')
    else:
        emit('phase', step='compile', phase='link')
        say('  linking')
        tmp = out / (EXE_NAME + '.new')
        lib_dir = ['-L', str(msys / 'lib')] if WIN else []
        p = subprocess.run([str(clangxx), *ld, '-o', str(tmp), '@' + str(rsp), *lib, *lib_dir, *LINK_LIBS],
                           capture_output=True, text=True, env=env, errors='replace')
        if p.returncode:
            print(p.stderr[-4000:], file=sys.stderr)
            tmp.unlink(missing_ok=True)
            fail('the link failed')
        try:
            os.replace(tmp, exe)
        except PermissionError:
            tmp.unlink(missing_ok=True)
            fail(f'{exe} is in use: close the game and try again')
        state_link.write_text(link_key, encoding='utf-8')
    copy_if_changed(gen / 'romfs-index.bin', out / 'romfs-index.bin')
    xlate = xlate_source()
    note_cache_translator(out)   # before the new translator replaces the one the shader cache was made with
    for name in (XLATE_NAME, 'nx2navi_xlate.id'):
        copy_if_changed(xlate / name, out / name)
    if (KIT / 'bin' / 'amd_fidelityfx_vk.dll').is_file():   # FSR 3.1 (runtime/vk_ffx.cpp)
        copy_if_changed(KIT / 'bin' / 'amd_fidelityfx_vk.dll', out / 'amd_fidelityfx_vk.dll')
    if all((KIT / 'bin' / n).is_file() for n in FFX4_DLLS):   # FSR 4 (runtime/vk_ffx4.cpp): both or neither
        for name in FFX4_DLLS:
            copy_if_changed(KIT / 'bin' / name, out / name)
        if (KIT / 'bin' / FFX4_FG_DLL).is_file():   # its frame generation (runtime/vk_fg4.cpp)
            copy_if_changed(KIT / 'bin' / FFX4_FG_DLL, out / FFX4_FG_DLL)
    kept = place_dlss(out, linux=not WIN)   # NVIDIA DLSS (runtime/vk_dlaa.c): this platform's library from the kit
    if kept:
        say(f'  {kept}')
    if WIN and place_dlssg(out):   # DLSS Frame Generation (runtime/vk_dlssg.c): Windows DLLs too
        say('  DLSS Frame Generation: nx2navi_dlssg.dll, Streamline and nvngx_dlssg.dll beside the game')
    beside = copy_dlls(exe, msys / 'bin', out) if WIN else []
    say(f'  {EXE_NAME} {exe.stat().st_size / 1e6:.0f} MB' + (f'; beside it: {", ".join(beside)}' if beside else ''))
    (out / 'profile').mkdir(exist_ok=True)


# ---------------------------------------------------------------- main

SHADER_PROGRESS = re.compile(r'^prepare-shaders: progress (translate|build) (\d+) (\d+)\s*$')
SHADER_STALL_S = 180   # no count moved for this long: the launcher says so (the step still runs)
SHADER_BEAT_S = 10     # a progress event at least this often while the step runs


def shader_progress(phase, done, total, quiet_s, elapsed_s, build_only=False):
    """The shaders step's progress event fields: the translation fills the first half of the bar, the build into the
    driver's cache the second (nx2navi.exe prints both counts, runtime/gfx_shader.c prepare_files and
    runtime/vk_pipeline.c lb_warm_prepared); before either count, reading the game's shaders. quiet_s: seconds since a
    count last moved, said once it passes SHADER_STALL_S, so a slow step never looks frozen without a word. The
    detail ends without a full stop: the launcher adds one (launcher/ui/app.py).
    build_only: a run that only builds (warm_driver_cache), whose count fills the whole bar: starting at one half,
    the bar of a 20 minute build read 50% in its first second, and the launcher's time left, which follows the bar's
    pace (launcher/jobs.py Job.eta), had half the bar to go by."""
    clock = f'{int(elapsed_s) // 60}:{int(elapsed_s) % 60:02d}'
    if phase is None:
        fraction, detail = 0.0, f'Reading the game\'s shaders ({clock} so far)'
    elif phase == 'translate':
        fraction, detail = 0.5 * done / max(total, 1), f'Translated {done:,} of {total:,} shaders'
    else:
        part = done / max(total, 1)
        fraction = part if build_only else 0.5 + 0.5 * part
        detail = f'Built {done:,} of {total:,} shaders for your graphics card'
    if quiet_s >= SHADER_STALL_S:
        detail += (f'. No change for {int(quiet_s) // 60} min, but it is still running: one large shader can take '
                   'a while')
    return {'step': 'shaders', 'fraction': fraction, 'detail': detail}


# In the game's shader-cache/: the translator identities its translations are of, newest first (runtime/gfx_shader.c,
# "Carry"): '#' comment lines, then one identity a line in 64 hex digits, the content of nx2navi_xlate.id.
SHADER_IDS = 'identities.txt'
HEX64 = re.compile(r'[0-9a-fA-F]{64}')


def cache_identities(out):
    """The translator identities the game's shader cache lists (SHADER_IDS), lower case; [] when it has none."""
    try:
        text = (out / 'shader-cache' / SHADER_IDS).read_text(encoding='ascii', errors='replace')
    except OSError:
        return []
    return [line.strip().lower() for line in text.splitlines() if HEX64.fullmatch(line.strip())]


def note_cache_translator(out):
    """Before a new translator goes beside the game: a shader cache that has no SHADER_IDS yet (made before the game
    carried translations over) gets one naming the translator that made it, the one still beside the game. The game
    then carries those translations over at this update too (runtime/gfx_shader.c ids_load), and the update need not
    prepare them all again (shader_carry). Nothing when there is no cache, the file is there, or the old id is no
    identity."""
    cache, ids = out / 'shader-cache', out / 'shader-cache' / SHADER_IDS
    if not cache.is_dir() or ids.exists():
        return
    try:
        old = (out / 'nx2navi_xlate.id').read_text(encoding='ascii', errors='replace').strip()[:64]
    except OSError:
        return
    if not HEX64.fullmatch(old):
        return
    tmp = ids.with_name(SHADER_IDS + '.tmp')
    tmp.write_text('# Translator identities this shader cache holds translations of, newest first\n' + old.lower() + '\n',
                   encoding='ascii', newline='\n')
    os.replace(tmp, ids)


def shader_carry(out):
    """Whether the game carries the shaders prepared before an update over (runtime/gfx_shader.c, NX2NAVI_SHADER_CARRY,
    on unless the environment or nx2navi.ini sets it to 0) and there are some: shader-prepared.bin is written by a
    finished preparation, and the cache must name the translator its translations are of (SHADER_IDS,
    note_cache_translator), else the game has nothing to carry them over by."""
    if os.environ.get('NX2NAVI_SHADER_CARRY', '').strip() == '0':
        return False
    try:
        for line in (out / 'nx2navi.ini').read_text(encoding='utf-8-sig', errors='replace').splitlines():
            k, eq, v = line.strip().partition('=')
            if eq and k.strip().upper().removeprefix('NX2NAVI_') == 'SHADER_CARRY' and v.strip().strip('"') == '0':
                return False
    except OSError:
        pass
    return (out / 'shader-prepared.bin').is_file() and bool(cache_identities(out))


# The head of shader-prepared.bin (runtime/prp_file.h): magic, version, record size, and from version 2 on what the
# list was prepared from: sha256 of seed-programs.bin, sha256 of seed-answers.bin, the translator identity.
PREPARED_HEAD = struct.Struct('<8sII')
SEED_FILES = ('seed-programs.bin', 'seed-answers.bin')


def prepared_from(out):
    """What the game folder's prepared shader list says it was prepared from: {'programs', 'answers', 'translator'}
    as hex digests; {} for a list that does not say (written before the list recorded it); None without a list."""
    try:
        with open(out / 'shader-prepared.bin', 'rb') as f:
            data = f.read(PREPARED_HEAD.size + 96)
    except OSError:
        return None
    if len(data) < PREPARED_HEAD.size:
        return {}
    magic, version, _rec = PREPARED_HEAD.unpack_from(data)
    if magic != b'NX2NSPRP' or version != 2 or len(data) < PREPARED_HEAD.size + 96:
        return {}
    body = data[PREPARED_HEAD.size:]
    return {'programs': body[:32].hex(), 'answers': body[32:64].hex(), 'translator': body[64:96].hex()}


def seed_changed(out):
    """Why the shaders prepared before cannot stay in use, given the seed just written into the game folder
    (SEED_FILES): a sentence, or '' when the list was prepared from this very seed. A list that does not say what it
    was prepared from counts as changed: nothing tells whether it holds what this seed names.
    The translator is not compared: input ids do not depend on it and the game carries the translations of the one
    before over (shader_carry)."""
    was = prepared_from(out)
    if not was:
        return 'the shaders prepared before do not say which shader seed they were prepared from'
    try:
        now = {'programs': sha256_file(out / SEED_FILES[0]), 'answers': sha256_file(out / SEED_FILES[1])}
    except OSError:
        return ''   # no seed to compare with: as before the list recorded it
    if (was['programs'], was['answers']) != (now['programs'], now['answers']):
        return 'the shader seed changed since the shaders were prepared'
    return ''


def shader_run(cmd, cwd=None, build_only=False):
    """Runs a shader preparation command and returns its exit code. Its output is read as it comes: the counts
    nx2navi.exe prints become progress events, the other lines go on to the log, and while nothing moves an event
    every few seconds keeps saying the step is running and for how long. build_only: shader_progress."""
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding='utf-8', errors='replace', bufsize=1)
    t0 = time.monotonic()
    state = {'phase': None, 'done': 0, 'total': 0, 'moved': t0}
    lock = threading.Lock()
    stop = threading.Event()

    def send():
        now = time.monotonic()
        emit('progress', **shader_progress(state['phase'], state['done'], state['total'], now - state['moved'],
                                           now - t0, build_only))

    def heartbeat():
        while not stop.wait(SHADER_BEAT_S):
            with lock:
                send()

    beat = threading.Thread(target=heartbeat, daemon=True)
    beat.start()
    last_log = 0.0
    try:
        for line in proc.stdout:
            m = SHADER_PROGRESS.match(line)
            if not m:
                say(line.rstrip('\n'))
                continue
            phase, done, total = m.group(1), int(m.group(2)), int(m.group(3))
            with lock:
                if (phase, done) != (state['phase'], state['done']):
                    state['moved'] = time.monotonic()
                state.update(phase=phase, done=done, total=total)
                send()
            if time.monotonic() - last_log >= 30 or done == total:   # the log gets a count now and then, not each one
                last_log = time.monotonic()
                say(f'  shaders: {"translated" if phase == "translate" else "built"} {done} of {total}, '
                    f'{time.monotonic() - t0:.0f} s')
    finally:
        rc = proc.wait()
        proc.stdout.close()
        stop.set()
        beat.join()
    return rc


def refresh_pipeline_seed(out):
    """The kit's pipeline list (tools/pipeline_list.py) as the game folder's seed-pipeline-keys.bin, so that an update
    which carries the shaders over builds a newer list's pipelines too: `--warm-shaders` builds from that file. The
    list's shader variants are not translated on this path, so a record whose variant this folder has neither prepared
    nor met is left out by the game, which counts those in its line."""
    lst = KIT / 'tools' / 'data' / 'pipeline_list.bin'
    if not lst.is_file():
        return
    import pipeline_list
    got = pipeline_list.unpack(lst.read_bytes())
    if got:
        (out / 'seed-pipeline-keys.bin').write_bytes(got[1])


def warm_driver_cache(out):
    """After an update that carries the shaders over: every prepared stage library is built into the graphics driver's
    cache again (`nx2navi.exe --warm-shaders`). It takes seconds while the driver still has them. It takes minutes when
    the update changed what a library is built with, which gives each one another entry in the driver's cache: without
    this run every pipeline a draw first needed then compiled in play, which skipped that draw for some frames or made
    it wait (on an RX 7900 XT 100 to 340 ms a library against 3 ms from the cache, after dynamic vertex input was
    turned off for that driver). A run that fails is not an error of the build: the game checks the cache at every
    start and warms it in the background itself (runtime/vk_pipeline.c lw_start)."""
    exe = out / EXE_NAME
    if not exe.is_file() or not (out / 'shader-prepared.bin').is_file():
        return
    refresh_pipeline_seed(out)
    say("  the prepared shaders are built into the graphics driver's cache again")
    t0 = time.monotonic()
    try:
        rc = shader_run([str(exe), '--warm-shaders'], cwd=str(out), build_only=True)
    except OSError as e:
        rc = str(e)
    if rc:
        say(f'  that did not finish ({rc}); the game does it in the background while it runs')
    else:
        say(f"  driver's cache warm in {time.monotonic() - t0:.0f} s")


def step_shaders(out, update=False):
    """Every shader of the game files into the game's shader cache and the driver's own cache, before the first start
    (shader_run reports its progress).
    update: after a kit update the shaders prepared before stay in use and the game translates them again in the
    background (shader_carry), so the update does not wait the minutes a full preparation takes; the driver's cache
    gets their stage libraries again (warm_driver_cache). Only a list prepared from the seed this kit writes stays
    (seed_changed): with another seed, or a list that does not say, everything is prepared again as at the install."""
    # The game translates with the copy beside it. The compile step refreshes that copy, but a kit that changes only
    # the translator runs this step again alone (step_key), which would then prepare with the old one. Only a kit that
    # has one: the compile step, which always runs first, stops a build without it.
    for d in (KIT / 'bin', KIT / 'build' / 'xlate'):
        if (d / XLATE_NAME).is_file():
            note_cache_translator(out)
            for name in (XLATE_NAME, 'nx2navi_xlate.id'):
                copy_if_changed(d / name, out / name)
            break
    cmd = [sys.executable, '-u', str(KIT / 'tools' / 'prepare_shaders.py'), str(out), '--romfs', str(out / 'romfs'),
           '--exefs', str(out / 'exefs'), '--no-list', '--exe', EXE_NAME]
    why = ''
    carry = update and shader_carry(out)
    if carry and prepared_from(out) is not None:
        # The seed this kit makes from the game files, written without preparing anything (seconds): the list
        # prepared before stays in use only when it was prepared from this seed. Another seed names other shader
        # variants (a newer pipeline list, a change to which variants are prepared), and those the old list lacks
        # would be translated when play first draws them. A seed that cannot be written leaves things as they were.
        if shader_run(cmd + ['--no-run']) == 0:
            why = seed_changed(out)
        if why:
            say(f'  {why}: preparing them again')
    if carry and not why:
        for seed in SEED_FILES:
            (out / seed).unlink(missing_ok=True)
        say('  the shaders prepared before this update stay in use; the game translates them again in the background')
        if loose_shaders(out) >= PACK_LOOSE_MIN:
            # minutes without a count (4 on a player's 328,000 files): the row says what runs meanwhile
            emit('progress', step='shaders', fraction=0.0, detail='Folding the shader cache into one file')
            pack_shader_cache(out)
        warm_driver_cache(out)
        return
    rc = shader_run(cmd)
    if rc:
        fail('preparing the shaders failed (the lines above say why); the game still runs, compiling shaders as it '
             'meets them. Run the build again to retry this step')
    # translated now: the shader cache holds them. seed-pipeline-keys.bin stays: --warm-shaders builds from it again
    for seed in SEED_FILES:
        (out / seed).unlink(missing_ok=True)
    pack_shader_cache(out)


# An update that carries the shaders over packs the cache when it has this many loose translations: the game writes
# each translation it makes in play, and those of its background pass after a translator change (about 42,000), as loose
# files, and reading one the first time cost 5.6 ms on Windows against a fraction of a millisecond from the pack.
PACK_LOOSE_MIN = 1000


def loose_shaders(out):
    """The loose translations (<key>.spv) in the game's shader cache."""
    try:
        with os.scandir(out / 'shader-cache') as it:
            return sum(1 for e in it if e.name.endswith('.spv'))
    except OSError:
        return 0


def pack_shader_cache(out):
    """The prepared shader cache folded into shader-cache/shader-cache.pack (tools/pack_shader_cache.py): the
    preparation leaves four files per shader, about 175,000 for the whole game, which the game looks up one by one at
    every start (gfx_shader.c gfx_shader_prewarm, the use masks read after it) and which are slow to scan, copy and
    back up. The game reads the pack first and the loose files after it, so a cache left unpacked still works: a
    failure here is said and the build goes on."""
    tool = KIT / 'tools' / 'pack_shader_cache.py'
    if not tool.is_file() or not (out / 'shader-cache').is_dir():
        return
    t0 = time.monotonic()
    r = subprocess.run([sys.executable, '-u', str(tool), str(out / 'shader-cache')], stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
    for line in r.stdout.splitlines():
        say('  ' + line)
    if r.returncode:
        say(f'  packing the shader cache failed (code {r.returncode}); the game reads the loose files instead')
    else:
        say(f'  shader cache packed in {time.monotonic() - t0:.0f} s')


def prompt_art_dir():
    """The kit's button art (<kit>/prompt-art, tools/prompt_art.py), or None when the kit has none."""
    art = KIT / prompt_art.FOLDER
    return art if art.is_dir() else None


def bind_lines(out):
    """The bind_ lines of the game folder's nx2navi.ini, comments cut off: the keys the keyboard and mouse prompts
    show (tools/pad_prompts.py pc_keys reads the same lines)."""
    try:
        text = (out / 'nx2navi.ini').read_text(encoding='utf-8', errors='replace')
    except OSError:
        return []
    lines = (ln.split('#')[0].split(';')[0].strip() for ln in text.splitlines())
    return [ln for ln in lines if ln.lower().startswith('bind_')]


def step_prompts(out):
    """The button prompts drawn from the kit's art into the game folder: tools/pad_prompts.py writes one overlay
    folder per style and layout (prompt_art.OVERLAYS), which runtime/romfs.c loads in place of the game's button font
    and prompt textures when the Controls menu's Button prompts is Xbox, PlayStation, Steam Deck or Keyboard and mouse. Its lines
    go on to the log, each folder written is a progress event."""
    total = len(prompt_art.OVERLAYS)
    proc = subprocess.Popen(py_cmd(KIT / 'tools' / 'pad_prompts.py', '--art', prompt_art_dir(), '--game', out),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                            errors='replace', bufsize=1)
    done = 0
    emit('progress', step='prompts', fraction=0.0, detail=f'Drawing {total} sets of button prompts')
    try:
        for line in proc.stdout:
            line = line.rstrip('\n')
            if line.startswith('{'):   # its closing report, one long JSON line
                continue
            say(f'  {line}')
            if PROMPTS_WRITTEN.match(line):
                done += 1
                emit('progress', step='prompts', fraction=round(done / total, 4),
                     detail=f'Drew {done} of {total} sets of button prompts')
    finally:
        rc = proc.wait()
        proc.stdout.close()
    if rc:
        fail('drawing the button prompts failed (the lines above say why); the game still runs. Run the build again '
             'to retry this step')


AWAKE = None   # keep_awake's systemd-inhibit process on Linux; dropping it would close the pipe and end the hold


def keep_awake():
    """Hold off sleep for this process's life: the build runs for many minutes with no input, and a PC that sleeps in
    the middle stops the compiles. The hold ends with the process. Only sleep: the screen may still blank and lock, as
    with Windows' ES_SYSTEM_REQUIRED. Linux: systemd-inhibit takes a lock on sleep and keeps it while its `cat` reads
    the pipe this process holds open, so the lock also goes when this process is killed. Without systemd-inhibit (or a
    login manager to take the lock) nothing happens."""
    global AWAKE
    if os.name != 'nt':
        if AWAKE is None:
            try:
                AWAKE = subprocess.Popen(['systemd-inhibit', '--what=sleep', '--mode=block', '--who=NX2navi',
                                          '--why=Building the game', 'cat'], stdin=subprocess.PIPE,
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                         start_new_session=True)   # a Ctrl+C meant for the build leaves the lock
            except OSError:
                pass
        return
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    except (OSError, AttributeError):
        pass


def load_state(path):
    try:
        state = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
    except (OSError, ValueError):
        state = {}
    if state.get('version') == 1:   # the first kit: a list of finished steps, all tied to one kit version
        done = state.get('done', [])
        state = {'version': STATE_VERSION, 'input': state.get('input'), 'steps': {}}
        for step in ('exefs', 'romfs'):
            if step in done:
                state['steps'][step] = step_key(step, state['input'])
    return state


CPUINFO = Path('/proc/cpuinfo')
# /proc/cpuinfo's names for the features of CPU_V2 and CPU_V3 (cpu_level's fallback)
CPUINFO_MACROS = {'pni': '__SSE3__', 'ssse3': '__SSSE3__', 'sse4_1': '__SSE4_1__', 'sse4_2': '__SSE4_2__',
                  'popcnt': '__POPCNT__', 'lahf_lm': '__LAHF_SAHF__', 'cx16': '__GCC_HAVE_SYNC_COMPARE_AND_SWAP_16',
                  'avx': '__AVX__', 'avx2': '__AVX2__', 'bmi1': '__BMI__', 'bmi2': '__BMI2__', 'f16c': '__F16C__',
                  'fma': '__FMA__', 'abm': '__LZCNT__', 'movbe': '__MOVBE__', 'xsave': '__XSAVE__'}


def cpu_flags():
    """The features every logical processor has, as Linux lists them in /proc/cpuinfo (a feature is listed only when
    the kernel runs with it); None where the file cannot be read or is not an x86 one (Windows, other processors).
    One flags line per logical processor: a hybrid processor's cores can differ and the game runs on any of them, so
    only what all of them have counts."""
    if os.name == 'nt':
        return None
    try:
        lines = CPUINFO.read_text(encoding='ascii', errors='replace').splitlines()
    except OSError:
        return None
    flags = []
    for line in lines:
        key, _, rest = line.partition(':')
        if key.strip() == 'flags':
            flags.append(set(rest.split()))
    return set.intersection(*flags) if flags else None   # no flags line: not x86 (its list is called Features)


def cpu_tag():
    """This processor's features the lifted code's level follows (cpu_level), as the system reports them: Windows
    through IsProcessorFeaturePresent, Linux through the flags of /proc/cpuinfo (which lists a feature only when the
    kernel runs with it). One digit each for SSE4.2, AVX2 and AVX-512F: '110' is AVX2 without AVX-512. Part of the
    compile key, so a game folder moved to a PC without them (or a new processor) is compiled again rather than kept
    with instructions the processor lacks. 'any' where it cannot be read."""
    if os.name != 'nt':
        have = cpu_flags()
        if have is None:
            return 'any'
        return ''.join('1' if f in have else '0' for f in ('sse4_2', 'avx2', 'avx512f'))
    try:
        import ctypes
        present = ctypes.windll.kernel32.IsProcessorFeaturePresent
        # PF_SSE4_2_INSTRUCTIONS_AVAILABLE 38, PF_AVX2_INSTRUCTIONS_AVAILABLE 40, PF_AVX512F_INSTRUCTIONS_AVAILABLE 41
        return ''.join('1' if present(f) else '0' for f in (38, 40, 41))
    except (OSError, AttributeError):
        return 'any'


def step_scan(out):
    """Every file of the game data decompressed once, and stand-ins made for the damaged textures: tools/romfs_check.py,
    in worker threads here, so nothing of it runs while the game does."""
    proc = subprocess.Popen(py_cmd(KIT / 'tools' / 'romfs_check.py', '--game', out),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                            errors='replace', bufsize=1)
    emit('progress', step='scan', fraction=0.0, detail='Checking the game files')
    try:
        for line in proc.stdout:
            line = line.rstrip('\n')
            m = SCAN_CHECKED.match(line)
            if m:
                emit('progress', step='scan', fraction=round(int(m[1]) / max(int(m[2]), 1), 4),
                     detail=f'Checked {m[1]} of {m[2]} files')
            else:
                say(f'  {line}')
    finally:
        rc = proc.wait()
        proc.stdout.close()
    if rc:
        fail('checking the game files failed (the lines above say why); the game still runs. Run the build again to '
             'retry this step')


def step_key(step, identity, out=None):
    """What a step's output depends on. exefs and romfs: the game files only; lift: the unpacked ExeFS, the lifter and
    the mods' code cheats and ExeFS patches (mod_digest); compile: the lift, the runtime sources, this script and the
    processor (cpu_tag); shaders: the compile without the mods' code, the preparing script and the translator;
    prompts: the game files, the kit's button art, the version of the tools that draw it (PROMPT_TOOLS), the bind_
    lines of nx2navi.ini, and whether every folder it writes is still there; scan: the game files, the check's version
    (SCAN_TOOL) and whether its report is still there."""
    if step in ('exefs', 'romfs'):
        return f'{step}-1|{identity}'
    if step == 'scan':
        written = 'written' if (out / 'romfs-check.json').is_file() else 'missing'
        tool = KIT / 'tools' / 'romfs_check.py'
        return f'scan-1|{identity}|{SCAN_TOOL if tool.is_file() else "no tool"}|{written}'
    if step == 'prompts':
        binds = sha256_bytes(json.dumps(bind_lines(out)).encode())
        written = all((out / f / 'manifest.tsv').is_file() for f in prompt_art.OVERLAYS)
        return (f'prompts-1|{identity}|{tree_digest(prompt_art_dir(), ("**/*.png",))}|{PROMPT_TOOLS}|{binds}'
                f'|{"written" if written else "missing"}')
    base = f'lift|{exefs_digest(out)}|{lift_digest()}'
    mods = mod_digest(out) if step in ('lift', 'compile') else ''
    lift = f'{base}|mods {mods}' if mods else base
    if step == 'lift':
        return lift

    def compile_of(lift):
        return (f'compile|{lift}|{tree_digest(KIT / "runtime", ("**/*",))}|{sha256_file(Path(__file__))}'
                f'|{kit_version()}|cpu {cpu_tag()}')
    if step == 'compile':
        native_tag = '|native' if (not WIN and os.environ.get('NX2NAVI_NATIVE_CPU') == '1') else ''
        return f'{compile_of(lift)}{native_tag}'
    # shaders: the built game without the mods' code (UNAFFECTED) and the translator that prepared them, the one
    # xlate_source finds (the kit's bin/, or a source checkout's build/xlate/)
    return (f'shaders-1|{compile_of(base)}|{sha256_file(KIT / "tools" / "prepare_shaders.py")}'
            f'|{xlate_id()}')


def out_path(arg):
    """--out as an absolute path. On Windows a mapped drive letter stays as given: resolve() would name the folder by
    its server and share instead (a UNC path), which is a second connection to the same share, and some servers drop
    that one under the compilers' load while the mapped drive stays up."""
    return Path(os.path.abspath(arg)) if WIN else Path(arg).resolve()


def main():
    """The build (build_main), with every error it does not report itself reported as fail reports one: the launcher
    shows the message of the error event (launcher/build.py Run), and an exception that only printed its traceback
    reached the player as "the build stopped (exit code 1)". The traceback still goes to the log."""
    try:
        build_main()
    except nup.NupError as e:
        fail(f'the dump is damaged: {e}. Copy the dump to this PC again, or dump the game again, and then try again')
    except Exception as e:   # noqa: BLE001 - whatever it is, the player gets its message
        with OUT_LOCK:
            traceback.print_exc()
        fail(f'the build stopped: {error_text(e)}')


def build_main():
    global EVENTS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--nup', help='the game as one .nup file')
    ap.add_argument('--exefs', help='the game\'s exefs folder (with --romfs, instead of --nup)')
    ap.add_argument('--romfs', help='the game\'s romfs folder')
    ap.add_argument('--rebuild', action='store_true',
                    help='build again from the game files the last run unpacked into --out (after a kit update)')
    ap.add_argument('--retune', action='store_true',
                    help="with --rebuild: compile the lifted code for this processor's level even when the objects "
                         'kept from before an update were compiled for another (Rebuild the Game)')
    ap.add_argument('--out', help='the folder for the finished game')
    ap.add_argument('--check', action='store_true', help='only check the game\'s version; builds nothing')
    ap.add_argument('--msys', default=r'C:\msys64\ucrt64',
                    help='MSYS2 UCRT64 folder (default C:\\msys64\\ucrt64; Windows only: Linux takes clang and lld from PATH)')
    ap.add_argument('--jobs', type=int,
                    help='parallel compiles (default: one per processor, fewer when the free memory is short)')
    ap.add_argument('--events', action='store_true', help='machine-readable progress lines on stdout ("@@" + JSON)')
    ap.add_argument('--make-runtime-lib', metavar='DIR', help='build the prebuilt runtime library of a release into DIR')
    ap.add_argument('--obj', help='with --make-runtime-lib: the folder for its objects')
    ap.add_argument('--until', choices=STEPS, help=argparse.SUPPRESS)   # tests: stop after this step
    a = ap.parse_args()
    EVENTS = a.events
    jobs_why = '--jobs'
    if not a.jobs or a.jobs < 1:
        a.jobs, jobs_why = default_jobs()
    if a.make_runtime_lib:
        make_runtime_lib(a.make_runtime_lib, Path(a.msys), a.jobs, a.obj)
        return
    given = bool(a.nup) + bool(a.exefs or a.romfs) + bool(a.rebuild)
    if given != 1 or (not a.nup and not a.rebuild and not (a.exefs and a.romfs)):
        ap.error('give --nup, both --exefs and --romfs, or --rebuild')
    if a.rebuild and not a.out:
        ap.error('--rebuild needs --out')
    out = out_path(a.out) if a.out else None
    cache = out / 'build-cache' if out else None
    state_path = cache / 'state.json' if cache else None
    state = load_state(state_path) if state_path else {}

    emit('step', step='check', state='start')
    src = BuiltSource(out, state) if a.rebuild else Source(a)
    check(src)
    say('check: Ocarina of Time 1.1.1 (build e0f3ccfa)')
    emit('check', version='1.1.1', build=MAIN_BUILD_ID[:8])
    emit('step', step='check', state='done')
    if a.check:
        return
    if not out:
        ap.error('--out is needed to build')
    msys = Path(a.msys)
    if WIN:
        for need in ('bin/clang.exe', 'bin/clang++.exe', 'bin/SDL3.dll', 'lib/libSDL3.dll.a', 'include/vma'):
            if not (msys / need).exists():
                fail(f'{msys / need} is missing: install MSYS2 and the packages README-BuildKit.md lists')
    else:
        for tool in ('clang', 'clang++', 'ld.lld'):
            if not shutil.which(tool):
                fail(f'{tool} is not on PATH: install the packages README-BuildKit.md lists under "Linux"')
    packages = {'numpy': 'numpy', 'lz4': 'lz4', **({} if a.rebuild else {'zbic': 'zbic'}),
                **(PROMPT_PACKAGES if prompt_art_dir() else {})}
    for pip_name, mod in packages.items():
        try:
            __import__(mod)
        except ImportError as e:   # its own message too: an installed package can fail to load (a DLL, a clash)
            extra = ' fonttools pillow' if prompt_art_dir() else ''
            fail(f'the Python package {pip_name} is missing or does not load ({e}): '
                 f'pip install numpy lz4 zstandard zbic{extra}')
    cache.mkdir(parents=True, exist_ok=True)
    reset = state.get('version') != STATE_VERSION or state.get('input') != src.identity
    if reset:
        state = {'version': STATE_VERSION, 'input': src.identity, 'steps': {}}
    state['kit'] = kit_version()
    if a.rebuild and a.retune and retune_due(cache / 'obj', msys):
        # the compile step's key does not name the level: without this a finished compile reads as done before
        state['steps'].pop('compile', None)
    keep_awake()
    # What can be known to fail later is looked at now: the disk space of every step that will run, and on Windows
    # that the build tools compile and link (not for a build the tests stop before its compile step, --until)
    ahead = steps_to_run(state, src.identity, out, a.until)
    check_space(out, space_needs(ahead, out, sum(src.romfs_size(ref) for _, ref in src.romfs) if 'romfs' in ahead else 0,
                                 a.rebuild and shader_carry(out)))
    if WIN and 'compile' in ahead:
        probe_toolchain(msys, cache / 'toolchain-check')
    # Linux: the compiler, headers and libraries are the system's, so they are tried on the runtime before the long
    # steps of a build that will reach the compile step (the first step whose key is not current makes the rest run)
    if not WIN and any(state['steps'].get(n) != step_key(n, src.identity, out) for n in STEPS[:STEPS.index('compile') + 1]):
        preflight(msys, a.jobs)
    say(f'compiles at a time: {a.jobs} ({jobs_why})')
    t0 = time.time()
    romfs_job = None
    runs = {'exefs': lambda: step_exefs(src, out), 'romfs': lambda: step_romfs(src, out, min(a.jobs, 8)),
            'lift': lambda: step_lift(out, cache),
            'compile': lambda: step_compile(out, cache, msys, a.jobs, romfs_job, a.rebuild and not a.retune),
            'shaders': lambda: step_shaders(out, a.rebuild), 'prompts': lambda: step_prompts(out),
            'scan': lambda: step_scan(out)}
    state_lock = threading.Lock()   # the RomFS step records itself from its own thread

    def save():
        # whole or not at all: a cut write (a full disk, Cancel ending the process) would leave a state.json that
        # loads as {}, and the launcher would take the built game for none and ask for the dump again
        with state_lock:
            tmp = state_path.with_name(state_path.name + '.tmp')
            tmp.write_text(json.dumps(state, indent=1), encoding='utf-8')
            os.replace(tmp, state_path)

    if reset:
        # on disk at once: the exefs step rewrites <out>/exefs in place, and a run cut short there would leave the
        # old state.json saying the old game's steps are done, which --rebuild trusts
        save()

    def run(name):
        t = time.time()
        runs[name]()
        with state_lock:
            state['steps'][name] = step_key(name, src.identity, out)
        save()
        say(f'{name}: {time.time() - t:.0f} s')
        emit('step', step=name, state='done', seconds=round(time.time() - t, 1))

    def lift_gone(name):
        # Keep Build Cache off (the launcher) deletes the lift and the objects after a build and keeps this state:
        # the lift's key still matches then, and a compile step that runs would find no lifted code to compile
        return (name == 'lift' and not (cache / 'lift' / 'layout.json').is_file()
                and state['steps'].get('compile') != step_key('compile', src.identity, out))

    try:
        for name in STEPS:
            if name == 'prompts' and not prompt_art_dir():
                say('prompts: none to draw, the kit has no button art')
                emit('step', step=name, state='skip')
            elif state['steps'].get(name) == step_key(name, src.identity, out) and not lift_gone(name):
                say(f'{name}: done before')
                emit('step', step=name, state='skip')
            else:
                # a step that runs makes every step after it run too: their keys hash its output, not its key (but
                # for OWN_KEY_STEPS, whose keys name their inputs, and a step that reads nothing of it, UNAFFECTED)
                with state_lock:
                    for later in STEPS[STEPS.index(name):]:
                        if later == name or not ({name, later} & set(OWN_KEY_STEPS) or (name, later) in UNAFFECTED):
                            state['steps'].pop(later, None)
                say(f'{name}:')
                emit('step', step=name, state='start')
                if name in ('shaders', 'prompts', 'scan') and romfs_job:   # they read the RomFS image
                    romfs_job.join()
                if name == 'romfs' and a.until != 'romfs':
                    # The RomFS image is mostly disk work, the lift and the compile are processor work: they run side
                    # by side, and only the RomFS index and the files that include it wait for the image
                    # (step_compile). A first build takes about as long as the longer of the two, not their sum.
                    romfs_job = Background(lambda: run('romfs'))
                else:
                    run(name)
            if romfs_job and romfs_job.failed():
                romfs_job.join()
            if name == a.until:
                break
        if romfs_job:
            romfs_job.join()
    except (SystemExit, Exception):
        # a failed lift or compile, or an error of this script: the RomFS image still finishes and is kept, so the
        # next run starts after it (the thread is a daemon, and the process ending would cut its .part file short)
        if romfs_job and romfs_job.thread.is_alive():
            say('waiting for the RomFS image before stopping')
            romfs_job.thread.join()
        raise
    save()
    say(f'\nfinished in {(time.time() - t0) / 60:.1f} min: {out / EXE_NAME}')
    emit('end', exe=str(out / EXE_NAME), seconds=round(time.time() - t0, 1), kit=kit_version())


if __name__ == '__main__':
    main()
