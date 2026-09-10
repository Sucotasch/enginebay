#!/usr/bin/env python3
"""Engine diagnostics for the EngineBay launcher.

Port of Quartermaster's `internal/peimports` (MIT, github.com/Quartermaster-
Labs/Quartermaster). Answers one question about a Windows binary: can the
loader actually load it, and if not, which DLL is missing.

Why this exists: a failed DLL load is invisible. When a backend's dependency
chain is incomplete the process dies with STATUS_DLL_NOT_FOUND (0xC0000135)
before main() runs — no stdout, no stderr. Everything upstream only sees a
process that exited immediately. This reads the PE import table and names the
missing DLL with actionable advice.

Also exposes VRAM reading (PDH / DXGI) for the launcher's VRAM guard.

Usage:
    python scripts/engine_diag.py check <exe>      # missing-DLL report
    python scripts/engine_diag.py vram             # free/total VRAM (+ holders)
"""

from __future__ import annotations

import argparse
import os
import re
import struct
import sys
from pathlib import Path

# ── PE import-table parsing (load-time imports only, no deps) ──────────────

MAX_DESCRIPTORS = 4096  # a much larger count means a malformed header
MAX_NAME_LEN = 512      # import name string bound


def _le32(b: bytes, off: int) -> int:
    return struct.unpack_from("<I", b, off)[0]


def _all_zero(b: bytes) -> bool:
    return not any(b)


def pe_imports(path: str | os.PathLike) -> list[str]:
    """Return the load-time import DLL names of a PE file, in header order.

    A file that is not a PE (or has no import directory) yields no names.
    Delay-loaded imports (data directory 13) are deliberately ignored: a
    missing delay import fails at first call, not at load.
    """
    data = Path(path).read_bytes()
    if len(data) < 0x40 or data[:2] != b"MZ":
        return []
    e_lfanew = _le32(data, 0x3C)
    if data[e_lfanew : e_lfanew + 4] != b"PE\x00\x00":
        return []
    coff = e_lfanew + 4
    if coff + 24 > len(data):
        return []
    n_sections = _le32(data, coff + 2) & 0xFFFF
    size_opt = _le32(data, coff + 16) & 0xFFFF
    opt = coff + 20
    if opt + size_opt > len(data):
        return []
    magic = _le32(data, opt) & 0xFFFF
    if magic == 0x20B:      # PE32+
        dd_off = opt + 112
    elif magic == 0x10B:    # PE32
        dd_off = opt + 96
    else:
        return []
    # DataDirectory[1] = import directory. Each entry is (RVA, Size).
    imp_rva = _le32(data, dd_off + 1 * 8)
    if imp_rva == 0:
        return []
    # Section table: after the optional header.
    sec_off = opt + size_opt
    sections: list[tuple[int, int, int, int]] = []  # va, vsize, raw_off, raw_size
    for i in range(n_sections):
        base = sec_off + i * 40
        if base + 40 > len(data):
            break
        vsize = _le32(data, base + 8)
        va = _le32(data, base + 12)
        raw_size = _le32(data, base + 16)
        raw_off = _le32(data, base + 20)
        sections.append((va, vsize, raw_off, raw_size))

    def rva_to_off(rva: int) -> int | None:
        for va, vsize, raw_off, raw_size in sections:
            size = max(vsize, raw_size)
            if va <= rva < va + size:
                off = raw_off + (rva - va)
                if off + 1 <= len(data):
                    return off
        return None

    def read_cstr(rva: int) -> str | None:
        off = rva_to_off(rva)
        if off is None:
            return None
        end = data.find(b"\x00", off, off + MAX_NAME_LEN)
        if end < 0:
            return None
        try:
            return data[off:end].decode("ascii")
        except UnicodeDecodeError:
            return None

    names: list[str] = []
    for i in range(MAX_DESCRIPTORS):
        off = rva_to_off(imp_rva + i * 20)
        if off is None or off + 20 > len(data):
            break
        desc = data[off : off + 20]
        if _all_zero(desc):
            break
        name_rva = _le32(desc, 12)
        if name_rva == 0:
            break
        name = read_cstr(name_rva)
        if not name:
            break
        names.append(name)
    return names


# ── Missing-DLL walk ───────────────────────────────────────────────────────

def system_search_dirs() -> list[str]:
    dirs: list[str] = []
    root = os.environ.get("SystemRoot", "")
    if root:
        dirs += [os.path.join(root, "System32"), os.path.join(root, "SysWOW64"), root]
    dirs += [d for d in os.environ.get("PATH", "").split(os.pathsep) if d]
    return dirs


def _lookup(name: str, dirs: list[str]) -> str:
    for d in dirs:
        if not d:
            continue
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return ""


def missing_dlls(exe: str) -> list[tuple[str, str]]:
    """Walk the import graph rooted at exe; report DLLs the loader would miss.

    Returns [(missing dll name, <module that imports it>)]. Best-effort: an
    unreadable or non-PE dependency is skipped, so "[]" means not *proved*
    broken. The search path mirrors the default loader order: app dir first,
    then system dirs, then PATH.
    """
    exe_dir = os.path.dirname(os.path.abspath(exe))
    search = [exe_dir] + system_search_dirs()
    seen: set[str] = set()
    missing: dict[str, tuple[str, str]] = {}
    queue = [exe]
    while queue:
        cur = queue.pop(0)
        key = os.path.basename(cur).lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            imports = pe_imports(cur)
        except Exception:
            continue
        for imp in imports:
            lower = imp.lower()
            # API sets are loader-resolved virtual names with no file on disk.
            if lower.startswith("api-ms-") or lower.startswith("ext-ms-"):
                continue
            if lower in seen:
                continue
            found = _lookup(imp, search)
            if found:
                queue.append(found)
                continue
            if lower not in missing:
                missing[lower] = (imp, os.path.basename(cur))
    return sorted(missing.values(), key=lambda t: t[0].lower())


def runtime_advice(names: list[str]) -> str:
    """Map missing DLL names to a fix. Empty when unrecognised."""
    hip = cuda = driver = vc = False
    for n in names:
        l = n.lower()
        # nvcuda/nvml ship with the NVIDIA display driver, not the toolkit.
        if l.startswith("nvcuda") or l.startswith("nvml"):
            driver = True
        elif (l.startswith("amdhip") or l.startswith("hip") or l.startswith("roc")
              or l.startswith("libhip") or l.startswith("amd_comgr")):
            hip = True
        elif (l.startswith("cudart") or l.startswith("cublas") or l.startswith("cudnn")
              or l.startswith("nvrtc")):
            cuda = True
        elif l.startswith("msvcp") or l.startswith("vcruntime") or l.startswith("concrt"):
            vc = True
    if hip:
        return ("copy the AMD ROCm/HIP runtime libraries next to the executable "
                "(PATH does not help: the HIP SDK names them lib*.dll)")
    if cuda:
        return "this build needs the NVIDIA CUDA runtime next to the executable or on PATH"
    if driver:
        return "this is a CUDA build and needs an NVIDIA driver; it cannot run on this GPU"
    if vc:
        return "install the Microsoft Visual C++ redistributable"
    return ""


def hint(exe: str) -> str:
    """One-line actionable description of why exe cannot load, or ''."""
    deps = missing_dlls(exe)
    if not deps:
        return ""
    names = [d[0] for d in deps]
    needed = deps[0][1]
    same_source = all(d[1] == needed for d in deps)
    base = os.path.basename(exe)
    msg = f"{base} cannot load: missing {', '.join(names)}"
    if same_source and needed.lower() != base.lower():
        msg += f" (imported by {needed})"
    advice = runtime_advice(names)
    if advice:
        msg += f"; {advice}"
    return msg


# ── VRAM reading (PDH + DXGI, vendor-neutral, ctypes) ─────────────────────
#
# Port of Quartermaster's internal/perf (MIT). Sources:
#  - PDH counter `\GPU Adapter Memory(*)\Dedicated Usage` → system-wide
#    dedicated VRAM in use (bytes) — the number Task Manager shows. Free =
#    adapter total (DXGI GetDesc1) − this usage.
#  - PDH counter `\GPU Process Memory(*)\Dedicated Usage` → per-process VRAM,
#    instance names embed the PID. Used for the "who holds VRAM" list.
#  PDH does not sample hardware perf counters, so it does not stall generation.
#  No vendor tools (nvidia-smi) and nothing to install.

import ctypes
from ctypes import wintypes

_PDH = ctypes.windll.pdh  # type: ignore[attr-defined]
_DFMT_DOUBLE = 0
_PDH_MORE_DATA = -2147481646  # 0x800007D2
_PDH_NO_DATA = -2147481643    # 0x800007D5


class _PDH_FMT_COUNTERVALUE_ITEM_DOUBLE(ctypes.Structure):
    _fields_ = [("name", wintypes.LPWSTR), ("status", ctypes.c_uint32),
                ("value", ctypes.c_double)]


class _PdhCounter:
    """One PDH query over a wildcard GPU counter with a single handle."""

    def __init__(self, counter_path: str):
        self.query = ctypes.c_void_p()
        if _PDH.PdhOpenQueryW(None, 0, ctypes.byref(self.query)) != 0:
            raise RuntimeError("PdhOpenQuery failed")
        self._counter = ctypes.c_void_p()
        rc = _PDH.PdhAddEnglishCounterW(
            self.query, counter_path, 0, ctypes.byref(self._counter))
        if rc != 0:
            _PDH.PdhCloseQuery(self.query)
            raise RuntimeError(f"PdhAddEnglishCounter({counter_path}): 0x{rc & 0xFFFFFFFF:x}")
        _PDH.PdhCollectQueryData(self.query)

    def items(self) -> list[tuple[str, float]]:
        """[(instance_name, value)] — names embed luid/pid, values raw."""
        rc = _PDH.PdhCollectQueryData(self.query)
        if rc != 0 and rc != _PDH_NO_DATA:
            return []
        buf_size = ctypes.c_uint32(0)
        item_count = ctypes.c_uint32(0)
        rc = _PDH.PdhGetFormattedCounterArrayW(
            self._counter, _DFMT_DOUBLE, ctypes.byref(buf_size),
            ctypes.byref(item_count), None)
        if rc != _PDH_MORE_DATA or item_count.value == 0:
            return []
        buf = ctypes.create_string_buffer(buf_size.value)
        rc = _PDH.PdhGetFormattedCounterArrayW(
            self._counter, _DFMT_DOUBLE, ctypes.byref(buf_size),
            ctypes.byref(item_count), ctypes.cast(buf, ctypes.c_void_p))
        if rc != 0:
            return []
        arr = ctypes.cast(buf, ctypes.POINTER(_PDH_FMT_COUNTERVALUE_ITEM_DOUBLE))
        out: list[tuple[str, float]] = []
        for i in range(item_count.value):
            it = arr[i]
            if it.status == 0:
                out.append((it.name or "", it.value))
        return out

    def close(self) -> None:
        try:
            _PDH.PdhCloseQuery(self.query)
        except Exception:
            pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _parse_luid(name: str) -> tuple[int, int] | None:
    """(high, low) from a PDH instance name: '..._luid_0x%08x_0x%08x_...'."""
    idx = name.find("luid_0x")
    if idx < 0:
        return None
    rest = name[idx + 7 :]
    parts = rest.split("_", 3)
    if len(parts) < 3:
        return None
    try:
        high = int(parts[0], 16)
        low = int(parts[1].lstrip("0x") or "0", 16)
    except ValueError:
        return None
    return high, low


class DxgiAdapter:
    """One physical GPU (DXGIEnumAdapters1 + GetDesc1) with VRAM total."""

    def __init__(self) -> None:
        self.adapters: list[tuple[int, str, int]] = []  # (luid_key, name, total_mb)

    def refresh(self) -> None:
        self.adapters = []
        from ctypes import HRESULT

        dxgi = ctypes.windll.dxgi  # type: ignore[attr-defined]

        class _DXGI_ADAPTER_DESC(ctypes.Structure):
            # Classic DXGI_ADAPTER_DESC (same VRAM fields as Desc1, no LUID).
            _fields_ = [
                ("Description", ctypes.c_wchar * 128),
                ("VendorId", ctypes.c_uint),
                ("DeviceId", ctypes.c_uint),
                ("SubSysId", ctypes.c_uint),
                ("Revision", ctypes.c_uint),
                ("DedicatedVideoMemory", ctypes.c_size_t),
                ("DedicatedSystemMemory", ctypes.c_size_t),
                ("SharedSystemMemory", ctypes.c_size_t),
                ("AdapterLuid", ctypes.c_ulonglong),
            ]

        # {7b7166ec-21c7-44ae-b21a-c9ae321ae369} little-endian.
        IID_IDXGIFACTORY = bytes([
            0xEC, 0x66, 0x71, 0x7B, 0xC7, 0x21, 0xAE, 0x44,
            0xB2, 0x1A, 0xC9, 0xAE, 0x32, 0x1A, 0xE3, 0x69,
        ])
        dxgi.CreateDXGIFactory.argtypes = [
            ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        dxgi.CreateDXGIFactory.restype = HRESULT
        factory = ctypes.c_void_p()
        if dxgi.CreateDXGIFactory(IID_IDXGIFACTORY, ctypes.byref(factory)) != 0:
            return
        if not factory:
            return
        try:
            # COM object: first member points to the vtable (array of fns).
            vtbl_ptr = ctypes.cast(factory, ctypes.POINTER(ctypes.c_void_p)).contents.value
            if not vtbl_ptr:
                return
            vtbl = ctypes.cast(vtbl_ptr, ctypes.POINTER(ctypes.c_void_p))
            # IDXGIFactory vtable: 0..6 IDXGIObject, 7 EnumAdapters,
            # 8 MakeWindowAssociation, 9 GetWindowAssociation,
            # 10 CreateSwapChain, 11 CreateSoftwareAdapter.
            enum_adapters = ctypes.cast(
                vtbl[7], ctypes.WINFUNCTYPE(
                    HRESULT, ctypes.c_void_p, ctypes.c_uint,
                    ctypes.POINTER(ctypes.c_void_p)))
            release = ctypes.cast(
                vtbl[2], ctypes.WINFUNCTYPE(HRESULT, ctypes.c_void_p))
            i = 0
            while True:
                adapter = ctypes.c_void_p()
                if enum_adapters(factory, i, ctypes.byref(adapter)) != 0 or not adapter:
                    break
                avtp = ctypes.cast(adapter, ctypes.POINTER(ctypes.c_void_p)).contents.value
                if not avtp:
                    release(adapter)
                    break
                avt = ctypes.cast(avtp, ctypes.POINTER(ctypes.c_void_p))
                # IDXGIAdapter vtable: 0..6 IDXGIObject, 7 EnumOutputs,
                # 8 GetDesc, 9 CheckInterfaceSupport.
                get_desc = ctypes.cast(
                    avt[8], ctypes.WINFUNCTYPE(
                        HRESULT, ctypes.c_void_p,
                        ctypes.POINTER(_DXGI_ADAPTER_DESC)))
                rel_a = ctypes.cast(
                    avt[2], ctypes.WINFUNCTYPE(HRESULT, ctypes.c_void_p))
                desc = _DXGI_ADAPTER_DESC()
                if get_desc(adapter, ctypes.byref(desc)) == 0:
                    total_mb = desc.DedicatedVideoMemory // (1024 * 1024)
                    if total_mb > 0:
                        # AdaptorLuid: low 32 bits = LowPart, high = HighPart —
                        # exactly the key _parse_luid() builds from PDH names.
                        self.adapters.append((desc.AdapterLuid, desc.Description.strip(), total_mb))
                rel_a(adapter)
                i += 1
            release(factory)
        except Exception:
            pass  # DXGI enumeration best-effort; total falls back to PDH or None


def vram_snapshot(timeout: float = 2.0) -> dict:
    """Return a VRAM snapshot dict; never raises.

    Keys:
      ok        bool   — True when a usable free/total pair was obtained
      total_mb  int|None
      used_mb   int|None   (system-wide PDH Dedicated Usage)
      free_mb   int|None   (total − used, when both known)
      holders   list[(pid:int, name:str, mb:int)] — top per-process VRAM users
      source    str        'pdh+dxgi' | 'pdh' | 'dxgi' | 'none'
    """
    import time as _t

    result: dict = {"ok": False, "total_mb": None, "used_mb": None,
                    "free_mb": None, "holders": [], "source": "none"}
    used_counter = None
    proc_counter = None
    try:
        used_counter = _PdhCounter(r"\GPU Adapter Memory(*)\Dedicated Usage")
        used_items = used_counter.items()
        # Sum per adapter LUID; adapter memory may surface on any mirror LUID.
        per_luid: dict[int, float] = {}
        for name, val in used_items:
            luid = _parse_luid(name)
            if luid is None:
                continue
            per_luid[luid] = per_luid.get(luid, 0.0) + val
        used_mb = max((int(v / (1024 * 1024)) for v in per_luid.values()), default=0)
        if used_mb > 0:
            result["used_mb"] = used_mb
            result["source"] = "pdh"

        # Per-process holders (for the red dialog "who holds VRAM" list).
        try:
            proc_counter = _PdhCounter(r"\GPU Process Memory(*)\Dedicated Usage")
            per_proc: dict[int, tuple[str, int]] = {}
            for name, val in proc_counter.items():
                if val <= 0:
                    continue
                pid = _parse_pid(name)
                if pid is None:
                    continue
                mb = int(val / (1024 * 1024))
                if mb <= 0 and val > 0:
                    mb = 1
                cur = per_proc.get(pid)
                if cur is None or mb > cur[1]:
                    per_proc[pid] = (name, mb)
            result["holders"] = [
                {"pid": pid, "name": _pid_name(pid), "mb": mb}
                for pid, (_, mb) in sorted(per_proc.items(), key=lambda kv: -kv[1][1])
            ][:20]
        except Exception:
            pass

        dxgi = DxgiAdapter()
        dxgi.refresh()
        _t.sleep(0)  # (DXGI enumeration is synchronous here)
        if dxgi.adapters:
            # Pick the adapter the PDH usage belongs to (match by LUID when
            # possible), else the one with the most VRAM (the discrete GPU).
            total_mb = None
            if per_luid:
                for luid_key, name, total in dxgi.adapters:
                    if luid_key in per_luid:
                        total_mb = total
                        break
            if total_mb is None:
                _lk, _name, total_mb = max(dxgi.adapters, key=lambda a: a[2])
            result["total_mb"] = total_mb
            if result["source"] == "pdh":
                free_mb = total_mb - result["used_mb"]
                if free_mb < 0:
                    free_mb = 0
                result["free_mb"] = free_mb
                result["source"] = "pdh+dxgi"
            else:
                result["source"] = "dxgi"
        if result["free_mb"] is not None:
            result["ok"] = True
    except Exception:
        pass
    finally:
        if used_counter:
            used_counter.close()
        if proc_counter:
            proc_counter.close()
    return result


def _parse_pid(name: str) -> int | None:
    """pid from a PDH GPU Process Memory instance name: 'pid_1234_luid_...'."""
    if not name.startswith("pid_"):
        return None
    rest = name[4:]
    parts = rest.split("_", 1)
    try:
        return int(parts[0])
    except ValueError:
        return None


def _pid_name(pid: int) -> str:
    """Exe name for a PID via Windows Toolhelp (no psutil needed)."""
    try:
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        TH32CS_SNAPPROCESS = 0x00000002
        snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
        if not snap or snap == -1:
            return f"pid {pid}"

        class _PROCESSENTRY32W(ctypes.Structure):
            _fields_ = [
                ("dwSize", ctypes.c_uint32),
                ("cntUsage", ctypes.c_uint32),
                ("th32ProcessID", ctypes.c_uint32),
                ("th32DefaultHeapID", ctypes.c_size_t),
                ("th32ModuleID", ctypes.c_uint32),
                ("cntThreads", ctypes.c_uint32),
                ("th32ParentProcessID", ctypes.c_uint32),
                ("pcPriClassBase", ctypes.c_long),
                ("dwFlags", ctypes.c_uint32),
                ("szExeFile", ctypes.c_wchar * 260),
            ]

        entry = _PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(_PROCESSENTRY32W)
        ok = kernel32.Process32FirstW(snap, ctypes.byref(entry))
        while ok:
            if entry.th32ProcessID == pid:
                kernel32.CloseHandle(snap)
                return entry.szExeFile or f"pid {pid}"
            ok = kernel32.Process32NextW(snap, ctypes.byref(entry))
        kernel32.CloseHandle(snap)
        return f"pid {pid}"
    except Exception:
        return f"pid {pid}"


# ── Job Object: kill the whole child tree when THIS process dies ──────────
#
# Port of Quartermaster's internal/process/treecleanup_windows.go (MIT).
# The parent assigns ITSELF to a Job Object with KILL_ON_JOB_CLOSE and leaks
# the handle for its lifetime; every spawned child inherits the job, so when
# the parent exits — graceful close, crash, kill -9, logoff — the OS reaps the
# entire tree. Orphaned llama-server.exe processes holding GBs of VRAM become
# physically impossible, no matter how the parent died.
#
# BREAKAWAY_OK keeps self-update/relaunch legal: a child may escape the job by
# spawning itself with CREATE_BREAKAWAY_FROM_JOB (used by updaters; EngineBay
# does not spawn successors today, but the flag costs nothing).
#
# Windows 8+ supports nested jobs, so being inside an existing job (task
# scheduler, CI runners) is fine; on failure we return False and callers keep
# their manual-kill fallbacks — the job is a safety net, never a requirement.

JobObjectExtendedLimitInformation = 9
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x1000


def setup_job_tree() -> bool:
    """Put this process (and all future children) in a kill-on-close job.

    Returns True when the job is active. The job handle is deliberately NOT
    closed: KILL_ON_JOB_CLOSE fires when the LAST handle is released, which
    the OS does at process exit — that is the whole mechanism.
    """
    if os.name != "nt":
        return False
    try:
        k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        # Explicit signatures: default c_int restype/args TRUNCATE 64-bit
        # handles (GetCurrentProcess' pseudo-handle is 0xFFFF...FFFF and
        # arrives as 0x00000000FFFFFFFF → ERROR_INVALID_HANDLE on x64).
        k32.CreateJobObjectW.restype = ctypes.c_void_p
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        k32.SetInformationJobObject.restype = ctypes.c_int
        k32.SetInformationJobObject.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
        k32.AssignProcessToJobObject.restype = ctypes.c_int
        k32.AssignProcessToJobObject.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p]

        class _IO_COUNTERS(ctypes.Structure):
            _fields_ = [("ReadOperationCount", ctypes.c_ulonglong),
                        ("WriteOperationCount", ctypes.c_ulonglong),
                        ("OtherOperationCount", ctypes.c_ulonglong),
                        ("ReadTransferCount", ctypes.c_ulonglong),
                        ("WriteTransferCount", ctypes.c_ulonglong),
                        ("OtherTransferCount", ctypes.c_ulonglong)]

        class _JOBOCKET_BASIC_LIMIT(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_ulonglong),
                        ("PerJobUserTimeLimit", ctypes.c_ulonglong),
                        ("LimitFlags", ctypes.c_uint),
                        ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", ctypes.c_uint),
                        ("Affinity", ctypes.c_size_t),
                        ("PriorityClass", ctypes.c_uint),
                        ("SchedulingClass", ctypes.c_uint)]

        class _JOBOCKET_EXTENDED_LIMIT(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", _JOBOCKET_BASIC_LIMIT),
                        ("IoInfo", _IO_COUNTERS),
                        ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        job = k32.CreateJobObjectW(None, None)
        if not job:
            return False
        info = _JOBOCKET_EXTENDED_LIMIT()
        info.BasicLimitInformation.LimitFlags = (
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_BREAKAWAY_OK)
        cur = k32.GetCurrentProcess()  # pseudo-handle, stays c_void_p-wide
        if not k32.AssignProcessToJobObject(job, cur):
            return False
        if not k32.SetInformationJobObject(
                job, JobObjectExtendedLimitInformation,
                ctypes.byref(info), ctypes.sizeof(info)):
            return False
        return True  # handle intentionally leaked — see docstring
    except Exception:
        return False


# ── GGUF header reader (metadata only, bounded read) ───────────────────────
#
# GGUF layout: magic "GGUF", u32 version, u64 tensor_count, u64 kv_count,
# then kv_count pairs of (u64-len key string, u32 value type, value). The
# whole header is at the START of the file — typically < 200 KB even with a
# chat template embedded — so we stream at most a few MB and never touch the
# 14 GB of tensor data behind it. Same spirit as Quartermaster's
# autogen/gguf.go metadata pass, cut down to what the model library needs.

_GGUF_MAGIC = b"GGUF"
# value type -> fixed byte size; 8 (string) and 9 (array) are handled specially
_GGUF_SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4,
                      7: 1, 10: 8, 11: 8, 12: 8}
# llama.h LLAMA_FTYPE (stable classic values; newer entries stay raw "ft<N>")
GGUF_FTYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 4: "Q4_1_F16", 7: "Q8_0",
    8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M",
    13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M",
    18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS",
    23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ1_M", 26: "IQ4_NL", 27: "IQ4_XS",
    28: "IQ2_S", 29: "IQ2_M", 30: "IQ3_S", 31: "IQ3_M",
}
_GGUF_MAX_HEADER_BYTES = 8 * 1024 * 1024   # hard read cap
_GGUF_MAX_KV = 1024                        # sanity: real headers hold ~100
_GGUF_MAX_STRING = 64 * 1024               # materialize small strings only
_GGUF_MAX_ARRAY_ITEMS = 1_000_000


def gguf_metadata(path) -> dict:
    """Read GGUF header KV pairs we care about. Never raises, never reads
    past the first few MB. Returns {} when the file is not a readable GGUF.

    Result keys: ok, version, tensors, arch, name, ctx, blocks, size_label,
    file_type (raw int), quant (human name from file_type, 'ft<N>' if new).
    """
    out: dict = {}
    try:
        with open(path, "rb") as f:
            buf = bytearray()
            pos = 0

            def need(n: int) -> bool:
                nonlocal pos
                while len(buf) - pos < n:
                    if len(buf) > _GGUF_MAX_HEADER_BYTES:
                        return False
                    chunk = f.read(min(1 << 20, _GGUF_MAX_HEADER_BYTES - len(buf)))
                    if not chunk:
                        return False
                    buf.extend(chunk)
                return True

            def u(sz: int) -> int | None:
                nonlocal pos
                if not need(sz):
                    return None
                v = int.from_bytes(buf[pos:pos + sz], "little", signed=False)
                pos += sz
                return v

            def s() -> str | None:
                nonlocal pos
                ln = u(8)
                if ln is None or ln > _GGUF_MAX_STRING or not need(ln):
                    return None
                v = bytes(buf[pos:pos + ln]).decode("utf-8", "replace")
                pos += ln
                return v

            def skip_value(vt: int) -> bool:
                nonlocal pos
                if vt in _GGUF_SCALAR_SIZES:
                    if not need(_GGUF_SCALAR_SIZES[vt]):
                        return False
                    pos += _GGUF_SCALAR_SIZES[vt]
                    return True
                if vt == 8:  # string: parse length, skip bytes
                    ln = u(8)
                    if ln is None or ln > _GGUF_MAX_HEADER_BYTES or not need(ln):
                        return False
                    pos += ln
                    return True
                if vt == 9:  # array: element type + count, then elements
                    et = u(4)
                    cnt = u(8)
                    if et is None or cnt is None or cnt > _GGUF_MAX_ARRAY_ITEMS:
                        return False
                    if et in _GGUF_SCALAR_SIZES:
                        total = cnt * _GGUF_SCALAR_SIZES[et]
                        if total > _GGUF_MAX_HEADER_BYTES or not need(total):
                            return False
                        pos += total
                        return True
                    if et == 8:  # array of strings: walk lengths
                        for _ in range(cnt):
                            ln = u(8)
                            if ln is None or ln > _GGUF_MAX_HEADER_BYTES or not need(ln):
                                return False
                            pos += ln
                        return True
                    return False  # array of arrays: not in real headers
                return False

            if not need(4) or bytes(buf[:4]) != _GGUF_MAGIC:
                return out
            pos = 4
            version = u(4)
            tensors = u(8)
            kv_count = u(8)
            if version is None or tensors is None or kv_count is None:
                return out
            if kv_count > _GGUF_MAX_KV:
                kv_count = _GGUF_MAX_KV  # cap corrupt headers

            kv: dict[str, object] = {}
            arch: str | None = None
            # Early exit once everything the library needs is in hand — the
            # tokenizer arrays at the header tail cost seconds on HDDs and
            # hold nothing we display. file_type may be absent (ik-quant
            # builds often skip it), so it never gates the exit.
            for _ in range(kv_count):
                key = s()
                vt = u(4)
                if key is None or vt is None:
                    break
                if vt in _GGUF_SCALAR_SIZES:
                    if vt in (4, 10):  # u32/u64 → int
                        sz = _GGUF_SCALAR_SIZES[vt]
                        val = u(sz)
                    elif vt in (5, 11):  # i32/i64 → signed
                        if not need(_GGUF_SCALAR_SIZES[vt]):
                            break
                        sz = _GGUF_SCALAR_SIZES[vt]
                        val = int.from_bytes(buf[pos:pos + sz], "little", signed=True)
                        pos += sz
                    else:  # tiny scalars (u8/i8/u16/i16/f32/bool/f64) → raw
                        if not skip_value(vt):
                            break
                        val = None
                elif vt == 8:
                    val = s()
                    if val is None:
                        break
                else:  # arrays and everything else: skip, keep parsing
                    if not skip_value(vt):
                        break
                    val = None
                if key and val is not None and len(kv) < 256:
                    kv[key] = val
                if key == "general.architecture" and isinstance(val, str):
                    arch = val
                if (arch is not None and "general.name" in kv
                        and kv.get(f"{arch}.context_length") is not None):
                    break

            arch = kv.get("general.architecture")
            out["ok"] = True
            out["version"] = version
            out["tensors"] = tensors
            out["arch"] = arch if isinstance(arch, str) else None
            out["name"] = kv.get("general.name") if isinstance(kv.get("general.name"), str) else None
            out["size_label"] = (kv.get("general.size_label")
                                 if isinstance(kv.get("general.size_label"), str) else None)
            ft = kv.get("general.file_type")
            out["file_type"] = ft if isinstance(ft, int) else None
            out["quant"] = GGUF_FTYPE_NAMES.get(ft, f"ft{ft}") if isinstance(ft, int) else None
            if isinstance(arch, str):
                out["ctx"] = kv.get(f"{arch}.context_length")
                out["blocks"] = kv.get(f"{arch}.block_count")
            else:
                out["ctx"] = None
                out["blocks"] = None
            # MoE detection WITHOUT tensor parsing: the size_label "NNB-AxB"
            # pattern (total-active params) or an explicit expert-count KV.
            # A MoE model fits a small GPU only with expert offload
            # (--n-cpu-moe / -ot exps=CPU), so the launcher must know.
            moe = False
            if isinstance(out["size_label"], str) and re.match(
                    r"^\d+(?:\.\d+)?[BbMm]-A\d+(?:\.\d+)?[BbMm]$", out["size_label"]):
                moe = True
            ec = kv.get("general.expert_count")
            if isinstance(ec, int) and ec > 0:
                moe = True
            out["moe"] = moe
            return out
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    chk = sub.add_parser("check", help="report missing DLLs for a binary")
    chk.add_argument("exe", help="path to a Windows .exe/.dll")
    sub.add_parser("vram", help="print free/total VRAM (PDH/DXGI)")
    gg = sub.add_parser("gguf", help="print GGUF header metadata")
    gg.add_argument("path", help="path to a .gguf file")
    args = ap.parse_args()

    if args.cmd == "check":
        if not Path(args.exe).is_file():
            print(f"ERROR: {args.exe} not found", file=sys.stderr)
            return 1
        deps = missing_dlls(args.exe)
        if not deps:
            print(f"OK: {os.path.basename(args.exe)} — no missing load-time DLLs found")
        else:
            print(hint(args.exe))
            for name, by in deps:
                print(f"  - {name}  (imported by {by})")
        return 0

    if args.cmd == "vram":
        snap = vram_snapshot()
        print(f"ok={snap['ok']} source={snap['source']}")
        print(f"total={snap['total_mb']} MB used={snap['used_mb']} MB free={snap['free_mb']} MB")
        for h in snap["holders"]:
            print(f"  holder: pid={h['pid']} {h['name']} {h['mb']} MB")
        return 0

    if args.cmd == "gguf":
        if not Path(args.path).is_file():
            print(f"ERROR: {args.path} not found", file=sys.stderr)
            return 1
        m = gguf_metadata(args.path)
        if not m.get("ok"):
            print("ERROR: not a readable GGUF (bad magic or truncated header)",
                  file=sys.stderr)
            return 1
        for k in ("arch", "name", "size_label", "quant", "ctx", "blocks",
                  "tensors", "version", "moe"):
            if m.get(k) is not None:
                print(f"{k:12s} = {m[k]}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())