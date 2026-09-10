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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    chk = sub.add_parser("check", help="report missing DLLs for a binary")
    chk.add_argument("exe", help="path to a Windows .exe/.dll")
    sub.add_parser("vram", help="print free/total VRAM (PDH/DXGI)")
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

    return 0


if __name__ == "__main__":
    sys.exit(main())