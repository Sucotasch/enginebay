"""
LLM Launcher — GUI for launching llama-server with model selection and parameters.
Dependencies: Python 3 + PyQt6.
Portable — auto-discovers Hermes and llama-server.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
import io
import shlex
from collections import defaultdict
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPalette, QColor, QFont, QIcon
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLineEdit, QPushButton, QLabel, QComboBox,
    QTextEdit, QListWidget, QCheckBox, QFileDialog, QMessageBox,
    QInputDialog, QSizePolicy, QScrollArea, QFrame, QDialog, QMenu,
)

# --- Paths ---
SCRIPT_DIR = Path(__file__).parent
CONFIG_FILE = SCRIPT_DIR / "launcher_config.json"
HISTORY_FILE = SCRIPT_DIR / "launcher_history.json"
PRESETS_FILE = SCRIPT_DIR / "launcher_presets.json"

# --- Auto-discover Hermes ---
def find_hermes_home() -> Path | None:
    candidates = [
        Path.home() / ".hermes",
        Path(os.environ.get("APPDATA", "")) / "hermes",
        Path(os.environ.get("LOCALAPPDATA", "")) / "hermes",
    ]
    for p in candidates:
        if p.exists() and (p / "config.yaml").exists():
            return p
    return None


HERMES_HOME = find_hermes_home()
HERMES_CONFIG = HERMES_HOME / "config.yaml" if HERMES_HOME else None

# --- Auto-discover llama-server ---
def find_llama_server() -> Path | None:
    env_path = os.environ.get("LLAMA_SERVER_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    lmstudio_base = Path.home() / ".cache" / "lm-studio" / "extensions" / "backends"
    if lmstudio_base.exists():
        candidates = sorted(lmstudio_base.glob("llama.cpp-win-*/llama-server.exe"), reverse=True)
        if candidates:
            return candidates[0]

    which = shutil.which("llama-server") or shutil.which("llama-server.exe")
    if which:
        return Path(which)

    common = [
        Path.home() / "AppData" / "Local" / "Programs" / "llama.cpp" / "llama-server.exe",
        Path("C:/Program Files/llama.cpp/llama-server.exe"),
        Path("C:/Program Files (x86)/llama.cpp/llama-server.exe"),
    ]
    for p in common:
        if p.exists():
            return p
    return None


LLAMA_SERVER = find_llama_server()

# --- Engines (alternative llama-server sources) ---

def _extract_cuda_version(name: str) -> str:
    """Extract a 'cuda-13.3' / 'cuda13.3' style version token from an asset name."""
    parts = name.split("-")
    for i, part in enumerate(parts):
        if part == "cuda" and i + 1 < len(parts):
            return f"cuda-{parts[i + 1]}"
        if part.startswith("cuda") and len(part) > 4 and part[4:5].isdigit():
            return part
    return ""


def _classify_llama_asset(name: str):
    """llama.cpp upstream assets:
    llama-<tag>-bin-win-cuda-<v>-x64.zip  +  cudart-llama-<tag>-bin-win-cuda-<v>-x64.zip.
    Returns (kind, cuda) where kind in ('bin','cudart'), or None."""
    if not name.endswith(".zip") or "win" not in name.lower():
        return None
    if name.startswith("cudart-"):
        kind = "cudart"
    elif name.startswith("llama-"):
        kind = "bin"
    else:
        return None
    cuda = _extract_cuda_version(name)
    return (kind, cuda) if cuda else None


def _classify_beellama_asset(name: str):
    """BeeLlama.cpp fork assets:
    beellama-<tag>-bin-win-cuda-<v>-x64.zip  +  beellama-<tag>-cudart-win-cuda-<v>-x64.zip.
    Returns (kind, cuda) where kind in ('bin','cudart'), or None."""
    if not name.endswith(".zip") or "win" not in name.lower() or not name.startswith("beellama-"):
        return None
    if "-cudart-" in name:
        kind = "cudart"
    elif "-bin-" in name:
        kind = "bin"
    else:
        return None
    cuda = _extract_cuda_version(name)
    return (kind, cuda) if cuda else None


ENGINES = {
    "llama.cpp": {
        "label": "llama.cpp (upstream)",
        "repo": "ggml-org/llama.cpp",
        "api": "https://api.github.com/repos/ggml-org/llama.cpp/releases",
        "versions_dir": SCRIPT_DIR / "llama.cpp" / "versions",
        "classify": _classify_llama_asset,
        "default_params": (
            "-c 98304 -ngl 99 -b 2048 -ub 512 "
            "--kv-unified --cache-type-k q4_0 --cache-type-v q4_0 "
            "-t 5 --flash-attn on --reasoning off "
            "--temp 1.0 --min-p 0.05 --top-p 0.95 --top-k 64"
        ),
    },
    "beellama.cpp": {
        "label": "BeeLlama.cpp (fork — KVarN, precision tail)",
        "repo": "Anbeeld/beellama.cpp",
        "api": "https://api.github.com/repos/Anbeeld/beellama.cpp/releases",
        "versions_dir": SCRIPT_DIR / "beellama.cpp" / "versions",
        "classify": _classify_beellama_asset,
        "default_params": (
            "-c 98304 -ngl 99 -b 2048 -ub 512 "
            "--kv-unified --cache-type-k kvarn5 --cache-type-v kvarn4 "
            "--kv-tail-tokens 1024 "
            "-t 5 --flash-attn on --reasoning off "
            "--temp 1.0 --min-p 0.05 --top-p 0.95 --top-k 64"
        ),
    },
    "ik_llama.cpp": {
        "label": "ik_llama.cpp (fork — IQ4_KT/KS, trellis quants)",
        "repo": "ikawrakow/ik_llama.cpp",
        "api": "https://api.github.com/repos/ikawrakow/ik_llama.cpp/releases",
        "versions_dir": SCRIPT_DIR / "ik_llama.cpp" / "versions",
        "classify": None,
        "manual_build": True,
        "build_script": "build-ikllama.bat",
        "default_params": (
            "-c 98304 -ngl 99 -b 1024 -ub 256 "
            "--cache-type-k q4_0 --cache-type-v q4_0 "
            "-t 5 --flash-attn on --jinja --reasoning auto "
            "--temp 1.0 --min-p 0.05 --top-p 0.95 --top-k 64"
        ),
        # Flags that THIS engine does NOT support — used to detect stale params
        # when loading a saved config that was written by a different engine.
        "incompatible_params": ("--kv-unified", "kvarn", "--kv-tail-tokens",
                                "--spec-draft-n-max", "draft-mtp"),
    },
}

DEFAULT_ENGINE = "llama.cpp"
ENGINES[DEFAULT_ENGINE]["versions_dir"].mkdir(parents=True, exist_ok=True)
ENGINES["beellama.cpp"]["versions_dir"].mkdir(parents=True, exist_ok=True)
ENGINES["ik_llama.cpp"]["versions_dir"].mkdir(parents=True, exist_ok=True)

# Backward-compatible aliases
VERSIONS_DIR = ENGINES["llama.cpp"]["versions_dir"]


def get_installed_versions(versions_dir: Path, active_path: str | None = None) -> list[dict]:
    versions = []
    if not versions_dir.exists():
        return versions
    for d in sorted(versions_dir.iterdir(), reverse=True):
        if d.is_dir():
            exe = d / "llama-server.exe"
            if not exe.exists():
                for sub in d.iterdir():
                    if sub.is_dir():
                        exe = sub / "llama-server.exe"
                        if exe.exists():
                            break
            exe_str = str(exe) if exe.exists() else None
            is_active = False
            if active_path and exe_str:
                is_active = active_path == exe_str
            elif not active_path and LLAMA_SERVER:
                is_active = str(LLAMA_SERVER).startswith(str(d))
            versions.append({
                "name": d.name,
                "path": exe_str,
                "dir": str(d),
                "active": is_active,
            })
    return versions


# Per-engine default launch params live in ENGINES[...]["default_params"].
# The upstream llama.cpp default is retained below for reference / CLI reuse.
DEFAULT_PARAMS = ENGINES["llama.cpp"]["default_params"]

HOST = "127.0.0.1"
PORT = "8888"


def load_history() -> list[str]:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def save_history(model_path: str, history: list[str]):
    if model_path in history:
        history.remove(model_path)
    history.insert(0, model_path)
    history = history[:20]
    HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False), encoding="utf-8")


def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_config(cfg: dict):
    CONFIG_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def load_presets() -> dict:
    if PRESETS_FILE.exists():
        try:
            return json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_presets(presets: dict):
    PRESETS_FILE.write_text(json.dumps(presets, ensure_ascii=False, indent=2), encoding="utf-8")


# --- Dark theme stylesheet ---
DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1e1e1e;
    color: #d4d4d4;
    font-family: Consolas, monospace;
    font-size: 10pt;
}
QGroupBox {
    border: 1px solid #333;
    border-radius: 4px;
    margin-top: 10px;
    padding-top: 14px;
    font-weight: bold;
    color: #569cd6;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
}
QLineEdit {
    background-color: #2d2d2d;
    color: #d4d4d4;
    border: 1px solid #333;
    border-radius: 3px;
    padding: 4px 6px;
    font-family: Consolas, monospace;
    font-size: 10pt;
}
QLineEdit:read-only {
    color: #aaa;
}
QTextEdit {
    background-color: #2d2d2d;
    color: #d4d4d4;
    border: 1px solid #333;
    border-radius: 3px;
    font-family: Consolas, monospace;
}
QTextEdit#logView {
    background-color: #1e1e1e;
    color: #cccccc;
}
QComboBox {
    background-color: #2d2d2d;
    color: #d4d4d4;
    border: 1px solid #333;
    border-radius: 3px;
    padding: 4px 8px;
    font-family: Consolas, monospace;
    font-size: 9pt;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QComboBox QAbstractItemView {
    background-color: #2d2d2d;
    color: #d4d4d4;
    selection-background-color: #094771;
    border: 1px solid #333;
}
QPushButton {
    background-color: #333;
    color: #d4d4d4;
    border: 1px solid #444;
    border-radius: 3px;
    padding: 5px 12px;
    font-family: Consolas, monospace;
    font-size: 9pt;
}
QPushButton:hover {
    background-color: #444;
}
QPushButton:pressed {
    background-color: #555;
}
QPushButton:disabled {
    color: #666;
    background-color: #2a2a2a;
}
QPushButton#startBtn {
    background-color: #2ea043;
    color: white;
    font-size: 11pt;
    font-weight: bold;
    padding: 6px 20px;
}
QPushButton#startBtn:hover {
    background-color: #3fb950;
}
QPushButton#stopBtn {
    background-color: #da3633;
    color: white;
    font-size: 11pt;
    font-weight: bold;
    padding: 6px 20px;
}
QPushButton#stopBtn:hover {
    background-color: #f85149;
}
QPushButton#browseBtn {
    background-color: #0e639c;
    color: white;
    font-weight: bold;
}
QPushButton#browseBtn:hover {
    background-color: #1177bb;
}
QPushButton#downloadBtn {
    background-color: #2ea043;
    color: white;
}
QPushButton#activateBtn {
    background-color: #8957e5;
    color: white;
}
QPushButton#deleteBtn {
    background-color: #da3633;
    color: white;
}
QListWidget {
    background-color: #2d2d2d;
    color: #d4d4d4;
    border: 1px solid #333;
    border-radius: 3px;
    font-family: Consolas, monospace;
    font-size: 9pt;
}
QListWidget::item:selected {
    background-color: #094771;
    color: white;
}
QCheckBox {
    color: #888;
    spacing: 6px;
}
QCheckBox::indicator {
    width: 14px;
    height: 14px;
}
QLabel {
    color: #888;
}
QLabel#statusLabel {
    font-weight: bold;
    font-size: 10pt;
}
QLabel#previewLabel {
    background-color: #2d2d2d;
    color: #6a9955;
    font-size: 8pt;
    padding: 6px;
    border: 1px solid #333;
    border-radius: 3px;
}
QLabel#infoLabel {
    font-size: 8pt;
}
"""


class LLMLauncher(QMainWindow):
    _log_signal = pyqtSignal(str)
    _set_status_signal = pyqtSignal(str)
    _set_progress_signal = pyqtSignal(str)
    _show_updates_signal = pyqtSignal(list)
    _refresh_versions_signal = pyqtSignal()
    _on_exit_signal = pyqtSignal()
    _health_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("LLM Launcher — llama-server")
        self.setMinimumSize(700, 500)
        self.resize(820, 850)

        self.process: subprocess.Popen | None = None
        self.monitoring = False
        self.history = load_history()
        self.cfg = load_config()
        self.presets = load_presets()
        self.log_file = None
        self.installed_versions: list[dict] = []

        self._log_signal.connect(self._log)
        self._set_status_signal.connect(self._set_status)
        self._show_updates_signal.connect(self._show_updates)
        self._refresh_versions_signal.connect(self._refresh_versions_list)
        self._on_exit_signal.connect(self._on_process_exit)
        self._health_signal.connect(self._set_status)

        self._build_ui()

        self._set_progress_signal.connect(self.dl_progress_label.setText)

        self._ensure_provider()
        self._load_saved_state()
        self._start_health_monitor()

    def _ensure_provider(self):
        if not HERMES_CONFIG:
            self._log("Hermes not found - provider not registered")
            return

        try:
            content = HERMES_CONFIG.read_text(encoding="utf-8")
        except Exception as e:
            self._log(f"Cannot read {HERMES_CONFIG}: {e}")
            return

        if "local-llama:" in content:
            self._log(f"Hermes provider found: {HERMES_HOME}")
            return

        insert_marker = "providers:"
        if insert_marker not in content:
            self._log("providers section not found in config.yaml")
            return

        provider_block = """\
  local-llama:
    base_url: http://127.0.0.1:8888/v1
    api_key: not-needed
    models:
    - Local Model"""

        new_content = content.replace(
            insert_marker,
            insert_marker + "\n" + provider_block
        )

        if "fallback_providers:" in new_content and "local-llama" not in new_content.split("fallback_providers:")[1].split("\n")[0]:
            new_content = new_content.replace(
                "fallback_providers:",
                "fallback_providers:\n  - local-llama"
            )

        try:
            HERMES_CONFIG.write_text(new_content, encoding="utf-8")
            self._log(f"Provider local-llama added to {HERMES_CONFIG}")
        except Exception as e:
            self._log(f"Cannot write config: {e}")

    # ── UI ──────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 6, 10, 10)
        main_layout.setSpacing(4)

        # ── Status bar (auto-discovery info) ──
        if HERMES_HOME and LLAMA_SERVER:
            info_text = f"Hermes: {HERMES_HOME}  |  llama-server: {LLAMA_SERVER.name}"
            info_color = "#3fb950"
        elif HERMES_HOME:
            info_text = f"Hermes: {HERMES_HOME}  |  llama-server: NOT FOUND"
            info_color = "#d29922"
        elif LLAMA_SERVER:
            info_text = f"Hermes: NOT FOUND  |  llama-server: {LLAMA_SERVER.name}"
            info_color = "#d29922"
        else:
            info_text = "Hermes: NOT FOUND  |  llama-server: NOT FOUND"
            info_color = "#da3633"

        info_label = QLabel(info_text)
        info_label.setObjectName("infoLabel")
        info_label.setStyleSheet(f"color: {info_color}; font-size: 8pt;")
        main_layout.addWidget(info_label)

        # ── Model selector ──
        grp_model = QGroupBox("Model")
        grp_model_layout = QHBoxLayout(grp_model)

        self.model_entry = QLineEdit()
        self.model_entry.setPlaceholderText("Path to GGUF model...")
        grp_model_layout.addWidget(self.model_entry, 1)

        self.recent_btn = QPushButton(" \u25be ")
        self.recent_btn.setFixedWidth(36)
        self.recent_btn.clicked.connect(self._show_recent)
        grp_model_layout.addWidget(self.recent_btn)

        browse_model_btn = QPushButton(" Browse... ")
        browse_model_btn.setObjectName("browseBtn")
        browse_model_btn.clicked.connect(self._browse_model)
        grp_model_layout.addWidget(browse_model_btn)

        main_layout.addWidget(grp_model)

        # ── Server binary ──
        grp_bin = QGroupBox("llama-server")
        grp_bin_layout = QHBoxLayout(grp_bin)

        self.bin_entry = QLineEdit()
        self.bin_entry.setPlaceholderText("Path to llama-server.exe...")
        grp_bin_layout.addWidget(self.bin_entry, 1)

        browse_bin_btn = QPushButton(" ... ")
        browse_bin_btn.clicked.connect(self._browse_binary)
        grp_bin_layout.addWidget(browse_bin_btn)

        main_layout.addWidget(grp_bin)

        # ── Host / Port ──
        net_layout = QHBoxLayout()
        net_layout.setSpacing(8)

        host_lbl = QLabel("Host:")
        net_layout.addWidget(host_lbl)
        self.host_entry = QLineEdit(HOST)
        self.host_entry.setFixedWidth(150)
        net_layout.addWidget(self.host_entry)

        port_lbl = QLabel("Port:")
        net_layout.addWidget(port_lbl)
        self.port_entry = QLineEdit(PORT)
        self.port_entry.setFixedWidth(80)
        net_layout.addWidget(self.port_entry)

        net_layout.addStretch()
        main_layout.addLayout(net_layout)

        # ── Presets ──
        preset_layout = QHBoxLayout()
        preset_layout.setSpacing(6)

        preset_lbl = QLabel("Preset:")
        preset_layout.addWidget(preset_lbl)

        self.preset_combo = QComboBox()
        self.preset_combo.addItem("(none)")
        for name in self.presets:
            self.preset_combo.addItem(name)
        self.preset_combo.currentTextChanged.connect(self._on_preset_selected)
        preset_layout.addWidget(self.preset_combo, 1)

        save_preset_btn = QPushButton("\U0001f4be")
        save_preset_btn.setFixedWidth(36)
        save_preset_btn.clicked.connect(self._save_preset)
        preset_layout.addWidget(save_preset_btn)

        delete_preset_btn = QPushButton("\U0001f5d1")
        delete_preset_btn.setFixedWidth(36)
        delete_preset_btn.clicked.connect(self._delete_preset)
        preset_layout.addWidget(delete_preset_btn)

        main_layout.addLayout(preset_layout)

        # ── Launch params ──
        grp_params = QGroupBox("Launch Parameters")
        grp_params_layout = QHBoxLayout(grp_params)

        self.params_text = QTextEdit()
        self.params_text.setPlainText(ENGINES[DEFAULT_ENGINE]["default_params"])
        self.params_text.setMaximumHeight(90)
        self.params_text.textChanged.connect(self._update_preview)
        grp_params_layout.addWidget(self.params_text, 1)

        paste_btn = QPushButton("\U0001f4cb")
        paste_btn.setFixedWidth(36)
        paste_btn.clicked.connect(self._paste_params)
        grp_params_layout.addWidget(paste_btn)

        main_layout.addWidget(grp_params)

        # ── Engine / versions ──
        grp_versions = QGroupBox("Engine / llama-server Versions")
        grp_versions_layout = QVBoxLayout(grp_versions)

        engine_row = QHBoxLayout()
        engine_lbl = QLabel("Engine:")
        engine_row.addWidget(engine_lbl)
        self.engine_combo = QComboBox()
        for eid, eng in ENGINES.items():
            self.engine_combo.addItem(eng["label"], eid)
        self.engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        engine_row.addWidget(self.engine_combo, 1)
        grp_versions_layout.addLayout(engine_row)

        self.versions_list = QListWidget()
        self.versions_list.setMaximumHeight(80)
        grp_versions_layout.addWidget(self.versions_list)

        vbtn_layout = QHBoxLayout()
        vbtn_layout.setSpacing(6)

        check_btn = QPushButton("\U0001f504 Check Updates")
        check_btn.clicked.connect(self._check_updates)
        vbtn_layout.addWidget(check_btn)

        download_btn = QPushButton("\u2b07 Download")
        download_btn.setObjectName("downloadBtn")
        download_btn.clicked.connect(self._download_version)
        vbtn_layout.addWidget(download_btn)

        use_btn = QPushButton("\u2705 Use")
        use_btn.setObjectName("activateBtn")
        use_btn.clicked.connect(self._use_version)
        vbtn_layout.addWidget(use_btn)

        del_ver_btn = QPushButton("\U0001f5d1 Delete")
        del_ver_btn.setObjectName("deleteBtn")
        del_ver_btn.clicked.connect(self._delete_version)
        vbtn_layout.addWidget(del_ver_btn)

        vbtn_layout.addStretch()

        self.dl_progress_label = QLabel("")
        self.dl_progress_label.setStyleSheet("color: #888; font-size: 8pt;")
        vbtn_layout.addWidget(self.dl_progress_label)

        self.preview_cb = QCheckBox("Show preview")
        self.preview_cb.setToolTip("Show pre-release/preview builds (unstable)")
        vbtn_layout.addWidget(self.preview_cb)

        grp_versions_layout.addLayout(vbtn_layout)
        main_layout.addWidget(grp_versions)

        # ── Start/Stop buttons + status + log save ──
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)

        self.start_btn = QPushButton(" \u25b6  Start ")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.clicked.connect(self._start_server)
        btn_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton(" \u25a0  Stop ")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_server)
        btn_layout.addWidget(self.stop_btn)

        btn_layout.addStretch()

        self.log_save_cb = QCheckBox("\U0001f4be Save log")
        self.log_save_cb.setChecked(self.cfg.get("log_save", False))
        self.log_save_cb.stateChanged.connect(self._toggle_log_save)
        btn_layout.addWidget(self.log_save_cb)

        self.status_label = QLabel(" \u25cf Offline ")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setStyleSheet("color: #888; font-weight: bold; font-size: 10pt;")
        btn_layout.addWidget(self.status_label)

        main_layout.addLayout(btn_layout)

        # ── Launch command preview ──
        grp_preview = QGroupBox("Command")
        grp_preview_layout = QVBoxLayout(grp_preview)

        self.preview_label = QLabel("(select a model)")
        self.preview_label.setObjectName("previewLabel")
        self.preview_label.setWordWrap(True)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        grp_preview_layout.addWidget(self.preview_label)

        main_layout.addWidget(grp_preview)

        # ── Log ──
        grp_log = QGroupBox("Log")
        grp_log_layout = QVBoxLayout(grp_log)

        self.log_view = QTextEdit()
        self.log_view.setObjectName("logView")
        self.log_view.setReadOnly(True)
        grp_log_layout.addWidget(self.log_view)

        main_layout.addWidget(grp_log, 1)

        # ── Connect signals ──
        self.model_entry.textChanged.connect(self._update_preview)
        self.host_entry.textChanged.connect(self._update_preview)
        self.port_entry.textChanged.connect(self._on_port_changed)

        self._refresh_versions_list()
        self._update_preview()

    # ── Actions ─────────────────────────────────────────────────────

    def _browse_model(self):
        init_dir = str(Path.home() / "Ai" / "Models") if Path.home().joinpath("Ai/Models").exists() else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, "Select GGUF Model", init_dir,
            "GGUF files (*.gguf);;All files (*.*)"
        )
        if path:
            self.model_entry.setText(path)
            save_history(path, self.history)

    def _browse_binary(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select llama-server.exe", "",
            "Executable (*.exe);;All files (*.*)"
        )
        if path:
            self.bin_entry.setText(path)

    def _show_recent(self):
        if not self.history:
            QMessageBox.information(self, "History", "No models in history yet.")
            return

        menu = QMenu(self)
        for p in self.history:
            name = Path(p).name
            action = menu.addAction(name)
            action.triggered.connect(lambda checked, v=p: self.model_entry.setText(v))
        menu.exec(self.recent_btn.mapToGlobal(self.recent_btn.rect().bottomLeft()))

    def _on_preset_selected(self, name: str):
        if name == "(none)" or name not in self.presets:
            return
        p = self.presets[name]
        # Presets are model-agnostic (params only), but params are engine-specific:
        # warn if the selected preset carries flags the current engine cannot parse,
        # otherwise they would crash the server at launch (exit 1, help dump).
        bad = self._find_incompatible(p.get("params", ""))
        if bad:
            eng = self._current_engine()
            self._log(f'Preset "{name}" has flags incompatible with {eng["label"]}: '
                      f'{", ".join(bad)}')
            box = QMessageBox(self)
            box.setWindowTitle("Engine mismatch")
            box.setIcon(QMessageBox.Icon.Warning)
            box.setText(
                f'Preset "{name}" contains flags that {eng["label"]} does not support:\n'
                f"    {', '.join(bad)}\n\n"
                f"The server would fail at launch. Load engine defaults instead, "
                f"or apply the preset as-is (advanced)."
            )
            defaults_btn = box.addButton("Load engine defaults",
                                         QMessageBox.ButtonRole.AcceptRole)
            box.addButton("Apply preset anyway",
                          QMessageBox.ButtonRole.DestructiveRole)
            box.exec()
            if box.clickedButton() == defaults_btn:
                self.params_text.setPlainText(eng["default_params"])
                self._log(f"Loaded {eng['label']} defaults instead")
                self._update_preview()
                return
        if "params" in p:
            self.params_text.setPlainText(p["params"])
        if "host" in p:
            self.host_entry.setText(p["host"])
        if "port" in p:
            self.port_entry.setText(p["port"])
        self._update_preview()
        self._log(f'Preset "{name}" loaded')

    def _find_incompatible(self, params: str) -> list[str]:
        """Return incompatible-param substrings found in a params string for the
        current engine. Token-based: a flag like 'kvarn' also matches 'kvarn5'."""
        eng = self._current_engine()
        incompat = eng.get("incompatible_params")
        if not incompat or not params:
            return []
        tokens = params.split()
        found = []
        for tok in incompat:
            if any(tok in t for t in tokens):
                found.append(tok)
        return found

    def _save_preset(self):
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if not ok or not name.strip() or name.strip() == "(none)":
            return
        name = name.strip()
        self.presets[name] = {
            "params": self.params_text.toPlainText().strip(),
            "host": self.host_entry.text().strip(),
            "port": self.port_entry.text().strip(),
        }
        save_presets(self.presets)
        self._refresh_preset_menu()
        self._log(f'Preset "{name}" saved')

    def _delete_preset(self):
        name = self.preset_combo.currentText()
        if name not in self.presets:
            return
        reply = QMessageBox.question(self, "Delete Preset?", f'Delete "{name}"?')
        if reply == QMessageBox.StandardButton.Yes:
            del self.presets[name]
            save_presets(self.presets)
            self._refresh_preset_menu()
            self._log(f'Preset "{name}" deleted')

    def _refresh_preset_menu(self):
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem("(none)")
        for name in self.presets:
            self.preset_combo.addItem(name)
        self.preset_combo.setCurrentIndex(0)
        self.preset_combo.blockSignals(False)

    def _paste_params(self):
        clipboard = QApplication.clipboard()
        text = clipboard.text() if clipboard else ""
        if not text:
            return
        conflicts = []
        if "--port" in text:
            conflicts.append("--port")
        if "--host" in text:
            conflicts.append("--host")
        if "--alias" in text:
            conflicts.append("--alias")
        if conflicts:
            QMessageBox.warning(
                self, "Warning",
                f"Pasted text contains {', '.join(conflicts)}.\n"
                f"Remove them manually - host/port are set in the GUI."
            )
        self.params_text.setPlainText(text.strip())
        self._update_preview()

    # ── Llama.cpp versions ───────────────────────────────────────────

    def _current_engine(self) -> dict:
        eid = self.engine_combo.currentData() if hasattr(self, "engine_combo") else None
        return ENGINES.get(eid, ENGINES[DEFAULT_ENGINE])

    def _current_engine_id(self) -> str:
        eid = self.engine_combo.currentData() if hasattr(self, "engine_combo") else DEFAULT_ENGINE
        return eid if eid in ENGINES else DEFAULT_ENGINE

    def _refresh_versions_list(self):
        self.versions_list.clear()
        active = self.bin_entry.text().strip() if hasattr(self, 'bin_entry') else None
        eng = self._current_engine()
        self.installed_versions = get_installed_versions(eng["versions_dir"], active)
        if not self.installed_versions:
            self.versions_list.addItem("  (no installed versions)")
            return
        for v in self.installed_versions:
            marker = "\u25cf " if v["active"] else "  "
            status = "\u2713" if v["path"] else "\u2717 no exe"
            self.versions_list.addItem(f"{marker}{v['name']}  [{status}]")

    def _on_engine_changed(self):
        eng = self._current_engine()
        self._log(f"Engine: {eng['label']} — versions: {eng['versions_dir']}")
        # Load the new engine's default params — previous params are engine-specific
        # and may contain unsupported flags (e.g. beellama's kvarn5, --kv-unified).
        self.params_text.setPlainText(eng["default_params"])
        self._refresh_versions_list()
        # Persist engine selection so it survives restarts
        save_config({**self.cfg, "engine_id": self._current_engine_id()})

    def _check_updates(self):
        eng = self._current_engine()
        if eng.get("manual_build"):
            # No prebuilt binaries for this engine — show build instructions.
            self._log(f"{eng['label']}: no prebuilt releases — build from source")
            self.dl_progress_label.setText("")
            QMessageBox.information(
                self,
                "Build required",
                f"{eng['label']} has no prebuilt Windows binaries.\n\n"
                f"Build it once from source (see {eng.get('build_script', 'build-*.bat')}), "
                f"then place the compiled llama-server.exe into:\n"
                f"{eng['versions_dir']}\n\n"
                f"After that, click 'Use' to activate it."
            )
            self._refresh_versions_list()
            return
        self._log(f"Checking {eng['repo']}...")
        self.dl_progress_label.setText("loading...")
        QApplication.processEvents()

        def worker():
            try:
                req = urllib.request.Request(eng["api"], headers={"User-Agent": "LLM-Launcher"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    releases = json.loads(resp.read())

                installed_names = [v["name"] for v in self.installed_versions]
                show_preview = self.preview_cb.isChecked() if hasattr(self, "preview_cb") else False
                groups = defaultdict(dict)
                for r in releases[:10]:
                    if r.get("draft"):
                        continue
                    if r.get("prerelease") and not show_preview:
                        continue
                    tag = r["tag_name"]
                    for a in r.get("assets", []):
                        parsed = eng["classify"](a["name"])
                        if not parsed:
                            continue
                        kind, cuda = parsed
                        key = (tag, cuda)
                        if kind == "cudart":
                            groups[key]["cudart"] = {"url": a["browser_download_url"], "size": a["size"]}
                        else:
                            groups[key]["llama"] = {"url": a["browser_download_url"], "size": a["size"], "name": a["name"]}

                available = []
                for (tag, cuda), assets in groups.items():
                    if "cudart" not in assets or "llama" not in assets:
                        continue
                    total_size = assets["cudart"]["size"] + assets["llama"]["size"]
                    available.append({
                        "tag": tag,
                        "cuda": cuda,
                        "cudart_url": assets["cudart"]["url"],
                        "cudart_size": assets["cudart"]["size"],
                        "llama_url": assets["llama"]["url"],
                        "llama_size": assets["llama"]["size"],
                        "total_size": total_size,
                        "total_mb": total_size / 1024 / 1024,
                        "installed": tag in installed_names,
                    })

                self._set_progress_signal.emit("")
                if not available:
                    self._log_signal.emit("No Windows CUDA builds found")
                else:
                    self._show_updates_signal.emit(available)

            except urllib.error.URLError as e:
                self._log_signal.emit(f"No connection to GitHub: {e.reason}")
                self._set_progress_signal.emit("")
            except urllib.error.HTTPError as e:
                self._log_signal.emit(f"GitHub returned error {e.code}")
                self._set_progress_signal.emit("")
            except TimeoutError:
                self._log_signal.emit("Timeout - GitHub not responding")
                self._set_progress_signal.emit("")
            except Exception as e:
                self._log_signal.emit(f"Error: {type(e).__name__}: {e}")
                self._set_progress_signal.emit("")

        threading.Thread(target=worker, daemon=True).start()

    def _show_updates(self, available: list):
        self.dl_progress_label.setText("")
        if not available:
            self._log("No builds found")
            return

        eng = self._current_engine()
        self._updates_engine_id = self._current_engine_id()
        self._updates_dialog = QDialog(self)
        self._updates_dialog.setWindowTitle(f"Available {eng['label']} versions")
        self._updates_dialog.resize(600, 400)
        self._updates_dialog.setModal(False)

        layout = QVBoxLayout(self._updates_dialog)

        header = QLabel(f"Available builds for {eng['label']}:")
        header.setStyleSheet("color: #569cd6; font-weight: bold; font-size: 10pt;")
        layout.addWidget(header)

        # Column headers
        hdr_layout = QHBoxLayout()
        for text, w in [("  ", 2), ("Release", 10), ("CUDA", 12), ("Size", 10)]:
            lbl = QLabel(text)
            lbl.setStyleSheet("color: #888; font-weight: bold; font-size: 9pt;")
            lbl.setFixedWidth(w * 10)
            hdr_layout.addWidget(lbl)
        layout.addLayout(hdr_layout)

        list_widget = QListWidget()
        for v in available:
            marker = "\u25cf" if v["installed"] else " "
            line = f"{marker} {v['tag']:<10} {v['cuda']:<12} {v['total_mb']:.0f} MB"
            list_widget.addItem(line)
        layout.addWidget(list_widget, 1)

        info = QLabel("Downloads cudart + bins, extracts into one folder")
        info.setStyleSheet("color: #666; font-size: 8pt;")
        layout.addWidget(info)

        download_btn = QPushButton("\u2b07 Download selected")
        download_btn.setObjectName("downloadBtn")

        def on_download():
            sel = list_widget.currentRow()
            if sel < 0 or sel >= len(available):
                return
            chosen = available[sel]
            if chosen["installed"]:
                QMessageBox.information(self._updates_dialog, "Already installed", f"Version {chosen['tag']} is already installed.")
                return
            self._updates_dialog.close()
            self._download_paired(
                chosen["tag"],
                chosen["cuda"],
                chosen["cudart_url"],
                chosen["llama_url"],
                engine_id=self._updates_engine_id,
            )

        download_btn.clicked.connect(on_download)
        layout.addWidget(download_btn)

        self._updates_dialog.show()

    def _download_version(self):
        self._check_updates()

    def _download_paired(self, tag: str, cuda: str, cudart_url: str, llama_url: str | None,
                         engine_id: str | None = None):
        dest_name = f"{tag}-{cuda}" if cuda else tag
        eng = ENGINES.get(engine_id) if engine_id else self._current_engine()
        self._log(f"Downloading {dest_name} from {eng['repo']} (cudart + bins)...")
        self.dl_progress_label.setText("downloading 1/2...")
        QApplication.processEvents()

        def worker():
            try:
                req = urllib.request.Request(cudart_url, headers={"User-Agent": "LLM-Launcher"})
                with urllib.request.urlopen(req, timeout=300) as resp:
                    cudart_data = resp.read()

                llama_data = None
                if llama_url:
                    self._set_progress_signal.emit("downloading 2/2...")
                    req2 = urllib.request.Request(llama_url, headers={"User-Agent": "LLM-Launcher"})
                    with urllib.request.urlopen(req2, timeout=300) as resp2:
                        llama_data = resp2.read()

                self._set_progress_signal.emit("extracting...")

                dest = eng["versions_dir"] / dest_name
                dest.mkdir(parents=True, exist_ok=True)

                with zipfile.ZipFile(io.BytesIO(cudart_data)) as zf:
                    zf.extractall(dest)

                if llama_data:
                    with zipfile.ZipFile(io.BytesIO(llama_data)) as zf:
                        zf.extractall(dest)

                self._log_signal.emit(f"\u2713 {dest_name} installed")
                self._refresh_versions_signal.emit()
            except Exception as e:
                self._log_signal.emit(f"Download error: {e}")
            finally:
                self._set_progress_signal.emit("")

        threading.Thread(target=worker, daemon=True).start()

    def _use_version(self):
        row = self.versions_list.currentRow()
        if row < 0 or row >= len(self.installed_versions):
            return
        v = self.installed_versions[row]
        if not v["path"]:
            QMessageBox.warning(self, "Error", f"llama-server.exe not found in {v['name']}")
            return
        self.bin_entry.setText(v["path"])
        self._log(f"Active version: {v['name']}")
        self._refresh_versions_list()
        # Persist engine + binary so the choice survives restarts
        save_config({
            **self.cfg,
            "engine_id": self._current_engine_id(),
            "binary": v["path"],
        })

    def _delete_version(self):
        row = self.versions_list.currentRow()
        if row < 0 or row >= len(self.installed_versions):
            return
        v = self.installed_versions[row]
        if v["active"]:
            QMessageBox.warning(self, "Error", "Cannot delete active version.")
            return
        reply = QMessageBox.question(self, "Delete?", f"Delete {v['name']}?")
        if reply == QMessageBox.StandardButton.Yes:
            try:
                shutil.rmtree(v["dir"])
                self._log(f"\U0001f5d1 {v['name']} deleted")
                self._refresh_versions_list()
            except Exception as e:
                self._log(f"Delete error: {e}")

    def _start_server(self):
        model = self.model_entry.text().strip()
        binary = self.bin_entry.text().strip()
        params = self.params_text.toPlainText().strip()
        host = self.host_entry.text().strip() or HOST
        port = self.port_entry.text().strip() or PORT

        if not model:
            QMessageBox.warning(self, "Error", "Specify a GGUF model path.")
            return
        if not Path(model).exists():
            QMessageBox.warning(self, "Error", f"File not found:\n{model}")
            return
        if not Path(binary).exists():
            QMessageBox.warning(self, "Error", f"llama-server not found:\n{binary}")
            return

        try:
            # Build argv as a list (no shell) so nested quotes in params
            # (e.g. --chat-template-kwargs "{...}") survive verbatim.
            argv = [binary, "-m", model, "--host", host, "--port", port,
                    "--alias", "Local Model"]
            if params:
                argv += shlex.split(params, posix=True)
            self._log(f"Launching: {' '.join(argv)}")
            self.process = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                           | subprocess.CREATE_NO_WINDOW
            )
        except Exception as e:
            self._log(f"Launch error: {e}")
            return

        save_config({
            "model": model, "binary": binary, "host": host, "port": port,
            "params": params, "engine_id": self._current_engine_id()
        })
        save_history(model, self.history)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._set_status("loading")

        threading.Thread(target=self._read_output, daemon=True).start()

    def _stop_server(self):
        if self.process:
            try:
                subprocess.run(
                    f"taskkill /F /T /PID {self.process.pid}",
                    shell=True, capture_output=True, timeout=5
                )
            except Exception:
                pass
        self.process = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._set_status("offline")
        self._log("Server stopped")
        if self.log_file:
            self.log_file.close()
            self.log_file = None

    def _toggle_log_save(self, state):
        if self.log_save_cb.isChecked():
            log_dir = SCRIPT_DIR / "logs"
            log_dir.mkdir(exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            log_path = log_dir / f"llama_{ts}.log"
            # Line-buffered so output survives crashes without explicit flush
            self.log_file = open(log_path, "a", encoding="utf-8", buffering=1)
            self._log(f"Log saving: {log_path}")
            save_config({**self.cfg, "log_save": True})
        else:
            if self.log_file:
                self.log_file.close()
                self.log_file = None
            save_config({**self.cfg, "log_save": False})

    def _read_output(self):
        if not self.process or not self.process.stdout:
            return
        try:
            for line in iter(self.process.stdout.readline, b""):
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    self._log_signal.emit(text)
        except Exception:
            pass
        finally:
            self._on_exit_signal.emit()

    def _on_process_exit(self):
        self.process = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._set_status("offline")

    # ── Health monitor ──────────────────────────────────────────────

    def _start_health_monitor(self):
        self.monitoring = True
        self._health_timer = QTimer(self)
        self._health_timer.timeout.connect(self._check_health)
        self._health_timer.start(3000)
        self._check_health()

    def _check_health(self):
        if not self.monitoring:
            return
        host = self.host_entry.text().strip() or HOST
        port = self.port_entry.text().strip() or PORT

        def worker():
            try:
                with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=2) as resp:
                    if resp.status == 200:
                        self._health_signal.emit("online")
                    else:
                        self._health_signal.emit("loading")
            except Exception:
                self._health_signal.emit("loading" if self.process else "offline")

        threading.Thread(target=worker, daemon=True).start()

    # ── Helpers ─────────────────────────────────────────────────────

    def _set_status(self, state: str):
        if state == "online":
            self.status_label.setText(" \u25cf Online ")
            self.status_label.setStyleSheet("color: #3fb950; font-weight: bold; font-size: 10pt;")
        elif state == "loading":
            self.status_label.setText(" \u25cf Loading... ")
            self.status_label.setStyleSheet("color: #d29922; font-weight: bold; font-size: 10pt;")
        else:
            self.status_label.setText(" \u25cf Offline ")
            self.status_label.setStyleSheet("color: #888; font-weight: bold; font-size: 10pt;")

    def _log(self, text: str):
        self.log_view.append(text)
        # Write to file if enabled
        if self.log_file:
            try:
                self.log_file.write(text + "\n")
                self.log_file.flush()
            except Exception:
                pass
        # Limit log lines
        doc = self.log_view.document()
        if doc.blockCount() > 500:
            cursor = self.log_view.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            cursor.movePosition(cursor.MoveOperation.Down, cursor.MoveMode.KeepAnchor, doc.blockCount() - 300)
            cursor.removeSelectedText()

    def _on_port_changed(self):
        self._update_preview()
        port = self.port_entry.text().strip()
        host = self.host_entry.text().strip() or HOST
        if not port or not HERMES_CONFIG:
            return
        try:
            content = HERMES_CONFIG.read_text(encoding="utf-8")
            new_content = re.sub(
                r'(local-llama:.*?base_url:\s+)http://([^:]+):\d+',
                f'\\1http://\\2:{port}',
                content,
                flags=re.DOTALL
            )
            if new_content != content:
                HERMES_CONFIG.write_text(new_content, encoding="utf-8")
                self._log(f"Hermes: port -> {port}")
        except Exception:
            pass

    def _update_preview(self):
        model = self.model_entry.text().strip()
        binary = self.bin_entry.text().strip()
        params = self.params_text.toPlainText().strip()
        host = self.host_entry.text().strip() or HOST
        port = self.port_entry.text().strip() or PORT
        if model:
            name = Path(model).name
            cmd = f'"{binary}" -m "{name}" --host {host} --port {port} {params}'
            self.preview_label.setText(cmd)
        else:
            self.preview_label.setText("(select a model)")

    def _load_saved_state(self):
        if self.cfg:
            # Restore fields BEFORE refreshing the versions list, so the
            # active-version marker uses the saved binary path.
            if "model" in self.cfg:
                self.model_entry.setText(self.cfg["model"])
            if "binary" in self.cfg:
                self.bin_entry.setText(self.cfg["binary"])
            if "host" in self.cfg:
                self.host_entry.setText(self.cfg["host"])
            if "port" in self.cfg:
                self.port_entry.setText(self.cfg["port"])
            if "params" in self.cfg:
                self.params_text.setPlainText(self.cfg["params"])

            engine_id = self.cfg.get("engine_id") or self.cfg.get("engine")
            if engine_id:
                idx = self.engine_combo.findData(engine_id)
                if idx < 0:
                    # fallback: match by label for older configs that saved label
                    idx = self.engine_combo.findText(engine_id)
                if idx >= 0:
                    self.engine_combo.blockSignals(True)
                    self.engine_combo.setCurrentIndex(idx)
                    self.engine_combo.blockSignals(False)
                    self._refresh_versions_list()

            # Detect stale params written by a different engine and reset them
            # to this engine's defaults, so an old preset doesn't crash launch.
            saved_params = self.cfg.get("params", "")
            if engine_id in ENGINES and saved_params:
                eng = ENGINES[engine_id]
                incompat = eng.get("incompatible_params")
                if incompat and any(tok in saved_params for tok in incompat):
                    self._log(f"Params from a different engine detected — "
                              f"loading {eng['label']} defaults")
                    self.params_text.setPlainText(eng["default_params"])


def main():
    # High-DPI awareness on Windows
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app.setStyleSheet(DARK_STYLESHEET)

    launcher = LLMLauncher()
    launcher.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
