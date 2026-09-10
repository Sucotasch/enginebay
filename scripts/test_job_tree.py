"""Live test of setup_job_tree(): hard-kill the parent, child must die.

Spawns this very python as a parent that (a) joins a kill-on-close job,
(b) starts a persistent child, (c) kills ITSELF with TerminateProcess —
no atexit, no closeEvent, nothing graceful. The child surviving would mean
the job mechanism failed; the OS must have reaped it.
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import engine_diag as diag  # noqa: E402

if len(sys.argv) == 2 and sys.argv[1] == "child":
    # The child: just stay alive so the parent can be killed beneath it.
    print("child-alive", flush=True)
    time.sleep(300)
    sys.exit(0)

# ── parent ──────────────────────────────────────────────────────────
ok = diag.setup_job_tree()
print(f"job tree active: {ok}")
assert ok, "setup_job_tree() failed"

child = subprocess.Popen(
    [sys.executable, __file__, "child"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW,
)
time.sleep(1.5)

# Confirm the child actually started.
poll = child.poll()
print(f"child running: {poll is None} (pid {child.pid})")
assert poll is None, f"child already exited with {poll}"

# Hard self-kill: TerminateProcess(this), the most brutal death available.
# Nothing in Python gets to run afterwards — this is the orphan scenario.
import ctypes  # noqa: E402
print("parent: hard self-kill (TerminateProcess)", flush=True)
ctypes.windll.kernel32.TerminateProcess(ctypes.windll.kernel32.GetCurrentProcess(), 1)
