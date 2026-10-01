"""Windows launcher for the benchmark harness (does not modify any harness file).

Usage (from inside tanker/):
    python win_run.py self_check --adapter adapters.mine:MineSolver
    python win_run.py run --adapter adapters.mine:MineSolver --out report.json

Why a launcher and not runpy: on Windows, multiprocessing "spawn" starts a fresh
interpreter that re-imports the *main* script. The patches below sit at module
level, and this file is the real main script, so the child process gets them too.
"""
import os
import signal
import sys

# POSIX-only calls the harness makes; harmless stand-ins on Windows.
if os.name == "nt":
    if not hasattr(os, "setsid"):
        os.setsid = lambda: None
    if not hasattr(os, "getuid"):
        os.getuid = lambda: 1000
    if not hasattr(signal, "SIGKILL"):
        signal.SIGKILL = signal.SIGTERM
    if not hasattr(os, "killpg"):
        def _killpg(pid, sig):
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        os.killpg = _killpg


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("self_check", "run"):
        raise SystemExit("usage: python win_run.py {self_check|run} --adapter module:Class [--out file]")
    which = sys.argv.pop(1)
    import _paths  # noqa: F401
    import metrics
    from benchkit.cli import run_main, self_check_main
    raise SystemExit((self_check_main if which == "self_check" else run_main)(metrics.BENCH))


if __name__ == "__main__":
    main()