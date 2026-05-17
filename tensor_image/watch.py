from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from watchfiles import watch


def _run_script(script: Path, args: list[str]) -> int:
    print(f"\n--- running {script} ---\n", flush=True)

    result = subprocess.run(
        [sys.executable, str(script), *args],
        cwd=script.parent,
    )

    print(f"\n--- exited with code {result.returncode} ---\n", flush=True)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tensor-image-watch",
        description="Rerun a Python script whenever watched Python files change.",
    )
    parser.add_argument("script", help="Python script to run")
    parser.add_argument(
        "--watch",
        default=None,
        help="Directory or file to watch. Defaults to the script file itself.",
    )
    parser.add_argument(
        "--no-initial-run",
        action="store_true",
        help="Do not run the script immediately on startup.",
    )
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the script. Use -- before script args.",
    )

    args = parser.parse_args()

    script = Path(args.script).resolve()
    if not script.exists():
        raise FileNotFoundError(script)

    watch_path = Path(args.watch).resolve() if args.watch else script

    script_args = args.script_args
    if script_args and script_args[0] == "--":
        script_args = script_args[1:]

    if not args.no_initial_run:
        _run_script(script, script_args)

    print(f"Watching: {watch_path}")
    print("Press Ctrl+C to stop.\n")

    for changes in watch(watch_path):
        py_changes = [
            path
            for _, path in changes
            if str(path).endswith(".py")
        ]

        if not py_changes:
            continue

        for path in py_changes:
            print(f"changed: {path}")

        _run_script(script, script_args)


if __name__ == "__main__":
    main()
