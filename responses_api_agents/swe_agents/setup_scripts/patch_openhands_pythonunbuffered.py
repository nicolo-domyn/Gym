#!/usr/bin/env python3
import sys
from pathlib import Path

MARKER = "CUSTOM-PYTHONUNBUFFERED 2026-10-16"

OLD_COMMAND = """  COMMAND="poetry run python evaluation/benchmarks/swe_bench/run_infer.py \\"""

NEW_COMMAND = f"""  # [{MARKER}] PYTHONUNBUFFERED=1 so a crash during run_infer.py's own startup (before
  # the first flush) still shows a real traceback in the agent.log instead of being silently
  # discarded -- see this patch script's own module docstring for the full root-cause writeup.
  COMMAND="PYTHONUNBUFFERED=1 poetry run python evaluation/benchmarks/swe_bench/run_infer.py \\"""


def patch_file(path: Path) -> bool:
    text = path.read_text()

    if MARKER in text:
        print(f"[patch_openhands_pythonunbuffered] {path} already patched, skipping.")
        return True

    if OLD_COMMAND not in text:
        print(
            f"[patch_openhands_pythonunbuffered] ERROR: expected COMMAND= line not found in {path}.",
            file=sys.stderr,
        )
        return False

    text = text.replace(OLD_COMMAND, NEW_COMMAND, 1)
    path.write_text(text)
    print(f"[patch_openhands_pythonunbuffered] Patched {path}.")
    return True


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: patch_openhands_pythonunbuffered.py <path-to-openhands-dir>", file=sys.stderr)
        sys.exit(1)

    openhands_dir = Path(sys.argv[1])
    target = openhands_dir / "evaluation" / "benchmarks" / "swe_bench" / "scripts" / "run_infer.sh"

    if not target.is_file():
        print(f"[patch_openhands_pythonunbuffered] ERROR: {target} does not exist.", file=sys.stderr)
        sys.exit(1)

    if not patch_file(target):
        sys.exit(1)


if __name__ == "__main__":
    main()
