#!/usr/bin/env python3
import sys
from pathlib import Path

MARKER = "aa0f1ed4c3b57828c8b9b2821b0ce74e201cf813"
OLD_RESOLVED_REFERENCE = 'resolved_reference = "aa0f1ed4c3b57828c8b9b2821b0ce74e201cf813"'
NEW_RESOLVED_REFERENCE = 'resolved_reference = "74690d732f3008a15c5def0da8609d5117f55e79"'


def patch_file(path: Path) -> bool:
    text = path.read_text()

    if MARKER not in text:
        print(f"[patch_openhands_swebench_pin] {path} already patched (or pin changed upstream), skipping.")
        return True

    if OLD_RESOLVED_REFERENCE not in text:
        print(
            f"[patch_openhands_swebench_pin] ERROR: stale marker {MARKER!r} found in {path} but not in the "
            "expected 'resolved_reference = ...' form -- the lockfile's format may have changed.",
            file=sys.stderr,
        )
        return False

    text = text.replace(OLD_RESOLVED_REFERENCE, NEW_RESOLVED_REFERENCE, 1)
    path.write_text(text)
    print(f"[patch_openhands_swebench_pin] Patched {path}.")
    return True


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: patch_openhands_swebench_pin.py <path-to-openhands-dir>", file=sys.stderr)
        sys.exit(1)

    openhands_dir = Path(sys.argv[1])
    target = openhands_dir / "poetry.lock"

    if not target.is_file():
        print(f"[patch_openhands_swebench_pin] ERROR: {target} does not exist.", file=sys.stderr)
        sys.exit(1)

    if not patch_file(target):
        sys.exit(1)


if __name__ == "__main__":
    main()
