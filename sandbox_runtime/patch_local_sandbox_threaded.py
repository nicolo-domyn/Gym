#!/usr/bin/env python3
"""[CUSTOM-SANDBOX-THREADED 2026-10-23] local_sandbox_server.py's `app.run(port=6000)` has no
`threaded=True`, so Flask's dev server handles exactly one connection at a time. Under
concurrent load this drops connections causing "Server disconnected without sending a response".
Patch the file to make it run with `threaded=True`. Since the patched file is part of the venv,
and possibly refreshed at times, it needs to be patched at runtime.
"""
import sys
from pathlib import Path

MARKER = "CUSTOM-SANDBOX-THREADED 2026-10-23"

OLD_APP_RUN = "    app.run(port=6000)"

NEW_APP_RUN = f"""    # [{MARKER}] threaded=True: see this patch script's own module docstring.
    app.run(port=6000, threaded=True)"""


def patch_file(path: Path) -> bool:
    text = path.read_text()

    if MARKER in text:
        print(f"[patch_local_sandbox_threaded] {path} already patched, skipping.")
        return True

    if OLD_APP_RUN not in text:
        print(
            f"[patch_local_sandbox_threaded] ERROR: expected 'app.run(port=6000)' line not found in {path}.",
            file=sys.stderr,
        )
        return False

    text = text.replace(OLD_APP_RUN, NEW_APP_RUN, 1)
    path.write_text(text)
    print(f"[patch_local_sandbox_threaded] Patched {path}.")
    return True


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: patch_local_sandbox_threaded.py <path-to-sandbox-runtime-venv>", file=sys.stderr)
        sys.exit(1)

    venv_dir = Path(sys.argv[1])
    target = (
        venv_dir
        / "lib"
        / "python3.13"
        / "site-packages"
        / "nemo_skills"
        / "code_execution"
        / "local_sandbox"
        / "local_sandbox_server.py"
    )

    if not target.is_file():
        print(f"[patch_local_sandbox_threaded] ERROR: {target} does not exist.", file=sys.stderr)
        sys.exit(1)

    if not patch_file(target):
        sys.exit(1)


if __name__ == "__main__":
    main()
