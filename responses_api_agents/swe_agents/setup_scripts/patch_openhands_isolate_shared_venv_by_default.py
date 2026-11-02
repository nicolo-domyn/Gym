#!/usr/bin/env python3
import sys
from pathlib import Path

MARKER = "CUSTOM-ISOLATE-BY-DEFAULT 2026-10-16"

OLD_PYTHON_BRANCH = """    elif instance.get("repo_language", instance.get("language", "")).lower() == "python":
        # nv-internal-1 / SWE-rebench-V2: use base python in the container
        action = CmdRunAction(
            command=(
                "deactivate >/dev/null 2>&1 || true; unset VIRTUAL_ENV; "
                "export PATH=/usr/bin:/bin:/usr/local/bin:$PATH; which python"
            )
        )
        action.set_hard_timeout(600)
        logger.info(action, extra={"msg_type": "ACTION"})
        obs = runtime.run_action(action)
        logger.info(obs, extra={"msg_type": "OBSERVATION"})
        assert_and_raise(
            obs.exit_code == 0,
            f"Failed to deactivate and which python: {str(obs)}",
        )"""

NEW_DEFAULT_BRANCH = """    else:
        # [CUSTOM-ISOLATE-BY-DEFAULT 2026-10-16] Default for every dataset that reaches this
        # point without a dedicated branch above (e.g. plain 'SWE-rebench'/V1, 'SWE-smith', and
        # any future SKIP_ENTRY_SCRIPT_TYPES addition), not just ones with repo_language/language
        # == "python". Without this, the shared/persistent OpenHands orchestrator venv's own bin
        # dir stays first on PATH for the whole rollout (see local_runtime.py's PATH-prepend) --
        # confirmed root cause of a real cross-job venv corruption incident (wrapt downgraded,
        # unrelated OpenStack packages appearing) on SWE-rebench V1, whose upstream schema has no
        # language field at all so this branch never used to fire for it. Safe for genuinely
        # non-python instances too: this only ever displaces OpenHands's own python/pip, it
        # doesn't remove any other toolchain already on PATH.
        action = CmdRunAction(
            command=(
                "deactivate >/dev/null 2>&1 || true; unset VIRTUAL_ENV; "
                "export PATH=/usr/bin:/bin:/usr/local/bin:$PATH; which python"
            )
        )
        action.set_hard_timeout(600)
        logger.info(action, extra={"msg_type": "ACTION"})
        obs = runtime.run_action(action)
        logger.info(obs, extra={"msg_type": "OBSERVATION"})
        assert_and_raise(
            obs.exit_code == 0,
            f"Failed to deactivate and which python: {str(obs)}",
        )"""


def patch_file(path: Path) -> bool:
    text = path.read_text()

    if MARKER in text:
        print(f"[patch_openhands_isolate_shared_venv_by_default] {path} already patched, skipping.")
        return True

    if OLD_PYTHON_BRANCH not in text:
        print(
            f"[patch_openhands_isolate_shared_venv_by_default] ERROR: expected python-isolation "
            f"elif block not found in {path}.",
            file=sys.stderr,
        )
        return False

    text = text.replace(OLD_PYTHON_BRANCH, NEW_DEFAULT_BRANCH, 1)
    path.write_text(text)
    print(f"[patch_openhands_isolate_shared_venv_by_default] Patched {path}.")
    return True


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: patch_openhands_isolate_shared_venv_by_default.py <path-to-openhands-dir>", file=sys.stderr)
        sys.exit(1)

    openhands_dir = Path(sys.argv[1])
    target = openhands_dir / "evaluation" / "benchmarks" / "swe_bench" / "run_infer.py"

    if not target.is_file():
        print(f"[patch_openhands_isolate_shared_venv_by_default] ERROR: {target} does not exist.", file=sys.stderr)
        sys.exit(1)

    if not patch_file(target):
        sys.exit(1)


if __name__ == "__main__":
    main()
