#!/usr/bin/env python3
import sys
from pathlib import Path

MARKER = "CUSTOM-SWEREBENCH-V1-SKIP-ENTRY 2026-10-16"

OLD_SKIP_ENTRY = """    # These dataset types operate directly in their repo dir, skipping the copy-to-workspace step.
    # [CUSTOM-SWESMITH 2026-10-09] added 'SWE-smith' -- its own git-checkout-based setup above
    # already leaves /testbed in the right state; no entry script exists for this dataset.
    SKIP_ENTRY_SCRIPT_TYPES = ('nv-internal-1', 'SWE-rebench-V2', 'SWE-Gym', 'R2E-Gym', 'swe-bench-ext', 'SWE-smith')"""

NEW_SKIP_ENTRY = """    # These dataset types operate directly in their repo dir, skipping the copy-to-workspace step.
    # [CUSTOM-SWESMITH 2026-10-09] added 'SWE-smith' -- its own git-checkout-based setup above
    # already leaves /testbed in the right state; no entry script exists for this dataset.
    # [CUSTOM-SWEREBENCH-V1-SKIP-ENTRY 2026-10-16] added plain 'SWE-rebench' (V1) -- its
    # workspace (/workspace/<repo>__<version>) is already fully pre-populated by app.py's own
    # bind mount (CUSTOM-SWEREBENCH-V1-WORKSPACE); running the default entry script here is not
    # just redundant, it's destructive -- `rm -rf /workspace/*` empties that bind mount's
    # contents, then `ln -s /testbed /workspace/$WORKSPACE_NAME` lands INSIDE the now-empty
    # (still-mounted, so un-rmdir-able) directory instead of replacing it, leaving the agent
    # operating in an empty, non-git directory. Confirmed root cause of a real 1333/1333 rollout
    # failure (job 60677532), all "fatal: not a git repository".
    SKIP_ENTRY_SCRIPT_TYPES = ('nv-internal-1', 'SWE-rebench', 'SWE-rebench-V2', 'SWE-Gym', 'R2E-Gym', 'swe-bench-ext', 'SWE-smith')"""


def patch_file(path: Path) -> bool:
    text = path.read_text()

    if MARKER in text:
        print(f"[patch_openhands_swerebench_v1_skip_entry] {path} already patched, skipping.")
        return True

    if OLD_SKIP_ENTRY not in text:
        print(
            f"[patch_openhands_swerebench_v1_skip_entry] ERROR: expected SKIP_ENTRY_SCRIPT_TYPES "
            f"block not found in {path} (did patch_openhands_swesmith_dataset_type.py run first?).",
            file=sys.stderr,
        )
        return False

    text = text.replace(OLD_SKIP_ENTRY, NEW_SKIP_ENTRY, 1)
    path.write_text(text)
    print(f"[patch_openhands_swerebench_v1_skip_entry] Patched {path}.")
    return True


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: patch_openhands_swerebench_v1_skip_entry.py <path-to-openhands-dir>", file=sys.stderr)
        sys.exit(1)

    openhands_dir = Path(sys.argv[1])
    target = openhands_dir / "evaluation" / "benchmarks" / "swe_bench" / "run_infer.py"

    if not target.is_file():
        print(f"[patch_openhands_swerebench_v1_skip_entry] ERROR: {target} does not exist.", file=sys.stderr)
        sys.exit(1)

    if not patch_file(target):
        sys.exit(1)


if __name__ == "__main__":
    main()
