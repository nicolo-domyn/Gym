#!/usr/bin/env python
"""Run SWE-smith's own evaluation primitives locally, without Docker.

SWE-smith's own `swesmith.harness.eval` module (and `RepoProfile.get_container()`) launch a
NEW Docker container internally via `docker.from_env()`. We're already running inside the
one Apptainer container that holds the right pre-built `/testbed` state, so there's nothing to "launch."

Instead, this script calls the same underlying, Docker-independent primitives, directly
against the already-mounted `/testbed`:
  - `swesmith.profiles.registry.get_from_inst(instance)` -- dict lookup by `repo`.
  - `RepoProfile.get_test_cmd(instance)` -- builds the real per-repo test command
  - `RepoProfile.log_parser(log_text)` -- string parsing of raw test output.
  - `swesmith.harness.grading.get_eval_report(prediction, instance, log_path)` -- grading
     against FAIL_TO_PASS/PASS_TO_PASS, reused verbatim.

The one thing we DO reimplement (not vendor) is the container-state setup each instance needs.
SWE-smith's `patch` field in the instance dict contains the bug-injection diff (not the fix,
unlike other SWE-bench-style datasets). TO use it, `git reset --hard` then `git apply` that
`patch` onto the image branch (which is the bug-free default version).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", required=True, help="JSONL path, one instance dict per line")
    parser.add_argument("--predictions_path", required=True, help="JSON path, SWE-bench-format prediction dict")
    parser.add_argument("--instance_id", required=True)
    parser.add_argument("--testbed_dir", default="/testbed")
    parser.add_argument("--report_path", required=True)
    parser.add_argument("--timeout", type=int, default=1800, help="test command timeout (seconds)")
    args = parser.parse_args()

    from swesmith.constants import TEST_OUTPUT_END, TEST_OUTPUT_START
    from swesmith.harness.grading import get_eval_report
    from swesmith.profiles import registry

    with open(args.dataset_path) as f:
        instance = json.loads(f.readline())
    with open(args.predictions_path) as f:
        prediction = json.load(f)

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"patch_exists": False, "resolved": False}

    def write_report(d: dict) -> None:
        report_path.write_text(json.dumps({args.instance_id: d}))

    model_patch = prediction.get("model_patch")
    if not model_patch:
        write_report(report)
        return 0

    rp = registry.get_from_inst(instance)
    if rp is None:
        report["error"] = f"No SWE-smith profile registered for repo {instance.get('repo')!r}"
        write_report(report)
        return 0
    
    reset = subprocess.run(
        ["git", "reset", "--hard"],
        cwd=args.testbed_dir,
        capture_output=True,
        text=True,
    )
    if reset.returncode != 0:
        report["error"] = f"git reset --hard failed: {reset.stderr}"
        write_report(report)
        return 0

    bug_patch = instance.get("patch")
    if not bug_patch:
        report["error"] = f"instance {args.instance_id!r} has no 'patch' field"
        write_report(report)
        return 0
    bug_patch_path = Path(args.testbed_dir) / ".swesmith_bug.diff"
    bug_patch_path.write_text(bug_patch)
    bug_apply = subprocess.run(
        ["git", "apply", "--verbose", str(bug_patch_path)],
        cwd=args.testbed_dir,
        capture_output=True,
        text=True,
    )
    bug_patch_path.unlink(missing_ok=True)
    if bug_apply.returncode != 0:
        report["error"] = f"git apply (bug patch) failed: {bug_apply.stderr}"
        write_report(report)
        return 0

    # Apply the model's predicted patch. Best-effort, matching every other processor in this
    # agent (a bad/non-applying patch should surface as a failed eval, not crash this driver).
    patch_path = Path(args.testbed_dir) / ".model_patch.diff"
    patch_path.write_text(model_patch)
    apply_result = subprocess.run(
        ["git", "apply", "--verbose", str(patch_path)],
        cwd=args.testbed_dir,
        capture_output=True,
        text=True,
    )
    patch_path.unlink(missing_ok=True)
    if apply_result.returncode != 0:
        report["error"] = f"git apply failed: {apply_result.stderr}"
        write_report(report)
        return 0
    report["patch_exists"] = True

    test_cmd, _ = rp.get_test_cmd(instance)
    wrapped_test_cmd = "\n".join(
        [
            "set -uxo pipefail",
            f": '{TEST_OUTPUT_START}'",
            test_cmd,
            f": '{TEST_OUTPUT_END}'",
        ]
    )
    log_path = report_path.parent / "test_output.log"
    with open(log_path, "w") as log_f:
        try:
            subprocess.run(
                wrapped_test_cmd,
                shell=True,
                executable="/bin/bash",
                cwd=args.testbed_dir,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
            )
        except subprocess.TimeoutExpired:
            # Fall through -- get_eval_report grades whatever log exists, same as upstream's
            # own timeout handling (see swesmith/harness/eval.py's run_evaluation()).
            pass

    full_report = get_eval_report(prediction, instance, str(log_path))
    write_report(full_report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
