#!/usr/bin/env python3
import sys
from pathlib import Path

MARKER = "CUSTOM-SWESMITH 2026-10-09"

# --- 1. set_dataset_type(): add the 'SWE-smith' branch ---

OLD_DATASET_TYPE_BRANCH = """    elif 'r2e-gym' in name_lower:
        DATASET_TYPE = 'R2E-Gym'
    elif 'swe-bench-live' in name_lower:"""

NEW_DATASET_TYPE_BRANCH = """    elif 'r2e-gym' in name_lower:
        DATASET_TYPE = 'R2E-Gym'
    # [CUSTOM-SWESMITH 2026-10-09] SWE-smith previously had no branch here, so it silently
    # fell through to the generic 'SWE-bench' default below -- that default's
    # get_instance_docker_image() does instance_id.split('__') expecting exactly one '__',
    # but SWE-smith instance_ids (e.g. "pytest-dev__iniconfig.16793ead.combine_file__06k2m7dd")
    # contain two, crashing every single SWE-smith row before the agent loop ever starts
    # (confirmed 6/6 in a real smoke run, job 59806056). See the other [CUSTOM-SWESMITH] tags
    # in this file for the rest of this dataset's wiring.
    elif 'swe-smith' in name_lower:
        DATASET_TYPE = 'SWE-smith'
    elif 'swe-bench-live' in name_lower:"""

# --- 2. get_config(): route SWE-smith through the image_name-field lookup ---

OLD_GET_CONFIG = """    # We use a different instance image for the each instance of swe-bench eval
    if DATASET_TYPE in ('SWE-rebench', 'SWE-rebench-V2'):
        # SWE-rebench instances have their own image_name field
        base_container_image = instance.get('image_name', '')"""

NEW_GET_CONFIG = """    # We use a different instance image for the each instance of swe-bench eval
    # [CUSTOM-SWESMITH 2026-10-09] SWE-smith ships ONE SHARED IMAGE PER REPO (confirmed:
    # 59,136 instances, exactly 222 distinct image_name values) -- same image_name-field
    # lookup as SWE-rebench/SWE-rebench-V2, not the per-instance_id docker-image-name
    # construction below (which would crash on SWE-smith's instance_id format anyway).
    if DATASET_TYPE in ('SWE-rebench', 'SWE-rebench-V2', 'SWE-smith'):
        # SWE-rebench/SWE-smith instances have their own image_name field
        base_container_image = instance.get('image_name', '')"""

# --- 3. SKIP_ENTRY_SCRIPT_TYPES: add 'SWE-smith' ---

OLD_SKIP_ENTRY = """    # These dataset types operate directly in their repo dir, skipping the copy-to-workspace step.
    SKIP_ENTRY_SCRIPT_TYPES = ('nv-internal-1', 'SWE-rebench-V2', 'SWE-Gym', 'R2E-Gym', 'swe-bench-ext')"""

NEW_SKIP_ENTRY = """    # These dataset types operate directly in their repo dir, skipping the copy-to-workspace step.
    # [CUSTOM-SWESMITH 2026-10-09] added 'SWE-smith' -- its own git-checkout-based setup above
    # already leaves /testbed in the right state; no entry script exists for this dataset.
    SKIP_ENTRY_SCRIPT_TYPES = ('nv-internal-1', 'SWE-rebench-V2', 'SWE-Gym', 'R2E-Gym', 'swe-bench-ext', 'SWE-smith')"""

# --- 4. initialize_runtime(): dedicated checkout-by-instance_id branch instead of base_commit ---

OLD_BASE_COMMIT = """    if DATASET_TYPE != "swe-bench-ext":
        # Deep-clean the git state so the agent cannot inspect commits past
        # the task's base_commit via local refs, tags, reflog, or dangling
        # objects. Remote access is blocked by removing remotes (below) and
        # by runtime command blacklist rules that forbid --all/--branches/
        # --remotes/--tags, explicit SHAs, origin/*, etc.
        base_commit = instance['base_commit']"""

NEW_BASE_COMMIT = """    if DATASET_TYPE == "SWE-smith":
        # [CUSTOM-SWESMITH 2026-10-14] SWE-smith has no base_commit field at all (by design --
        # see swesmith/harness/grading.py, which never reads one). Originally this branch did
        # `git checkout -f <instance_id>` onto a remote tracking branch named after the instance
        # -- but Docker Hub's published per-repo images only bake in whatever branches existed
        # at build time, which is NOT the full per-instance branch set (confirmed: some repos'
        # images had 0 of the needed branches). That required a separate network-dependent
        # branch-refresh step for every repo.
        #
        # Confirmed empirically (2026-10-14, pytest-dev/iniconfig instance) that this is
        # unnecessary: SWE-smith's own `patch` field IS the bug-injection diff itself (not a
        # fix -- unlike SWE-bench-style datasets), and `git apply`-ing it directly onto the
        # image's already-present, bug-free default branch reproduces the exact same tree the
        # instance branch would have given, with zero network access. This is also exactly what
        # this project's own eval-phase driver (swesmith_eval_driver.py) does with the MODEL's
        # predicted patch -- same mechanism, just the dataset's bug patch instead.
        action = CmdRunAction(command='git reset --hard')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(obs.exit_code == 0, f'Failed to git reset --hard: {str(obs)}')

        bug_patch = instance['patch']
        with tempfile.TemporaryDirectory() as temp_dir:
            patch_file_path = os.path.join(temp_dir, 'swesmith_bug.patch')
            with open(patch_file_path, 'w') as f:
                f.write(bug_patch)
            runtime.copy_to(patch_file_path, '/tmp/')

        action = CmdRunAction(command='git apply --verbose /tmp/swesmith_bug.patch')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(obs.exit_code == 0, f'Failed to apply SWE-smith bug patch: {str(obs)}')

        # [CUSTOM-SWESMITH-COMMIT-BUG-PATCH 2026-10-20] `git apply` only ever modifies the
        # working tree, never creates a commit -- confirmed root cause of a real zero-reward
        # incident via a fast local repro (no full training job needed): `git tag -f
        # swebench_baseline` right after `git apply` (no commit in between) tags the SAME
        # pre-patch commit, since there's nothing new to tag yet. `git rev-parse
        # swebench_baseline^{tree}` then resolves to the COMMITTED (bug-free) tree, not the
        # working tree's actual (bugged) file content. complete_runtime() below does `git diff
        # swebench_baseline` to extract the model's final patch -- with the tag on the wrong
        # (pre-bug) commit, that diff silently bundles the uncommitted bug-injection change
        # together with the agent's own edits. If the agent's fix exactly reverses the bug,
        # that hunk nets to ZERO change in the extracted patch (the two cancel out), which is
        # exactly what real eval reports showed: "model's patch treats the actually-bugged line
        # as unchanged context, edits something else instead" -- not because the model never
        # saw the bug, but because its correct fix was invisibly erased by comparing against
        # the wrong baseline. swe-bench-ext's own analogous fallback branch below already avoids
        # this (`git add -A && git commit ... && git tag -f swebench_baseline HEAD`) -- this
        # branch was simply missing the commit step. Verified via local repro (pallets/click):
        # without this commit, `swebench_baseline^{tree}` == pristine tree even after a clean
        # `git apply`; with it, the tree hash correctly changes and the working tree is clean
        # afterward (so complete_runtime()'s later `git diff swebench_baseline` only ever shows
        # the agent's own real edits).
        action = CmdRunAction(command="git add -A && git commit -q -m 'swesmith bug injection'")
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(obs.exit_code == 0, f'Failed to commit SWE-smith bug patch: {str(obs)}')

        # Tag the agent's actual starting point so complete_runtime() (below) diffs against
        # THIS commit, not instance['base_commit'] (which doesn't exist for SWE-smith) and not
        # the default branch (which would wrongly exclude the synthetic bug from the diff).
        # Reuses the EXACT tag name complete_runtime() already looks for first -- originally
        # added for swe-bench-ext's flat-source case -- so no change is needed there at all;
        # -f guards against a stale tag if this ever runs against a non-fresh copy.
        action = CmdRunAction(command='git tag -f swebench_baseline')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(obs.exit_code == 0, f'Failed to tag swebench_baseline: {str(obs)}')
    elif DATASET_TYPE != "swe-bench-ext":
        # Deep-clean the git state so the agent cannot inspect commits past
        # the task's base_commit via local refs, tags, reflog, or dangling
        # objects. Remote access is blocked by removing remotes (below) and
        # by runtime command blacklist rules that forbid --all/--branches/
        # --remotes/--tags, explicit SHAs, origin/*, etc.
        base_commit = instance['base_commit']"""

# --- 5. initialize_runtime(): exempt SWE-smith from the "testbed" conda-env assertion ---

OLD_TESTBED_ASSERT = """    if DATASET_TYPE not in ('Multimodal', 'SWE-bench-Live', 'nv-internal-1', 'SWE-rebench', 'SWE-rebench-V2', 'SWE-bench_Multilingual', 'swe-bench-ext'):
        # Only for non-multimodal datasets, we need to activate the testbed environment for Python
        # SWE-Bench multimodal datasets, SWE-bench-Live, nv-internal-1, and SWE-rebench are not using the testbed environment"""

NEW_TESTBED_ASSERT = """    # [CUSTOM-SWESMITH 2026-10-09] SWE-smith's images have no conda env named "testbed" at all
    # (confirmed directly against a real pulled image -- no /opt/miniconda3 present) and span
    # ~10 languages, so there's no single "testbed" convention to assert on the way SWE-Gym/
    # R2E-Gym have. Exempting here is enough to unblock Python-language SWE-smith repos (which
    # just use the image's own system interpreter); a real per-language activation story is a
    # separate, not-yet-needed follow-up for non-Python SWE-smith repos.
    if DATASET_TYPE not in ('Multimodal', 'SWE-bench-Live', 'nv-internal-1', 'SWE-rebench', 'SWE-rebench-V2', 'SWE-bench_Multilingual', 'swe-bench-ext', 'SWE-smith'):
        # Only for non-multimodal datasets, we need to activate the testbed environment for Python
        # SWE-Bench multimodal datasets, SWE-bench-Live, nv-internal-1, and SWE-rebench are not using the testbed environment"""

# --- 6. _get_workspace_path(): add 'SWE-smith' -> "/testbed" ---

OLD_WORKSPACE_PATH = """    if DATASET_TYPE in ("SWE-Gym", "R2E-Gym"):"""

NEW_WORKSPACE_PATH = """    # [CUSTOM-SWESMITH 2026-10-09] SWE-smith images ship /testbed checked out at the repo's
    # plain default (bug-free) branch -- same path as SWE-Gym/R2E-Gym, just reached via a
    # `git checkout -f <instance_id>` in initialize_runtime instead of a base_commit reset.
    if DATASET_TYPE in ("SWE-Gym", "R2E-Gym", "SWE-smith"):"""

REPLACEMENTS = [
    ("set_dataset_type branch", OLD_DATASET_TYPE_BRANCH, NEW_DATASET_TYPE_BRANCH),
    ("get_config image_name lookup", OLD_GET_CONFIG, NEW_GET_CONFIG),
    ("SKIP_ENTRY_SCRIPT_TYPES", OLD_SKIP_ENTRY, NEW_SKIP_ENTRY),
    ("initialize_runtime base_commit/checkout", OLD_BASE_COMMIT, NEW_BASE_COMMIT),
    ("initialize_runtime testbed assertion", OLD_TESTBED_ASSERT, NEW_TESTBED_ASSERT),
    ("_get_workspace_path", OLD_WORKSPACE_PATH, NEW_WORKSPACE_PATH),
]


def patch_file(path: Path) -> bool:
    text = path.read_text()

    if MARKER in text:
        print(f"[patch_openhands_swesmith_dataset_type] {path} already patched, skipping.")
        return True

    for name, old, new in REPLACEMENTS:
        if old not in text:
            print(f"[patch_openhands_swesmith_dataset_type] ERROR: '{name}' block not found in {path}.", file=sys.stderr)
            return False

    for name, old, new in REPLACEMENTS:
        text = text.replace(old, new, 1)

    path.write_text(text)
    print(f"[patch_openhands_swesmith_dataset_type] Patched {path}.")
    return True


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: patch_openhands_swesmith_dataset_type.py <path-to-openhands-dir>", file=sys.stderr)
        sys.exit(1)

    openhands_dir = Path(sys.argv[1])
    target = openhands_dir / "evaluation" / "benchmarks" / "swe_bench" / "run_infer.py"

    if not target.is_file():
        print(f"[patch_openhands_swesmith_dataset_type] ERROR: {target} does not exist.", file=sys.stderr)
        sys.exit(1)

    if not patch_file(target):
        sys.exit(1)


if __name__ == "__main__":
    main()
