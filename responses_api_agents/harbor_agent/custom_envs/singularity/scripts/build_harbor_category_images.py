"""
[CUSTOM] Patch the existing Harbor category Singularity images with our own exec-server's
missing uvicorn/fastapi dependency, without rebuilding anything from scratch.

Rewritten 2026-10-06, twice. First rewrite tried to fully replay each category's Dockerfile
That does NOT work for Harbor: `apt-get`/`dpkg` hard-refuse non-root regardless of file
ownership -- unlike `chmod` or `pip install`, which succeed via ordinary file-ownership
writes.

This second rewrite sidesteps the problem entirely instead of solving it: the existing 6
category .sif images ALREADY have every category-specific apt-get/pip dependency
correctly baked in -- the ONLY thing missing from them is uvicorn/fastapi (our own
exec-server's dependency, not task/category content; see the "Real bug" section below).
uvicorn/fastapi are pure-Python packages, installable via `pip3 install`, which does NOT
need root.
So instead of rebuilding a category image from its Dockerfile, this script PATCHES the
already-correct existing image:

  1. `singularity build --force --sandbox <dir> <existing.sif>` -- convert the already-correct
     image to an editable sandbox. Unprivileged, standard Apptainer format conversion (same
     operation as a `docker://` pull into a sandbox, just from a local .sif source instead).
  2. `singularity exec --no-mount home,tmp,bind-paths --writable <dir> pip3 install
     --break-system-packages uvicorn fastapi` -- pure Python package install, no apt-get/dpkg
     involved, so no root needed. `--no-mount home,tmp,bind-paths`: without this, a plain
     `singularity exec` applies this cluster's admin-configured default bind-mounts (e.g.
     /leonardo*), which this minimal image has no mountpoint directory for (confirmed by a
     real failure, 2026-10-06: "destination /leonardo doesn't exist in container") -- matches
     the same flag singularity.py's own real `_start_server()` invocation already uses.
  3. `chmod -R a+rwX <dir>/app` (host-side, defensive) -- the existing image's workdir should
     already be world-writable from its original build (that fix is part of what's already
     baked in), but packing is confirmed to reset file OWNERSHIP even though it preserves mode
     bits (see the sibling SWE-Gym script's own header for the full "REAL GOTCHA" writeup) --
     redoing this costs nothing and removes any doubt.
  4. `singularity build --force <out>.sif <dir>` -- pack back into a new .sif.

Deliberately does NOT touch $TASK_ROOT or any category Dockerfile at all -- so the
separately-documented KNOWN GOTCHA (4 of 6 categories' medium_20000 tier shipping a Dockerfile
mismatched with the rest of that category, see scripts/prepare-env/build_harbor_task_images.sh)
is irrelevant to this fix. That gotcha only matters for a full rebuild-FROM-Dockerfile, which
this script no longer does.

Real bug this fixes: every trial failed with `_start_server()` exhausting all 3
port-retry attempts. When `_start_server()` exhausts its retries and raises, Harbor's own
Trial still writes a result.json (no trajectory.json, since the agent loop never started),
harbor_agent/app.py's `HarborAgent.run()` converts that into a "successful" (HTTP 200)
response with reward=0.0 and `output=[]` (see `HarborAgentUtils.trial_result_to_responses`'s
own docstring: "Returns an empty list when no trajectory is available").

Must run from a network-enabled context -- step 2's `pip3 install` needs it (PyPI, not a local
mirror); boost_usr_prod compute nodes have no internet. Use lrd_all_serial (see this
directory's sibling sbatch entry point) rather than the login node for the real patch run.

Usage:
    python build_harbor_category_images.py \\
      --existing-sif-dir /leonardo_scratch/fast/BOOST_EUROPA/images/harbor \\
      --out-dir /leonardo_scratch/fast/BOOST_EUROPA/$USER/singularity_images
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


# Our own in-container exec server's (server.py, via singularity.py's bootstrap.sh) runtime
# deps -- not task content, not test-harness content. None of the category base images ship
# these; see the module docstring's "Real bug" section for the bug this addresses.
SERVER_PACKAGES = ["uvicorn", "fastapi"]

DEFAULT_CATEGORIES = [
    "file_operations",
    "data_science",
    "data_processing",
    "scientific_computing",
    "security",
    "debugging",
]

# Login-node (and some compute-node) /tmp is too small for image extraction -- confirmed
# empirically on the sibling SWE-Gym rebuild ("no space left on device" mid-extract against
# plain /tmp). Default scratch location for sandbox extraction/packing; override with
# --build-tmp.
DEFAULT_BUILD_TMP = Path(f"/leonardo_scratch/fast/BOOST_EUROPA/{os.environ.get('USER', '')}/singularity_tmp")


def _run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def patch_category_image(
    category: str,
    existing_sif: Path,
    out_sif: Path,
    build_tmp: Path,
    workdir: str,
) -> None:
    """Run the 4-step local patch pipeline for one category -- see module docstring."""
    server_pip = " ".join(SERVER_PACKAGES)

    build_tmp.mkdir(parents=True, exist_ok=True)
    sandbox_dir = Path(tempfile.mkdtemp(prefix=f"sandbox.{category}.", dir=str(build_tmp)))
    try:
        print(f"[{category}] step 1/4: converting {existing_sif} -> sandbox")
        # --force: tempfile.mkdtemp above already CREATED sandbox_dir (empty) to get a safe
        # unique path; `singularity build --sandbox` treats a pre-existing target dir as a
        # conflict and refuses without --force ("build target '...' already exists") -- not a
        # TOCTOU risk, the dir is always freshly made moments earlier and never pre-populated.
        _run(["singularity", "build", "--force", "--sandbox", str(sandbox_dir), str(existing_sif)])

        print(f"[{category}] step 2/4: pip-installing {server_pip} (no apt-get needed)")
        _run(
            [
                "singularity",
                "exec",
                "--no-mount",
                "home,tmp,bind-paths",
                "--writable",
                str(sandbox_dir),
                "bash",
                "-c",
                f"pip3 install --no-cache-dir --break-system-packages {server_pip}",
            ]
        )

        # Step 3/4 -- host-side, defensive. See module docstring on why packing resets
        # ownership but preserves mode bits, so this is cheap insurance, not strictly new work.
        print(f"[{category}] step 3/4: chmod -R a+rwX {workdir} (host-side)")
        workdir_in_sandbox = sandbox_dir / workdir.lstrip("/")
        subprocess.run(["chmod", "-R", "a+rwX", str(workdir_in_sandbox)], check=False)

        tmp_sif = out_sif.with_suffix(out_sif.suffix + ".partial")
        print(f"[{category}] step 4/4: packing sandbox -> {out_sif}")
        _run(["singularity", "build", "--force", str(tmp_sif), str(sandbox_dir)])
        tmp_sif.rename(out_sif)
        print(f"[{category}] patched {out_sif}")
    except subprocess.CalledProcessError as e:
        print(f"  ✗ FAILED for {category}: {e}", file=sys.stderr)
        raise
    finally:
        shutil.rmtree(sandbox_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--existing-sif-dir",
        type=Path,
        required=True,
        help="Dir containing the existing (uvicorn/fastapi-missing) harbor_<category>.sif images to patch",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Where to write the patched .sif images")
    parser.add_argument(
        "--categories",
        nargs="*",
        default=DEFAULT_CATEGORIES,
        help="Categories to patch (default: the 6 currently built, see scripts/pull_harbor_tasks.sh)",
    )
    parser.add_argument(
        "--workdir",
        default="/app",
        help="Container workdir to chmod (default: %(default)s -- matches all 6 current categories' Dockerfiles)",
    )
    parser.add_argument(
        "--build-tmp",
        type=Path,
        default=DEFAULT_BUILD_TMP,
        help="Scratch dir for sandbox extraction/packing (default: %(default)s -- login-node "
        "/tmp is too small, confirmed empirically).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned steps per category without running them.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        args.build_tmp.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("SINGULARITY_TMPDIR", str(args.build_tmp))
        os.environ.setdefault("APPTAINER_TMPDIR", str(args.build_tmp))

    for category in args.categories:
        print(f"=== {category} ===")
        existing_sif = args.existing_sif_dir / f"harbor_{category}.sif"
        out_sif = args.out_dir / f"harbor_{category}.sif"

        if not existing_sif.exists():
            print(f"✗ {category}: no existing image at {existing_sif}, can't patch — skipping", file=sys.stderr)
            continue

        if args.dry_run:
            print(f"Would convert {existing_sif} -> sandbox")
            print(f"Would pip3 install --no-cache-dir --break-system-packages {' '.join(SERVER_PACKAGES)}")
            print(f"Would chmod -R a+rwX <sandbox>{args.workdir}")
            print(f"Would pack sandbox -> {out_sif}")
            continue

        if out_sif.exists():
            print(f"✅ {category} already patched — skipping")
            continue

        patch_category_image(category, existing_sif, out_sif, args.build_tmp, args.workdir)


if __name__ == "__main__":
    main()
