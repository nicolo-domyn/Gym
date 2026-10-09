"""
[CUSTOM] Point every task.toml's [environment].docker_image at the shared, local, per-category
.sif path -- the simpler sibling of rewrite_task_tomls.py for this project's registry-free,
one-image-per-category approach (see build_harbor_category_images.py's module docstring for
why one image per category, not per task).

Usage:
    python rewrite_task_tomls_for_category_images.py \\
      --task-root /leonardo_scratch/fast/BOOST_EUROPA/$USER/harbor_tasks \\
      --sif-dir /leonardo_scratch/fast/BOOST_EUROPA/images/harbor   # the shared, pre-built default (HARBOR_SIF_DIR in .env)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rewrite_task_tomls import update_task_toml_docker_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task-root", type=Path, required=True, help="Root dir holding <category>/<category>_task_NNNN/")
    parser.add_argument("--sif-dir", type=Path, required=True, help="Dir holding harbor_<category>.sif images")
    parser.add_argument(
        "--categories",
        nargs="*",
        default=["file_operations", "data_science", "data_processing", "scientific_computing", "security", "debugging"],
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total = 0
    missing_sif = []
    for category in args.categories:
        sif_path = args.sif_dir / f"harbor_{category}.sif"
        if not sif_path.is_file():
            missing_sif.append(str(sif_path))
            continue

        category_dir = args.task_root / category
        task_tomls = sorted(category_dir.glob("*/task.toml"))
        for task_toml in task_tomls:
            if args.dry_run:
                print(f"PLAN {task_toml} -> {sif_path}")
            else:
                update_task_toml_docker_image(task_toml, str(sif_path))
            total += 1
        print(f"{category}: {len(task_tomls)} task.toml files -> {sif_path}")

    if missing_sif:
        print("Skipped categories with no built .sif yet (run build_harbor_category_images.py first):")
        for m in missing_sif:
            print(f"  {m}")

    print(f"Done. {total} task.toml files {'would be ' if args.dry_run else ''}updated.")
    if missing_sif:
        sys.exit(1)


if __name__ == "__main__":
    main()
