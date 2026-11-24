#!/usr/bin/env python3
"""[CUSTOM-ATOMIC-NG-METRICS 2026-11-21] Serialize and make atomic every update OpenHands makes
to NeMo Gym's per-rollout metrics file ($NEMO_GYM_METRICS_FPATH, nemo_gym_metrics.json).

Three places in the vendored OpenHands did an unlocked read-modify-write on that file with a
plain open("w"):
  - openhands/runtime/base.py  Runtime.on_event                   (total_command_exec_time)
  - openhands/agenthub/nemo_gym_client.py  _update_model_call_time (total_model_call_time)
  - evaluation/benchmarks/swe_bench/run_infer.py  update_metrics   (runtime setup timings)
The first two run concurrently (runtime event callbacks vs. the agent loop). Two interleaved
writes both truncate the file at open() and then write from offset 0, so the shorter one
leaves the tail of the longer one behind: a valid JSON object followed by garbage. Job
64971699 (fidelity-val-smoke-mm-070): `JSONDecodeError: Extra data: line 1 column 186` in
swe_agents' host-side update_metrics, the /run answered 500 and the whole validation aborted.
Over virtio-9p (VM sidecar) the window is much wider than on a local filesystem.

This patch adds openhands/utils/nemo_gym_metrics.py (update_nemo_gym_metrics: a process-wide
lock plus a best-effort fcntl.flock, a tolerant read, an atomic temp-file + os.replace write)
and routes the three call sites through it. Idempotent: each file is skipped once it carries
the marker. Note: openhands/ is imaged into openhands_pkg.img for the VM sidecar, so that
image must be rebuilt after (re)applying this patch -- see scripts/presync.sbatch.
"""

import sys
from pathlib import Path

MARKER = "CUSTOM-ATOMIC-NG-METRICS 2026-11-21"
TAG = "[patch_openhands_atomic_nemo_gym_metrics]"

HELPER_RELPATH = Path("openhands") / "utils" / "nemo_gym_metrics.py"
HELPER_SOURCE = f'''"""[{MARKER}] Serialized, crash-safe updates of NeMo Gym's metrics file.

Added by Gym's responses_api_agents/swe_agents/setup_scripts/
patch_openhands_atomic_nemo_gym_metrics.py -- see that script's docstring for the incident.
"""

import fcntl
import json
import os
import threading
import uuid
from typing import Any, Callable

_LOCK = threading.Lock()


def _read_metrics(path: str) -> dict[str, Any]:
    """The metrics dict; tolerates trailing garbage (keeps the first JSON object) and an
    unreadable or missing file (empty dict), as written by older unpatched writers."""
    try:
        with open(path) as f:
            text = f.read()
    except FileNotFoundError:
        return {{}}
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text.lstrip())
        except json.JSONDecodeError:
            return {{}}
    return obj if isinstance(obj, dict) else {{}}


def update_nemo_gym_metrics(mutate: Callable[[dict[str, Any]], None]) -> None:
    """Apply `mutate` to the metrics dict at $NEMO_GYM_METRICS_FPATH and write it back.

    One process-wide lock serializes the read-modify-write cycles of this process (all the
    writers are threads of the same OpenHands process); the fcntl.flock on a sibling .lock
    file is best effort across processes (it may be a no-op over virtio-9p). The write goes
    to a temporary file in the same directory and is moved into place with os.replace, so a
    reader never sees a partially written file.
    """
    path = os.environ["NEMO_GYM_METRICS_FPATH"]
    with _LOCK:
        lock_file = None
        try:
            lock_file = open(path + ".lock", "a")
            fcntl.flock(lock_file, fcntl.LOCK_EX)
        except OSError:
            pass
        try:
            metrics = _read_metrics(path)
            mutate(metrics)
            directory, name = os.path.split(path)
            tmp_path = os.path.join(directory, f".{{name}}.{{os.getpid()}}.{{uuid.uuid4().hex[:8]}}.tmp")
            with open(tmp_path, "w") as f:
                json.dump(metrics, f)
            os.replace(tmp_path, path)
        finally:
            if lock_file is not None:
                lock_file.close()  # releases the flock too
'''

# (relative path, old text, new text) -- old text copied verbatim from nv-OpenHands 87fbbca1.
REPLACEMENTS = [
    (
        Path("openhands") / "runtime" / "base.py",
        '''            import os, json

            metrics_fpath = os.environ["NEMO_GYM_METRICS_FPATH"]
            with open(metrics_fpath) as f:
                existing_dict = json.loads(f.read())

            exec_time_taken = existing_dict.get("total_command_exec_time", 0.0)
            existing_dict["total_command_exec_time"] = exec_time_taken + time.time() - start_time

            with open(metrics_fpath, "w") as f:
                json.dump(existing_dict, f)
''',
        f'''            # [{MARKER}] serialized + atomic, see openhands/utils/nemo_gym_metrics.py
            from openhands.utils.nemo_gym_metrics import update_nemo_gym_metrics

            exec_time_taken = time.time() - start_time

            def _add_exec_time(metrics: dict) -> None:
                metrics["total_command_exec_time"] = (
                    metrics.get("total_command_exec_time", 0.0) + exec_time_taken
                )

            update_nemo_gym_metrics(_add_exec_time)
''',
    ),
    (
        Path("openhands") / "agenthub" / "nemo_gym_client.py",
        '''        metrics_fpath = os.environ["NEMO_GYM_METRICS_FPATH"]
        with open(metrics_fpath) as f:
            existing_dict = json.loads(f.read())

        model_call_time_taken = existing_dict.get("total_model_call_time", 0.0)
        existing_dict["total_model_call_time"] = (
            model_call_time_taken + time.time() - start_time
        )

        with open(metrics_fpath, "w") as f:
            json.dump(existing_dict, f)
''',
        f'''        # [{MARKER}] serialized + atomic, see openhands/utils/nemo_gym_metrics.py
        from openhands.utils.nemo_gym_metrics import update_nemo_gym_metrics

        model_call_time_taken = time.time() - start_time

        def _add_model_call_time(metrics: dict) -> None:
            metrics["total_model_call_time"] = (
                metrics.get("total_model_call_time", 0.0) + model_call_time_taken
            )

        update_nemo_gym_metrics(_add_model_call_time)
''',
    ),
    (
        Path("evaluation") / "benchmarks" / "swe_bench" / "run_infer.py",
        '''    import os, json

    metrics_fpath = os.environ["NEMO_GYM_METRICS_FPATH"]
    with open(metrics_fpath) as f:
        existing_dict = json.loads(f.read())

    existing_dict = {k: v for k, v in existing_dict.items() if v is not None}
    update_dict = {k: v for k, v in update_dict.items() if v is not None}

    with open(metrics_fpath, "w") as f:
        json.dump(existing_dict | update_dict, f)
''',
        f'''    # [{MARKER}] serialized + atomic, see openhands/utils/nemo_gym_metrics.py
    from openhands.utils.nemo_gym_metrics import update_nemo_gym_metrics

    update_dict = {{k: v for k, v in update_dict.items() if v is not None}}

    def _merge(metrics: dict) -> None:
        for k in [k for k, v in metrics.items() if v is None]:
            del metrics[k]
        metrics.update(update_dict)

    update_nemo_gym_metrics(_merge)
''',
    ),
]


def write_helper(openhands_dir: Path) -> bool:
    path = openhands_dir / HELPER_RELPATH
    if path.is_file() and MARKER in path.read_text():
        print(f"{TAG} {path} already present, skipping.")
        return True
    if not path.parent.is_dir():
        print(f"{TAG} ERROR: {path.parent} does not exist.", file=sys.stderr)
        return False
    path.write_text(HELPER_SOURCE)
    print(f"{TAG} Wrote {path}.")
    return True


def patch_file(path: Path, old: str, new: str) -> bool:
    if not path.is_file():
        print(f"{TAG} ERROR: {path} does not exist.", file=sys.stderr)
        return False
    text = path.read_text()
    if MARKER in text:
        print(f"{TAG} {path} already patched, skipping.")
        return True
    if text.count(old) != 1:
        print(
            f"{TAG} ERROR: expected exactly one occurrence of the metrics read-modify-write "
            f"block in {path}, found {text.count(old)}.",
            file=sys.stderr,
        )
        return False
    path.write_text(text.replace(old, new, 1))
    print(f"{TAG} Patched {path}.")
    return True


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: patch_openhands_atomic_nemo_gym_metrics.py <path-to-openhands-dir>", file=sys.stderr)
        sys.exit(1)

    openhands_dir = Path(sys.argv[1])
    ok = write_helper(openhands_dir)
    for relpath, old, new in REPLACEMENTS:
        ok = patch_file(openhands_dir / relpath, old, new) and ok
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
