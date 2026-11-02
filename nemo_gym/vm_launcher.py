"""Start and supervise a per-node QEMU/KVM VM sidecar, SSH into it to run commands.

[CUSTOM] Adapted from nemo_gym/sandbox_launcher.py -- this is the same shape of problem
(lazily and idempotently ensure a long-lived, shared sidecar is running on whichever node
this process happens to be on, survive multiple independent caller processes racing to spawn
it, self-heal via a health-check-based watchdog) applied to a VM instead of a Singularity
sandbox process. See sandbox_launcher.py's own module docstring for the full reasoning behind
the cross-process idempotency (HTTP/SSH health check + fcntl.flock-guarded spawn) and
health-check-based (not Popen.poll()-based) watchdog design -- both apply here verbatim.

Why a VM at all: the SWE-bench-family dataset types (SWE-Gym, R2E-Gym, SWE-smith,
SWE-rebench V1/V2) need a writable /testbed (or /{repo_name}) inside their Apptainer
containers, but that content is root-owned from the upstream Docker build, and this cluster
has unprivileged user namespaces disabled (/proc/sys/user/max_user_namespaces = 0), so
--fakeroot isn't available and --writable-tmpfs alone can't write to those paths. Booting a
real KVM VM sidesteps this entirely: inside the guest we are real root (no --fakeroot
gymnastics needed), so a plain `apptainer exec --writable-tmpfs` against the ORIGINAL,
unmodified image just works -- confirmed directly on this cluster, including the real,
multi-TB Lustre images directory reachable from the guest via a virtio-9p passthrough mount
with negligible overhead (sub-second `singularity exec` against a real image through the 9p
mount, cold or warm).

Lifecycle: one persistent VM per node, shared by every rollout that lands there for the
job's duration -- NOT one per rollout. A per-rollout VM was the first design considered and
rejected: each boot does real work (a qcow2 overlay, a cloud-init seed, a free hostfwd port),
and if enough rollouts start near-simultaneously on one node (plausible at this codebase's
concurrency settings), that's many concurrent QEMU boots competing for the same CPU and the
same Lustre metadata server at once -- the same *shape* of problem as the per-actor-venv-build
race documented in this repo's CLAUDE.md. A persistent per-node VM pays that boot cost exactly
once, at first touch, before any rollout traffic exists to contend with it. The cost: one VM
now serves many concurrent rollouts, so a restart affects everyone in flight on that node, not
just one -- the same tradeoff sandbox_launcher.py already accepts for ns_tools/reasoning_gym,
and for the same reason (each rollout's filesystem state is already isolated one layer down,
here by Apptainer's own --containall --writable-tmpfs --pid namespaces).

What is NOT ported from the igeniusai/domyn-inspect `terminal_bench_kvm` precedent (the
existing, validated prior art for unprivileged QEMU/KVM on this exact cluster/account): its
Docker-socket-over-SSH forwarding (we have no persistent daemon to forward a socket to --
`run_in_vm` just SSHes in and runs the same one-shot `apptainer exec` command the host would
have run directly) and its redsocks/iptables transparent-proxy layer (the existing host-side
code already threads HTTP_PROXY/HTTPS_PROXY into the apptainer command string for the one
case that needs real internet, SWE-rebench-V2's eval; `run_in_vm` just exports the same vars
before the SSH hop, reusing that existing mechanism instead of adding a new one).
"""

from __future__ import annotations

import asyncio
import fcntl
import logging
import os
import shlex
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)

_HEALTH_POLL = 2.0
_BOOT_TIMEOUT_S = 300.0
_WATCHDOG_INTERVAL = 120.0  # matches sandbox_launcher.py's post-incident value -- avoid
# killing a healthy-but-briefly-slow-to-respond VM under load.
_SSH_CONNECT_TIMEOUT_S = 5

# The default assumes this file's actual location in the domyn-rl repo
# (RL/3rdparty/Gym-workspace/Gym/nemo_gym/vm_launcher.py, six levels under the repo root where
# scripts/prepare-env/vm/lib.sh lives) -- override via VM_LAUNCHER_LIB_SH if the Gym submodule
# is ever vendored at a different depth, rather than relying on this guess silently being wrong.
_VM_LIB_SH = os.environ.get(
    "VM_LAUNCHER_LIB_SH",
    str(Path(__file__).resolve().parent.parent.parent.parent.parent.parent / "scripts" / "prepare-env" / "vm" / "lib.sh"),
)
_VM_IMAGE_DIR_DEFAULT = "/leonardo_scratch/fast/BOOST_EUROPA/images/vm"

# One VM per *node*, not per process -- every caller on a node must agree on the same
# node-local state dir regardless of which resources-server process it's in, exactly like
# sandbox_launcher.py's hardcoded port 6000 lets independent ns_tools/reasoning_gym processes
# find the same sandbox. /tmp is node-local (not shared storage), matching sandbox_launcher.py's
# own choice of /tmp for its lock/log paths.
_STATE_DIR = "/tmp/vm_launcher"


class VMConnectionInfo:
    __slots__ = ("ssh_key", "ssh_port")

    def __init__(self, ssh_key: str, ssh_port: int) -> None:
        self.ssh_key = ssh_key
        self.ssh_port = ssh_port


def _work_dir() -> str:
    d = _STATE_DIR
    os.makedirs(d, exist_ok=True)
    return d


def _hostname() -> str:
    return os.uname().nodename


def _spawn_lock_path() -> str:
    return os.path.join(_work_dir(), "spawn.lock")


def _ssh_key_path() -> str:
    return os.path.join(_work_dir(), "id")


def _ssh_port_path() -> str:
    return os.path.join(_work_dir(), "ssh_port")


def _pid_path() -> str:
    return os.path.join(_work_dir(), "vm.pid")


def _log_path() -> str:
    return os.path.join(_work_dir(), "vm.log")


def _current_connection_info() -> Optional[VMConnectionInfo]:
    key = _ssh_key_path()
    port_path = _ssh_port_path()
    if not (os.path.exists(key) and os.path.exists(port_path)):
        return None
    with open(port_path) as f:
        port = int(f.read().strip())
    return VMConnectionInfo(ssh_key=key, ssh_port=port)


def _check_health(info: VMConnectionInfo, timeout_s: float = _SSH_CONNECT_TIMEOUT_S) -> bool:
    try:
        result = subprocess.run(
            [
                "ssh",
                "-q",
                "-i",
                info.ssh_key,
                "-p",
                str(info.ssh_port),
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "UserKnownHostsFile=/dev/null",
                "-o",
                f"ConnectTimeout={int(timeout_s)}",
                "root@127.0.0.1",
                "true",
            ],
            capture_output=True,
            timeout=timeout_s + 5,
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def _tail_log(n: int = 60) -> str:
    path = _log_path()
    if not os.path.exists(path):
        return "(no log file)"
    try:
        with open(path) as f:
            lines = f.readlines()
        return "".join(lines[-n:])
    except OSError as e:
        return f"(could not read log: {e})"


def _kill_stale_vm() -> None:
    """[CUSTOM-VM-WATCHDOG-RESTART-FIX 2026-10-28] _spawn() always builds its disk overlay at
    the same fixed path (_work_dir()/disk.qcow2) regardless of whether this is a fresh first
    boot on this node or a watchdog-triggered restart of an unhealthy VM. If the previous VM's
    qemu-system-x86_64 process is still alive -- confirmed to happen in practice: a VM going
    `healthy=False` (SSH stops responding, almost certainly transient resource pressure, not a
    crash) does NOT mean the process has exited -- `vm_make_overlay`'s `qemu-img create` then
    fails outright: "Failed to get 'write' lock. Is another process using the image?", and the
    whole restart attempt aborts with no retry. Observed hitting this on two separate nodes in a
    single real job (62542185): both recovered on their own moments later (same SSH port, no new
    DEBUG_VM_BOOT success line), meaning the "restart" was chasing a VM that was never really
    dead -- but if a VM ever DOES genuinely crash instead of just stalling, there would be no
    working recovery path at all and that node would be stuck for the rest of the job.

    Fix: before ever touching the overlay, make sure nothing is still holding it. Does not rely
    on SSH (the guest not answering SSH is exactly the symptom that got us here) -- goes straight
    to the host-visible PID from the last spawn's pidfile: SIGTERM, a short grace period, then
    SIGKILL if it's still around. A completely fresh node (no pidfile yet, or a pidfile for a
    PID that's already gone) is a silent no-op.

    """
    pid_path = _pid_path()
    if not os.path.exists(pid_path):
        return
    try:
        pid = int(open(pid_path).read().strip())
    except (ValueError, OSError):
        return
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return  # already gone -- nothing to clean up
    except PermissionError:
        return  # not ours (e.g. pid reused by an unrelated process) -- don't touch it
    logger.warning("Killing stale VM process pid=%d before respawning on %s", pid, _hostname())
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(50):  # 5s grace period, matching vm_stop()'s own poll cadence in lib.sh
        time.sleep(0.1)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _spawn(
    golden_image_dir: str,
    lustre_share_roots: list[str],
    block_device_mounts: Optional[list[tuple[str, str]]] = None,
) -> None:
    """Boot this node's VM. Blocks until SSH is reachable or _BOOT_TIMEOUT_S elapses.

    Runs scripts/vm/lib.sh's vm_* helpers via a small inline bash script rather than
    reimplementing the boot sequence in Python -- the shell helpers are the same ones this
    change's build/smoke-test scripts use and were validated directly against a real compute
    node, including the virtio-9p share(s) and the cloud-init seed. Keeping one implementation
    (shell) for "how to boot this VM" avoids a second, Python-side copy that could drift from
    what was actually tested.

    block_device_mounts: (host_image_path, guest_mount_path) pairs for read-only ext4 images
    attached as extra virtio-blk drives (see scripts/vm/build_openhands_image.sh) -- used for
    directories with too many small files for 9p to serve without stalling (confirmed directly:
    tuned 9p msize/cache and even a newer QEMU with the 9pfs fid-hashtable fix both still stalled
    on importing OpenHands' full Python dependency tree; a dedicated local block device did not,
    in both single and concurrency=8 testing). Mounted AFTER the 9p shares, at the guest's exact
    absolute path, so it shadows just that one subtree of a 9p-shared parent (e.g. OpenHands'
    setup dir shadowing part of the wider Lustre share) -- same real-path convention as
    [CUSTOM-VM-MOUNT-SAME-PATH] below.
    """
    _kill_stale_vm()
    work = _work_dir()
    log_file = open(_log_path(), "a")  # noqa: SIM115
    qemu_dir = os.path.join(golden_image_dir, "qemu")
    golden = os.path.join(golden_image_dir, "golden.qcow2")
    block_device_mounts = block_device_mounts or []
    shares = " ".join(shlex.quote(f"{root}:{_mount_tag(root)}") for root in lustre_share_roots)
    # qemu.sh (lib.sh) only binds VM_QEMU_PREFIX_DIR and /dev/kvm into the Singularity container
    # QEMU itself runs inside -- every other host path QEMU needs to touch (the golden disk, the
    # per-node work dir, each Lustre root being shared via -virtfs, and each extra block-device
    # image's directory) has to be listed here explicitly, or qemu-system-x86_64 fails with a
    # plain "No such file or directory" that looks like a missing file rather than a missing
    # bind (hit directly while building this module).
    extra_binds = ":".join(
        [
            golden_image_dir,
            work,
            *lustre_share_roots,
            *{os.path.dirname(img) for img, _ in block_device_mounts},
        ]
    )
    extra_drives = ":".join(img for img, _ in block_device_mounts)
    # [CUSTOM-VM-MOUNT-SAME-PATH 2026-10-27] Each share must be mounted in the guest at the EXACT
    # same absolute path it has on the host, not some other guest-side path (e.g. /mnt/<tag>) --
    # that was this module's first real bug, caught by an actual job run: _build_apptainer_command
    # constructs --mount/container-path strings using the real host absolute paths (e.g.
    # /leonardo_scratch/large/.../images/swesmith/<instance>.sif), so if the guest doesn't have
    # something real at that exact path, `singularity exec` fails with "could not open image ...
    # lstat /leonardo_scratch: no such file or directory" -- which is exactly what happened.
    # -virtfs only makes a share AVAILABLE to the guest (via its mount_tag); it does not mount it
    # anywhere on its own, so this loop runs once per boot, right after SSH comes up.
    mount_cmds = "\n".join(
        f'vm_ssh "$WORK/id" "$VM_SSH_PORT" "mkdir -p {shlex.quote(root)} && '
        f'mount -t 9p -o trans=virtio,version=9p2000.L {_mount_tag(root)} {shlex.quote(root)}"'
        for root in lustre_share_roots
    )
    # Block devices attach, in VM_EXTRA_DRIVES order, right after vda=disk/vdb=seed -- so the
    # Nth entry here is /dev/vd<c, d, e, ...>. Mounted AFTER the 9p loop above (see docstring).
    block_device_cmds = "\n".join(
        f'vm_ssh "$WORK/id" "$VM_SSH_PORT" "mkdir -p {shlex.quote(mount_path)} && '
        f'mount -o ro /dev/vd{chr(ord("c") + i)} {shlex.quote(mount_path)}"'
        for i, (_, mount_path) in enumerate(block_device_mounts)
    )
    script = f"""
set -euo pipefail
export VM_QEMU_PREFIX_DIR={shlex.quote(os.path.join(qemu_dir, "prefix"))}
export VM_QEMU_SIF={shlex.quote(os.path.join(qemu_dir, "debian-bookworm.sif"))}
export VM_EXTRA_BINDS={shlex.quote(extra_binds)}
export VM_EXTRA_DRIVES={shlex.quote(extra_drives)}
source {shlex.quote(_VM_LIB_SH)}
WORK={shlex.quote(work)}
vm_make_overlay {shlex.quote(golden)} "$WORK/disk.qcow2"
vm_make_key "$WORK"
vm_root_user_data "$WORK/user-data" "$WORK/id.pub"
vm_make_seed "$WORK/user-data" "$WORK/seed.img" "$WORK"
vm_start {_vcpus()} {_mem_mb()} "$WORK/disk.qcow2" "$WORK/seed.img" "$WORK" {shares}
echo "$VM_SSH_PORT" > {shlex.quote(_ssh_port_path())}
echo "$VM_PID" > {shlex.quote(_pid_path())}
vm_wait_ssh "$WORK/id" "$VM_SSH_PORT" "$VM_PID" "$WORK" {int(_BOOT_TIMEOUT_S)}
{mount_cmds}
{block_device_cmds}
"""
    result = subprocess.run(["bash", "-c", script], stdout=log_file, stderr=subprocess.STDOUT)
    log_file.close()
    if result.returncode != 0:
        raise RuntimeError(f"VM boot failed (see {_log_path()}):\n{_tail_log()}")


def _mount_tag(host_path: str) -> str:
    # virtio-9p mount tags are limited in length/charset; a short stable hash of the path is
    # simpler than sanitizing the path itself and avoids collisions between similarly-named
    # roots (e.g. two different users' scratch dirs).
    import hashlib

    return "vmshare" + hashlib.sha1(host_path.encode()).hexdigest()[:8]  # noqa: S324 (not security-sensitive)


def _vcpus() -> int:
    # [CUSTOM-VM-SIZING placeholder] Not yet derived from real measurement -- see this change's
    # plan doc, "RAM sizing is now a genuinely open, unmeasured question". A modest fixed
    # allocation for now; revisit once real per-node concurrent-rollout telemetry exists.
    return int(os.environ.get("VM_LAUNCHER_VCPUS", "8"))


def _mem_mb() -> int:
    # [CUSTOM-VM-SIZING placeholder] Same caveat as _vcpus() -- --writable-tmpfs writes are
    # RAM-backed, so this has to accommodate the AGGREGATE working-tree/build-artifact
    # footprint of however many concurrent rollouts land on this node at once, not just one.
    # Not yet measured; this default is a starting point to unblock implementation, not a
    # validated sizing decision -- do not scale up real traffic onto this without first
    # measuring real per-rollout --writable-tmpfs usage (see the plan's Verification step 1).
    return int(os.environ.get("VM_LAUNCHER_MEM_MB", "32768"))


def mount_tag_for(host_path: str) -> str:
    """Public accessor for the virtio-9p mount_tag `_spawn` uses for `host_path` -- a pure
    internal protocol identifier, not a guest filesystem path (each share is mounted in the
    guest at the SAME absolute path as the host, see [CUSTOM-VM-MOUNT-SAME-PATH] in `_spawn`,
    so callers building mount strings never need this; it's exposed mainly for tests/debugging
    that want to inspect what tag a given root maps to)."""
    return _mount_tag(host_path)


def ensure_vm(
    golden_image_dir: str = _VM_IMAGE_DIR_DEFAULT,
    lustre_share_roots: Optional[list[str]] = None,
    block_device_mounts: Optional[list[tuple[str, str]]] = None,
) -> VMConnectionInfo:
    """Ensure this node's VM sidecar is running, booting it if needed.

    Safe to call from multiple independent rollouts/processes on the same node --
    idempotent via an SSH health check plus a cross-process file lock (see
    sandbox_launcher.py's ensure_sandbox() for the identical pattern), not a same-process
    singleton. Cheap after the first call on a given node: a health check, not a fresh boot.

    Args:
        golden_image_dir: directory holding golden.qcow2 and the unprivileged QEMU install
            (built once, offline, by scripts/vm/build_golden_vm_image.sh).
        lustre_share_roots: host directories to share into the guest via virtio-9p at boot,
            at a stable per-path mount tag (see mount_tag_for). Only read on the first
            (spawning) call on a node -- a VM already running was booted with whatever roots
            an earlier caller requested; passing a different set later has no effect until
            the VM is restarted. Callers needing a new root added should restart the node's
            VM (kill its pid file) rather than relying on a later ensure_vm() call to add it.
        block_device_mounts: (host_image_path, guest_mount_path) pairs attached as extra
            read-only virtio-blk drives, mounted at guest_mount_path after the 9p shares above
            -- see _spawn's docstring. Same "only read on the first/spawning call" caveat as
            lustre_share_roots.
    """
    lustre_share_roots = lustre_share_roots or []
    block_device_mounts = block_device_mounts or []
    info = _current_connection_info()
    if info is not None and _check_health(info):
        logger.info("VM already healthy on 127.0.0.1:%d, nothing to do", info.ssh_port)
    else:
        lock_path = _spawn_lock_path()
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                info = _current_connection_info()
                if info is not None and _check_health(info):
                    logger.info("VM became healthy while waiting for the spawn lock, skipping spawn")
                else:
                    _spawn(golden_image_dir, lustre_share_roots, block_device_mounts)
                    info = _current_connection_info()
                    if info is None or not _check_health(info):
                        raise RuntimeError(f"VM did not become healthy after boot (see {_log_path()}):\n{_tail_log()}")
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    watchdog = threading.Thread(
        target=_watchdog,
        args=(golden_image_dir, lustre_share_roots, block_device_mounts),
        daemon=True,
        name="vm-watchdog",
    )
    watchdog.start()

    logger.info("VM ready on 127.0.0.1:%d", info.ssh_port)
    return info


def _watchdog(
    golden_image_dir: str,
    lustre_share_roots: list[str],
    block_device_mounts: Optional[list[tuple[str, str]]] = None,
) -> None:
    """Health-check-based, not Popen.poll()-based -- same reasoning as
    sandbox_launcher.py's _watchdog: only the process that spawned the VM holds a valid
    Popen/pid handle for it, so every caller independently polls over SSH instead. Mitigates
    this design's main accepted cost (one VM restart affects every rollout in flight on that
    node) by making a VM failure self-healing rather than silently wedging the node.
    """
    while True:
        time.sleep(_WATCHDOG_INTERVAL)
        info = _current_connection_info()
        if info is not None and _check_health(info):
            continue
        logger.warning("VM unhealthy or not yet booted -- attempting restart...")
        lock_path = _spawn_lock_path()
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                info = _current_connection_info()
                if info is not None and _check_health(info):
                    logger.info("VM recovered via another caller, no restart needed")
                    continue
                _spawn(golden_image_dir, lustre_share_roots, block_device_mounts)
                logger.info("VM recovered")
            except (RuntimeError, OSError):
                logger.exception("VM failed to recover after restart")
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)


async def run_in_vm(
    info: VMConnectionInfo,
    command_str: str,
    env: Optional[dict[str, str]] = None,
    **popen_kwargs,
) -> "asyncio.subprocess.Process":
    """SSH into this node's VM and run `command_str` -- a near drop-in replacement for the
    existing `asyncio.create_subprocess_shell(apptainer_cmd, ...)` call sites in
    responses_api_agents/swe_agents/app.py, not a new execution model: these remain one-shot
    subprocess invocations, just one SSH hop away. No socket-forwarding/daemon complexity
    needed (unlike Docker in the terminal_bench_kvm precedent) because there's no persistent
    daemon on the other end, just the same apptainer exec that would have run directly on the
    host.

    `env` vars are exported in the remote shell before `command_str` runs (SSH does not punch
    through arbitrary environment variables by default) -- used for the same HTTP_PROXY/
    HTTPS_PROXY passthrough app.py's _build_apptainer_command already does for SWE-rebench-V2.
    """
    env_prefix = ""
    if env:
        env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + " "
    remote_cmd = f"{env_prefix}{command_str}"
    ssh_cmd = (
        f"ssh -q -i {shlex.quote(info.ssh_key)} -p {info.ssh_port} -o BatchMode=yes "
        f"-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=15 "
        f"root@127.0.0.1 {shlex.quote(remote_cmd)}"
    )
    return await asyncio.create_subprocess_shell(ssh_cmd, **popen_kwargs)
