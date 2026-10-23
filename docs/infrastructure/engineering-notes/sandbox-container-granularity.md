# Sandbox container granularity — deferred decisions

Companion note to `nemo_gym/sandbox_launcher.py` (the Singularity-based isolation launcher
for `nemo_skills`-backed code-execution sandboxes, currently used by `ns_tools` and
`reasoning_gym`). Two things were deliberately deferred when that module was built
(2026-09-29) rather than decided speculatively without production telemetry, plus one
correction to an earlier claim made during this module's design (see §3). Revisit both
once we have real usage data from a live job.

## 1. Shared-per-job container vs. isolated-per-rollout container

`ensure_sandbox()` currently runs **one** Singularity container per job, shared by every
rollout that uses `ns_tools`/`reasoning_gym` in that job (see the "Node-sharing note" and
"Extension point" paragraphs in `sandbox_launcher.py`'s module docstring). The alternative —
spawning a fresh, fully-isolated container per rollout (or even per device/node) — was
considered and explicitly deferred, not rejected. `ensure_sandbox()` has no built-in
assumption that it's called once per job: a future caller can invoke it with a
per-rollout/per-session `port` and lock path instead of a fixed per-job one, and get the same
isolation guarantees at rollout-scoped lifecycle. This is a real option, not a rewrite.

### Tradeoffs

| | Shared, per-job (current default) | Isolated, per-rollout |
|---|---|---|
| Filesystem isolation between concurrent rollouts | Shared writable-tmpfs overlay — two concurrent rollouts' generated code *could* collide on a fixed path (e.g. both writing `/tmp/out.txt`) | Fully separate overlay per rollout — closes this gap entirely |
| Startup cost | ~2s once per job (~0.003% of an 18h job) — confirmed via direct measurement this session | ~2s **per tool-using rollout**. At real scale this stops being negligible: e.g. 100k tool-using rollouts × 2s ≈ 55 CPU-hours of pure container-startup, which shows up as lost rollout-collection throughput, not a one-time cost |
| Aggregate resource accounting | Needs an explicit throttling layer (the proxy semaphore in `sandbox_launcher.py`) to stay safe on a node shared with vLLM/training workers | Still needs a cap on concurrent *containers* — the same problem, just moved up a layer, not eliminated |
| Blast radius of a crash | A watchdog-triggered restart briefly affects every concurrently-active session, not just the one that crashed | One rollout's crash is fully contained to itself |
| Operational surface | One process tree, one log file, one watchdog per job | N concurrent containers, N log files, N watchdogs — meaningfully more moving parts to debug when something breaks |

### Why shared-per-job is the current default

For the *current* tool (pure Python code execution via a forked IPython kernel per
`session_id`), the marginal isolation benefit of per-rollout containers is narrower than it
first looks: `local_sandbox_server.py` already forks a separate OS process per session, so
each rollout's *variable state* is already isolated from every other rollout. The only thing
a full per-rollout container adds on top is filesystem-overlay separation — which only
matters if generated code happens to write to a fixed, guessable path that another
*concurrently active* rollout's code also touches. Possible, but not the common failure mode
for this kind of task, and not something we've observed causing a wrong verification result.

Given that, and given the real (not hypothetical) throughput cost of per-rollout startup at
scale, shared-per-job is the safer default until we have evidence the shared-overlay
assumption actually causes incorrect rewards in practice.

### When to revisit

- A future resources server whose task genuinely depends on isolated, persistent workspace
  state across a rollout's own multiple turns (SWE-bench-style tasks: a checked-out repo,
  build artifacts, etc.) — this is the case explicitly flagged as needing non-shared state,
  and per-rollout containers (or a different mechanism, e.g. a fresh workspace directory per
  rollout inside the *same* long-lived container) are worth designing for then, once that
  server exists and its actual requirements are concrete.
- If production telemetry ever shows two concurrent rollouts colliding on a shared filesystem
  path inside the sandbox (e.g. a suspicious pattern of one rollout's tool output containing
  another rollout's data) — that would be direct evidence the shared-overlay assumption is
  actually costing us correctness, not just a theoretical gap.
- If per-rollout container startup cost, measured against real tool-usage-rate telemetry from
  a live job (see the open question below), turns out to be small enough that the throughput
  argument against it doesn't hold.

**Open question, not yet answered**: we don't have real numbers on what fraction of rollouts
in a given mix actually invoke a Python tool, or how many concurrent tool-using rollouts a
real job sustains. Both numbers are needed to make the per-rollout-cost tradeoff quantitative
instead of a guess. Pull these from a live job's timing metrics
(`num_tool_calls`/`total_tool_execution_time_seconds`, already returned by both resources
servers' `verify()`) once one is running against real production data.

"Per device" (one sandbox per GPU/training node, rather than one per whole job) is a
structurally different and larger change than per-rollout — today, the single `NemoGym` Ray
actor (and everything under it, including the sandbox) lives on exactly one node per job,
wherever Ray's scheduler places it; rollout collection on every other node reaches that one
node's HTTP endpoints over the network regardless of GPU topology. Moving to per-device
sandboxes would require rollout collection itself to know "call my own node's sandbox"
instead of always reaching the one actor's node — a change to rollout-collection routing,
well beyond `sandbox_launcher.py`'s scope. Not under active consideration unless a specific
latency/locality motivation shows up.

## 2. The throttling proxy queues *calls*, not *live sessions*

`_SandboxProxyServer`'s semaphore (`proxy_max_concurrency`) bounds how many `/execute`
requests can be actively running at once — `server.semaphore.acquire(timeout=...)` **blocks**
(queues) the calling thread rather than rejecting immediately, up to
`proxy_request_timeout_s` (120s default), so it already behaves like a queue with a bounded
wait, not a hard reject-on-full gate.

What it does *not* do: bound how many **sessions are alive at once**. A rollout that calls
the tool once and then goes quiet (forked IPython kernel still alive, session not yet
cleaned up) isn't tracked by the semaphore at all once its request completes — it already
left the queue. Only two things currently bound standing session count:
- `NEMO_SKILLS_SANDBOX_SESSION_TIMEOUT` (set by `sandbox_launcher.py`'s `_spawn()`, default
  1800s) — evicts idle sessions, but only after they've been idle that long. This bounds
  *leaked* sessions (e.g. from a rollout that crashed without calling
  `DELETE /sessions/<id>`), not the peak concurrent count at any instant.
- Indirectly, the natural concurrency of the training job's own rollout collection — if the
  job only ever has, say, 200 rollouts in flight at once, session count can't exceed that
  regardless of the proxy.

**Deferred, not implemented**: a proper admission-control gate on *new session creation*
(distinct from call-execution throttling) — since the proxy already sees every request, it
could track distinct session IDs in flight and cap concurrent *live* ones, separately from
`proxy_max_concurrency`'s call-level cap. Not built now because we don't have evidence it's
needed — `SESSION_TIMEOUT` plus the job's natural rollout concurrency may be sufficient in
practice. Revisit if node memory pressure or an unexpectedly high standing-session count
shows up in a real job (the sandbox log's periodic `active_sessions=N` line, emitted by
`local_sandbox_server.py` itself, is the place to look for this signal).

## 3. Network isolation: real namespace isolation is not achievable here (correction)

Earlier in this module's design, filesystem and network isolation were validated together as
"both work" via `singularity exec --containall --writable-tmpfs --net --network none`, tested
by running a health check *from inside the same container invocation*. That test was real but
incomplete: it never checked whether a process *outside* the container — which is exactly
what `ns_tools`/`reasoning_gym` are, since they run as plain host processes, not inside any
container themselves — could reach the sandboxed server's port. It can't.

Confirmed directly on a real compute node: `--network none` gives the container a fully
isolated network namespace with no veth/bridge to the host — a port bound inside is invisible
outside. `bridge`/`ptp` CNI networking (which would give the container a routable IP the host
could reach while still isolating it from the wider internet) is restricted to root by this
cluster's `singularity.conf` (`allow net networks` unset); `--fakeroot` doesn't help because
it itself needs unprivileged user namespaces, which are cluster-wide disabled (`/proc/sys/user/
max_user_namespaces = 0`, the same restriction that ruled out generic DIY containerization at
the very start of this investigation). There is no unprivileged path on this cluster to "network-
isolated but host-reachable" via Singularity's own namespace mechanism.

**Current behavior** (`nemo_gym/sandbox_launcher.py`, `_build_singularity_command()`): no
`--net`/`--network` flags at all — the container shares the host's network namespace, which is
what makes host↔container HTTP reachability work. `network="none"` vs `"host"` now controls
only `NEMO_SKILLS_SANDBOX_BLOCK_NETWORK` (the sandbox server's own in-process `socket` patch)
— the same mechanism and the same weaker guarantee (catches naive/in-process network calls,
not a deliberately adversarial subprocess-based bypass) that existed before this module. Real
filesystem isolation (`--containall --writable-tmpfs`) is unaffected by any of this and still
gives a genuinely ephemeral, host-invisible overlay — confirmed with a real write-inside/
read-from-host test end to end through the actual launcher, not just a standalone check.

**Deferred, not implemented**: a Unix-domain-socket bridge would recover real network
isolation without needing CNI or root. Sketch: run `local_sandbox_server.py` bound to its
(hardcoded, see §4) TCP port entirely inside the isolated netns as today, plus a small
asyncio TCP↔Unix-socket relay process in the same container invocation, listening on a
socket file under a `--bind`-mounted directory (bind mounts are a *filesystem* mechanism,
unaffected by network namespace isolation). The host side would connect via `httpx.
HTTPTransport(uds=...)` (a real, already-supported httpx feature) instead of a TCP URL. Not
built now: real added complexity (a new relay process to write, test, and keep alive; the
throttling proxy would also need UDS support) for a security property whose main practical
gap — deliberately adversarial network egress via a subprocess bypassing the Python patch —
hasn't been observed as an actual problem. Revisit if that changes, or if a future
dangerous-tool env's threat model specifically requires it.

## 4. The sandbox server's bind port is hardcoded, not configurable

`local_sandbox_server.py`'s `__main__` block is a bare `app.run(port=6000)` — no env var, no
CLI flag, nothing else reads a configurable port. `ensure_sandbox()`'s `port` argument must
always be `6000` (enforced with a `ValueError` if not, added after a real spawn against a
non-default port silently listened on 6000 while a health check against the *requested* port
timed out for the full 120s with no indication why). A caller wanting a different
externally-visible port should use `proxy_port` instead — a real, independently-listening
port this module controls via `_start_proxy()` — not `port` itself.
