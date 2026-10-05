# Brave Search Resources Server

## Table of Contents
- [Brave Search Resources Server](#brave-search-resources-server)
  - [Table of Contents](#table-of-contents)
  - [Description](#description)
    - [Overview](#overview)
    - [Dataset Description](#dataset-description)
    - [Environment Tools](#environment-tools)
    - [Verifier](#verifier)
  - [Requirements](#requirements)
    - [Outbound internet access](#outbound-internet-access)
  - [Environment Inheritance](#environment-inheritance)
  - [Usage Commands](#usage-commands)
    - [Running the Server](#running-the-server)

## Description

### Overview
Same intent as `resources_servers/google_search`, backed by the official
[Brave Search MCP server](https://github.com/brave/brave-search-mcp-server) instead of Google's
Programmable Search Engine API.

### Dataset Description
Uses the same dataset as `google_search`:
https://huggingface.co/datasets/nvidia/Nemotron-RL-knowledge-web_search-mcqa

Multiple Choice Question Answering (MCQA) tasks in the STEM domain, filtered for difficulty
with Qwen3-32B so samples are not easily answerable without external search assistance.

### Environment Tools
- **search**: Calls the Brave MCP server's `brave_web_search` tool and returns up to
  `max_results` (default 10) results.
- **browse**: Identical to `google_search`'s `browse()` -- fetches a URL and extracts cleaned
  text via `trafilatura`. Brave's MCP server has no page-fetch tool of its own, and
  page-cleaning is orthogonal to which search backend produced the URL.

### Verifier
Same as `google_search`: extracts the boxed MCQ option (`\boxed{X}`) from the last assistant
message and checks it against `expected_answer`.

## Requirements

You will need a `brave_api_key` (from the [Brave Search API](https://brave.com/search/api/)).
This repo's secrets all flow through one gitignored `.env` at the domyn-rl repo root (see
`.env.example`), sourced by `launch.sh` and exported into the job's environment:
```bash
# .env
BRAVE_API_KEY=<your_api_key>
```
The resources server config (`configs/brave_search.yaml`) reads it via
`${oc.env:BRAVE_API_KEY}` -- no separate `env.yaml` needed for this secret specifically
(a deliberate departure from Gym's usual `env.yaml`-interpolation convention, matching
`ns_tools`'s own `oc.env` precedent elsewhere in this fork).

By default, this server spawns the Brave MCP server itself as a local subprocess
(`npx -y @brave/brave-search-mcp-server --transport http`) and connects to it over
streamable-HTTP. `npx` and network access to fetch the package are required wherever the
resources server process runs.

### Outbound internet access

`boost_usr_prod` compute nodes have no outbound internet access, so the default local-spawn
mode will not be able to reach Brave's API when this server runs there -- the same gap that
also affects `google_search` and `tavily_search`. The repo now has a working fix for this:
`nemo_rl/utils/proxy.py` (opt-in via the experiment YAML's `proxy:` block) keeps a squidward
forward-proxy alive for the whole training run and exports `HTTP_PROXY`/`HTTPS_PROXY` into
every Ray worker's environment (see `RL/CLAUDE.md`-adjacent docs / the proxy module's own
docstring for how). `requests`-based servers (`google_search`) and `httpx`-based clients pick
this up automatically; Tavily's aiohttp-based client now does too (`trust_env=True`).
**Brave's local-spawn mode is not yet confirmed to benefit from this**: the internet-reaching
leg is the spawned Node.js `npx` subprocess, and Node's fetch/undici does not automatically
honor `HTTP_PROXY` the way Python's HTTP stacks do -- untested, needs a real `BRAVE_API_KEY`
to verify. `brave_mcp_base_url` (config field, or the `BRAVE_MCP_BASE_URL` env var) remains
the fallback: point this server at an already-running Brave MCP server elsewhere (e.g. one
started on a network-enabled node) instead of local-spawn.

### Rate limiting

Enabled by default (`rate_limiter_enabled: true` in `configs/brave_search.yaml`, currently
set to Brave's own 30 requests/second limit). Backed by a distributed Redis-based limiter
(`rate_limiter.py`, a vendored copy of `nemo_rl/utils/rate_limiter.py`) so the limit is
enforced across every worker/node, not just locally. Requires:
1. A `redis:` block with `enabled: true` in the training experiment YAML (see
   `nemo_rl/utils/redis_job.py`'s docstring) -- this submits the shared Redis SLURM job
   (`images/redis/submit.sh`) with walltime matched to the training run and exports its host
   as `NEMO_RL_REDIS_HOST`.
2. `REDIS_PASSWORD` set in your `.env` (see `.env.example`) -- the static, shared password
   from `images/redis/redis.conf`, not a per-run secret.

Without both of these, `redis_host` stays null and every request fails fast at startup --
set `rate_limiter_enabled: false` in `configs/brave_search.yaml` to disable if not using the
shared Redis job. Tune `rate_limit_window_sec`/`rate_limit_count`/`rate_limit_max_wait_sec`
if Brave's rate limit changes.

## Environment Inheritance
This environment can be inherited to integrate search functionality into your own custom
environment. The following tool definitions are available for inheritance:
```python
tools = [
    {
        "type": "function",
        "name": "search",
        "description": "Search the web via Brave Search and return up to 10 results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The term to search for",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "browse",
        "description": "Returns the cleaned content of a webpage. If the page is too long, it will be truncated to 10,000 words.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "The url of the page to get the content of"}},
            "required": ["url"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]
```

## Usage Commands

### Running the Server
```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/brave_search/configs/brave_search.yaml"
ng_run "+config_paths=[$config_paths]"

ng_collect_rollouts +agent_name=simple_agent \
    +input_jsonl_fpath=resources_servers/brave_search/data/example.jsonl \
    +output_jsonl_fpath=results/example_rollouts.jsonl \
    +limit=1
```

## Licensing Information

Dependencies
- nemo_gym: Apache 2.0
- trafilatura: [Apache 2.0 license](https://trafilatura.readthedocs.io/en/latest/#license)
- mcp (official MCP Python SDK): MIT license
