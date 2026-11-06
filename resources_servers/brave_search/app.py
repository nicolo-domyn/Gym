"""
Brave Search Resources Server.

Wraps the official Brave Search MCP server (https://github.com/brave/brave-search-mcp-server)
as a Gym resources server, mirroring google_search's tool surface (`search`/`browse`) and
verify() logic against the same nvidia/Nemotron-RL-knowledge-web_search-mcqa dataset.

MCP integration note: unlike tavily_search's aiohttp-adapter-over-httpx pattern, the MCP
client session here is connected once at server startup (not per-request) and kept alive for
the process lifetime, using the official `mcp` Python SDK's streamable-HTTP transport. This
transport is httpx-based internally; we intentionally do not attempt to swap it for an aiohttp
adapter here (unlike tavily_search) since MCP tool calls happen at most a few times per
rollout turn, not at the 16k-concurrent-request scale the aiohttp mandate is about -- see
docs/infrastructure/engineering-notes/aiohttp-vs-httpx.md. google_search's use of sync
`requests` for the same reason is a precedent for this kind of low-volume exception.
"""

import importlib.util
import logging
import os
import re
import subprocess
import time
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import httpx
import trafilatura
from fastapi import FastAPI
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ResourcesServerRef


if TYPE_CHECKING:
    # Only for static type checkers -- never executed at runtime, so this never triggers
    # nemo_rl's package-level import side effects. See `_load_rate_limiter_class()` below
    # for how the real class is loaded at runtime.
    from nemo_rl.utils.rate_limiter import RateLimiter


logger = logging.getLogger(__name__)


def _load_rate_limiter_class() -> type:
    """Load nemo_rl's `RateLimiter` directly from its source file, bypassing `nemo_rl` as a
    package.

    This resources server builds its own isolated venv and has no normal import path to
    `nemo_rl`. We  deliberately do not do `import nemo_rl.utils.rate_limiter` (even 
    with `nemo_rl` on `sys.path`): importing the `nemo_rl` package runs 
    `nemo_rl/__init__.py`'s module-level side effects (reconfiguring the root logger via
    `logging.basicConfig()`, a container fingerprint check, a nsight.py patch, etc.), none
    of which this process has any reason to inherit.
    Instead, load just the one file via `importlib.util`, which has zero intra-package
    imports and is fully standalone.

    Only called from the `rate_limiter_enabled` branch below, so this import is attempted --
    and can only fail -- when the feature is actually in use.
    """
    this_file = Path(__file__).resolve()
    # this_file is .../RL/3rdparty/Gym-workspace/Gym/resources_servers/brave_search/app.py
    # parents[5] walks up: brave_search -> resources_servers -> Gym -> Gym-workspace ->
    # 3rdparty -> RL.
    rl_root = this_file.parents[5]
    rate_limiter_path = rl_root / "nemo_rl" / "utils" / "rate_limiter.py"
    if not rate_limiter_path.is_file():
        raise FileNotFoundError(
            f"rate_limiter_enabled=true but could not find nemo_rl's rate_limiter.py at the "
            f"expected path {rate_limiter_path}. This resources server expects to be running "
            "as part of RL's own Gym submodule (RL/3rdparty/Gym-workspace/Gym)."
        )
    spec = importlib.util.spec_from_file_location("_brave_search_nemo_rl_rate_limiter", rate_limiter_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RateLimiter


class BraveSearchResourcesServerConfig(BaseResourcesServerConfig):
    brave_api_key: str

    # If set, connect to an already-running Brave MCP server (streamable-HTTP transport) at
    # this URL instead of spawning one locally. This is the hook for routing through an
    # outbound-internet gateway: boost_usr_prod compute nodes have no internet access, and the
    # Brave MCP server itself needs it to reach Brave's real search API.
    brave_mcp_base_url: Optional[str] = None

    # Only used when brave_mcp_base_url is None (local spawn). --offline: `npx -y` alone
    # still hits the npm registry even when the package is already cached (confirmed --
    # it hangs against a blocked network) -- --offline forces cache-only resolution,
    # required on air-gapped compute nodes. scripts/presync_baremetal_extras.sbatch
    # warms that cache from a network-enabled node beforehand.
    brave_mcp_command: List[str] = Field(
        default_factory=lambda: [
            "npx",
            "--offline",
            "-y",
            "@brave/brave-search-mcp-server",
            "--transport",
            "http",
        ]
    )
    brave_mcp_host: str = "127.0.0.1"
    brave_mcp_port: int = 8766

    max_results: int = 10
    debug: bool = False

    # Optional distributed rate limiter for Brave's own API rate limit, backed by a shared
    # Redis instance (see nemo_rl/utils/redis_job.py in the outer repo, which submits it as a
    # SLURM job and exports its host into NEMO_RL_REDIS_HOST). Disabled by default -- every
    # request to Brave's search API passes through this when enabled, matching the
    # per-request/`key`-scoped design of rate_limiter.py.
    rate_limiter_enabled: bool = False
    redis_host: Optional[str] = None
    redis_port: int = 6380
    redis_password: Optional[str] = None
    rate_limit_window_sec: int = 1
    rate_limit_count: int = 30  # Brave Search API: 30 requests/second
    rate_limit_max_wait_sec: int = 10
    rate_limit_raise_after: int = 60

    # [CUSTOM-VERIFIER-DISPATCH 2026-11-06] Per-row verifier selection, mirroring ns_tools's
    # own default_verifier/verifiers dispatch (resources_servers/ns_tools/app.py) -- added so
    # brave_search's shared tools (search/browse) can be reused by datasets needing a different
    # grading strategy than MCQA's box_parser exact-match (e.g. wikidata-triplet-qa's free-text
    # answers, graded via equivalence_llm_judge) without forking a near-duplicate resources
    # server. "box_parser" (not "exact_match"/"mcqa") is the name used here since no other
    # multi-verifier config in this repo has already established a canonical name for this kind
    # of default/parser-based path -- ns_tools's own default_verifier is math_with_judge itself,
    # not a plain-parser fallback, so there's no existing convention to match.
    default_verifier: str = "box_parser"
    verifiers: Dict[str, ResourcesServerRef] = Field(default_factory=dict)


class BaseSearchQueryRequest(BaseModel):
    query: str


class BaseGetPageContentRequest(BaseModel):
    url: str


class BaseGetPageContentResponse(BaseModel):
    page_content: str


class BaseGetSearchQueryResponse(BaseModel):
    search_results: str


class BraveSearchRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    expected_answer: str

    # [CUSTOM-VERIFIER-DISPATCH 2026-11-06] Optional per-row verifier selection -- see
    # BraveSearchResourcesServerConfig.default_verifier/verifiers. None (the default) preserves
    # today's behavior for all existing MCQA rows, which carry no such field.
    verifier_type: Optional[str] = None


class BraveSearchVerifyRequest(BraveSearchRunRequest, BaseVerifyRequest):
    pass


class BraveSearchVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    parsed_option: Optional[str] = None
    delegated_response: Optional[Dict[str, Any]] = None


def box_parser(output_str: str) -> Optional[str]:
    if not isinstance(output_str, str):
        output_str = str(output_str)

    if output_str is None:
        return None
    try:
        match = re.search(r"\\boxed\{(.*?)\}", output_str)
        parsed_option = match.group(1) if match else None
        return parsed_option
    except Exception as e:
        print(f"Regex error: {e}")
        return None


def _extract_last_assistant_text(body: BraveSearchVerifyRequest) -> Optional[str]:
    last_message = body.response.output[-1]
    if last_message.type == "message" and last_message.role == "assistant":
        return last_message.content
    else:
        return None


class BraveSearchResourcesServer(SimpleResourcesServer):
    config: BraveSearchResourcesServerConfig

    _mcp_exit_stack: Optional[AsyncExitStack] = None
    _mcp_session: Optional[ClientSession] = None
    _mcp_process: Optional[subprocess.Popen] = None
    _rate_limiter: Optional["RateLimiter"] = None

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        if self.config.rate_limiter_enabled:
            if not self.config.redis_host:
                raise ValueError("rate_limiter_enabled=true requires redis_host to be set.")
            RateLimiter = _load_rate_limiter_class()
            self._rate_limiter = RateLimiter(
                redis_host=self.config.redis_host,
                redis_port=self.config.redis_port,
                redis_pswd=self.config.redis_password,
                rate_limit_window_sec=self.config.rate_limit_window_sec,
                rate_limit_count=self.config.rate_limit_count,
                key="brave_search",
                raise_after=self.config.rate_limit_raise_after,
                max_wait_sec=self.config.rate_limit_max_wait_sec,
            )

        app.post("/search")(self.search)
        app.post("/browse")(self.browse)

        main_app_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan_wrapper(app):
            if self.config.brave_mcp_base_url is None:
                self._start_local_mcp_server()
            await self._connect_mcp_session()
            try:
                async with main_app_lifespan(app) as maybe_state:
                    yield maybe_state
            finally:
                await self._close_mcp_session()
                self._stop_local_mcp_server()

        app.router.lifespan_context = lifespan_wrapper

        return app

    def _mcp_url(self) -> str:
        if self.config.brave_mcp_base_url is not None:
            return self.config.brave_mcp_base_url
        return f"http://{self.config.brave_mcp_host}:{self.config.brave_mcp_port}/mcp"

    def _start_local_mcp_server(self) -> None:
        env = dict(os.environ)
        env["BRAVE_API_KEY"] = self.config.brave_api_key
        env["BRAVE_MCP_HOST"] = self.config.brave_mcp_host
        env["BRAVE_MCP_PORT"] = str(self.config.brave_mcp_port)
        # Node's global fetch (undici) does not honor HTTP_PROXY/HTTPS_PROXY on its own
        # (unlike Python's requests/httpx) -- confirmed via a real air-gapped test: MCP
        # session connect worked, but the server's own outbound call to Brave's API
        # failed with "fetch failed" until this hook patched in a ProxyAgent. No-ops if
        # no proxy env var is set. See proxy-hook.cjs and its sibling node_modules/undici
        # (installed by scripts/presync_baremetal_extras.sbatch in the outer repo).
        hook_path = Path(__file__).parent / "proxy-hook.cjs"
        node_options = f"--require {hook_path}"
        if env.get("NODE_OPTIONS"):
            node_options = f"{env['NODE_OPTIONS']} {node_options}"
        env["NODE_OPTIONS"] = node_options
        cmd = list(self.config.brave_mcp_command)
        logger.info(f"Starting Brave MCP server: {' '.join(cmd)}")
        # Don't pipe stdout/stderr so we can see output directly in logs (matches ns_tools).
        self._mcp_process = subprocess.Popen(cmd, env=env)
        self._wait_for_local_mcp_server_ready()

    def _wait_for_local_mcp_server_ready(self, timeout: float = 30.0, poll_interval: float = 0.5) -> None:
        url = self._mcp_url()
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self._mcp_process is not None and self._mcp_process.poll() is not None:
                raise RuntimeError(
                    f"Brave MCP server died during startup (exit code: {self._mcp_process.returncode}). "
                    "Check logs above for details."
                )
            try:
                # trust_env=False: this is a loopback call to our own just-spawned subprocess,
                # never meant to leave the machine. Without it, httpx inherits HTTP_PROXY/
                # HTTPS_PROXY when proxy.enabled=true and routes this through the forward proxy,
                # which rejects it (squidward's Safe_ports ACL only allows 80/443, and our local
                # MCP server binds an arbitrary port) -- confirmed via a real 403. NO_PROXY alone
                # doesn't fix this: httpx's own NO_PROXY parsing doesn't do CIDR-range matching
                # despite its docstring's claim, only exact IPs/"localhost" (confirmed
                # empirically), so an explicit opt-out here is the only reliable fix.
                with httpx.Client(timeout=2.0, trust_env=False) as client:
                    response = client.post(url, json={})
                logger.info(f"Brave MCP server is ready (status: {response.status_code})")
                return
            except (httpx.ConnectError, httpx.ConnectTimeout):
                time.sleep(poll_interval)
            except Exception as e:
                # Other errors might indicate server is up but returned an error - that's ok.
                logger.info(f"Brave MCP server responded with error (server is ready): {e}")
                return

        if self._mcp_process is not None and self._mcp_process.poll() is None:
            self._mcp_process.terminate()
            self._mcp_process.wait(timeout=5)
        raise TimeoutError(f"Brave MCP server did not start within {timeout}s. Check logs above for details.")

    def _stop_local_mcp_server(self) -> None:
        if self._mcp_process is None:
            return
        logger.info(f"Terminating Brave MCP server (PID: {self._mcp_process.pid})")
        self._mcp_process.terminate()
        try:
            self._mcp_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Brave MCP server did not terminate gracefully, killing...")
            self._mcp_process.kill()
        self._mcp_process = None

    @staticmethod
    def _mcp_http_client_factory(
        headers: Optional[dict] = None,
        timeout: Optional[httpx.Timeout] = None,
        auth: Optional[httpx.Auth] = None,
    ) -> httpx.AsyncClient:
        # trust_env=False: this session talks only to our own just-spawned local MCP server,
        # never meant to leave the machine -- see _wait_for_local_mcp_server_ready's own comment
        # on why this matters and why NO_PROXY alone doesn't fix it for httpx. Reimplements
        # create_mcp_http_client's own kwarg-building (can't just wrap it and mutate the result:
        # httpx.AsyncClient resolves env proxies into its transport mounts once, in __init__ --
        # trust_env is only a read-only property afterwards, setting it post-construction is a
        # no-op for the already-built mounts).
        kwargs: dict = {"trust_env": False}
        if timeout is not None:
            kwargs["timeout"] = timeout
        if headers is not None:
            kwargs["headers"] = headers
        if auth is not None:
            kwargs["auth"] = auth
        return httpx.AsyncClient(**kwargs)

    async def _connect_mcp_session(self) -> None:
        self._mcp_exit_stack = AsyncExitStack()
        read_stream, write_stream, _ = await self._mcp_exit_stack.enter_async_context(
            streamablehttp_client(self._mcp_url(), httpx_client_factory=self._mcp_http_client_factory)
        )
        session = await self._mcp_exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()
        self._mcp_session = session
        logger.info(f"Connected to Brave MCP server at {self._mcp_url()}")

    async def _close_mcp_session(self) -> None:
        if self._mcp_exit_stack is not None:
            await self._mcp_exit_stack.aclose()
            self._mcp_exit_stack = None
            self._mcp_session = None

    async def _call_search_tool(self, body: BaseSearchQueryRequest):
        return await self._mcp_session.call_tool(
            "brave_web_search", {"query": body.query, "count": self.config.max_results}
        )

    async def search(self, body: BaseSearchQueryRequest) -> BaseGetSearchQueryResponse:
        try:
            if self._rate_limiter is not None:
                await self._rate_limiter.acquire_lock()
            try:
                result = await self._call_search_tool(body)
            except Exception as e:
                # [CUSTOM-SEARCH-MCP-RECONNECT 2026-11-05] `_mcp_session` is connected once at
                # startup and kept alive for the whole process lifetime (see the module
                # docstring), with no reconnect logic -- confirmed directly against a real job's
                # rollout data that once this one long-lived session breaks (any transport-level
                # exception here), every subsequent search() call fails forever, for the rest of
                # the job. Since we never pinned down the exact exception that kills it, self-heal
                # regardless of cause: tear down the stale session/exit stack and reconnect once,
                # then retry the call exactly once before falling back to the generic error
                # response below. `_close_mcp_session()` must run first -- `_connect_mcp_session()`
                # just overwrites `_mcp_exit_stack` with a fresh `AsyncExitStack()` and would
                # otherwise leak the old streams/process pipes on every reconnect over a long job.
                logger.warning(
                    f"brave_search.search() call_tool failed: {type(e).__name__}: {e} -- "
                    "attempting one MCP session reconnect + retry."
                )
                await self._close_mcp_session()
                await self._connect_mcp_session()
                result = await self._call_search_tool(body)
            text = "\n".join(part.text for part in result.content if hasattr(part, "text"))
            if result.isError:
                return BaseGetSearchQueryResponse(search_results=f"Error: {text}")
            return BaseGetSearchQueryResponse(search_results=text)
        except Exception as e:
            # This used to fail silently into the response body only -- the agent would see
            # "Error: ..." as if it were a search result, but nothing in our own logs would show
            # it. A connection failure here means we've actually lost internet access,
            # log it more clearly.
            logger.error(f"brave_search.search() failed: {type(e).__name__}: {e}")
            return BaseGetSearchQueryResponse(search_results=f"Error: Unexpected error - {str(e)}")

    async def browse(self, body: BaseGetPageContentRequest) -> BaseGetPageContentResponse:
        # Brave's MCP server exposes no page-fetch tool, so this is unchanged from
        # google_search's browse() -- fetching/cleaning a page is orthogonal to which search
        # backend produced the URL.
        #
        # [CUSTOM-BROWSE-HTTPX-FETCH 2026-11-05] Was trafilatura.fetch_url(body.url) -- its own
        # urllib3-based downloader cannot use our plain HTTP forward proxy under any
        # configuration: confirmed directly on a real air-gapped compute node. trafilatura's
        # PROXY_URL detection (trafilatura/downloads.py) is gated behind
        # `from urllib3.contrib.socks import SOCKSProxyManager` succeeding (silently falls back
        # to None, i.e. no proxy at all, if the optional `pysocks` dependency isn't installed --
        # it isn't, here), and even with pysocks installed, SOCKSProxyManager only understands
        # socks4/socks5 URLs and raises "Unable to determine SOCKS version" against our plain
        # http://squid proxy URL. Fetching with httpx instead sidesteps this entirely --
        # confirmed working end-to-end on the same compute node (real 200s, real extracted
        # content) -- and reuses this module's own existing, already-reasoned exception to the
        # aiohttp mandate (see the module docstring: MCP tool calls, and now this fetch, happen
        # at most a few times per rollout turn, not at the 16k-concurrent-request scale that
        # mandate is about). trafilatura is still used for the network-free HTML-to-text
        # extraction step below, which is the part actually worth keeping it for.
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
                response = await client.get(body.url)
            html = response.text if response.status_code == 200 else None
            if html:
                text = trafilatura.extract(html)
                if text and len(text.split()) > 10000:
                    text = text[:5000] + "..." + text[-5000:]
                if text:
                    return BaseGetPageContentResponse(page_content=text)
                else:
                    return BaseGetPageContentResponse(page_content="No text found")
            else:
                return BaseGetPageContentResponse(page_content="No HTML found")
        except Exception as e:
            # [ADDED 2026-11-04] See the matching comment in search() -- trafilatura.fetch_url
            # is a genuine internet-bound call (through the proxy when enabled), so a failure
            # here is worth a loud log line, not just a string buried in the response body.
            logger.error(f"brave_search.browse() failed for url={body.url!r}: {type(e).__name__}: {e}")
            return BaseGetPageContentResponse(page_content=f"Error: Unexpected error = {str(e)}")

    async def verify(self, body: BraveSearchVerifyRequest) -> BraveSearchVerifyResponse:
        # [CUSTOM-VERIFIER-DISPATCH 2026-11-06] Mirrors ns_tools.app.py's verify() dispatch
        # (resources_servers/ns_tools/app.py) -- the default/box_parser path stays in-process
        # (no self-HTTP-call) since it's this server's own original logic, not a delegation
        # target; any other verifier_type (e.g. equivalence_llm_judge, for wikidata-triplet-qa's
        # free-text answers sharing these same search/browse tools) delegates over HTTP to the
        # resources server registered for it in `verifiers`, exactly like ns_tools delegates to
        # math_with_judge.
        verifier_type = body.verifier_type or self.config.default_verifier

        if verifier_type == self.config.default_verifier:
            expected_answer = body.expected_answer
            response_text = _extract_last_assistant_text(body)
            parsed_option = box_parser(response_text)
            reward = 1.0 if parsed_option == expected_answer else 0.0
            return BraveSearchVerifyResponse(**body.model_dump(), reward=reward, parsed_option=parsed_option)

        if verifier_type not in self.config.verifiers:
            raise ValueError(
                f"Unknown verifier: {verifier_type}. Configure it in 'verifiers' or check 'default_verifier'."
            )

        verifier_ref = self.config.verifiers[verifier_type]
        response = await self.server_client.post(
            server_name=verifier_ref.name,
            url_path="/verify",
            json=body.model_dump(),
        )
        result = await response.json()
        if "reward" not in result:
            raise ValueError(f"Verifier did not return 'reward' field. Response: {result}")

        return BraveSearchVerifyResponse(**body.model_dump(), reward=result["reward"], delegated_response=result)


if __name__ == "__main__":
    BraveSearchResourcesServer.run_webserver()
