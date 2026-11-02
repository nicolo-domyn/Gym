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

import logging
import os
import re
import subprocess
import time
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import List, Optional

import httpx
import trafilatura
from fastapi import FastAPI
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from pydantic import BaseModel, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from rate_limiter import RateLimiter


logger = logging.getLogger(__name__)


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


class BaseSearchQueryRequest(BaseModel):
    query: str


class BaseGetPageContentRequest(BaseModel):
    url: str


class BaseGetPageContentResponse(BaseModel):
    page_content: str


class BaseGetSearchQueryResponse(BaseModel):
    search_results: str


class BraveSearchRunRequest(BaseRunRequest):
    expected_answer: str


class BraveSearchVerifyRequest(BraveSearchRunRequest, BaseVerifyRequest):
    pass


class BraveSearchVerifyResponse(BaseVerifyResponse):
    parsed_option: Optional[str] = None


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
    _rate_limiter: Optional[RateLimiter] = None

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        if self.config.rate_limiter_enabled:
            if not self.config.redis_host:
                raise ValueError("rate_limiter_enabled=true requires redis_host to be set.")
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
                with httpx.Client(timeout=2.0) as client:
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

    async def _connect_mcp_session(self) -> None:
        self._mcp_exit_stack = AsyncExitStack()
        read_stream, write_stream, _ = await self._mcp_exit_stack.enter_async_context(
            streamablehttp_client(self._mcp_url())
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

    async def search(self, body: BaseSearchQueryRequest) -> BaseGetSearchQueryResponse:
        try:
            if self._rate_limiter is not None:
                await self._rate_limiter.acquire_lock()
            result = await self._mcp_session.call_tool(
                "brave_web_search", {"query": body.query, "count": self.config.max_results}
            )
            text = "\n".join(part.text for part in result.content if hasattr(part, "text"))
            if result.isError:
                return BaseGetSearchQueryResponse(search_results=f"Error: {text}")
            return BaseGetSearchQueryResponse(search_results=text)
        except Exception as e:
            return BaseGetSearchQueryResponse(search_results=f"Error: Unexpected error - {str(e)}")

    async def browse(self, body: BaseGetPageContentRequest) -> BaseGetPageContentResponse:
        # Brave's MCP server exposes no page-fetch tool, so this is unchanged from
        # google_search's browse() -- fetching/cleaning a page is orthogonal to which search
        # backend produced the URL.
        try:
            html = trafilatura.fetch_url(body.url)
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
            return BaseGetPageContentResponse(page_content=f"Error: Unexpected error = {str(e)}")

    async def verify(self, body: BraveSearchVerifyRequest) -> BraveSearchVerifyResponse:
        expected_answer = body.expected_answer
        response_text = _extract_last_assistant_text(body)
        parsed_option = box_parser(response_text)
        if parsed_option == expected_answer:
            reward = 1.0
        else:
            reward = 0.0
        return BraveSearchVerifyResponse(**body.model_dump(), reward=reward, parsed_option=parsed_option)


if __name__ == "__main__":
    BraveSearchResourcesServer.run_webserver()
