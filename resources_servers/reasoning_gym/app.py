# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import logging
import re
import time
import uuid
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional

import reasoning_gym
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from nemo_skills.mcp.tool_manager import ToolManager
from pydantic import Field, model_validator

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.sandbox_launcher import ensure_sandbox
from nemo_gym.server_utils import SESSION_ID_KEY

logger = logging.getLogger(__name__)

# [CUSTOM-REASONING-GYM-SCORING 2026-11-27] Answer extraction and normalization helpers, see
# ReasoningGymResourcesServer.verify() for how they combine.
# Last <answer>...</answer> whose body contains no other "<answer>", so a literal mention of the
# tag earlier in the text ("I'll wrap it in <answer> tags: <answer>3</answer>") can't swallow it.
_ANSWER_TAG_RE = re.compile(r"<answer>((?:(?!<answer>).)*?)</answer>", re.DOTALL)
# Plain decimal numeral without leading zeros, so "0101" (a bit string) never parses as 101.
_PLAIN_DECIMAL_RE = re.compile(r"[+-]?(?:0|[1-9]\d*)(?:\.\d+)?")
_APPROX_PREFIX_RE = re.compile(r"^(?:≈|~|\\approx)\s*")
_EXACT_SCORE = 1.0 - 1e-9


class ReasoningGymResourcesServerConfig(BaseResourcesServerConfig):
    # NeMo Skills tool modules to load (e.g., "nemo_skills.mcp.servers.python_tool::DirectPythonTool")
    nemo_skills_tools: List[str] = Field(default_factory=list)

    # Per-tool overrides for nemo_skills tools
    nemo_skills_tool_overrides: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # Sandbox configuration for code execution tools. Independently configurable from
    # ns_tools's own sandbox_host/sandbox_port -- this resources server must work standalone
    # (no ns_tools in the same job). Set the same NEMO_SKILLS_SANDBOX_HOST/PORT values as
    # ns_tools's config when co-deploying to share one sandbox instance.
    sandbox_host: str = "127.0.0.1"
    sandbox_port: str = "6000"

    # If set, the sandbox is launched by nemo_gym.sandbox_launcher.ensure_sandbox() inside a
    # Singularity container instead of being assumed to already be running externally. Empty
    # (default) preserves the old behavior. When co-deployed with ns_tools pointed at the same
    # sandbox_host/sandbox_port, whichever resources server starts first wins the spawn race
    # (ensure_sandbox() is idempotent across processes -- see that module's docstring); set the
    # same sandbox_sif_path/sandbox_venv_path on both so either one can spawn it correctly.
    sandbox_sif_path: str = ""
    sandbox_venv_path: str = ""

    # [CUSTOM-REASONING-GYM-SANDBOX-PROXY 2026-10-23] local_sandbox_server.py was a bare
    # single-threaded `app.run(port=6000)` -- it served exactly one request at a time. 
    # Under concurrent load, excess connections got dropped with
    # "Server disconnected without sending a response" (httpcore.RemoteProtocolError)
    # instead of queued. Two fixes combined:
    # (1) sandbox_runtime/patch_local_sandbox_threaded.py makes the upstream server itself
    #     threaded, so it can actually serve many requests concurrently instead of just
    #     queuing them one at a time;
    # (2) sandbox_proxy_port routes traffic through sandbox_launcher.py's existing throttling
    #     proxy (_start_proxy) instead of the raw port, which turns any remaining overload
    #     into a clean, catchable 503 instead of a dropped connection.
    # (3) With the upstream no longer single-threaded, proxy_max_concurrency=16 is safe.
    sandbox_proxy_port: Optional[int] = 6001
    sandbox_proxy_max_concurrency: int = 16
    # [CUSTOM-REASONING-GYM-TIMEOUT 2026-11-27] How long the proxy waits for the sandbox (and for
    # a free slot). Must exceed every tool's exec_timeout_s: at or below it the proxy cuts off a
    # still-running execution and, since ReadTimeout is in its retry list, re-runs it from scratch.
    sandbox_proxy_request_timeout_s: float = 150.0

    # Verbose logging for tool execution timing (disabled by default)
    verbose_tool_logging: bool = False

    # When True, skip replaying session history after sandbox worker restarts.
    disable_session_restore: bool = False

    # [CUSTOM-REASONING-GYM-SCORING 2026-11-27] Agentic rollout audit (items #16/#32).
    # Only the last <answer> tag of the final assistant message is graded. Without a tag, a bare
    # final message or its last \boxed{} still counts, but only for full credit (many questions say
    # "respond with only your answer", which contradicts the system prompt's <answer> requirement).
    # True disables that fallback: no tag, no reward.
    strict_answer_tag: bool = False
    # reasoning_gym scorers hand out 0.01-0.05 for any (even unparseable) answer; rewards below
    # this are set to 0 so GRPO doesn't reward format alone. 0 disables it.
    min_reward: float = 0.06

    @model_validator(mode="after")
    def _check_proxy_timeout(self) -> "ReasoningGymResourcesServerConfig":
        if not (self.sandbox_sif_path and self.sandbox_proxy_port):
            return self
        for tool_name, overrides in self.nemo_skills_tool_overrides.items():
            exec_timeout_s = overrides.get("exec_timeout_s")
            if exec_timeout_s is not None and exec_timeout_s >= self.sandbox_proxy_request_timeout_s:
                raise ValueError(
                    f"{tool_name}.exec_timeout_s={exec_timeout_s} must be below "
                    f"sandbox_proxy_request_timeout_s={self.sandbox_proxy_request_timeout_s}"
                )
        return self


class ReasoningGymVerifyRequest(BaseVerifyRequest):
    question: str
    answer: Optional[str]
    metadata: dict[str, Any]


class ReasoningGymVerifyResponse(BaseVerifyResponse):
    task_name: str
    score: float
    extracted_answer: Optional[str]

    # Timing metrics for tool execution
    total_tool_execution_time_seconds: float = 0.0
    num_tool_calls: int = 0
    avg_tool_call_time_seconds: float = 0.0
    tool_timeout_count: int = 0  # Internal sandbox timeouts (process_status == "timeout")
    tool_request_timeout_count: int = 0  # HTTP/request-level timeouts


class ReasoningGymResourcesServer(SimpleResourcesServer):
    config: ReasoningGymResourcesServerConfig
    tool_manager: Optional[Any] = None
    _tool_name_map: Dict[str, str] = {}  # Maps tool names to qualified names
    _timing_by_session: Dict[str, list] = {}  # session_id -> list of timing records

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        # Initialize nemo_skills ToolManager if tools are configured
        if self.config.nemo_skills_tools:
            if self.config.sandbox_sif_path:
                # Isolate the sandbox in a Singularity container before anything tries to
                # reach it -- see nemo_gym/sandbox_launcher.py for the isolation rationale.
                # network="none": no legitimate reason for generated code in this sandbox to
                # reach the network.
                ensure_sandbox(
                    sif_path=self.config.sandbox_sif_path,
                    venv_path=self.config.sandbox_venv_path,
                    port=int(self.config.sandbox_port),
                    network="none",
                    proxy_port=self.config.sandbox_proxy_port,
                    proxy_max_concurrency=self.config.sandbox_proxy_max_concurrency,
                    proxy_request_timeout_s=self.config.sandbox_proxy_request_timeout_s,
                )
                # [CUSTOM-REASONING-GYM-SANDBOX-PROXY 2026-10-23] see sandbox_proxy_port's own
                # comment above -- this server's own tool calls (_initialize_nemo_skills_tools
                # below reads sandbox_port) must target the proxy's advertised port, not the raw
                # single-threaded sandbox port, once a proxy is in use. The background watchdog
                # started inside ensure_sandbox() always health-checks the raw port directly by
                # design (it needs to know if the underlying process itself is alive, independent
                # of the proxy) -- that part is unaffected and unchanged.
                if self.config.sandbox_proxy_port:
                    self.config.sandbox_port = str(self.config.sandbox_proxy_port)
            self._initialize_nemo_skills_tools()

            # Register a catch-all endpoint for tool execution
            # This handles any tool name dynamically
            app.post("/{tool_name}")(self.execute_tool)

        return app

    def _initialize_nemo_skills_tools(self):
        """Initialize the nemo_skills ToolManager with configured tools."""

        # Reduce verbosity of MCP and httpx loggers (they log every HTTP request at INFO)
        for noisy_logger in [
            "mcp.server.streamable_http_manager",
            "mcp.server.streamable_http",
            "mcp.server.lowlevel.server",
            "mcp.server",
            "mcp.client.streamable_http",
            "httpx",
        ]:
            logging.getLogger(noisy_logger).setLevel(logging.WARNING)

        logger.info(f"Initializing NeMo Skills ToolManager with tools: {self.config.nemo_skills_tools}")

        context = {
            "sandbox": {
                "sandbox_type": "local",
                "host": self.config.sandbox_host,
                "port": self.config.sandbox_port,
                "disable_session_restore": self.config.disable_session_restore,
            }
        }

        overrides = {
            tool_name: dict(tool_config) for tool_name, tool_config in self.config.nemo_skills_tool_overrides.items()
        }

        self.tool_manager = ToolManager(
            module_specs=self.config.nemo_skills_tools,
            overrides=overrides,
            context=context,
        )

        # Load tools and build name mapping
        async def _load_tools():
            tools = await self.tool_manager.list_all_tools()
            for tool in tools:
                self._tool_name_map[tool["name"]] = tool["name"]
            logger.info(f"Loaded {len(tools)} nemo_skills tools: {list(self._tool_name_map.keys())}")

        asyncio.get_event_loop().run_until_complete(_load_tools())
        logger.info("NeMo Skills ToolManager initialized successfully")

    async def execute_tool(self, tool_name: str, request: Request) -> PlainTextResponse:
        """
        Execute a nemo_skills tool by name.

        Uses the nemo-gym session ID as the request_id for stateful tools.
        Returns the result as plain text for simple_agent compatibility.
        Tracks execution timing and timeout detection per session.
        """
        if not self.tool_manager:
            return PlainTextResponse(json.dumps({"error": "No tools configured"}))

        # Check if tool is in our known tools
        if tool_name not in self._tool_name_map:
            logger.error(f"Unknown tool requested: {tool_name}")
            return PlainTextResponse(json.dumps({"error": f"Unknown tool: {tool_name}"}))

        # Get session ID for stateful execution
        session_id = request.session.get(SESSION_ID_KEY)
        if not session_id:
            session_id = str(uuid.uuid4())
            logger.warning(f"No session ID found, using fallback: {session_id}")

        if session_id not in self._timing_by_session:
            self._timing_by_session[session_id] = []

        start_time = time.perf_counter()
        is_internal_timeout = False
        is_request_timeout = False
        result = None

        try:
            body = await request.json()

            # Execute the tool
            result = await self.tool_manager.execute_tool(
                raw_name=tool_name,
                args=body,
                extra_args={"request_id": session_id},
            )

            # Check for internal sandbox timeout (process_status == "timeout")
            try:
                if isinstance(result, str):
                    result_dict = json.loads(result)
                elif isinstance(result, dict):
                    result_dict = result
                else:
                    result_dict = {}
                is_internal_timeout = result_dict.get("process_status") == "timeout"
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass

        except TimeoutError as e:
            is_request_timeout = True
            logger.warning(f"Request timeout executing tool {tool_name}: {e}")
            result = {"error": "Request timeout", "process_status": "timeout"}

        # [CUSTOM-REASONING-GYM-TOOLARGS 2026-10-24] A model omitting a required tool
        # argument (e.g. calling the python-exec tool with no "code" key) previously fell
        # through to the generic `except Exception` below: it was *not* an uncaught crash
        # (that branch already returns a clean response), but `str(KeyError('code'))` is
        # just `"'code'"` -- a cryptic, unhelpful message that gives the model no signal
        # about what argument was missing or that the fix is to include it. Logged at
        # `exception` level too, which is misleading severity for a routine malformed
        # tool call, not an internal bug. Give the model an actionable message instead, and
        # log at `warning` since this is an expected, recoverable model-output issue, not
        # unexpected server state.
        except KeyError as e:
            missing_arg = str(e).strip("'\"")
            logger.warning(f"Tool '{tool_name}' called without required argument '{missing_arg}': {e}")
            result = {"error": f"Missing required argument: '{missing_arg}'"}

        except Exception as e:
            logger.exception(f"Error executing tool {tool_name}: {e}")
            result = {"error": str(e)}

        elapsed = time.perf_counter() - start_time
        self._timing_by_session[session_id].append(
            {
                "tool_name": tool_name,
                "execution_time_seconds": elapsed,
                "is_internal_timeout": is_internal_timeout,
                "is_request_timeout": is_request_timeout,
            }
        )
        if self.config.verbose_tool_logging:
            timeout_info = ""
            if is_internal_timeout:
                timeout_info = " [INTERNAL_TIMEOUT]"
            elif is_request_timeout:
                timeout_info = " [REQUEST_TIMEOUT]"
            logger.info(f"Tool '{tool_name}' executed in {elapsed:.3f}s{timeout_info} (session={session_id[:8]}...)")

        # Return result as plain text to avoid double JSON serialization
        if isinstance(result, str):
            # [CUSTOM-REASONING-GYM-SURROGATE-ENCODE 2026-10-23] Model-executed code can
            # construct a Python str containing lone/unpaired UTF-16 surrogate code points
            # (e.g. a bad \uD8xx-style escape or buggy surrogate-pair arithmetic it wrote
            # itself) -- valid as an in-memory str, but PlainTextResponse.render()'s own
            # `content.encode(self.charset)` has no such tolerance and raises
            # UnicodeEncodeError, which was previously uncaught here and surfaced as an
            # unrelated HTTP 500 instead of the tool's real output. Round-tripping through
            # utf-8 with errors="replace" keeps the real output (garbled characters become
            # U+FFFD, not dropped) so the model still gets to see and reason about what it
            # actually produced, instead of a meaningless transport error.
            result = result.encode("utf-8", errors="replace").decode("utf-8")
            return PlainTextResponse(result)
        return PlainTextResponse(json.dumps(result))

    def _aggregate_timing_metrics(self, session_id: Optional[str]) -> Dict[str, Any]:
        """Aggregate tool execution timing metrics for a session."""
        tool_timings = self._timing_by_session.pop(session_id, []) if session_id else []

        total_tool_time = sum(t["execution_time_seconds"] for t in tool_timings)
        num_tool_calls = len(tool_timings)
        avg_tool_time = total_tool_time / num_tool_calls if num_tool_calls > 0 else 0.0
        tool_timeout_count = sum(1 for t in tool_timings if t.get("is_internal_timeout"))
        tool_request_timeout_count = sum(1 for t in tool_timings if t.get("is_request_timeout"))

        return {
            "total_tool_execution_time_seconds": total_tool_time,
            "num_tool_calls": num_tool_calls,
            "avg_tool_call_time_seconds": avg_tool_time,
            "tool_timeout_count": tool_timeout_count,
            "tool_request_timeout_count": tool_request_timeout_count,
        }

    async def verify(self, request: Request, body: ReasoningGymVerifyRequest) -> ReasoningGymVerifyResponse:
        """Uses reasoning gym verifier"""
        session_id = request.session.get(SESSION_ID_KEY) if request else None
        metrics = self._aggregate_timing_metrics(session_id)
        if self.config.verbose_tool_logging:
            logger.info(
                f"Session {session_id[:8] if session_id else 'unknown'}... metrics: "
                f"{metrics['num_tool_calls']} tool calls, total={metrics['total_tool_execution_time_seconds']:.3f}s, "
                f"avg={metrics['avg_tool_call_time_seconds']:.3f}s, "
                f"internal_timeouts={metrics['tool_timeout_count']}, request_timeouts={metrics['tool_request_timeout_count']}"
            )

        task_name = body.metadata.get("source_dataset")

        if not task_name:
            raise ValueError(f"No task name found in metadata: {body.metadata}")

        entry = {
            "question": body.question,
            "answer": body.answer,
            "metadata": body.metadata,
        }

        # [CUSTOM-REASONING-GYM-SCORING 2026-11-27] Previously all assistant messages were
        # concatenated and, with no <answer> tag (31% of audited rollouts), the whole text was
        # graded: the default scorer's substring credit plus the 0.01-0.05 floor many scorers give
        # to any answer rewarded format, not solving. Now only the final message counts, the
        # untagged fallback earns full credit or nothing, and floor credit is zeroed.
        final_text = self._final_message_text(body.response)
        model_answer = self._extract_tagged_answer(final_text)
        score = 0.0
        if model_answer is not None:
            model_answer = self._normalize_answer(model_answer, body.answer)
            score = self._score(task_name, model_answer, entry)
        elif final_text and not self.config.strict_answer_tag:
            for candidate in (final_text.strip(), self._last_boxed(final_text)):
                if not candidate:
                    continue
                candidate = self._normalize_answer(candidate, body.answer)
                candidate_score = self._score(task_name, candidate, entry)
                if candidate_score >= _EXACT_SCORE:
                    model_answer, score = candidate, candidate_score
                    break

        reward = score if score >= self.config.min_reward else 0.0

        return ReasoningGymVerifyResponse(
            **body.model_dump(),
            reward=reward,
            task_name=task_name,
            score=score,
            extracted_answer=model_answer,
            **metrics,
        )

    @staticmethod
    def _score(task_name: str, answer: str, entry: Dict[str, Any]) -> float:
        try:
            score_fn = reasoning_gym.get_score_answer_fn(task_name)
            return float(score_fn(answer=answer, entry=entry))
        except Exception as e:
            logger.warning(f"Error scoring answer for task {task_name}: {e}")
            return 0.0

    @staticmethod
    def _final_message_text(response) -> Optional[str]:
        """Text of the assistant message that ends the rollout.

        None when the rollout ended on a tool call or tool output (step cap, truncation): an
        answer written before further tool use is not the model's final answer.
        """
        for output_item in reversed(response.output):
            if output_item.type in ("function_call", "function_call_output"):
                return None
            if output_item.type != "message":
                continue
            if isinstance(output_item.content, str):
                return output_item.content
            return "".join(c.text for c in output_item.content if c.type == "output_text")
        return None

    @staticmethod
    def _extract_tagged_answer(text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        matches = _ANSWER_TAG_RE.findall(text)
        return matches[-1].strip() if matches else None

    @staticmethod
    def _last_boxed(text: str) -> Optional[str]:
        """Content of the last \\boxed{...}, with balanced braces (e.g. \\boxed{\\frac{1}{2}})."""
        start = text.rfind("\\boxed{")
        if start < 0:
            return None
        depth = 0
        for i in range(start + len("\\boxed"), len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start + len("\\boxed{") : i].strip()
        return None

    @staticmethod
    def _normalize_answer(answer: str, oracle: Optional[str]) -> str:
        """Fix formatting differences the reasoning_gym scorers grade as wrong.

        - a leading "≈"/"~"/"\\approx" (coin_flip's Fraction() parse rejects it -> 0);
        - a decimal numeral exactly equal to a decimal oracle ("83.0" vs "83": simple_geometry's
          default substring scorer gives 0.5) is replaced by the oracle string.
        """
        if not isinstance(oracle, str):
            return answer
        oracle = oracle.strip()
        stripped = _APPROX_PREFIX_RE.sub("", answer)
        if stripped != answer and not _APPROX_PREFIX_RE.match(oracle):
            answer = stripped
        if _PLAIN_DECIMAL_RE.fullmatch(answer) and _PLAIN_DECIMAL_RE.fullmatch(oracle):
            try:
                if Decimal(answer) == Decimal(oracle):
                    return oracle
            except InvalidOperation:
                pass
        return answer

    async def shutdown(self):
        """Cleanup resources on server shutdown."""
        if self.tool_manager:
            await self.tool_manager.shutdown()


if __name__ == "__main__":
    ReasoningGymResourcesServer.run_webserver()
