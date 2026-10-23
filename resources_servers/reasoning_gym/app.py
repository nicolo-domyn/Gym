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
from typing import Any, Dict, List, Optional

import reasoning_gym
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from nemo_skills.mcp.tool_manager import ToolManager
from pydantic import Field
from reasoning_gym.utils import extract_answer

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.sandbox_launcher import ensure_sandbox
from nemo_gym.server_utils import SESSION_ID_KEY

logger = logging.getLogger(__name__)


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

    # Verbose logging for tool execution timing (disabled by default)
    verbose_tool_logging: bool = False

    # When True, skip replaying session history after sandbox worker restarts.
    disable_session_restore: bool = False


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
                )
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

        model_answer = self._extract_answer_from_response(body.response)

        task_name = body.metadata.get("source_dataset")

        if not task_name:
            raise ValueError(f"No task name found in metadata: {body.metadata}")

        entry = {
            "question": body.question,
            "answer": body.answer,
            "metadata": body.metadata,
        }
        try:
            score_fn = reasoning_gym.get_score_answer_fn(task_name)
            score = float(score_fn(answer=model_answer, entry=entry))
        except Exception as e:
            print(f"Error scoring answer for task {task_name}: {e}")
            score = 0.0

        return ReasoningGymVerifyResponse(
            **body.model_dump(),
            reward=score,
            task_name=task_name,
            score=score,
            extracted_answer=model_answer,
            **metrics,
        )

    def _extract_answer_from_response(self, response) -> str:
        assistant_responses = []
        for output_item in response.output:
            if output_item.type != "message":
                continue

            if isinstance(output_item.content, str):
                assistant_responses.append(output_item.content)
            else:
                for content_item in output_item.content:
                    if content_item.type != "output_text":
                        continue
                    assistant_responses.append(content_item.text)

        full_text = "".join(assistant_responses)

        # Try <answer> tags first (reasoning gym default)
        extracted = extract_answer(full_text, tag_name="answer")
        if extracted is not None:
            return extracted

        # Try \boxed{} if <answer> tags fail
        # this could be a slight instruction following issue, if model is prompted to use <answer> but uses boxed instead
        # found for deepseek-distill-qwen-1.5b it fails to use <answer> tags in favor of boxed, hence this fallback
        # may advise commenting this out for large models who follow instructions to use <answer> well
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", full_text)
        if boxed_match:
            return boxed_match.group(1).strip()

        # return full text if <answer> or \boxed{} fail
        return full_text.strip() if full_text.strip() else ""

    async def shutdown(self):
        """Cleanup resources on server shutdown."""
        if self.tool_manager:
            await self.tool_manager.shutdown()


if __name__ == "__main__":
    ReasoningGymResourcesServer.run_webserver()
