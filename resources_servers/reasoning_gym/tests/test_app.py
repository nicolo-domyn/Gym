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

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
import reasoning_gym

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.reasoning_gym.app import (
    ReasoningGymResourcesServer,
    ReasoningGymResourcesServerConfig,
    ReasoningGymVerifyRequest,
)


class _FakeRequest:
    """Minimal stand-in for fastapi.Request -- just what execute_tool()/verify() read."""

    def __init__(self, session: dict, body: dict):
        self.session = session
        self._body = body

    async def json(self):
        return self._body


class TestApp:
    @pytest.fixture
    def config(self) -> ReasoningGymResourcesServerConfig:
        return ReasoningGymResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )

    @pytest.fixture
    def server(self, config) -> ReasoningGymResourcesServer:
        return ReasoningGymResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def _create_response(self, text: str, msg_id: str = "test_msg") -> NeMoGymResponse:
        return NeMoGymResponse(
            id="test_response_id",
            created_at=1234567890.0,
            model="test_model",
            object="response",
            output=[
                NeMoGymResponseOutputMessage(
                    id=msg_id,
                    role="assistant",
                    type="message",
                    content=[
                        NeMoGymResponseOutputText(
                            type="output_text",
                            text=text,
                            annotations=[],
                        )
                    ],
                )
            ],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )

    @pytest.mark.asyncio
    async def test_reasoning_gym_verify_correct_answer(self, server):
        dataset = reasoning_gym.create_dataset("knights_knaves", size=1, seed=42)
        entry = dataset[0]

        response = self._create_response(text=entry["answer"], msg_id="test_msg_1")

        verify_request = ReasoningGymVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": entry["question"]}]
            ),
            response=response,
            question=entry["question"],
            answer=entry["answer"],
            metadata=entry["metadata"],
        )

        verify_response = await server.verify(None, verify_request)

        assert verify_response.reward >= 0.9, f"Expected high reward for correct answer, got {verify_response.reward}"

    @pytest.mark.asyncio
    async def test_reasoning_gym_verify_incorrect_answer(self, server):
        dataset = reasoning_gym.create_dataset("knights_knaves", size=1, seed=42)
        entry = dataset[0]

        response = self._create_response(text="This is completely wrong", msg_id="test_msg_2")

        verify_request = ReasoningGymVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": entry["question"]}]
            ),
            response=response,
            question=entry["question"],
            answer=entry["answer"],
            metadata=entry["metadata"],
        )

        verify_response = await server.verify(None, verify_request)

        assert verify_response.reward <= 0.1, f"Expected low reward for incorrect answer, got {verify_response.reward}"

    @pytest.mark.asyncio
    async def test_reasoning_gym_verify_multiple_tasks(self, server):
        tasks_to_test = ["knights_knaves", "leg_counting", "basic_arithmetic"]

        for task_name in tasks_to_test:
            dataset = reasoning_gym.create_dataset(task_name, size=1, seed=42)
            entry = dataset[0]

            response = self._create_response(text=entry["answer"], msg_id=f"test_msg_{task_name}")

            verify_request = ReasoningGymVerifyRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                    input=[{"role": "user", "content": entry["question"]}]
                ),
                response=response,
                question=entry["question"],
                answer=entry["answer"],
                metadata=entry["metadata"],
            )

            verify_response = await server.verify(None, verify_request)

            assert verify_response.reward >= 0.9, f"Task {task_name} failed: reward={verify_response.reward}"
            assert verify_response.task_name == task_name

    def test_config_with_tools(self) -> None:
        """Test that tool-related config fields are accepted and construct the server."""
        config = ReasoningGymResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="reasoning_gym",
            nemo_skills_tools=["nemo_skills.mcp.servers.python_tool::DirectPythonTool"],
            nemo_skills_tool_overrides={"DirectPythonTool": {"exec_timeout_s": 10}},
            sandbox_host="127.0.0.1",
            sandbox_port="6000",
        )
        server = ReasoningGymResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        assert server.config.nemo_skills_tools == ["nemo_skills.mcp.servers.python_tool::DirectPythonTool"]
        assert server.config.sandbox_host == "127.0.0.1"
        assert server.config.sandbox_port == "6000"

    @pytest.mark.asyncio
    async def test_execute_tool_dispatches_and_tracks_timing(self, server) -> None:
        """execute_tool() should dispatch to the ToolManager and record per-session timing."""
        server.tool_manager = MagicMock()
        server.tool_manager.execute_tool = AsyncMock(return_value=json.dumps({"process_status": "completed", "stdout": "4\n"}))
        server._tool_name_map = {"stateful_python_code_exec": "stateful_python_code_exec"}

        request = _FakeRequest(session={SESSION_ID_KEY: "test-session"}, body={"code": "print(2+2)"})
        response = await server.execute_tool("stateful_python_code_exec", request)

        assert json.loads(response.body)["process_status"] == "completed"
        server.tool_manager.execute_tool.assert_called_once_with(
            raw_name="stateful_python_code_exec",
            args={"code": "print(2+2)"},
            extra_args={"request_id": "test-session"},
        )
        assert "test-session" in server._timing_by_session
        assert len(server._timing_by_session["test-session"]) == 1
        assert server._timing_by_session["test-session"][0]["is_internal_timeout"] is False

    @pytest.mark.asyncio
    async def test_execute_tool_unknown_tool(self, server) -> None:
        """Requesting a tool name not in the loaded tool map should error, not crash."""
        server.tool_manager = MagicMock()
        server._tool_name_map = {}

        request = _FakeRequest(session={SESSION_ID_KEY: "test-session"}, body={})
        response = await server.execute_tool("not_a_real_tool", request)

        assert "error" in json.loads(response.body)

    @pytest.mark.asyncio
    async def test_execute_tool_records_internal_timeout(self, server) -> None:
        """A sandbox-reported timeout (process_status == 'timeout') should be tracked, not raised."""
        server.tool_manager = MagicMock()
        server.tool_manager.execute_tool = AsyncMock(
            return_value=json.dumps({"process_status": "timeout", "stdout": "", "stderr": "Client timed out\n"})
        )
        server._tool_name_map = {"stateful_python_code_exec": "stateful_python_code_exec"}

        request = _FakeRequest(session={SESSION_ID_KEY: "timeout-session"}, body={"code": "while True: pass"})
        response = await server.execute_tool("stateful_python_code_exec", request)

        assert json.loads(response.body)["process_status"] == "timeout"
        assert server._timing_by_session["timeout-session"][0]["is_internal_timeout"] is True
