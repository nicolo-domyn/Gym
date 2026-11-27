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
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
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


def _message(text: str, msg_id: str = "msg") -> NeMoGymResponseOutputMessage:
    return NeMoGymResponseOutputMessage(
        id=msg_id, content=[NeMoGymResponseOutputText(text=text, annotations=[])]
    )


def _tool_call(call_id: str = "call_1") -> NeMoGymResponseFunctionToolCall:
    return NeMoGymResponseFunctionToolCall(
        arguments='{"code": "print(1)"}', call_id=call_id, name="stateful_python_code_exec"
    )


def _tool_output(call_id: str = "call_1") -> NeMoGymFunctionCallOutput:
    return NeMoGymFunctionCallOutput(call_id=call_id, output='{"process_status": "completed", "stdout": "1"}')


class TestScoring:
    """[CUSTOM-REASONING-GYM-SCORING 2026-11-27] agentic rollout audit items #16 and #32."""

    def _server(self, **config_kwargs) -> ReasoningGymResourcesServer:
        config = ReasoningGymResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="", **config_kwargs)
        return ReasoningGymResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    async def _verify(self, server, task_name: str, output: list, entry: dict | None = None, seed: int = 42):
        if entry is None:
            entry = reasoning_gym.create_dataset(task_name, size=1, seed=seed)[0]
        response = NeMoGymResponse(
            id="resp", created_at=0.0, model="m", object="response", output=output,
            parallel_tool_calls=False, tool_choice="auto", tools=[],
        )
        request = ReasoningGymVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": entry["question"]}]
            ),
            response=response,
            question=entry["question"],
            answer=entry["answer"],
            metadata=entry["metadata"],
        )
        return await server.verify(None, request), entry

    def _entry(self, task_name: str, seed: int = 42) -> dict:
        return reasoning_gym.create_dataset(task_name, size=1, seed=seed)[0]

    @pytest.mark.asyncio
    async def test_tagged_answer_in_final_message(self) -> None:
        entry = self._entry("leg_counting")
        output = [_tool_call(), _tool_output(), _message(f"Counting legs... <answer>{entry['answer']}</answer>")]
        result, _ = await self._verify(self._server(), "leg_counting", output, entry)
        assert result.reward == 1.0
        assert result.extracted_answer == entry["answer"]

    @pytest.mark.asyncio
    async def test_last_tag_wins_and_literal_tag_mention_ignored(self) -> None:
        entry = self._entry("leg_counting")
        text = f"I'll wrap it in <answer> tags. Draft <answer>999</answer>, final <answer>{entry['answer']}</answer>"
        result, _ = await self._verify(self._server(), "leg_counting", [_message(text)], entry)
        assert result.extracted_answer == entry["answer"]
        assert result.reward == 1.0

    @pytest.mark.asyncio
    async def test_answer_before_more_tool_use_is_not_final(self) -> None:
        """A tagged answer followed by more tool calls (rollout cut by the step cap) scores 0."""
        entry = self._entry("leg_counting")
        output = [_message(f"<answer>{entry['answer']}</answer>"), _tool_call(), _tool_output()]
        result, _ = await self._verify(self._server(), "leg_counting", output, entry)
        assert result.reward == 0.0
        assert result.extracted_answer is None

    @pytest.mark.asyncio
    async def test_earlier_message_tag_not_used(self) -> None:
        entry = self._entry("leg_counting")
        output = [
            _message(f"Maybe <answer>{entry['answer']}</answer>", msg_id="m1"),
            _tool_call(),
            _tool_output(),
            _message("I am not sure anymore.", msg_id="m2"),
        ]
        result, _ = await self._verify(self._server(), "leg_counting", output, entry)
        assert result.reward == 0.0

    @pytest.mark.asyncio
    async def test_bare_exact_answer_gets_full_credit(self) -> None:
        """Questions often say "respond with only your answer": a bare exact answer still counts."""
        entry = self._entry("leg_counting")
        result, _ = await self._verify(self._server(), "leg_counting", [_message(f"\n\n{entry['answer']}")], entry)
        assert result.reward == 1.0

    @pytest.mark.asyncio
    async def test_bare_answer_rejected_in_strict_mode(self) -> None:
        entry = self._entry("leg_counting")
        server = self._server(strict_answer_tag=True)
        result, _ = await self._verify(server, "leg_counting", [_message(entry["answer"])], entry)
        assert result.reward == 0.0

    @pytest.mark.asyncio
    async def test_untagged_long_text_gets_no_substring_credit(self) -> None:
        """Old behaviour: whole text graded, default scorer gave len(oracle)/len(text) > 0."""
        entry = self._entry("leg_counting")
        text = f"After counting every animal carefully, the total number of legs is {entry['answer']}."
        result, _ = await self._verify(self._server(), "leg_counting", [_message(text)], entry)
        assert result.reward == 0.0

    @pytest.mark.asyncio
    async def test_untagged_last_balanced_boxed(self) -> None:
        entry = self._entry("leg_counting")
        text = f"First guess \\boxed{{1}}. Final: \\boxed{{{entry['answer']}}}"
        result, _ = await self._verify(self._server(), "leg_counting", [_message(text)], entry)
        assert result.reward == 1.0

    def test_last_boxed_handles_nested_braces(self) -> None:
        assert ReasoningGymResourcesServer._last_boxed("x \\boxed{a} y \\boxed{\\frac{1}{2}} z") == "\\frac{1}{2}"
        assert ReasoningGymResourcesServer._last_boxed("\\boxed{unclosed") is None
        assert ReasoningGymResourcesServer._last_boxed("no box") is None

    @pytest.mark.asyncio
    async def test_floor_credit_zeroed(self) -> None:
        """puzzle24 gives 0.01 to any answer, even garbage: reward must be 0, score keeps the raw value."""
        result, _ = await self._verify(self._server(), "puzzle24", [_message("<answer>banana</answer>")])
        assert result.score == pytest.approx(0.01)
        assert result.reward == 0.0

    @pytest.mark.asyncio
    async def test_floor_threshold_can_be_disabled(self) -> None:
        result, _ = await self._verify(self._server(min_reward=0.0), "puzzle24", [_message("<answer>banana</answer>")])
        assert result.reward == pytest.approx(0.01)

    @pytest.mark.asyncio
    async def test_no_final_message_scores_zero(self) -> None:
        result, _ = await self._verify(self._server(), "puzzle24", [_tool_call(), _tool_output()])
        assert result.reward == 0.0
        assert result.extracted_answer is None

    @pytest.mark.asyncio
    async def test_simple_geometry_trailing_zero(self) -> None:
        entry = self._entry("simple_geometry", seed=1)
        assert "." not in entry["answer"]
        result, _ = await self._verify(
            self._server(), "simple_geometry", [_message(f"<answer>{entry['answer']}.0</answer>")], entry
        )
        assert result.reward == 1.0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("prefix", ["≈", "≈ ", "~", "\\approx "])
    async def test_coin_flip_approx_prefix(self, prefix: str) -> None:
        entry = self._entry("coin_flip", seed=1)
        result, _ = await self._verify(
            self._server(), "coin_flip", [_message(f"<answer>{prefix}{entry['answer']}</answer>")], entry
        )
        assert result.reward == 1.0

    @pytest.mark.parametrize(
        "answer,oracle,expected",
        [
            ("83.0", "83", "83"),
            ("-4.50", "-4.5", "-4.5"),
            ("0101", "101", "0101"),  # bit strings / zero-padded values are not numbers
            ("83.1", "83", "83.1"),
            ("≈0.25", "0.25", "0.25"),
            ("~x", "~x", "~x"),  # oracle itself starts with the prefix: leave it alone
            ("A", None, "A"),
        ],
    )
    def test_normalize_answer(self, answer, oracle, expected) -> None:
        assert ReasoningGymResourcesServer._normalize_answer(answer, oracle) == expected


class TestProxyTimeoutConfig:
    """[CUSTOM-REASONING-GYM-TIMEOUT 2026-11-27]"""

    def _config(self, exec_timeout_s, **kwargs) -> ReasoningGymResourcesServerConfig:
        return ReasoningGymResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            nemo_skills_tools=["nemo_skills.mcp.servers.python_tool::DirectPythonTool"],
            nemo_skills_tool_overrides={"DirectPythonTool": {"exec_timeout_s": exec_timeout_s}},
            **kwargs,
        )

    def test_exec_timeout_at_or_above_proxy_timeout_rejected(self) -> None:
        with pytest.raises(ValueError, match="sandbox_proxy_request_timeout_s"):
            self._config(150, sandbox_sif_path="/x.sif", sandbox_proxy_request_timeout_s=150)

    def test_exec_timeout_below_proxy_timeout_accepted(self) -> None:
        config = self._config(120, sandbox_sif_path="/x.sif")
        assert config.sandbox_proxy_request_timeout_s > 120

    def test_check_skipped_without_launched_sandbox(self) -> None:
        """No sif path -> no proxy is started by this server, so nothing to check."""
        self._config(500)
