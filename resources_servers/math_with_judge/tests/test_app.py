# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import json
import multiprocessing as mp
from copy import deepcopy
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

from math_verify.errors import TimeoutException
from pydantic import ValidationError
from pytest import approx, fixture, raises, skip

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputRefusal,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.math_with_judge.app import (
    AnswerFormatRule,
    JudgeEvaluation,
    LibraryJudgeMathResourcesServer,
    LibraryJudgeMathResourcesServerConfig,
    LibraryJudgeMathVerifyRequest,
    _run_math_verify,
)


def _geometric_series_terms(num_terms: int) -> str:
    return "+".join("1" if exponent == 0 else f"x^{exponent}" for exponent in range(num_terms))


def _verify_pathological_sympy_expression(result_connection: Any) -> None:
    config = LibraryJudgeMathResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        should_use_judge=False,
        library_verifier_timeout_seconds=1.0,
        judge_model_server=ModelServerRef(
            type="responses_api_models",
            name="math_judge",
        ),
        judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
    )
    resources_server = LibraryJudgeMathResourcesServer(
        config=config,
        server_client=MagicMock(spec=ServerClient),
    )
    generated_answer = "\\boxed{\\frac{x^{800}-1}{x-1}-(" + _geometric_series_terms(800) + ")}"

    try:
        result_connection.send(asyncio.run(resources_server._verify_answer("question", "0", generated_answer)))
    except BaseException as e:
        result_connection.send(e)
    finally:
        result_connection.close()


def _sleeping_library_verifier_process(result_connection: Any) -> None:
    import time

    try:
        time.sleep(60)
    finally:
        result_connection.close()


class TestApp:
    @fixture
    def config(self) -> LibraryJudgeMathResourcesServerConfig:
        return LibraryJudgeMathResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            judge_model_server=ModelServerRef(
                type="responses_api_models",
                name="math_judge",
            ),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )

    def _create_response(self, id: str, output_item: NeMoGymResponseOutputItem) -> dict[str, Any]:
        return NeMoGymResponse(
            id=id,
            created_at=1234.5,
            model="response_model",
            object="response",
            output=[output_item],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        ).model_dump()

    def _check_judge_evaluation(
        self,
        judge_evaluation: JudgeEvaluation,
        question: str,
        first_answer: str,
        second_answer: str,
        expected_create_params: dict[str, Any],
        expected_response_id: str,
        expected_output_item: NeMoGymResponseOutputItem,
    ) -> None:
        expected_prompt = LibraryJudgeMathResourcesServer.JUDGE_PROMPT_TEMPLATE.format(
            question=question, first_answer=first_answer, second_answer=second_answer
        )
        assert judge_evaluation.responses_create_params == NeMoGymResponseCreateParamsNonStreaming(
            **expected_create_params,
            input=[
                {
                    "role": "system",
                    "content": LibraryJudgeMathResourcesServer.JUDGE_SYSTEM_MESSAGE,
                },
                {
                    "role": "user",
                    "content": expected_prompt,
                },
            ],
        )

        actual_response = judge_evaluation.response
        assert actual_response.output == [expected_output_item]
        response_map = actual_response.model_dump(exclude_none=True)
        assert response_map.pop("id") == expected_response_id
        assert response_map.pop("created_at") == approx(1234.5)
        assert response_map.pop("model") == "response_model"
        assert response_map.pop("object") == "response"
        assert response_map.pop("parallel_tool_calls") is False
        assert response_map.pop("tool_choice") == "none"
        assert response_map.pop("tools") == []
        assert list(response_map) == ["output"]

    def _create_response_output_message(self, message_text: str) -> NeMoGymResponseOutputMessage:
        return NeMoGymResponseOutputMessage(
            id=f"ID for {message_text}",
            content=[NeMoGymResponseOutputText(annotations=[], text=message_text, type="output_text")],
            role="assistant",
            status="in_progress",
            type="message",
        )

    async def test_verify(self, config: LibraryJudgeMathResourcesServerConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        resources_server = LibraryJudgeMathResourcesServer(config=config, server_client=server_mock)
        response_mock = AsyncMock()
        post_mock = MagicMock()
        post_mock.read = response_mock
        server_mock.post = AsyncMock(return_value=post_mock)
        not_equal_item = self._create_response_output_message(
            f"{LibraryJudgeMathResourcesServer.JUDGE_NOT_EQUAL_LABEL} done"
        )
        response_mock.return_value = json.dumps(self._create_response("verify_not_equal_id", not_equal_item))

        question = "Simplify the expression x + 7 - 6"
        expected_answer = "x + 1"
        model_create_params = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {
                    "role": "user",
                    "content": question,
                }
            ]
        )
        first_part = "$1"
        first_part_item = self._create_response_output_message(first_part)
        first_model_response = NeMoGymResponse(**self._create_response("first_part_id", first_part_item))
        not_equal_verify_request = LibraryJudgeMathVerifyRequest(
            responses_create_params=deepcopy(model_create_params),
            response=first_model_response.model_copy(deep=True),
            question=question,
            expected_answer=expected_answer,
        )
        not_equal_verify_response = await resources_server.verify(not_equal_verify_request)
        assert not_equal_verify_response.responses_create_params == model_create_params
        assert not_equal_verify_response.response == first_model_response
        assert not_equal_verify_response.reward == approx(0.0)
        assert not_equal_verify_response.expected_answer == expected_answer
        assert not_equal_verify_response.extracted_answer == "1"
        assert not_equal_verify_response.library_reward == approx(0.0)
        # No system prompt, so no format is required.
        assert not_equal_verify_response.format_ok is None
        assert not_equal_verify_response.required_answer_format is None
        judge_evaluations = not_equal_verify_response.judge_evaluations
        assert len(judge_evaluations) == 1
        self._check_judge_evaluation(
            judge_evaluations[0],
            question,
            expected_answer,
            not_equal_verify_response.extracted_answer,
            {},
            "verify_not_equal_id",
            not_equal_item,
        )
        assert sorted(list(not_equal_verify_response.model_dump())) == [
            "expected_answer",
            "extracted_answer",
             "format_ok",
            "judge_evaluations",
            "library_reward",
            "required_answer_format",
            "response",
            "responses_create_params",
            "reward",
        ]

        second_model_response = first_model_response.model_copy(deep=True)
        second_model_response.output = second_model_response.output + [
            NeMoGymResponseReasoningItem(id="extra_reasoning", summary=[], type="reasoning"),
            self._create_response_output_message(" + x$"),
            NeMoGymResponseOutputMessage(
                id="refusal_finish",
                content=[
                    NeMoGymResponseOutputRefusal(refusal="no response", type="refusal"),
                ],
                role="assistant",
                status="completed",
                type="message",
            ),
        ]
        equal_verify_request = LibraryJudgeMathVerifyRequest(
            responses_create_params=deepcopy(model_create_params),
            response=second_model_response.model_copy(deep=True),
            question=question,
            expected_answer=expected_answer,
        )
        equal_verify_response = await resources_server.verify(equal_verify_request)
        assert equal_verify_response.responses_create_params == model_create_params
        assert equal_verify_response.response == second_model_response
        assert equal_verify_response.reward == approx(1.0)
        assert equal_verify_response.expected_answer == expected_answer
        assert equal_verify_response.extracted_answer == "x + 1"
        assert equal_verify_response.library_reward == approx(1.0)
        assert equal_verify_response.judge_evaluations is None
        assert equal_verify_response.format_ok is None
        assert equal_verify_response.required_answer_format is None
        assert sorted(list(equal_verify_response.model_dump())) == [
            "expected_answer",
            "extracted_answer",
            "format_ok",
            "judge_evaluations",
            "library_reward",
            "required_answer_format",
            "response",
            "responses_create_params",
            "reward",
        ]

    BOXED_SYSTEM_PROMPT = "Write your final answer within \\boxed{}."
    ANSWER_LINE_SYSTEM_PROMPT = (
        'Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) '
        "where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command."
    )

    async def _verify_text(
        self,
        server: LibraryJudgeMathResourcesServer,
        text: str,
        expected_answer: str,
        system_prompt: Optional[str] = None,
    ):
        question = "What is the answer?"
        messages = [{"role": "user", "content": question}]
        if system_prompt is not None:
            messages.insert(0, {"role": "system", "content": system_prompt})
        output_item = self._create_response_output_message(text)
        return await server.verify(
            LibraryJudgeMathVerifyRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=messages),
                response=NeMoGymResponse(**self._create_response("format_id", output_item)),
                question=question,
                expected_answer=expected_answer,
            )
        )

    def test_has_answer_format(self, config: LibraryJudgeMathResourcesServerConfig) -> None:
        resources_server = LibraryJudgeMathResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        # (response text, has a closed \boxed{}, has an ANSWER: line)
        cases = [
            ("So the result is \\boxed{5}.", True, False),
            ("$\\boxed{\\frac{1}{2}}$", True, False),
            ("\\fbox{5}", True, False),
            ("\\boxed {x^{2}}", True, False),
            ("Unclosed \\boxed{5 then \\boxed{7}", True, False),
            ("\\boxed{\\frac{1}{2}", False, False),
            ("Work.\n\nANSWER: 5", False, True),
            ("answer: 5", False, True),
            ("**Answer:** 5", False, True),
            ("Final answer: 5", False, True),
            ("> ANSWER: $5$", False, True),
            ("\\boxed{3}\n\nANSWER: 5", True, True),
            ("", False, False),
            ("We get 2 + 3 = 5", False, False),
            ("The answer is 5.", False, False),
            ("The final answer is $5$.", False, False),
            ("Let me check my answer: 2 + 2 = 4", False, False),
        ]
        for text, has_boxed, has_answer_line in cases:
            assert resources_server._has_answer_format(text, "boxed") is has_boxed, text
            assert resources_server._has_answer_format(text, "answer_line") is has_answer_line, text

    def test_required_answer_format(self, config: LibraryJudgeMathResourcesServerConfig) -> None:
        resources_server = LibraryJudgeMathResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        user_message = {"role": "user", "content": "What is 2 + 3?"}

        def required_answer_format(input: Any, instructions: Optional[str] = None) -> Optional[str]:
            params = NeMoGymResponseCreateParamsNonStreaming(input=input, instructions=instructions)
            return resources_server._required_answer_format(params)

        boxed_system = {"role": "system", "content": self.BOXED_SYSTEM_PROMPT}
        assert required_answer_format([boxed_system, user_message]) == "boxed"
        # The ANSWER: rule comes first: this prompt also mentions \boxed.
        answer_line_system = {"role": "system", "content": self.ANSWER_LINE_SYSTEM_PROMPT}
        assert required_answer_format([answer_line_system, user_message]) == "answer_line"
        assert required_answer_format([{"role": "developer", "content": self.BOXED_SYSTEM_PROMPT}]) == "boxed"
        content_list_system = {
            "role": "system",
            "content": [{"type": "input_text", "text": self.BOXED_SYSTEM_PROMPT}],
        }
        assert required_answer_format([content_list_system, user_message]) == "boxed"
        # `instructions` is not sent to the model by vllm_model, so it does not count.
        assert required_answer_format([user_message], instructions=self.ANSWER_LINE_SYSTEM_PROMPT) is None
        # Only the system prompt counts, not a format asked for in the user message.
        assert (
            required_answer_format([{"role": "user", "content": "What is 2 + 3? " + self.BOXED_SYSTEM_PROMPT}]) is None
        )
        # A prompt with neither text has no format check.
        assert required_answer_format([{"role": "system", "content": "You are a helpful assistant."}]) is None
        assert required_answer_format([{"role": "system", "content": "End with a line ANSWER: <value>."}]) is None
        assert required_answer_format("What is 2 + 3?") is None

        custom_rules_config = config.model_copy(
            update={"answer_format_rules": [AnswerFormatRule(prompt_text="\\boxed{}", answer_format="boxed")]}
        )
        custom_rules_server = LibraryJudgeMathResourcesServer(
            config=custom_rules_config, server_client=MagicMock(spec=ServerClient)
        )
        params = NeMoGymResponseCreateParamsNonStreaming(input=[answer_line_system, user_message])
        assert custom_rules_server._required_answer_format(params) is None
        params = NeMoGymResponseCreateParamsNonStreaming(input=[boxed_system, user_message])
        assert custom_rules_server._required_answer_format(params) == "boxed"

        with raises(ValidationError):
            LibraryJudgeMathResourcesServerConfig(**(config.model_dump() | {"format_penalty": 1.5}))
        with raises(ValidationError):
            AnswerFormatRule(prompt_text="ANSWER: $ANSWER", answer_format="latex")

    def test_answer_line_extraction(self) -> None:
        # (response text, last ANSWER: line content, normalized content)
        cases = [
            ("ANSWER: \\sqrt{2}", "\\sqrt{2}", "\\sqrt{2}"),
            ("**Answer:** $x^2 + 1$", "$x^2 + 1$", "x^2 + 1"),
            ("**Final Answer:**\nThere are **81** days.", "There are **81** days.", "There are **81** days"),
            ("ANSWER: \\(5\\).", "\\(5\\).", "5"),
            ("ANSWER: $$\\frac{1}{2}$$", "$$\\frac{1}{2}$$", "\\frac{1}{2}"),
            ("ANSWER: **5**", "5**", "5"),
            ("ANSWER: 0.5", "0.5", "0.5"),
            ("ANSWER: \\frac{1}{2}\\,", "\\frac{1}{2}\\,", "\\frac{1}{2}\\,"),
            ("ANSWER: $3$ and $5$", "$3$ and $5$", "$3$ and $5$"),
            ("ANSWER: D<|im_end|>", "D<|im_end|>", "D"),
            ("ANSWER: 5 <|endoftext|><|im_end|>", "5 <|endoftext|><|im_end|>", "5"),
            # Only token-like names at the end of the line are special tokens, not math.
            ("ANSWER: 0<|x|>1", "0<|x|>1", "0<|x|>1"),
            ("ANSWER: E = <|x|>", "E = <|x|>", "E = <|x|>"),
            ("ANSWER: <|x - 1|>", "<|x - 1|>", "<|x - 1|>"),
            ("ANSWER: <|im_start|>5", "<|im_start|>5", "<|im_start|>5"),
            ("ANSWER:\n\n2\\sqrt{2}\nDone.", "2\\sqrt{2}", "2\\sqrt{2}"),
            ("ANSWER: 1\nWait, no.\nANSWER: 2", "2", "2"),
            ("ANSWER:", "", ""),
            # Display math opened on (or after) the ANSWER: line is read up to its closing delimiter.
            ("ANSWER:\n\\[\n1280\n\\]\nDone.", "\\[ 1280 \\]", "1280"),
            ("ANSWER: $$\n\\frac{1}{2}\n$$", "$$ \\frac{1}{2} $$", "\\frac{1}{2}"),
            # Unicode math and \big-style size modifiers are rewritten as LaTeX.
            ("ANSWER: 2√5/5**", "2√5/5**", "2\\sqrt{5}/5"),
            ("ANSWER: $√14$", "$√14$", "\\sqrt{14}"),
            ("ANSWER: √(x+1)", "√(x+1)", "\\sqrt{x+1}"),
            ("ANSWER: −3π/4", "−3π/4", "-3\\pi /4"),
            (
                "ANSWER: \\big\\lfloor \\frac{2n}{3} \\big\\rfloor",
                "\\big\\lfloor \\frac{2n}{3} \\big\\rfloor",
                "\\lfloor \\frac{2n}{3} \\rfloor",
            ),
            ("ANSWER: \\bigcup_i A_i", "\\bigcup_i A_i", "\\bigcup_i A_i"),
            # A second "ANSWER:", also inside math delimiters, backticks or \text{...}.
            ("**Final answer:**\n\\(ANSWER: \\frac18\\)", "\\(ANSWER: \\frac18\\)", "\\frac18"),
            ("Answer:\n`ANSWER: 1,2,3,4`", "`ANSWER: 1,2,3,4`", "1,2,3,4"),
            ("ANSWER: \\(\\displaystyle\\text{ANSWER: 168}\\)", "\\(\\displaystyle\\text{ANSWER: 168}\\)", "168"),
            ("ANSWER: \\text{No}", "\\text{No}", "\\text{No}"),
            # A math delimiter whose pair is missing, but not an escaped dollar.
            ("ANSWER: 1+3\\sqrt{2}\\)", "1+3\\sqrt{2}\\)", "1+3\\sqrt{2}"),
            ("ANSWER: \\$4.34", "\\$4.34", "\\$4.34"),
        ]
        for text, content, normalized in cases:
            assert LibraryJudgeMathResourcesServer._extract_answer_line(text) == content, text
            assert LibraryJudgeMathResourcesServer._normalize_answer_line(content) == normalized, text

    async def test_verify_format_penalty(self, config: LibraryJudgeMathResourcesServerConfig) -> None:
        no_judge_config = config.model_copy(update={"should_use_judge": False})
        server = LibraryJudgeMathResourcesServer(config=no_judge_config, server_client=MagicMock(spec=ServerClient))
        # Off by default: format_ok is reported but the reward is unchanged.
        default_response = await self._verify_text(server, "We get 2 + 3 = 5", "5", self.BOXED_SYSTEM_PROMPT)
        assert default_response.required_answer_format == "boxed"
        assert default_response.format_ok is False
        assert default_response.reward == approx(1.0)

        penalized_config = no_judge_config.model_copy(update={"penalize_answer_format": True})
        server = LibraryJudgeMathResourcesServer(config=penalized_config, server_client=MagicMock(spec=ServerClient))
        boxed, answer_line = self.BOXED_SYSTEM_PROMPT, self.ANSWER_LINE_SYSTEM_PROMPT
        # (system prompt, response text, format_ok, reward, library_reward)
        cases = [
            (boxed, "So 2 + 3 = \\boxed{5}.", True, 1.0, 1.0),
            # The format the system prompt did not ask for does not count.
            (boxed, "ANSWER: 5", False, 0.5, 1.0),
            (boxed, "We get 2 + 3 = 5", False, 0.5, 1.0),
            (boxed, "\\boxed{6}", True, 0.0, 0.0),
            (boxed, "We get 2 + 3 = 6", False, 0.0, 0.0),
            (answer_line, "ANSWER: 5", True, 1.0, 1.0),
            (answer_line, "So 2 + 3 = \\boxed{5}.", False, 0.5, 1.0),
            # No format asked for: no penalty.
            (None, "We get 2 + 3 = 5", None, 1.0, 1.0),
        ]
        for system_prompt, text, format_ok, reward, library_reward in cases:
            response = await self._verify_text(server, text, "5", system_prompt)
            assert response.format_ok is format_ok, (system_prompt, text)
            assert response.reward == approx(reward), (system_prompt, text)
            assert response.library_reward == approx(library_reward), (system_prompt, text)

        full_penalty_config = penalized_config.model_copy(update={"format_penalty": 1.0})
        server = LibraryJudgeMathResourcesServer(
            config=full_penalty_config, server_client=MagicMock(spec=ServerClient)
        )
        assert (await self._verify_text(server, "We get 2 + 3 = 5", "5", boxed)).reward == approx(0.0)

    async def test_verify_grade_answer_line(self, config: LibraryJudgeMathResourcesServerConfig) -> None:
        no_judge_config = config.model_copy(update={"should_use_judge": False})
        whole_text_server = LibraryJudgeMathResourcesServer(
            config=no_judge_config, server_client=MagicMock(spec=ServerClient)
        )
        answer_line_config = no_judge_config.model_copy(update={"grade_answer_line": True})
        answer_line_server = LibraryJudgeMathResourcesServer(
            config=answer_line_config, server_client=MagicMock(spec=ServerClient)
        )
        # (response text, expected answer, reward grading the whole text, reward grading the ANSWER: line)
        cases = [
            # LaTeX without $...$ after "ANSWER:" is only read inside \boxed{}.
            ("ANSWER: \\sqrt{2}", "\\sqrt{2}", 0.0, 1.0),
            ("ANSWER: 2\\sqrt{2}", "2\\sqrt{2}", 0.0, 1.0),
            ("ANSWER: x^2 + 1", "x^2+1", 0.0, 1.0),
            ("ANSWER: (2, 3)", "(2,3)", 0.0, 1.0),
            ("ANSWER: D", "D", 0.0, 1.0),
            ("**ANSWER:** $\\frac{1}{2}$.", "\\frac{1}{2}", 1.0, 1.0),
            # Falls back to the line as written when \boxed{} cannot read it.
            ("ANSWER: there are **81** days", "81", 1.0, 1.0),
            ("**Final Answer:**\nThere are **81** days.", "81", 1.0, 1.0),
            # Only the ANSWER: line is graded, not a number or a \boxed{} elsewhere in the response.
            ("We first find $x = 5$.\n\nANSWER: \\sqrt{3}", "5", 1.0, 0.0),
            ("\\boxed{3}\n\nANSWER: 5", "5", 0.0, 1.0),
            ("ANSWER: 3 or 5", "5", 0.0, 0.0),
            # Answer lines that math-verify cannot read as written (seen in training rollouts).
            ("We find $AB = 16\\sqrt{3}$.\n\nANSWER: 16√3", "16\\sqrt{3}", 0.0, 1.0),
            ("**Final answer:**\n\\(ANSWER: \\frac18\\)", "\\frac{1}{8}", 0.0, 1.0),
            (
                "ANSWER: \\(\\big\\lfloor \\frac{2n}{3} \\big\\rfloor + 1\\)",
                "\\left\\lfloor \\frac{2n}{3} \\right\\rfloor + 1",
                0.0,
                1.0,
            ),
            ("ANSWER: 1+3\\sqrt{2}\\)", "1+3\\sqrt{2}", 0.0, 1.0),
            ("ANSWER:\n\\[\n1280\n\\]", "1280", 1.0, 1.0),
            # Without an ANSWER: line the whole text is graded.
            ("So 2 + 3 = \\boxed{5}.", "5", 1.0, 1.0),
        ]
        for text, expected_answer, whole_text_reward, answer_line_reward in cases:
            whole_text_response = await self._verify_text(
                whole_text_server, text, expected_answer, self.ANSWER_LINE_SYSTEM_PROMPT
            )
            assert whole_text_response.reward == approx(whole_text_reward), text
            answer_line_response = await self._verify_text(
                answer_line_server, text, expected_answer, self.ANSWER_LINE_SYSTEM_PROMPT
            )
            assert answer_line_response.reward == approx(answer_line_reward), text

        # Only when the system prompt asks for an ANSWER: line.
        response = await self._verify_text(
            answer_line_server, "ANSWER: \\sqrt{2}", "\\sqrt{2}", self.BOXED_SYSTEM_PROMPT
        )
        assert response.reward == approx(0.0)
        response = await self._verify_text(answer_line_server, "ANSWER: \\sqrt{2}", "\\sqrt{2}")
        assert response.reward == approx(0.0)

    async def test_verify_answer(self, config: LibraryJudgeMathResourcesServerConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        resources_server = LibraryJudgeMathResourcesServer(config=config, server_client=server_mock)
        response_mock = AsyncMock()
        post_mock = MagicMock()
        post_mock.read = response_mock
        server_mock.post = AsyncMock(return_value=post_mock)

        (
            equal_reward,
            equal_extracted_answer,
            equal_library_reward,
            equal_judge_evaluations,
        ) = await resources_server._verify_answer("What is 3 plus 5?", "8", "3 + 5 = \\boxed{8}")
        assert equal_reward == approx(1.0)
        assert equal_extracted_answer == "8"
        assert equal_library_reward == approx(1.0)
        assert equal_judge_evaluations is None

        not_equal_item = self._create_response_output_message(
            f"Conclusion: {LibraryJudgeMathResourcesServer.JUDGE_NOT_EQUAL_LABEL}"
        )
        response_mock.side_effect = [json.dumps(self._create_response("verify_answer_not_equal_id", not_equal_item))]
        not_equal_question = "What is 1 + 1?"
        not_equal_expected_answer = "2"
        not_equal_generated_answer = "3"
        (
            not_equal_reward,
            not_equal_extracted_answer,
            not_equal_library_reward,
            not_equal_judge_evaluations,
        ) = await resources_server._verify_answer(
            not_equal_question,
            not_equal_expected_answer,
            not_equal_generated_answer,
        )
        assert not_equal_reward == approx(0.0)
        assert not_equal_extracted_answer == "3"
        assert not_equal_library_reward == approx(0.0)
        assert len(not_equal_judge_evaluations) == 1
        self._check_judge_evaluation(
            not_equal_judge_evaluations[0],
            not_equal_question,
            not_equal_expected_answer,
            not_equal_generated_answer,
            {},
            "verify_answer_not_equal_id",
            not_equal_item,
        )

        first_judge_equal_item = self._create_response_output_message(
            f"I say {LibraryJudgeMathResourcesServer.JUDGE_EQUAL_LABEL} as the verdict"
        )
        second_judge_equal_item = self._create_response_output_message(
            LibraryJudgeMathResourcesServer.JUDGE_EQUAL_LABEL
        )
        response_mock.side_effect = [
            json.dumps(self._create_response("verify_answer_first_judge_equal_id", first_judge_equal_item)),
            json.dumps(self._create_response("verify_answer_second_judge_equal_id", second_judge_equal_item)),
        ]
        judge_equal_question = "What is 14 divided by 10?"
        judge_equal_expected_answer = "1.4"
        judge_equal_generated_answer = "Final answer: {7 / 5}"
        (
            judge_equal_reward,
            judge_equal_extracted_answer,
            judge_equal_library_reward,
            judge_equal_judge_evaluations,
        ) = await resources_server._verify_answer(
            judge_equal_question,
            judge_equal_expected_answer,
            judge_equal_generated_answer,
        )
        assert judge_equal_reward == approx(1.0)
        assert judge_equal_extracted_answer == "5"
        assert judge_equal_library_reward == approx(0.0)
        assert len(judge_equal_judge_evaluations) == 2
        self._check_judge_evaluation(
            judge_equal_judge_evaluations[0],
            judge_equal_question,
            judge_equal_expected_answer,
            judge_equal_extracted_answer,
            {},
            "verify_answer_first_judge_equal_id",
            first_judge_equal_item,
        )
        self._check_judge_evaluation(
            judge_equal_judge_evaluations[1],
            judge_equal_question,
            judge_equal_extracted_answer,
            judge_equal_expected_answer,
            {},
            "verify_answer_second_judge_equal_id",
            second_judge_equal_item,
        )

    def test_library_verifier_returns_promptly_for_pathological_sympy_expression(self) -> None:
        if "fork" not in mp.get_all_start_methods():
            skip("This regression test requires fork to bound a stuck SymPy verifier.")

        ctx = mp.get_context("fork")
        result_connection, child_connection = ctx.Pipe(duplex=False)
        process = ctx.Process(target=_verify_pathological_sympy_expression, args=(child_connection,))
        process.start()
        child_connection.close()
        process.join(timeout=2.0)

        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
            raise AssertionError("library verifier did not return promptly for a real math_verify/SymPy expression")

        assert process.exitcode == 0
        if not result_connection.poll():
            raise AssertionError("library verifier process exited without returning a result") from None
        result = result_connection.recv()
        result_connection.close()

        if isinstance(result, BaseException):
            raise result

        reward, extracted_answer, library_reward, judge_evaluations = result
        # This is a real SymPy/math_verify pathological case, so the exact result
        # depends on whether it finishes before the subprocess timeout. The
        # regression is that either outcome returns promptly instead of hanging.
        assert reward == approx(0.0) or reward == approx(1.0)
        assert extracted_answer is None or isinstance(extracted_answer, str)
        assert library_reward == approx(reward)
        assert judge_evaluations is None

    async def test_library_verifier_process_is_cleaned_up_on_cancellation(self) -> None:
        if "fork" not in mp.get_all_start_methods():
            skip("This cancellation test requires fork.")

        ctx = mp.get_context("fork")
        result_connection, child_connection = ctx.Pipe(duplex=False)
        process = ctx.Process(target=_sleeping_library_verifier_process, args=(child_connection,))
        process.start()
        child_connection.close()

        try:
            task = asyncio.create_task(
                LibraryJudgeMathResourcesServer._wait_for_library_verifier_process(
                    process,
                    result_connection,
                    60.0,
                )
            )
            await asyncio.sleep(0.05)
            task.cancel()
            with raises(asyncio.CancelledError):
                await task

            assert not process.is_alive()
        finally:
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)

    async def test_library_verifier_process_is_cleaned_up_on_timeout(self) -> None:
        if "fork" not in mp.get_all_start_methods():
            skip("This timeout test requires fork.")

        ctx = mp.get_context("fork")
        result_connection, child_connection = ctx.Pipe(duplex=False)
        process = ctx.Process(target=_sleeping_library_verifier_process, args=(child_connection,))
        process.start()
        child_connection.close()

        try:
            assert await LibraryJudgeMathResourcesServer._wait_for_library_verifier_process(
                process,
                result_connection,
                0.01,
            ) == (approx(0.0), None)
            assert not process.is_alive()
        finally:
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)

    async def test_library_verifier_process_errors_return_zero_reward(self) -> None:
        process = MagicMock()
        process.is_alive.return_value = False
        process.exitcode = 1
        result_connection = MagicMock()

        assert await LibraryJudgeMathResourcesServer._wait_for_library_verifier_process(
            process,
            result_connection,
            1.0,
        ) == (approx(0.0), None)
        result_connection.close.assert_called_once()

        process = MagicMock()
        process.is_alive.return_value = False
        process.exitcode = 0
        result_connection = MagicMock()
        result_connection.poll.return_value = True
        result_connection.recv.side_effect = EOFError()

        assert await LibraryJudgeMathResourcesServer._wait_for_library_verifier_process(
            process,
            result_connection,
            1.0,
        ) == (approx(0.0), None)
        result_connection.close.assert_called_once()

    async def test_verify_answer_with_library_async(self, config: LibraryJudgeMathResourcesServerConfig) -> None:
        resources_server = LibraryJudgeMathResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        assert await resources_server._verify_answer_with_library_async("4", "2 + 2 = \\boxed{4}") == (
            approx(1.0),
            "4",
        )
        assert await resources_server._verify_answer_with_library_async("\\boxed{12}", "3 * 4 = \\boxed{12}") == (
            approx(1.0),
            "12",
        )
        assert await resources_server._verify_answer_with_library_async("\\boxed{5}", "10 - 5 = \\boxed{5}") == (
            approx(1.0),
            "5",
        )
        assert await resources_server._verify_answer_with_library_async("4.0", "2 + 2 = \\boxed{\\frac{8}{2}}") == (
            approx(1.0),
            "4",
        )

        assert await resources_server._verify_answer_with_library_async("\\boxed{12}", "3 * 4 = 13") == (
            approx(0.0),
            "13",
        )
        assert await resources_server._verify_answer_with_library_async("17.001", "17") == (
            approx(0.0),
            "17",
        )

        assert await resources_server._verify_answer_with_library_async("", "") == (
            approx(0.0),
            None,
        )

        assert await resources_server._verify_answer_with_library_async("3", "3") == (
            approx(1.0),
            "3",
        )

    def test_run_math_verify_handles_timeout_exception(self) -> None:
        timeout_mock = MagicMock(side_effect=TimeoutException())
        assert _run_math_verify(timeout_mock, "3", "3") == (
            approx(0.0),
            None,
        )

    async def test_verify_answer_with_judge(self, config: LibraryJudgeMathResourcesServerConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        resources_server = LibraryJudgeMathResourcesServer(config=config, server_client=server_mock)
        response_mock = AsyncMock()
        post_mock = MagicMock()
        post_mock.read = response_mock
        server_mock.post = AsyncMock(return_value=post_mock)

        first_not_equal_item = self._create_response_output_message(
            f"{LibraryJudgeMathResourcesServer.JUDGE_NOT_EQUAL_LABEL} is the evaluation"
        )
        response_mock.side_effect = [json.dumps(self._create_response("first_not_equal_id", first_not_equal_item))]
        first_not_equal_question = "What is 2 + 2?"
        first_not_equal_expected_answer = "4"
        first_not_equal_generated_answer = "5"
        (
            first_not_equal_reward,
            first_not_equal_evaluations,
        ) = await resources_server._verify_answer_with_judge(
            first_not_equal_question,
            first_not_equal_expected_answer,
            first_not_equal_generated_answer,
        )
        assert first_not_equal_reward == approx(0.0)
        assert len(first_not_equal_evaluations) == 1
        self._check_judge_evaluation(
            first_not_equal_evaluations[0],
            first_not_equal_question,
            first_not_equal_expected_answer,
            first_not_equal_generated_answer,
            {},
            "first_not_equal_id",
            first_not_equal_item,
        )

        first_equal_item = self._create_response_output_message(LibraryJudgeMathResourcesServer.JUDGE_EQUAL_LABEL)
        second_equal_item = self._create_response_output_message(
            f"I conclude that {LibraryJudgeMathResourcesServer.JUDGE_EQUAL_LABEL}"
        )
        response_mock.side_effect = [
            json.dumps(self._create_response("second_equal_first_id", first_equal_item)),
            json.dumps(self._create_response("second_equal_second_id", second_equal_item)),
        ]
        second_equal_question = "What is 3 divided by 6?"
        second_equal_expected_answer = "1/2"
        second_equal_generated_answer = "0.5"
        (
            second_equal_reward,
            second_equal_evaluations,
        ) = await resources_server._verify_answer_with_judge(
            second_equal_question,
            second_equal_expected_answer,
            second_equal_generated_answer,
        )
        assert second_equal_reward == approx(1.0)
        assert len(second_equal_evaluations) == 2
        self._check_judge_evaluation(
            second_equal_evaluations[0],
            second_equal_question,
            second_equal_expected_answer,
            second_equal_generated_answer,
            {},
            "second_equal_first_id",
            first_equal_item,
        )
        self._check_judge_evaluation(
            second_equal_evaluations[1],
            second_equal_question,
            second_equal_generated_answer,
            second_equal_expected_answer,
            {},
            "second_equal_second_id",
            second_equal_item,
        )

        second_not_equal_item = self._create_response_output_message(
            LibraryJudgeMathResourcesServer.JUDGE_NOT_EQUAL_LABEL
        )
        response_mock.side_effect = [
            json.dumps(self._create_response("second_not_equal_first_id", second_equal_item)),
            json.dumps(self._create_response("second_not_equal_second_id", second_not_equal_item)),
        ]
        second_not_equal_question = "What is 4 times 5?"
        second_not_equal_expected_answer = "20"
        second_not_equal_generated_answer = "20.0"
        (
            second_not_equal_reward,
            second_not_equal_evaluations,
        ) = await resources_server._verify_answer_with_judge(
            second_not_equal_question,
            second_not_equal_expected_answer,
            second_not_equal_generated_answer,
        )
        assert second_not_equal_reward == approx(0.0)
        assert len(second_not_equal_evaluations) == 2
        self._check_judge_evaluation(
            second_not_equal_evaluations[0],
            second_not_equal_question,
            second_not_equal_expected_answer,
            second_not_equal_generated_answer,
            {},
            "second_not_equal_first_id",
            second_equal_item,
        )
        self._check_judge_evaluation(
            second_not_equal_evaluations[1],
            second_not_equal_question,
            second_not_equal_generated_answer,
            second_not_equal_expected_answer,
            {},
            "second_not_equal_second_id",
            second_not_equal_item,
        )

    async def _generate_and_check_judge_evaluation(
        self,
        resources_server: LibraryJudgeMathResourcesServer,
        question: str,
        expected_answers_equal: bool,
        expected_response_id: str,
        expected_output_item: NeMoGymResponseOutputItem,
    ) -> None:
        first_answer = f"{question}_1"
        second_answer = f"{question}_2"
        (
            actual_answers_equal,
            judge_evaluation,
        ) = await resources_server._generate_judge_evaluation(question, first_answer, second_answer)
        assert actual_answers_equal == expected_answers_equal
        self._check_judge_evaluation(
            judge_evaluation,
            question,
            first_answer,
            second_answer,
            {"max_output_tokens": 1024},
            expected_response_id,
            expected_output_item,
        )

    async def test_generate_judge_evaluation(self, config: LibraryJudgeMathResourcesServerConfig) -> None:
        judge_config = config.model_copy(deep=True)
        judge_config.judge_responses_create_params.max_output_tokens = 1024
        server_mock = MagicMock(spec=ServerClient)
        resources_server = LibraryJudgeMathResourcesServer(config=judge_config, server_client=server_mock)
        response_mock = AsyncMock()
        post_mock = MagicMock()
        post_mock.read = response_mock
        server_mock.post = AsyncMock(return_value=post_mock)

        response_mock.return_value = json.dumps({})
        with raises(ValidationError, match="Field required"):
            await resources_server._generate_judge_evaluation("invalid_response", "invalid_1", "invalid_2")

        reasoning_item = NeMoGymResponseReasoningItem(id="reasoning_item", summary=[], type="reasoning")
        response_mock.return_value = json.dumps(self._create_response("reasoning_id", reasoning_item))
        await self._generate_and_check_judge_evaluation(
            resources_server,
            "reasoning_question",
            False,
            "reasoning_id",
            reasoning_item,
        )

        refusal_item = NeMoGymResponseOutputMessage(
            id="refusal_item",
            content=[
                NeMoGymResponseOutputRefusal(refusal="I refuse", type="refusal"),
            ],
            role="assistant",
            status="completed",
            type="message",
        )
        response_mock.return_value = json.dumps(self._create_response("refusal_id", refusal_item))
        await self._generate_and_check_judge_evaluation(
            resources_server, "refusal_question", False, "refusal_id", refusal_item
        )

        no_evaluation_item = self._create_response_output_message("no evaluation")
        response_mock.return_value = json.dumps(self._create_response("no_evaluation_id", no_evaluation_item))
        await self._generate_and_check_judge_evaluation(
            resources_server,
            "no_evaluation_question",
            False,
            "no_evaluation_id",
            no_evaluation_item,
        )

        not_equal_item = self._create_response_output_message(
            f"Evaluation: {LibraryJudgeMathResourcesServer.JUDGE_NOT_EQUAL_LABEL}"
        )
        response_mock.return_value = json.dumps(self._create_response("not_equal_id", not_equal_item))
        await self._generate_and_check_judge_evaluation(
            resources_server,
            "not_equal_question",
            False,
            "not_equal_id",
            not_equal_item,
        )

        equal_item = self._create_response_output_message(
            f"The evaluation is {LibraryJudgeMathResourcesServer.JUDGE_EQUAL_LABEL}"
        )
        response_mock.return_value = json.dumps(self._create_response("equal_id", equal_item))
        await self._generate_and_check_judge_evaluation(
            resources_server, "equal_question", True, "equal_id", equal_item
        )

        equal_first_item = self._create_response_output_message(
            f"First {LibraryJudgeMathResourcesServer.JUDGE_EQUAL_LABEL}, "
            f"then {LibraryJudgeMathResourcesServer.JUDGE_NOT_EQUAL_LABEL}"
        )
        response_mock.return_value = json.dumps(self._create_response("equal_first_id", equal_first_item))
        await self._generate_and_check_judge_evaluation(
            resources_server,
            "equal_first_question",
            True,
            "equal_first_id",
            equal_first_item,
        )

        not_equal_first_item = self._create_response_output_message(
            f"{LibraryJudgeMathResourcesServer.JUDGE_NOT_EQUAL_LABEL} "
            f"{LibraryJudgeMathResourcesServer.JUDGE_EQUAL_LABEL}"
        )
        response_mock.return_value = json.dumps(self._create_response("not_equal_first_id", not_equal_first_item))
        await self._generate_and_check_judge_evaluation(
            resources_server,
            "not_equal_first_question",
            False,
            "not_equal_first_id",
            not_equal_first_item,
        )


# ──────────────────────────────────────────────────────────
# Math metrics tests
# ──────────────────────────────────────────────────────────


class TestComputeMetricsIntegration:
    """Test the full compute_metrics method on LibraryJudgeMathResourcesServer."""

    @fixture
    def server(self) -> LibraryJudgeMathResourcesServer:
        config = LibraryJudgeMathResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            judge_model_server=ModelServerRef(type="responses_api_models", name="math_judge"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )
        return LibraryJudgeMathResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def _make_tasks(self):
        """3 tasks × 4 rollouts with varying correctness and some no_answer."""
        return [
            # Task 0: 3 correct, 1 no_answer
            [
                {"reward": 1.0, "library_reward": 1.0, "extracted_answer": "204"},
                {"reward": 1.0, "library_reward": 1.0, "extracted_answer": "204"},
                {"reward": 1.0, "library_reward": 1.0, "extracted_answer": "204"},
                {"reward": 0.0, "library_reward": 0.0, "extracted_answer": None},
            ],
            # Task 1: 1 correct, 1 wrong, 2 no_answer
            [
                {"reward": 1.0, "library_reward": 1.0, "extracted_answer": "113"},
                {"reward": 0.0, "library_reward": 0.0, "extracted_answer": "42"},
                {"reward": 0.0, "library_reward": 0.0, "extracted_answer": None},
                {"reward": 0.0, "library_reward": 0.0, "extracted_answer": None},
            ],
            # Task 2: all wrong, 1 no_answer
            [
                {"reward": 0.0, "library_reward": 0.0, "extracted_answer": "99"},
                {"reward": 0.0, "library_reward": 0.0, "extracted_answer": "42"},
                {"reward": 0.0, "library_reward": 0.0, "extracted_answer": "7"},
                {"reward": 0.0, "library_reward": 0.0, "extracted_answer": None},
            ],
        ]

    def test_pass_at_k(self, server) -> None:
        tasks = self._make_tasks()
        result = server.compute_metrics(tasks)
        # pass@1: avg reward across all rollouts = (3+1+0)/3 tasks, each avg'd over 4 = 33.3%
        assert result["pass@1/symbolic_accuracy"] == approx(100.0 / 3.0, abs=0.01)
        # pass@4: binary per-task (any correct?) = 2/3 tasks = 66.7%
        assert result["pass@4/symbolic_accuracy"] == approx(200.0 / 3.0, abs=0.01)

    def test_majority_at_k(self, server) -> None:
        tasks = self._make_tasks()
        result = server.compute_metrics(tasks)
        assert "majority@4/symbolic_accuracy" in result

    def test_per_sample_aggregate(self, server) -> None:
        tasks = self._make_tasks()
        result = server.compute_metrics(tasks)
        psa = result["per_sample_aggregate"]
        assert "symbolic_accuracy" in psa
        assert len(psa["symbolic_accuracy"]) == 4

    def test_no_answer_tracking(self, server) -> None:
        tasks = self._make_tasks()
        result = server.compute_metrics(tasks)
        assert "pass@1/no_answer" in result
        assert "pass@4/no_answer" in result
        assert "pass@1[avg-of-4]/no_answer" in result
        psa = result["per_sample_aggregate"]
        assert "no_answer" in psa
        assert len(psa["no_answer"]) == 4
        assert psa["no_answer"][0] == approx(0.0)
        assert psa["no_answer"][3] == approx(100.0)
        assert "pass@1[avg-of-2]/no_answer/std_dev_across_runs" in result
        assert "pass@1[avg-of-4]/no_answer/std_dev_across_runs" in result

    def test_no_answer_stats(self, server) -> None:
        tasks = self._make_tasks()
        result = server.compute_metrics(tasks)
        assert "pass@1[avg-of-4]/no_answer/std_dev_across_runs" in result
        assert "pass@1[avg-of-4]/no_answer/std_err_across_runs" in result
        assert result["pass@1[avg-of-4]/no_answer/std_dev_across_runs"] > 0

    def test_stat_key_separator(self, server) -> None:
        tasks = self._make_tasks()
        result = server.compute_metrics(tasks)
        stat_keys = [k for k in result if "std_dev_across_runs" in k]
        for k in stat_keys:
            assert "/std_dev_across_runs" in k, f"Expected / separator in {k}"

    def test_stats_for_all_k_values(self, server) -> None:
        tasks = self._make_tasks()
        result = server.compute_metrics(tasks)
        for k_val in [2, 3, 4]:
            key = f"pass@1[avg-of-{k_val}]/symbolic_accuracy/std_dev_across_runs"
            assert key in result, f"Missing stats for k={k_val}: {key}"

    def test_multi_score(self, server) -> None:
        tasks = self._make_tasks()
        result = server.compute_metrics(tasks)
        assert "pass@1/symbolic_accuracy" in result
        assert "accuracy" not in str(
            [k for k in result if "accuracy" in k and "symbolic" not in k and "judge" not in k]
        )

    def test_empty_tasks(self, server) -> None:
        result = server.compute_metrics([])
        assert result == {}


class TestGetKeyMetrics:
    def test_selects_headlines(self) -> None:
        agent_metrics = {
            "mean/reward": 0.5,
            "mean/library_reward": 0.5,
            "mean/input_tokens": 100.0,
            "mean/output_tokens": 500.0,
            "mean/total_tokens": 600.0,
            "pass@1/symbolic_accuracy": 50.0,
            "pass@1/no_answer": 10.0,
            "pass@1[avg-of-1]/symbolic_accuracy": 50.0,
            "pass@1[avg-of-1]/no_answer": 10.0,
            "pass@1[avg-of-4]/symbolic_accuracy": 45.0,
            "pass@1[avg-of-4]/symbolic_accuracy/std_dev_across_runs": 3.0,
            "pass@1[avg-of-4]/no_answer": 12.0,
            "pass@1[avg-of-4]/no_answer/std_dev_across_runs": 2.0,
            "pass@4/symbolic_accuracy": 70.0,
            "pass@4/no_answer": 15.0,
            "majority@4/symbolic_accuracy": 60.0,
            "majority@4/no_answer": 5.0,
        }
        result = LibraryJudgeMathResourcesServer.get_key_metrics(None, agent_metrics)
        assert "mean/input_tokens" in result
        assert "mean/output_tokens" in result
        assert "mean/reward" not in result
        assert "mean/library_reward" not in result
        assert "mean/total_tokens" not in result
        assert "pass@1[avg-of-4]/symbolic_accuracy" in result
        assert "pass@1[avg-of-4]/no_answer" in result
        assert "pass@4/symbolic_accuracy" in result
        assert "pass@4/no_answer" not in result
        assert "majority@4/symbolic_accuracy" in result
        assert "majority@4/no_answer" not in result
        assert "pass@1[avg-of-4]/symbolic_accuracy/std_dev_across_runs" not in result
        assert "pass@1[avg-of-4]/no_answer/std_dev_across_runs" not in result


class TestAggregateMetrics:
    """Test the full aggregate_metrics route on the math server."""

    async def test_produces_symbolic_and_judge_accuracy(self) -> None:
        from nemo_gym.base_resources_server import AggregateMetricsRequest
        from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME

        config = LibraryJudgeMathResourcesServerConfig(
            host="127.0.0.1",
            port=12345,
            entrypoint="app.py",
            name="math_with_judge",
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )
        server = LibraryJudgeMathResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        responses = [
            {
                TASK_INDEX_KEY_NAME: 0,
                ROLLOUT_INDEX_KEY_NAME: 0,
                "reward": 1.0,
                "library_reward": 1.0,
                "judge_evaluations": [{"verdict": "A=B"}],
                "extracted_answer": "42",
            },
            {
                TASK_INDEX_KEY_NAME: 0,
                ROLLOUT_INDEX_KEY_NAME: 1,
                "reward": 0.0,
                "library_reward": 0.0,
                "judge_evaluations": [{"verdict": "A!=B"}],
                "extracted_answer": "43",
            },
            {
                TASK_INDEX_KEY_NAME: 1,
                ROLLOUT_INDEX_KEY_NAME: 0,
                "reward": 1.0,
                "library_reward": 1.0,
                "judge_evaluations": None,
                "extracted_answer": "7",
            },
            {
                TASK_INDEX_KEY_NAME: 1,
                ROLLOUT_INDEX_KEY_NAME: 1,
                "reward": 0.0,
                "library_reward": 0.0,
                "judge_evaluations": None,
                "extracted_answer": "8",
            },
        ]
        body = AggregateMetricsRequest(verify_responses=responses)
        result = await server.aggregate_metrics(body)
        am = result.agent_metrics

        assert "pass@1/symbolic_accuracy" in am
        assert "pass@1[avg-of-2]/symbolic_accuracy" in am
        assert "majority@2/symbolic_accuracy" in am
        assert "pass@1/judge_accuracy" in am
        assert "pass@1[avg-of-2]/symbolic_accuracy/std_dev_across_runs" in am
        assert "pass@2/symbolic_accuracy" in result.key_metrics
        assert "majority@2/symbolic_accuracy" in result.key_metrics
