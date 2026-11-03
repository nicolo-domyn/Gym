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
import contextlib
import logging
import multiprocessing as mp
import re
from io import StringIO
from typing import Any, ClassVar, Dict, List, Literal, Optional, Union

from fastapi import FastAPI
from math_verify import grader
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.reward_profile import compute_pass_majority_metrics, highest_k_metrics
from nemo_gym.server_utils import get_response_json


class AnswerFormatRule(BaseModel):
    # Text searched as is (case-sensitive) in the system prompt, i.e. the system/developer messages.
    prompt_text: str
    # The format the response must then use:
    #   - "boxed": a closed \boxed{...} (or \fbox{...});
    #   - "answer_line": a line starting with "ANSWER:" in any case, optionally after markdown markers
    #     or "Final", e.g. "ANSWER: 5", "**Answer:** 5", "Final answer: 5".
    answer_format: Literal["boxed", "answer_line"]



class LibraryJudgeMathResourcesServerConfig(BaseResourcesServerConfig):
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    should_use_judge: bool = True
    library_verifier_timeout_seconds: PositiveFloat = 10.0
    library_verifier_max_concurrency: PositiveInt = 32
    # Answer format check. The response must use the format its system prompt asks for: the first rule whose
    # prompt_text is in the system prompt gives it, reported as `required_answer_format`. `format_ok` tells
    # whether the response uses it. When no rule matches, the sample has no format check: format_ok is None.
    # The ANSWER: rule comes first because such prompts may also say that \boxed is not needed.
    answer_format_rules: List[AnswerFormatRule] = [
        AnswerFormatRule(prompt_text="ANSWER: $ANSWER", answer_format="answer_line"),
        AnswerFormatRule(prompt_text="\\boxed", answer_format="boxed"),
    ]
    # When True, a response with format_ok False gets max(0, reward - format_penalty): a correct answer in the
    # wrong format is penalized, a wrong answer stays at 0. Off by default, so the reward is unchanged.
    penalize_answer_format: bool = False
    format_penalty: float = Field(default=0.5, ge=0.0, le=1.0)
    # When True, a response whose system prompt asks for an ANSWER: line and that has one is graded on its last
    # ANSWER: line only: its content wrapped in \boxed{} (after "ANSWER:", math-verify reads LaTeX only inside
    # $...$), then the line as written if that fails. Off by default: the whole response is graded.
    grade_answer_line: bool = False


class LibraryJudgeMathRunRequest(BaseRunRequest):
    question: str
    expected_answer: str


class LibraryJudgeMathVerifyRequest(LibraryJudgeMathRunRequest, BaseVerifyRequest):
    pass


class JudgeEvaluation(BaseModel):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse


class LibraryJudgeMathVerifyResponse(BaseVerifyResponse):
    expected_answer: str
    extracted_answer: Optional[str]
    library_reward: float
    judge_evaluations: Optional[list[JudgeEvaluation]]
    # Optional so that subclasses building their own verify response keep working.
    format_ok: Optional[bool] = None
    required_answer_format: Optional[Literal["boxed", "answer_line"]] = None


def _run_math_verify(
    library_verifier: Any, expected_answer: str, generated_answer: str
) -> tuple[float, Optional[str]]:
    # This functionality is migrated from Nemo RL.
    # https://github.com/NVIDIA-NeMo/RL/blob/e1f56c42ae175d3863ccaf4e21b7de7e9c46c2e1/nemo_rl/environments/math_environment.py
    try:
        stripped = LibraryJudgeMathResourcesServer._strip_math_delimiters(expected_answer)
        ground_truth_parsable = "\\boxed{" + stripped + "}"
        with LibraryJudgeMathResourcesServer._mute_output():
            ret_score, extracted_answer = library_verifier([ground_truth_parsable], [generated_answer])

        reward = float(ret_score)

        if extracted_answer is not None:
            # Make sure the extracted answer has two elements.
            assert len(extracted_answer) == 2

            extracted_gold, extracted_prediction = extracted_answer

            # Get the extracted answer.
            for pred in extracted_prediction:
                if any(grader.verify(gold, pred) for gold in extracted_gold):
                    extracted_answer = pred
                    break
            else:
                # If no match is found, that means all the answers are
                # incorrect.  The first prediction is used as the extracted
                # answer.
                extracted_answer = extracted_prediction[0] if extracted_prediction else None

        return reward, extracted_answer

    # It's possible to emit a TimeoutException and that wouldn't be caught since
    # it actually subclasses from BaseException and math-verify itself does not
    # catch it.
    except (Exception, TimeoutException):
        return 0.0, None


def _run_math_verify_in_subprocess(expected_answer: str, generated_answer: str, result_connection: Any) -> None:
    # Keep math_verify construction inside the child process. A wedged SymPy call
    # can then be killed by terminating this process without poisoning the server.
    library_verifier = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(
            ExprExtractionConfig(),
            LatexExtractionConfig(),
        ),
    )
    try:
        result_connection.send(_run_math_verify(library_verifier, expected_answer, generated_answer))
    finally:
        result_connection.close()


class LibraryJudgeMathResourcesServer(SimpleResourcesServer):
    # These judge messages are adapted from ones used in Arena Hard.
    # https://github.com/lmarena/arena-hard-auto/blob/196f6b826783b3da7310e361a805fa36f0be83f3/utils/judge_utils.py
    # They are intended to serve as example messages for an LLM judge, and have not
    # been customized for a specific judge model.
    JUDGE_SYSTEM_MESSAGE: ClassVar[
        str
    ] = """Please act as an impartial judge and evaluate the equivalence of the solutions given by two AI assistants to the mathematical problem displayed below. You will be given AI assistant A's answer and AI assistant B's answer. Your job is to evaluate whether assistant A's answer is equivalent to assistant B's answer.

Consider the mathematical equivalence of the AI assistants' answers above all other considerations. If the problem requests special formatting instructions, you may disregard any formatting considerations when evaluating the answers -- consider only mathematical equivalence.

After evaluating both answers for equivalence, you must output only one of the following choices as your final verdict with a label:

1.  The AI assistants' answers are equivalent: [[A=B]]
2.  The AI assistants' answers are different: [[A!=B]]

Example output: "My final verdict is different [[A!=B]]"."""

    JUDGE_PROMPT_TEMPLATE: ClassVar[str] = (
        "<|Problem|>\n{question}\n\n<|Start of Assistant A's Answer|>\n{first_answer}\n<|End of Assistant A's Answer|>\n\n<|Start of Assistant B's Answer|>\n{second_answer}\n<|End of Assistant B's Answer|>"
    )

    JUDGE_EQUAL_LABEL: ClassVar[str] = "[[A=B]]"
    JUDGE_NOT_EQUAL_LABEL: ClassVar[str] = "[[A!=B]]"

    BOXED_START_PATTERN: ClassVar[re.Pattern] = re.compile(r"\\(?:boxed|fbox)\s*\{")
    ANSWER_LINE_PATTERN: ClassVar[re.Pattern] = re.compile(
        r"^[ \t>*_#]*(?:final[ \t]+)?answer[ \t*_]*:[ \t*]*(?P<content>.*)$", re.IGNORECASE | re.MULTILINE
    )
    # Special tokens such as <|im_end|> left at the end of the generation. Only token-like names at the end of
    # the line, so that math such as "0<|x|>1" or "<|x - 1|>" is kept.
    SPECIAL_TOKEN_PATTERN: ClassVar[re.Pattern] = re.compile(r"(?:\s*<\|[A-Za-z][A-Za-z0-9_]{2,}\|>)+\s*$")
    # A trailing sentence period or separator, but not LaTeX spacing such as "\,".
    TRAILING_PUNCTUATION_PATTERN: ClassVar[re.Pattern] = re.compile(r"(?<!\\)[.,;:]\s*$")
    MATH_DELIMITERS: ClassVar[tuple[tuple[str, str], ...]] = (("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)"), ("$", "$"))
    # Display math opened on the ANSWER: line (or the line after it) and closed on a later line.
    DISPLAY_MATH_DELIMITERS: ClassVar[dict[str, str]] = {"\\[": "\\]", "$$": "$$"}
    # Unicode math that math-verify does not read, e.g. "2√5/5", "−3" or "2π".
    UNICODE_SQRT_PATTERN: ClassVar[re.Pattern] = re.compile(r"√\s*(\d+(?:\.\d+)?|[A-Za-z]|\{[^{}]*\}|\([^()]*\))")
    UNICODE_SYMBOLS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("−", "-"),
        ("×", "\\times "),
        ("·", "\\cdot "),
        ("π", "\\pi "),
    )
    # \big, \Bigl, \biggr, ... size modifiers, which math-verify does not read: "\big\lfloor x \big\rfloor".
    SIZE_MODIFIER_PATTERN: ClassVar[re.Pattern] = re.compile(r"\\[Bb]igg?[lrm]?(?![A-Za-z])")
    # A second "ANSWER:" inside the line, e.g. "**Final answer:** \(ANSWER: \frac{1}{8}\)", possibly behind
    # \displaystyle or inside \text{...}.
    NESTED_ANSWER_PATTERN: ClassVar[re.Pattern] = re.compile(r"^(?:final[ \t]+)?answer[ \t]*:[ \t]*", re.IGNORECASE)
    DISPLAYSTYLE_PATTERN: ClassVar[re.Pattern] = re.compile(r"^\\displaystyle(?![A-Za-z])\s*")
    TEXT_WRAPPER_PATTERN: ClassVar[re.Pattern] = re.compile(r"^\\(?:text|textbf|mathrm|mathbf)\s*\{")


    config: LibraryJudgeMathResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)

        logging.getLogger("math_verify").setLevel(logging.CRITICAL)

        # The async path no longer blocks the event loop while SymPy runs, so
        # cap child-process fanout explicitly.
        self._library_verifier_semaphore = asyncio.Semaphore(value=self.config.library_verifier_max_concurrency)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        # Additional server routes go here! e.g.:
        # app.post("/get_weather")(self.get_weather)

        return app

    async def verify(self, body: LibraryJudgeMathVerifyRequest) -> LibraryJudgeMathVerifyResponse:
        assistant_responses = []
        for output_item in body.response.output:
            if output_item.type != "message":
                continue

            for content_item in output_item.content:
                if content_item.type != "output_text":
                    continue

                assistant_responses.append(content_item.text)

        combined_response = "".join(assistant_responses)
        required_answer_format = self._required_answer_format(body.responses_create_params)
        format_ok = None
        if required_answer_format is not None:
            format_ok = self._has_answer_format(combined_response, required_answer_format)
        generated_answer = combined_response
        if self.config.grade_answer_line and required_answer_format == "answer_line" and format_ok:
            generated_answer = self._answer_line_to_grade(body.expected_answer, combined_response)
        (
            reward,
            extracted_answer,
            library_reward,
            judge_evaluations,
        ) = await self._verify_answer(body.question, body.expected_answer, generated_answer)
        if self.config.penalize_answer_format and format_ok is False:
            reward = max(0.0, reward - self.config.format_penalty)
        return LibraryJudgeMathVerifyResponse(
            **body.model_dump(),
            reward=reward,
            extracted_answer=extracted_answer,
            library_reward=library_reward,
            judge_evaluations=judge_evaluations,
            format_ok=format_ok,
            required_answer_format=required_answer_format,
        )

    def _required_answer_format(self, params: NeMoGymResponseCreateParamsNonStreaming) -> Optional[str]:
        """The answer format the system prompt asks for, from the first matching answer_format_rules entry."""
        system_prompt = self._system_prompt(params)
        for rule in self.config.answer_format_rules:
            if rule.prompt_text in system_prompt:
                return rule.answer_format
        return None

    @staticmethod
    def _system_prompt(params: NeMoGymResponseCreateParamsNonStreaming) -> str:
        """Text of the system/developer messages of the request.

        `instructions` is left out: vllm_model does not send it to the model, so it cannot ask for a format.
        """
        texts = []
        if not isinstance(params.input, str):
            for item in params.input:
                if getattr(item, "role", None) not in ("system", "developer"):
                    continue
                if isinstance(item.content, str):
                    texts.append(item.content)
                else:
                    texts.extend(part.get("text") or "" for part in item.content if isinstance(part, dict))
        return "\n".join(texts)

    def _has_answer_format(self, text: str, answer_format: str) -> bool:
        """Whether the response gives its answer in the given format."""
        if answer_format == "boxed":
            return self._has_closed_boxed(text)
        return self.ANSWER_LINE_PATTERN.search(text) is not None

    def _answer_line_to_grade(self, expected_answer: str, text: str) -> str:
        """The part of a response answering on an "ANSWER:" line that gets graded.

        The line content wrapped in \\boxed{} when math-verify accepts it there, otherwise the line as
        written. The rest of the response is never graded, so a number elsewhere in it cannot match.
        """
        content = self._extract_answer_line(text)
        boxed_answer = "\\boxed{" + self._normalize_answer_line(content) + "}"
        library_reward, _ = self._verify_answer_with_library(expected_answer, boxed_answer)
        return boxed_answer if library_reward > 0.5 else "ANSWER: " + content

    @classmethod
    def _extract_answer_line(cls, text: str) -> str:
        """Content of the last "ANSWER:" line, or of the next non-empty line when nothing follows the colon.

        When that content only opens display math ("\\[" or "$$"), the whole block up to its closing delimiter.
        """
        match = list(cls.ANSWER_LINE_PATTERN.finditer(text))[-1]
        content = match.group("content")
        following_lines = iter(text[match.end() :].splitlines())
        if not content.strip():
            content = next((line for line in following_lines if line.strip()), "")
        closing = cls.DISPLAY_MATH_DELIMITERS.get(content.strip())
        if closing is not None:
            block = [content.strip()]
            for line in following_lines:
                block.append(line.strip())
                if closing in line:
                    break
            content = " ".join(part for part in block if part)
        return content.strip()

    @classmethod
    def _normalize_answer_line(cls, content: str) -> str:
        """Drop special tokens, markdown emphasis, a repeated "ANSWER:", outer or unpaired math delimiters and
        trailing punctuation, and rewrite unicode math and \\big-style size modifiers as LaTeX."""
        content = cls.SPECIAL_TOKEN_PATTERN.sub("", content).strip()
        content = cls.UNICODE_SQRT_PATTERN.sub(lambda m: "\\sqrt{" + cls._strip_group(m.group(1)) + "}", content)
        for symbol, latex in cls.UNICODE_SYMBOLS:
            content = content.replace(symbol, latex)
        content = cls.SIZE_MODIFIER_PATTERN.sub("", content)
        previous = None
        while content != previous:
            previous = content
            content = content.strip("*`").strip()
            content = cls.TRAILING_PUNCTUATION_PATTERN.sub("", content).strip()
            content = cls.DISPLAYSTYLE_PATTERN.sub("", content)
            content = cls._unwrap_nested_answer(content)
            for left, right in cls.MATH_DELIMITERS:
                inner = content[len(left) : -len(right)]
                if (
                    content.startswith(left)
                    and content.endswith(right)
                    and inner.strip()
                    and left not in inner
                    and right not in inner
                ):
                    content = inner.strip()
                    break
            else:
                content = cls._strip_unpaired_delimiter(content)
        return content

    @staticmethod
    def _strip_group(group: str) -> str:
        """The inside of a (...) or {...} group, or the group itself."""
        return group[1:-1].strip() if group[:1] in "({" else group

    @classmethod
    def _unwrap_nested_answer(cls, content: str) -> str:
        """Drop a second "ANSWER:" at the start, also when \\text{...} wraps it together with the answer."""
        match = cls.TEXT_WRAPPER_PATTERN.match(content)
        if match is not None and content.endswith("}") and cls.NESTED_ANSWER_PATTERN.match(content[match.end() :]):
            depth = 0
            for index in range(match.end() - 1, len(content)):
                depth += {"{": 1, "}": -1}.get(content[index], 0)
                if depth == 0:
                    break
            if index == len(content) - 1:
                content = content[match.end() : -1].strip()
        return cls.NESTED_ANSWER_PATTERN.sub("", content)

    @classmethod
    def _strip_unpaired_delimiter(cls, content: str) -> str:
        """Drop a math delimiter at either end whose pair is missing, e.g. "1+3\\sqrt{2}\\)"."""
        for left, right in cls.MATH_DELIMITERS:
            if left == right:
                if content.count(left) != 1:
                    continue
                if content.startswith(left):
                    return content[len(left) :].strip()
                if content.endswith(right) and not content.endswith("\\" + right):
                    return content[: -len(right)].strip()
            elif content.endswith(right) and left not in content:
                return content[: -len(right)].strip()
            elif content.startswith(left) and right not in content:
                return content[len(left) :].strip()
        return content

    @classmethod
    def _has_closed_boxed(cls, text: str) -> bool:
        # A \boxed{ whose braces never balance (e.g. the output was cut) does not count.
        for match in cls.BOXED_START_PATTERN.finditer(text):
            depth = 0
            for char in text[match.end() - 1 :]:
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        return True
        return False

    async def _verify_answer(
        self, question: str, expected_answer: str, generated_answer: str
    ) -> tuple[float, Optional[str], float, Optional[list[JudgeEvaluation]]]:
        """Verify the correctness of a generated answer.

        Verify the correctness of the specified model-generated answer to the
        specified question in comparison with the specified expected answer.
        """

        library_reward, extracted_answer = await self._verify_answer_with_library_async(
            expected_answer, generated_answer
        )
        if not self.config.should_use_judge or library_reward > 0.5:
            return library_reward, extracted_answer, library_reward, None

        judge_answer = extracted_answer if extracted_answer else generated_answer
        judge_reward, judge_evaluations = await self._verify_answer_with_judge(question, expected_answer, judge_answer)
        return judge_reward, extracted_answer, library_reward, judge_evaluations

    @classmethod
    @contextlib.contextmanager
    def _mute_output(cls):
        devnull_out, devnull_err = StringIO(), StringIO()
        with (
            contextlib.redirect_stdout(devnull_out),
            contextlib.redirect_stderr(devnull_err),
        ):
            yield

    @staticmethod
    def _strip_math_delimiters(s: str) -> str:
        """Strip outer math delimiters from expected answers.

        Many expected_answer values are wrapped in \\(...\\) or $...$,
        which causes the math_verify parser to fail when we wrap them
        in \\boxed{}.  Removing these outer delimiters fixes parsing.
        """
        s = s.strip()
        if s.startswith("\\(") and s.endswith("\\)"):
            s = s[2:-2].strip()
        if s.startswith("$") and s.endswith("$") and len(s) > 1:
            s = s[1:-1].strip()
        return s

    async def _verify_answer_with_library_async(
        self, expected_answer: str, generated_answer: str
    ) -> tuple[float, Optional[str]]:
        async with self._library_verifier_semaphore:
            # The production rollout workers run on Linux. Pin fork so the
            # verifier child is cheap to start and can be killed independently
            # if SymPy wedges.
            ctx = mp.get_context("fork")
            result_connection, child_connection = ctx.Pipe(duplex=False)
            process = ctx.Process(
                target=_run_math_verify_in_subprocess,
                args=(expected_answer, generated_answer, child_connection),
            )
            process.start()
            child_connection.close()
            return await self._wait_for_library_verifier_process(
                process,
                result_connection,
                self.config.library_verifier_timeout_seconds,
            )

    @staticmethod
    async def _wait_for_library_verifier_process(
        process: Any, result_connection: Any, timeout_seconds: float
    ) -> tuple[float, Optional[str]]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        try:
            while process.is_alive():
                remaining = deadline - loop.time()
                if remaining <= 0:
                    process.terminate()
                    terminate_deadline = loop.time() + 1.0
                    while process.is_alive() and loop.time() < terminate_deadline:
                        await asyncio.sleep(0.05)
                    if process.is_alive():
                        process.kill()
                    while process.is_alive():
                        await asyncio.sleep(0.05)
                    process.join(timeout=0)
                    return 0.0, None
                await asyncio.sleep(min(0.05, remaining))

            process.join(timeout=0)
            if process.exitcode != 0 or not result_connection.poll():
                return 0.0, None
            try:
                return result_connection.recv()
            except EOFError:
                return 0.0, None
        finally:
            with contextlib.suppress(OSError, AssertionError):
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=1.0)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=1.0)
            result_connection.close()

    async def _verify_answer_with_judge(
        self, question: str, expected_answer: str, generated_answer: str
    ) -> tuple[float, list[JudgeEvaluation]]:
        # The judge is asked to evaluate whether the answers are equal using both
        # orders of the answers, in case there is any positional bias in terms of
        # the order in which the answers are presented to the judge model.
        (
            first_order_equal,
            first_judge_evaluation,
        ) = await self._generate_judge_evaluation(question, expected_answer, generated_answer)
        if not first_order_equal:
            return 0.0, [first_judge_evaluation]

        (
            second_order_equal,
            second_judge_evaluation,
        ) = await self._generate_judge_evaluation(question, generated_answer, expected_answer)
        if second_order_equal:
            reward = 1.0
        else:
            reward = 0.0
        return reward, [first_judge_evaluation, second_judge_evaluation]

    async def _generate_judge_evaluation(
        self, question: str, first_answer: str, second_answer: str
    ) -> tuple[bool, JudgeEvaluation]:
        config = self.config
        responses_create_params = config.judge_responses_create_params.model_copy(deep=True)

        judge_prompt = self.JUDGE_PROMPT_TEMPLATE.format(
            question=question, first_answer=first_answer, second_answer=second_answer
        )
        responses_create_params.input = [
            NeMoGymEasyInputMessage(
                role="system",
                content=self.JUDGE_SYSTEM_MESSAGE,
            ),
            NeMoGymEasyInputMessage(
                role="user",
                content=judge_prompt,
            ),
        ]

        response = await self.server_client.post(
            server_name=config.judge_model_server.name,
            url_path="/v1/responses",
            json=responses_create_params,
        )
        judge_response = NeMoGymResponse.model_validate(await get_response_json(response))
        judge_evaluation = JudgeEvaluation(responses_create_params=responses_create_params, response=judge_response)

        # Currently, for all the cases in which the response from the LLM judge
        # does not conform to the expected format, the judge's evaluation is
        # treated as if the answers are not equal.  This may not be ideal, but it
        # is intended to minimize the number of failures for verify requests.
        last_output = judge_response.output[-1]
        if last_output.type != "message":
            return False, judge_evaluation

        last_content = last_output.content[-1]
        if last_content.type != "output_text":
            return False, judge_evaluation

        output_text = last_content.text
        equal_choice_position = output_text.find(self.JUDGE_EQUAL_LABEL)
        not_equal_choice_position = output_text.find(self.JUDGE_NOT_EQUAL_LABEL)

        # The first label that appears in the text is used for the evaluation.
        if equal_choice_position < 0:
            if not_equal_choice_position < 0:
                return False, judge_evaluation
            else:
                return False, judge_evaluation
        else:
            if not_equal_choice_position < 0:
                return True, judge_evaluation
            elif equal_choice_position < not_equal_choice_position:
                return True, judge_evaluation
            else:
                return False, judge_evaluation

    # ──────────────────────────────────────────────────────────
    # Aggregate metrics overrides
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _math_score_fn(r: dict) -> Dict[str, Union[float, bool]]:
        scores: Dict[str, Union[float, bool]] = {}
        if "library_reward" in r:
            scores["symbolic_accuracy"] = r["library_reward"]
        if "judge_evaluations" in r and r["judge_evaluations"] is not None:
            scores["judge_accuracy"] = r["reward"]
        return scores

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Compute math-specific metrics: pass@k, majority@k, per-sample statistics."""
        return compute_pass_majority_metrics(
            tasks,
            score_fn=self._math_score_fn,
            answer_key="extracted_answer",
        )[0]

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Select headline metrics for this math benchmark."""
        key: Dict[str, Any] = {}

        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]

        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]"))
        key.update(highest_k_metrics(agent_metrics, "pass@{k}", exclude_names=["no_answer"]))
        key.update(highest_k_metrics(agent_metrics, "majority@{k}", exclude_names=["no_answer"]))

        return key


if __name__ == "__main__":
    LibraryJudgeMathResourcesServer.run_webserver()
