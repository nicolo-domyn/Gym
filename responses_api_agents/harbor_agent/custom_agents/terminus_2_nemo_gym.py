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
import time
from pathlib import Path
from typing import Any, Literal

from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.environments.base import BaseEnvironment
from harbor.llms.base import BaseLLM
from harbor.models.agent.context import AgentContext

from responses_api_agents.harbor_agent.custom_agents.llms.nemo_gym_llm import NemoGymLLM
from responses_api_agents.harbor_agent.custom_envs.singularity.singularity import MemoryLimitExceededError


class Terminus2NemoGym(Terminus2):
    """Terminus2 variant that uses a NeMo Gym model server-compatible BaseLLM."""

    @staticmethod
    def name() -> str:
        return "terminus-2-nemo-gym"

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        max_turns: int | None = None,
        parser_name: str = "json",
        api_base: str | None = None,
        temperature: float = 0.7,
        reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "default"] | None = None,
        collect_rollout_details: bool = False,
        session_id: str | None = None,
        enable_summarize: bool = True,
        proactive_summarization_threshold: int = 8000,
        max_thinking_tokens: int | None = None,
        model_info: dict | None = None,
        trajectory_config: dict | None = None,
        tmux_pane_width: int = 160,
        tmux_pane_height: int = 40,
        store_all_messages: bool = False,
        record_terminal_session: bool = True,
        llm: BaseLLM | None = None,
        interleaved_thinking: bool = False,
        responses_create_params: dict[str, Any] | None = None,
        nemo_model_server_timeout_sec: float = 120.0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if llm is None:
            if model_name is None:
                raise ValueError("model_name is required for Terminus2NemoGym")
            if api_base is None:
                raise ValueError("api_base is required for Terminus2NemoGym when llm is not provided")

            llm = NemoGymLLM(
                model_name=model_name,
                api_base=api_base,
                collect_rollout_details=collect_rollout_details,
                model_info=model_info,
                responses_create_params=responses_create_params,
                timeout_sec=nemo_model_server_timeout_sec,
                # [CUSTOM] trial.py's AgentFactory.create_agent_from_config
                # passes the real per-trial logger (job.log FileHandler, DEBUG level)
                # as a `logger` kwarg, which lands in **kwargs here since this __init__
                # doesn't declare it explicitly. super().__init__() below still forwards
                # it on unchanged (normal self.logger wiring) -- reading it early too just
                # lets NemoGymLLM's own diagnostic logging land in the same job.log.
                logger=kwargs.get("logger"),
            )

        super().__init__(
            logs_dir=logs_dir,
            model_name=model_name,
            max_turns=max_turns,
            parser_name=parser_name,
            api_base=api_base,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            collect_rollout_details=collect_rollout_details,
            session_id=session_id,
            enable_summarize=enable_summarize,
            proactive_summarization_threshold=proactive_summarization_threshold,
            max_thinking_tokens=max_thinking_tokens,
            model_info=model_info,
            trajectory_config=trajectory_config,
            tmux_pane_width=tmux_pane_width,
            tmux_pane_height=tmux_pane_height,
            store_all_messages=store_all_messages,
            record_terminal_session=record_terminal_session,
            llm=llm,
            interleaved_thinking=interleaved_thinking,
            *args,
            **kwargs,
        )

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        """Override run() to gracefully handle agent errors.

        The parent's run() has a finally block that saves rollout_details and
        dumps the trajectory before any exception propagates. By catching
        exceptions here, we let Harbor's trial system proceed normally with the
        verifier — returning the agent's conversation history from all completed
        turns (reward will be 0 for incomplete work) instead of crashing the
        entire rollout batch.
        """
        self._memory_limit_exceeded = False
        run_start = time.monotonic()
        try:
            await super().run(instruction, environment, context)
        except MemoryLimitExceededError as e:
            self._memory_limit_exceeded = True
            self.logger.info(f"Agent error: {type(e).__name__}: {e}. Returning history from completed turns.")
        except asyncio.CancelledError:
            # [CUSTOM] Log the live counters here, at the exact moment of cancellation. 
            # Must re-raise not to break asyncio's cancellation contract.
            elapsed = time.monotonic() - run_start
            self.logger.info(
                f"[terminus_2_nemo_gym] Agent cancelled after {elapsed:.1f}s wall-clock, "
                f"n_episodes={self._n_episodes}, n_llm_calls={len(self._api_request_times)}, "
                f"last_call_durations_msec={self._api_request_times[-3:]}"
            )
            raise
        except Exception as e:
            self.logger.info(f"Agent error: {type(e).__name__}: {e}. Returning history from completed turns.")
        finally:
            # [CUSTOM] Work around laude-institute/harbor bug, fixed upstream in PR #686
            # (laude-institute/harbor@8c040e1b, 2026-02-13) but absent from our pinned
            # commit (9dddd797b, 2026-01-27 -- 43 commits before the fix).
            #
            # The bug: Terminus2.run()'s own finally block reports n_episodes from a local
            # `actual_episodes` var that only gets assigned if _run_agent_loop() returns
            # normally. On ANY exception out of the loop -- including the
            # AgentTimeoutError/CancelledError case above -- it silently stays at its
            # pre-loop default of 0, even if many episodes actually completed.
            # self._n_episodes is updated live inside the loop
            # (`self._n_episodes = episode + 1` after each turn) and is always correct;
            # overwriting with it here is a no-op in the non-buggy case.
            if context.metadata is not None:
                context.metadata["n_episodes"] = self._n_episodes
            self._write_agent_error_flags()

    def _write_agent_error_flags(self) -> None:
        """Write agent error flags to disk for app.py to pick up."""
        try:
            flags: dict[str, bool] = {
                "memory_limit_exceeded": self._memory_limit_exceeded,
            }
            llm = getattr(self, "_llm", None)
            if llm and isinstance(llm, NemoGymLLM):
                flags["context_length_exceeded"] = llm.context_length_exceeded
            (self.logs_dir / "agent_error_flags.json").write_text(json.dumps(flags))
        except Exception:
            pass  # Don't let flag-writing failures break the agent
