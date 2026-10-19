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
import multiprocessing
import queue
import threading
from typing import List, Literal, Optional

from fastapi import FastAPI
from verifiable_instructions import instructions_registry

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


# [CUSTOM]
# check_following() is plain CPU-bound Python (regexes, nltk), and some checks are
# quadratic in the response length: on a degenerate ~100k-token response a single call
# runs for >1h, blocking this server's event loop and wedging every pending rollout.
# Each verify therefore runs its checks in a worker process that is killed on timeout.
CHECK_TIMEOUT_S = 60.0
MAX_CONCURRENT_CHECKS = 16


class InstructionFollowingResourcesServerConfig(BaseResourcesServerConfig):
    pass


class InstructionFollowingRunRequest(BaseRunRequest):
    id: int
    instruction_id_list: List
    prompt: str
    kwargs: List
    grading_mode: Literal[
        "binary",
        "fraction",
    ] = "binary"


class InstructionFollowingVerifyRequest(InstructionFollowingRunRequest, BaseVerifyRequest):
    pass


class InstructionFollowingVerifyResponse(BaseVerifyResponse):
    follow_all_instructions: bool
    follow_instruction_list: List[bool]
    kwargs: List
    instruction_id_list: List
    prompt: str
    grading_mode: Literal[
        "binary",
        "fraction",
    ] = "binary"


def _check_instructions(instruction_list: List, kwargs_list: List, final_response_text: str) -> List[bool]:
    is_following_list = []

    for instruction_id, kwargs in zip(instruction_list, kwargs_list):
        try:
            # Create instruction instance
            instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
            instruction = instruction_cls(instruction_id)

            # Handle None kwargs
            if kwargs is None:
                kwargs = {}

            # Filter out None values from kwargs
            filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}

            # Build the instruction description with the provided kwargs
            instruction.build_description(**filtered_kwargs)

            # Check if the response follows the instruction
            if instruction.check_following(final_response_text):
                is_following_list.append(True)
            else:
                is_following_list.append(False)

        except Exception as e:
            # If there's an error processing the instruction, mark as failed
            print(f"Error processing instruction {instruction_id}: {e}")
            is_following_list.append(False)

    return is_following_list


def _check_instructions_child(conn, *args) -> None:
    # One-shot child of the previous version (a fresh process per check). Kept so servers
    # started before this version, whose children import this file from disk, keep working.
    conn.send(_check_instructions(*args))
    conn.close()


def _checker_worker(conn) -> None:
    conn.send("ready")
    while True:
        try:
            args = conn.recv()
        except EOFError:  # the server closed the pipe
            return
        conn.send(_check_instructions(*args))


class _CheckerPool:
    """Persistent checker processes, each running one check at a time.

    A fresh process per check re-imported this module (and nemo_gym) every time, ~6s of CPU:
    set_forkserver_preload(["__main__"]) does nothing on CPython 3.13, whose forkserver looks
    up a "main_path" key that get_preparation_data never sets. It also made every check load
    this file from disk mid-run, so a check that read it while it was being rewritten (job
    61445978) found no _check_instructions_child and scored every instruction as failed.
    Workers import once and are reused; one is replaced only after a timeout or a crash.
    """

    STARTUP_TIMEOUT_S = 600.0

    def __init__(self, size: int):
        self._size = size
        self._idle: "queue.SimpleQueue" = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._live = 0

    def _spawn(self):
        # forkserver, not fork: this process runs uvicorn plus several threads.
        ctx = multiprocessing.get_context("forkserver")
        conn, child_conn = ctx.Pipe()
        proc = ctx.Process(target=_checker_worker, args=(child_conn,), daemon=True)
        proc.start()
        child_conn.close()
        if not conn.poll(self.STARTUP_TIMEOUT_S) or conn.recv() != "ready":
            raise RuntimeError(f"Instruction checker did not start within {self.STARTUP_TIMEOUT_S}s")
        return proc, conn

    def _acquire(self, fresh: bool = False):
        while not fresh:
            try:
                worker = self._idle.get_nowait()
            except queue.Empty:
                break
            if worker[0].is_alive():
                return worker
            self._discard(worker)
        with self._lock:
            # A retry after a crash goes one over the size rather than wait for a worker.
            grow = fresh or self._live < self._size
            if grow:
                self._live += 1
        if not grow:
            return self._idle.get()
        try:
            return self._spawn()
        except BaseException:
            with self._lock:
                self._live -= 1
            raise

    def _discard(self, worker) -> None:
        proc, conn = worker
        conn.close()
        if proc.is_alive():
            proc.kill()
        proc.join()
        with self._lock:
            self._live -= 1

    def check(
        self, instruction_list: List, kwargs_list: List, final_response_text: str, timeout_s: float
    ) -> Optional[List[bool]]:
        """Returns None if the checks did not finish within timeout_s.

        A worker that dies mid-check is retried once on a fresh one before the instructions
        are scored as failed, so a lost process is not silently a zero reward.
        """
        for attempt in range(2):
            worker = self._acquire(fresh=attempt > 0)
            _, conn = worker
            try:
                conn.send((instruction_list, kwargs_list, final_response_text))
                if not conn.poll(timeout_s):
                    self._discard(worker)
                    return None
                result = conn.recv()
            except (EOFError, OSError) as e:  # the worker died (BrokenPipeError is an OSError)
                self._discard(worker)
                print(f"Instruction checker died ({e!r}), attempt {attempt + 1} of 2")
                continue
            except BaseException:
                self._discard(worker)
                raise
            self._idle.put(worker)
            return result
        return [False] * len(instruction_list)


_CHECKERS = _CheckerPool(MAX_CONCURRENT_CHECKS)


class InstructionFollowingResourcesServer(SimpleResourcesServer):
    config: InstructionFollowingResourcesServerConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_nltk_data()
        self._check_slots = asyncio.Semaphore(MAX_CONCURRENT_CHECKS)

    def _ensure_nltk_data(self):
        """Ensure required NLTK data is available at startup.

        nltk.download() always fetches the remote package index even when the
        data is already present. Guard with a local find() first to skip the
        download when the data already exists.
        """
        try:
            import nltk

            try:
                nltk.data.find("tokenizers/punkt_tab")
            except LookupError:
                nltk.download("punkt_tab", quiet=True)
        except ImportError:
            # ifbench not available, skip
            pass
        except Exception as e:
            print(f"NLTK setup warning: {e}")

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        return app

    async def verify(self, body: InstructionFollowingVerifyRequest) -> InstructionFollowingVerifyResponse:
        # Get the final text response from the last output item
        final_response_text = ""
        if body.response.output:
            last_output = body.response.output[-1]
            if hasattr(last_output, "content") and last_output.content:
                # Extract text from the nested content structure
                final_response_text = last_output.content[0].text

        # Verify each instruction using the verifiable instructions
        instruction_list = body.instruction_id_list
        kwargs_list = body.kwargs
        async with self._check_slots:
            is_following_list = await asyncio.to_thread(
                _CHECKERS.check,
                instruction_list,
                kwargs_list,
                final_response_text,
                CHECK_TIMEOUT_S,
            )
        if is_following_list is None:
            print(
                f"Instruction checks timed out after {CHECK_TIMEOUT_S}s (id={body.id}, "
                f"instructions={instruction_list}, response_chars={len(final_response_text)}); "
                "marking all as not followed"
            )
            is_following_list = [False] * len(instruction_list)

        # Calculate overall success
        reward_mode = getattr(body, "grading_mode", "binary")
        if reward_mode == "binary":
            reward = float(all(is_following_list))
        elif reward_mode == "fraction":
            reward = float((sum(is_following_list) / len(is_following_list)) if is_following_list else 0.0)
        else:
            raise ValueError(f"Invalid reward mode: {reward_mode}")

        return InstructionFollowingVerifyResponse(
            **body.model_dump(),
            reward=float(reward),
            follow_all_instructions=all(is_following_list),
            follow_instruction_list=is_following_list,
        )


if __name__ == "__main__":
    InstructionFollowingResourcesServer.run_webserver()
