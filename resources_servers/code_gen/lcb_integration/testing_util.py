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
import ast
import faulthandler
import json
import platform

# to run the solution files we're using a timing based approach
import signal
import sys
import time
import traceback

# used for debugging to time steps
from datetime import datetime
from decimal import Decimal
from enum import Enum
from io import BytesIO, StringIO

# from pyext import RuntimeModule
from types import ModuleType

# used for testing the code that reads from input
from unittest.mock import mock_open, patch

import math
import os

import numpy as np


import_string = "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(50000)\n"


def truncatefn(s, length=300):
    if isinstance(s, str):
        pass
    else:
        s = str(s)
    if len(s) <= length:
        return s

    return s[: length // 2] + "...(truncated) ..." + s[-length // 2 :]


class CODE_TYPE(Enum):
    call_based = 0
    standard_input = 1


# stuff for setting up signal timer
class TimeoutException(Exception):
    pass


def timeout_handler_factory(debug: bool):
    def timeout_handler(signum, frame):
        if debug:
            print("timeout occured: alarm went off")
        raise TimeoutException

    return timeout_handler


# used to capture stdout as a list
# from https://stackoverflow.com/a/16571630/6416660
# alternative use redirect_stdout() from contextlib
class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        # Make closing the StringIO a no-op
        self._stringio.close = lambda x: 1
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


# sys.stdin replacement for stdin programs. It behaves like a real stream:
# readline() keeps the newline and returns "" at EOF, iteration yields lines,
# and sys.stdin.buffer is a separate byte stream over the same input (the old
# mock was not iterable and its buffer.readline() always returned line 1).
class MockStdinWithBuffer:
    def __init__(self, inputs: str):
        self.inputs = inputs
        self._stringio = StringIO(inputs)
        self.buffer = MockBuffer(inputs)

    def read(self, *args):
        return self._stringio.read(*args)

    def readline(self, *args):
        return self._stringio.readline(*args)

    def readlines(self, *args):
        return self._stringio.readlines(*args)

    def __iter__(self):
        return iter(self._stringio.readline, "")

    def fileno(self):
        return 0

    def isatty(self):
        return False

    def __getattr__(self, name):
        # Delegate other attributes to StringIO
        return getattr(self._stringio, name)


class MockBuffer:
    def __init__(self, inputs: str):
        self.inputs = inputs.encode("utf-8")
        self._bytesio = BytesIO(self.inputs)

    def read(self, *args):
        return self._bytesio.read(*args)

    def readline(self, *args):
        return self._bytesio.readline(*args)

    def readlines(self, *args):
        return self._bytesio.readlines(*args)

    def __iter__(self):
        return iter(self._bytesio.readline, b"")

    def __getattr__(self, name):
        return getattr(self._bytesio, name)


_REAL_OPEN = open
_REAL_OS_READ = os.read
_STDIN_PATHS = (0, "/dev/stdin", "/proc/self/fd/0")


def _make_stdin_open(inputs: str):
    """`open` that serves the test input for fd 0 and /dev/stdin.

    Solutions write `open(0).read()` or `open(0, "rb")`. The exec'd code
    resolves `open` through `from io import *` (io.open), so patching
    builtins.open alone never reached it, and fd 0 of the grader is empty.
    Other files keep the previous behaviour (mock_open over the input).
    """
    fallback = mock_open(read_data=inputs)

    def _open(file, mode="r", *args, **kwargs):
        if file in _STDIN_PATHS:
            if "b" in mode:
                return BytesIO(inputs.encode("utf-8"))
            return StringIO(inputs)
        return fallback(file, mode, *args, **kwargs)

    return _open


def _make_os_read(stream: "MockBuffer"):
    def _read(fd, n):
        if fd == 0:
            return stream.read(n)
        return _REAL_OS_READ(fd, n)

    return _read


def clean_if_name(code: str) -> str:
    try:
        astree = ast.parse(code)
        last_block = astree.body[-1]
        if isinstance(last_block, ast.If):
            condition = last_block.test
            if ast.unparse(condition).strip() == "__name__ == '__main__'":
                code = (
                    ast.unparse(astree.body[:-1]) + "\n" + ast.unparse(last_block.body)  # type: ignore
                )
    except:
        pass

    return code


def make_function(code: str) -> str:
    try:
        import_stmts = []
        all_other_stmts = []
        astree = ast.parse(code)
        for stmt in astree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                import_stmts.append(stmt)
            else:
                all_other_stmts.append(stmt)

        function_ast = ast.FunctionDef(
            name="wrapped_function",
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=all_other_stmts,
            decorator_list=[],
            lineno=-1,
        )
        main_code = (
            import_string
            + "\n"
            + ast.unparse(import_stmts)  # type: ignore
            + "\n"
            + ast.unparse(function_ast)  # type: ignore
        )
        return main_code
    except Exception:
        return code


def call_method(method, inputs):
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)

    mock_stdin = MockStdinWithBuffer(inputs)
    stdin_open = _make_stdin_open(inputs)
    # The exec'd module binds `open` from `from io import *`; patch that name
    # in the solution's own globals as well as in builtins.
    method_globals = getattr(method, "__globals__", {})
    saved_open = method_globals.get("open", None)
    had_open = "open" in method_globals
    # make_function hoists imports to module level, so `from sys import stdin`
    # bound the real stream before sys.stdin is patched: rebind those names.
    real_stdin = {id(sys.stdin), id(sys.__stdin__)}
    saved_stdin = {k: v for k, v in method_globals.items() if id(v) in real_stdin}

    @patch("builtins.open", stdin_open)
    @patch("io.open", stdin_open)
    @patch("sys.stdin", mock_stdin)
    @patch("os.read", _make_os_read(mock_stdin.buffer))
    def _inner_call_method(_method):
        method_globals["open"] = stdin_open
        for name in saved_stdin:
            method_globals[name] = mock_stdin
        try:
            return _method()
        except SystemExit:
            pass
        finally:
            method_globals.update(saved_stdin)
            if had_open:
                method_globals["open"] = saved_open
            else:
                method_globals.pop("open", None)

    return _inner_call_method(method)


def get_function(compiled_sol, fn_name: str):  # type: ignore
    try:
        assert hasattr(compiled_sol, fn_name)
        return getattr(compiled_sol, fn_name)
    except Exception:
        return


def compile_code(code: str, timeout: int, fn_name: str | None = None):
    """Exec the code; return the object `fn_name` is looked up on.

    Call-based LeetCode-style answers define `class Solution`: the method is
    taken from an instance when Solution has `fn_name`. The old check
    (`"class Solution" in code`) also fired on stdin programs that merely
    mention the text, and on solutions whose `Solution` lacks the function,
    and failed them.
    """
    signal.alarm(timeout)
    try:
        tmp_sol = ModuleType("tmp_sol", "")
        exec(code, tmp_sol.__dict__)
        solution_cls = getattr(tmp_sol, "Solution", None)
        if fn_name and isinstance(solution_cls, type) and hasattr(solution_cls, fn_name):
            compiled_sol = solution_cls()
        else:
            compiled_sol = tmp_sol

        assert compiled_sol is not None
    finally:
        signal.alarm(0)

    return compiled_sol


# Absolute/relative tolerance for floating-point answers. Statements that print
# reals almost always accept 1e-6 (or looser); exact decimal comparison failed
# correct answers such as 0.400000000 vs 0.400000000000000022.
FLOAT_TOL = 1e-6


def _floats_close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=FLOAT_TOL, abs_tol=FLOAT_TOL)


def values_match(prediction, expected) -> bool:
    """Compare a call-based return value with the expected JSON value.

    Lists/tuples and dicts compare element-wise; floats with FLOAT_TOL (also
    inside lists); ints exactly; bools only with bools. numpy values are
    converted with .tolist() first, so an array return value no longer makes
    the comparison (and the reward computation) raise.
    """
    if hasattr(prediction, "tolist") and not isinstance(prediction, (str, bytes)):
        try:
            prediction = prediction.tolist()
        except Exception:
            pass
    if isinstance(expected, bool) or isinstance(prediction, bool):
        return isinstance(expected, bool) and isinstance(prediction, bool) and prediction == expected
    if isinstance(expected, (int, float)) and isinstance(prediction, (int, float)):
        if isinstance(expected, int) and isinstance(prediction, int):
            return prediction == expected
        return _floats_close(float(prediction), float(expected))
    if isinstance(expected, (list, tuple)) and isinstance(prediction, (list, tuple)):
        return len(prediction) == len(expected) and all(
            values_match(p, e) for p, e in zip(prediction, expected)
        )
    if isinstance(expected, dict) and isinstance(prediction, dict):
        return prediction.keys() == expected.keys() and all(values_match(prediction[k], expected[k]) for k in expected)
    try:
        return bool(prediction == expected)
    except Exception:
        return False


def convert_line_to_decimals(line: str) -> tuple[bool, list[Decimal]]:
    try:
        decimal_line = [Decimal(elem) for elem in line.split()]
    except:
        return False, []
    return True, decimal_line


def get_stripped_lines(val: str):
    ## you don't want empty lines to add empty list after splitlines!
    val = val.strip()

    return [val_line.strip() for val_line in val.split("\n")]


def grade_call_based(code: str, all_inputs: list, all_outputs: list, fn_name: str, timeout: int):
    # call-based clean up logic
    # need to wrap in try-catch logic after to catch the correct errors, but for now this is fine.
    code = import_string + "\n\n" + code
    compiled_sol = compile_code(code, timeout, fn_name=fn_name)

    if compiled_sol is None:
        return

    method = get_function(compiled_sol, fn_name)

    if method is None:
        return

    all_inputs = [[json.loads(line) for line in inputs.split("\n")] for inputs in all_inputs]

    all_outputs = [json.loads(output) for output in all_outputs]

    total_execution = 0
    all_results = []
    for idx, (gt_inp, gt_out) in enumerate(zip(all_inputs, all_outputs)):
        signal.alarm(timeout)
        faulthandler.enable()
        try:
            # can lock here so time is useful
            start = time.time()
            prediction = method(*gt_inp)
            total_execution += time.time() - start
            signal.alarm(0)

            tmp_result = values_match(prediction, gt_out)
            if not tmp_result:
                # Previous rules, kept as fallbacks: json.dumps equality and
                # np.allclose of scalars.
                try:
                    tmp_result = json.dumps(prediction) == json.dumps(gt_out)
                except Exception:
                    pass
            if not tmp_result:
                try:
                    tmp_result = bool(np.allclose(float(prediction), float(gt_out)))
                except Exception:
                    pass
            tmp_result = bool(tmp_result)

            all_results.append(tmp_result)

            if not tmp_result:
                return all_results, {
                    "output": truncatefn(prediction),
                    "inputs": truncatefn(gt_inp),
                    "expected": truncatefn(gt_out),
                    "error_code": -2,
                    "error_message": "Wrong Answer",
                }
        except Exception as e:
            signal.alarm(0)
            if "timeoutexception" in repr(e).lower():
                all_results.append(-3)
                return all_results, {
                    "error": repr(e),
                    "error_code": -3,
                    "error_message": "Time Limit Exceeded",
                    "inputs": truncatefn(gt_inp),
                    "expected": truncatefn(gt_out),
                }
            else:
                all_results.append(-4)
                return all_results, {
                    "error": repr(e),
                    "error_code": -4,
                    "error_message": "Runtime Error",
                    "inputs": truncatefn(gt_inp),
                    "expected": truncatefn(gt_out),
                    "traceback": traceback.format_exc(),
                }

        finally:
            signal.alarm(0)
            faulthandler.disable()

    return all_results, {"execution time": total_execution}


def grade_stdio(
    code: str,
    all_inputs: list,
    all_outputs: list,
    timeout: int,
    case_insensitive: bool = False,
):
    ## runtime doesn't interact well with __name__ == '__main__'
    code = clean_if_name(code)

    ## we wrap the given code inside another function
    code = make_function(code)

    compiled_sol = compile_code(code, timeout)
    if compiled_sol is None:
        return

    method = get_function(compiled_sol, "wrapped_function")

    if method is None:
        return

    all_results = []
    total_execution_time = 0
    for idx, (gt_inp, gt_out) in enumerate(zip(all_inputs, all_outputs)):
        signal.alarm(timeout)
        faulthandler.enable()

        signal.alarm(timeout)
        with Capturing() as captured_output:
            try:
                start = time.time()
                call_method(method, gt_inp)
                total_execution_time += time.time() - start
                # reset the alarm
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                if "timeoutexception" in repr(e).lower():
                    all_results.append(-3)
                    return all_results, {
                        "error": repr(e),
                        "error_code": -3,
                        "error_message": "Time Limit Exceeded",
                        "inputs": truncatefn(gt_inp),
                        "expected": truncatefn(gt_out),
                    }
                else:
                    all_results.append(-4)
                    return all_results, {
                        "error": repr(e),
                        "error_code": -4,
                        "error_message": "Runtime Error",
                        "inputs": truncatefn(gt_inp),
                        "expected": truncatefn(gt_out),
                        "traceback": traceback.format_exc(),
                    }

            finally:
                signal.alarm(0)
                faulthandler.disable()

        prediction = captured_output[0]

        stripped_prediction_lines = get_stripped_lines(prediction)
        stripped_gt_out_lines = get_stripped_lines(gt_out)

        ## WA happens in multiple circumstances
        ## so cache the return to make it clean!
        WA_send_args = {
            "output": truncatefn(prediction),
            "inputs": truncatefn(gt_inp),
            "expected": truncatefn(gt_out),
            "error_code": -2,
        }

        if len(stripped_prediction_lines) != len(stripped_gt_out_lines):
            all_results.append(-2)
            WA_send_args["error_message"] = "Wrong answer: mismatched output length"
            return all_results, WA_send_args

        for output_line_idx, (
            stripped_prediction_line,
            stripped_gt_out_line,
        ) in enumerate(zip(stripped_prediction_lines, stripped_gt_out_lines)):
            WA_send_args["error_message"] = (
                f"Wrong answer at {output_line_idx=}: {truncatefn(stripped_prediction_line)} != {truncatefn(stripped_gt_out_line)}"
            )

            ## CASE 1: exact match (any letter case when the statement allows it)
            if stripped_prediction_line == stripped_gt_out_line:
                continue
            if case_insensitive and stripped_prediction_line.lower() == stripped_gt_out_line.lower():
                continue

            ## CASE 2: element-wise comparision
            ## if there are floating elements
            ## use `decimal` library for good floating point comparision
            ## otherwise gotcha: np.isclose(50000000000000000, 50000000000000001) = True
            ## note that we should always be able to convert to decimals

            success, decimal_prediction_line = convert_line_to_decimals(stripped_prediction_line)
            if not success:
                all_results.append(-2)
                return all_results, WA_send_args
            success, decimal_gtout_line = convert_line_to_decimals(stripped_gt_out_line)
            if not success:
                all_results.append(-2)
                return all_results, WA_send_args

            if decimal_prediction_line == decimal_gtout_line:
                continue

            ## CASE 3: reals within FLOAT_TOL. Only when a token is written as
            ## a real (has '.', 'e' or 'E'), so integer answers stay exact.
            tokens = stripped_prediction_line.split() + stripped_gt_out_line.split()
            if (
                len(decimal_prediction_line) == len(decimal_gtout_line)
                and any(c in tok for tok in tokens for c in ".eE")
                and all(
                    p == g or (p.is_finite() and g.is_finite() and _floats_close(float(p), float(g)))
                    for p, g in zip(decimal_prediction_line, decimal_gtout_line)
                )
            ):
                continue

            all_results.append(-2)
            return all_results, WA_send_args
        all_results.append(True)

    return all_results, {"execution time": total_execution_time}


def run_test(in_outs, test=None, debug=False, timeout=6):
    """
    if test(generated_code) is not None it'll try to run the code.
    otherwise it'll just return an input and output pair.
    """
    signal.signal(signal.SIGALRM, timeout_handler_factory(debug))

    # Disable functionalities that can make destructive changes to the test.
    # max memory is set to 4GB
    reliability_guard(maximum_memory_bytes=4 * 1024**3)

    if debug:
        print(f"start = {datetime.now().time()}")

    # The in_outs is already loaded from the sample from the parent process
    # try:
    #     in_outs = json.loads(sample["input_output"])
    # except ValueError as e:
    #     raise e
    #     in_outs = None

    if in_outs:
        if in_outs.get("fn_name") is None:
            which_type = CODE_TYPE.standard_input  # Standard input
            method_name = None

        else:
            which_type = CODE_TYPE.call_based  # Call-based
            method_name = in_outs["fn_name"]

    if debug:
        print(f"loaded input_output = {datetime.now().time()}")

    if test is None:
        assert False, "should not happen: test code is none"
        return in_outs, {"error": "no test code provided"}
    elif test is not None:
        results = []
        if debug:
            print(f"loading test code = {datetime.now().time()}")

        if which_type == CODE_TYPE.call_based:
            signal.alarm(timeout)
            try:
                results, metadata = grade_call_based(
                    code=test,
                    all_inputs=in_outs["inputs"],
                    all_outputs=in_outs["outputs"],
                    fn_name=method_name,
                    timeout=timeout,
                )
                return results, metadata
            except Exception as e:
                return [-4], {
                    "error_code": -4,
                    "error_message": f"Error during testing: {e}",
                }
            finally:
                signal.alarm(0)
        elif which_type == CODE_TYPE.standard_input:
            # sol
            # if code has if __name__ == "__main__": then remove it

            signal.alarm(timeout)
            try:
                results, metadata = grade_stdio(
                    code=test,
                    all_inputs=in_outs["inputs"],
                    all_outputs=in_outs["outputs"],
                    timeout=timeout,
                    case_insensitive=bool(in_outs.get("case_insensitive")),
                )
                return results, metadata
            except Exception as e:
                return [-4], {
                    "error_code": -4,
                    "error_message": f"Error during testing: {e}",
                }
            finally:
                signal.alarm(0)


def _set_resource_limit(resource_type: str, target_soft_limit):
    import resource

    current_soft, current_hard = resource.getrlimit(resource_type)
    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as e:
            print(repr(e))


def reliability_guard(maximum_memory_bytes=None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)
    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource

        # The resource limit on RLIMIT_AS has been disabled because setting it caused additional out-of-memory (OOM) issues with Ray.
        # This happens since Ray and its subprocesses share the same virtual address space.
        # _set_resource_limit(resource.RLIMIT_AS, maximum_memory_bytes)
        _set_resource_limit(resource.RLIMIT_DATA, maximum_memory_bytes)

        if not platform.uname().system == "Darwin":
            _set_resource_limit(resource.RLIMIT_STACK, maximum_memory_bytes)

    faulthandler.disable()

    import builtins

    # builtins.exit = None
    builtins.quit = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    __builtins__["help"] = None

    import sys

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None
