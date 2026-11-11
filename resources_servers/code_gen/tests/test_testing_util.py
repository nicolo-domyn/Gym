# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grading rules of lcb_integration.testing_util.

The grade_* functions are called directly: run_test() applies
reliability_guard(), which disables os functions for the whole process.
"""

import json
import signal

import numpy as np
import pytest

from resources_servers.code_gen.app import all_tests_passed
from resources_servers.code_gen.lcb_integration.testing_util import (
    grade_call_based,
    grade_stdio,
    timeout_handler_factory,
    values_match,
)


@pytest.fixture(autouse=True)
def _alarm_handler():
    old = signal.signal(signal.SIGALRM, timeout_handler_factory(False))
    yield
    signal.signal(signal.SIGALRM, old)


def stdio(code: str, inp: str, out: str) -> list:
    return grade_stdio(code, [inp], [out], timeout=5)[0]


@pytest.mark.parametrize(
    "code",
    [
        "import sys\nfor line in sys.stdin:\n    print(int(line) * 2)",
        "import sys\ninput = sys.stdin.buffer.readline\nprint(int(input()) * 2)\nprint(int(input()) * 2)",
        "for x in open(0).read().split():\n    print(int(x) * 2)",
        "data = open(0, 'rb').read().split()\nprint('\\n'.join(str(int(x) * 2) for x in data))",
        "import os\nfor x in os.read(0, 10**6).split():\n    print(int(x) * 2)",
        "from sys import stdin\nprint(int(stdin.readline()) * 2)\nprint(int(stdin.readline()) * 2)",
        "import sys\nlines = sys.stdin.readlines()\nprint(int(lines[0]) * 2)\nprint(int(lines[1]) * 2)",
        "a = int(input())\nb = int(input())\nprint(a * 2)\nprint(b * 2)",
    ],
)
def test_stdin_idioms(code):
    assert stdio(code, "1\n2\n", "2\n4") == [True]


def test_stdin_readline_returns_empty_at_eof():
    code = "import sys\nn = 0\nwhile sys.stdin.readline():\n    n += 1\nprint(n)"
    assert stdio(code, "a\nb\n", "2") == [True]


def test_stdin_program_mentioning_class_solution():
    code = "class Solution:\n    def one(self):\n        return 1\nprint(int(input()) + Solution().one())"
    assert stdio(code, "1\n", "2") == [True]


@pytest.mark.parametrize(
    ("printed", "expected", "ok"),
    [
        ("0.400000000", "0.400000000000000022", True),
        ("3.1415927", "3.14159265", True),
        ("0.41", "0.4", False),
        ("50000000000000001", "50000000000000000", False),
        ("1 2.0000001", "1 2", True),
        ("YES", "Yes", False),
    ],
)
def test_stdout_float_tolerance(printed, expected, ok):
    res = stdio(f"print({printed!r})", "", expected)
    assert (res == [True]) is ok


def call(code: str, args: list, expected) -> list:
    inp = "\n".join(json.dumps(a) for a in args)
    return grade_call_based(code, [inp], [json.dumps(expected)], "f", timeout=5)[0]


def test_call_based_float_list_and_numpy():
    assert call("def f(x):\n    return [x / 3, 2.0]", [1], [0.3333333, 2]) == [True]
    assert call("import numpy as np\ndef f(x):\n    return np.array([x, x])", [1], [1, 1]) == [True]
    assert call("import numpy as np\ndef f(x):\n    return np.array([x, 0])", [1], [1, 1]) == [False]


def test_call_based_bool_vs_int_unchanged():
    # Unchanged from before these fixes: Python's 1 == True, so the
    # fallback rules accept 1 for True.
    assert call("def f(x):\n    return 1", [1], True) == [True]
    assert call("def f(x):\n    return False", [1], True) == [False]


def test_call_based_solution_class_and_module_function():
    assert call("class Solution:\n    def f(self, x):\n        return x + 1", [1], 2) == [True]
    # A helper class named Solution without the method: use the function.
    assert call("class Solution:\n    pass\n\ndef f(x):\n    return x + 1", [1], 2) == [True]


def test_values_match_nested():
    assert values_match({"a": [1.0000000001, (2, 3)]}, {"a": [1, [2, 3]]})
    assert not values_match([1, 2], [1, 2, 3])


def test_all_tests_passed():
    assert all_tests_passed([True, True])
    assert not all_tests_passed([True, -2])
    assert not all_tests_passed([True, np.array([True, False])])
    assert not all_tests_passed([])
