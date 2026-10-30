# [CUSTOM] Each ifx:* checker on one response that follows it and one that does not.
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ifx_instructions import INSTRUCTION_DICT  # noqa: E402


CASES = [
    ("ifx:max_word_length", {"max_len": 7}, "Cats nap in warm sun.", "Cats nap in extraordinarily warm sun."),
    (
        "ifx:long_word_share",
        {"min_len": 7, "percent": 50},
        "Remarkable creatures sleep peacefully.",
        "The cat is on the mat today.",
    ),
    ("ifx:capital_letter_range", {"low": 2, "high": 3}, "Alice met Bob.", "ALICE met Bob."),
    (
        "ifx:letter_words",
        {"letter": "z", "num_words": 2},
        "The lazy zebra dozed.",
        "The lazy dog and the sleepy cat.",
    ),
    ("ifx:vowel_start_share", {"percent": 50}, "An owl ate eight insects.", "The cat sat on the mat."),
    (
        "ifx:sentence_word_cap",
        {"max_words": 4},
        "Short one here. Another short one.",
        "Short one here. This sentence is far too long.",
    ),
    (
        "ifx:sentence_word_floor",
        {"min_words": 5},
        "This sentence has six words. So does this one, friend.",
        "This sentence has six words. Too short.",
    ),
    (
        "ifx:acrostic_sentences",
        {"word": "SUN"},
        "Sunlight warms. Under it we rest. Nothing else matters.",
        "Sunlight warms. Nothing else matters. Under it we rest.",
    ),
    (
        "ifx:sentence_final_letter",
        {"letter": "e"},
        "We like the shade. Nobody can hide.",
        "We like the shade. It is hot.",
    ),
    ("ifx:one_comma_per_sentence", {}, "Yes, we can. Today, we rest.", "Yes, we can. We rest."),
    ("ifx:digit_sum", {"total": 10}, "We bought 4 apples and 6 pears.", "We bought 4 apples and 7 pears."),
    (
        "ifx:double_letter_words",
        {"num_words": 3},
        "The little book fell off the tall shelf.",
        "The little cat sat on a mat.",
    ),
    (
        "ifx:paragraph_word_range",
        {"low": 3, "high": 5},
        "One two three four.\n\nFive six seven.",
        "One two three four.",
    ),
    ("ifx:sentence_same_initials", {}, "Birds fly by. Cats nap, curled.", "Birds fly by. Cats nap."),
    ("ifx:no_repeat_in_sentence", {}, "The cat sat. The dog ran.", "The cat sat on the mat."),
    ("ifx:letter_share_cap", {"letter": "e", "percent": 10}, "A cat ran far away.", "Everyone here sees trees."),
    (
        "ifx:alternating_word_length",
        {},
        "A bright cat jumped onto broken old fences.",
        "A cat jumped onto broken fences today.",
    ),
    ("ifx:nondecreasing_word_length", {}, "I am the best cook around tonight.", "I am the best cook in town."),
]


@pytest.mark.parametrize("instruction_id,kwargs,follows,breaks", CASES, ids=[c[0] for c in CASES])
def test_ifx_instruction(instruction_id, kwargs, follows, breaks):
    instruction = INSTRUCTION_DICT[instruction_id](instruction_id)
    description = instruction.build_description(**kwargs)
    assert description
    assert instruction.check_following(follows)
    assert not instruction.check_following(breaks)
    assert not instruction.check_following("")


def test_every_instruction_is_tested():
    assert {c[0] for c in CASES} == set(INSTRUCTION_DICT)


def test_args_round_trip():
    # Training rows store get_instruction_args(); building again from them must give the same text.
    for instruction_id, cls in INSTRUCTION_DICT.items():
        first = cls(instruction_id)
        description = first.build_description()
        second = cls(instruction_id)
        assert second.build_description(**(first.get_instruction_args() or {})) == description
