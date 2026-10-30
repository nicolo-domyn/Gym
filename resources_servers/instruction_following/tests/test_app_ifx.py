# [CUSTOM] app._check_instructions resolves ifx:* ids after the verifiable_instructions registry.
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import _check_instructions  # noqa: E402


def test_registry_and_ifx_ids_are_both_checked():
    response = "Sunlight warms us. Under it we rest. Nothing else matters 3."
    result = _check_instructions(
        ["punctuation:no_comma", "ifx:acrostic_sentences", "ifx:digit_sum", "ifx:not_a_real_id"],
        [{}, {"word": "SUN"}, {"total": 3}, {}],
        response,
    )
    assert result == [True, True, True, False]
