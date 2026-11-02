# [CUSTOM] Extra verifiable instructions ("ifx:*"), looked up by app.py after the
# verifiable_instructions registry.
#
# Letter- and word-level constraints in the spirit of IFBench's OOD set (word lengths, letter
# and vowel counts, per-sentence shape, ratios), none identical to an IFBench test constraint
# (allenai/IFBench IFBench_test.jsonl, 58 ids) nor to the 54 already trained on. Each class has
# the verifiable_instructions interface: build_description(**kwargs) -> str and
# check_following(response) -> bool. Training data is built from the same classes
# (domyn-rl scripts/build_ifx_data.py), so prompt wording and grading cannot drift apart.
#
# Definitions used by every check:
#   word     = a run of ASCII letters, apostrophes inside it allowed ("don't" is one word);
#              numbers and symbols are not words
#   sentence = nltk sent_tokenize on the response, keeping sentences that contain a word
#   letter   = A-Z / a-z
import random
import re
from typing import Dict, List, Optional

import nltk


_WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")
_VOWELS = set("aeiou")


def _words(text: str) -> List[str]:
    return _WORD_RE.findall(text)


def _letters(word: str) -> str:
    return re.sub(r"[^A-Za-z]", "", word)


def _sentences(text: str) -> List[str]:
    return [s for s in nltk.sent_tokenize(text) if _words(s)]


def _paragraphs(text: str) -> List[str]:
    return [p for p in re.split(r"\n\s*\n", text.strip()) if _words(p)]


class Instruction:
    """Same interface as verifiable_instructions.instructions.Instruction."""

    def __init__(self, instruction_id: str):
        self.id = instruction_id

    def build_description(self, **kwargs) -> str:
        raise NotImplementedError

    def get_instruction_args(self) -> Optional[Dict]:
        raise NotImplementedError

    def get_instruction_args_keys(self) -> List[str]:
        return list((self.get_instruction_args() or {}).keys())

    def check_following(self, value: str) -> bool:
        raise NotImplementedError


class MaxWordLength(Instruction):
    def build_description(self, *, max_len: Optional[int] = None) -> str:
        self._max_len = max_len if max_len is not None else random.randint(7, 10)
        return f"Do not use any word longer than {self._max_len} letters."

    def get_instruction_args(self):
        return {"max_len": self._max_len}

    def check_following(self, value):
        words = _words(value)
        return bool(words) and all(len(_letters(w)) <= self._max_len for w in words)


class LongWordShare(Instruction):
    def build_description(self, *, min_len: Optional[int] = None, percent: Optional[int] = None) -> str:
        self._min_len = min_len if min_len is not None else 7
        self._percent = percent if percent is not None else random.choice([20, 25, 30, 35])
        return f"At least {self._percent}% of the words in your response must be {self._min_len} or more letters long."

    def get_instruction_args(self):
        return {"min_len": self._min_len, "percent": self._percent}

    def check_following(self, value):
        words = _words(value)
        long_words = sum(len(_letters(w)) >= self._min_len for w in words)
        return bool(words) and 100 * long_words >= self._percent * len(words)


class CapitalLetterRange(Instruction):
    def build_description(self, *, low: Optional[int] = None, high: Optional[int] = None) -> str:
        self._low = low if low is not None else random.randint(10, 40)
        self._high = high if high is not None else self._low + random.choice([10, 15, 20])
        return f"Your response must contain between {self._low} and {self._high} capital letters in total."

    def get_instruction_args(self):
        return {"low": self._low, "high": self._high}

    def check_following(self, value):
        return self._low <= sum(c.isascii() and c.isupper() for c in value) <= self._high


class LetterWords(Instruction):
    def build_description(self, *, letter: Optional[str] = None, num_words: Optional[int] = None) -> str:
        self._letter = letter if letter is not None else random.choice("zqxjkvwy")
        self._num_words = num_words if num_words is not None else random.randint(3, 8)
        return f"At least {self._num_words} different words in your response must contain the letter '{self._letter}'."

    def get_instruction_args(self):
        return {"letter": self._letter, "num_words": self._num_words}

    def check_following(self, value):
        distinct = {w.lower() for w in _words(value) if self._letter in w.lower()}
        return len(distinct) >= self._num_words


class VowelStartShare(Instruction):
    def build_description(self, *, percent: Optional[int] = None) -> str:
        self._percent = percent if percent is not None else random.choice([35, 40, 45])
        return f"At least {self._percent}% of the words in your response must begin with a vowel (a, e, i, o or u)."

    def get_instruction_args(self):
        return {"percent": self._percent}

    def check_following(self, value):
        words = _words(value)
        vowel_start = sum(w[0].lower() in _VOWELS for w in words)
        return bool(words) and 100 * vowel_start >= self._percent * len(words)


class SentenceWordCap(Instruction):
    def build_description(self, *, max_words: Optional[int] = None) -> str:
        self._max_words = max_words if max_words is not None else random.randint(8, 14)
        return f"Every sentence in your response must have at most {self._max_words} words."

    def get_instruction_args(self):
        return {"max_words": self._max_words}

    def check_following(self, value):
        sentences = _sentences(value)
        return bool(sentences) and all(len(_words(s)) <= self._max_words for s in sentences)


class SentenceWordFloor(Instruction):
    def build_description(self, *, min_words: Optional[int] = None) -> str:
        self._min_words = min_words if min_words is not None else random.randint(12, 18)
        return f"Every sentence in your response must have at least {self._min_words} words."

    def get_instruction_args(self):
        return {"min_words": self._min_words}

    def check_following(self, value):
        sentences = _sentences(value)
        return bool(sentences) and all(len(_words(s)) >= self._min_words for s in sentences)


ACROSTIC_WORDS = [
    "LIGHT",
    "RIVER",
    "STONE",
    "BRAVE",
    "CLOUD",
    "MAPLE",
    "OCEAN",
    "TIGER",
    "GRACE",
    "SMILE",
    "PLANT",
    "HONEY",
    "FLAME",
    "DREAM",
    "SPARK",
    "BLOOM",
    "TRAIL",
    "NORTH",
    "QUIET",
    "EARTH",
    "MOON",
    "STAR",
    "WIND",
    "LAKE",
    "BIRD",
    "SNOW",
    "ROAD",
    "HOPE",
    "GOLD",
    "WAVE",
    "GARDEN",
    "SILVER",
    "BRIDGE",
    "FOREST",
    "CASTLE",
    "PLANET",
]


class AcrosticSentences(Instruction):
    def build_description(self, *, word: Optional[str] = None) -> str:
        self._word = (word if word is not None else random.choice(ACROSTIC_WORDS)).upper()
        return (
            f"Your response must have exactly {len(self._word)} sentences, and their first letters, "
            f"in order, must spell {self._word}."
        )

    def get_instruction_args(self):
        return {"word": self._word}

    def check_following(self, value):
        sentences = _sentences(value)
        initials = "".join(_words(s)[0][0].upper() for s in sentences)
        return initials == self._word


class SentenceFinalLetter(Instruction):
    def build_description(self, *, letter: Optional[str] = None) -> str:
        self._letter = letter if letter is not None else random.choice("estdny")
        return f"The last word of every sentence must end with the letter '{self._letter}'."

    def get_instruction_args(self):
        return {"letter": self._letter}

    def check_following(self, value):
        sentences = _sentences(value)
        return bool(sentences) and all(_words(s)[-1].lower().endswith(self._letter) for s in sentences)


class OneCommaPerSentence(Instruction):
    def build_description(self) -> str:
        return "Every sentence in your response must contain exactly one comma."

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        sentences = _sentences(value)
        return bool(sentences) and all(s.count(",") == 1 for s in sentences)


class DigitSum(Instruction):
    def build_description(self, *, total: Optional[int] = None) -> str:
        self._total = total if total is not None else random.randint(15, 45)
        return f"The digits (0-9) that appear in your response must add up to exactly {self._total}."

    def get_instruction_args(self):
        return {"total": self._total}

    def check_following(self, value):
        return sum(int(c) for c in value if c in "0123456789") == self._total


class DoubleLetterWords(Instruction):
    def build_description(self, *, num_words: Optional[int] = None) -> str:
        self._num_words = num_words if num_words is not None else random.randint(5, 12)
        return (
            f"Use at least {self._num_words} different words that contain the same letter twice in a row "
            "(such as 'book' or 'little')."
        )

    def get_instruction_args(self):
        return {"num_words": self._num_words}

    def check_following(self, value):
        distinct = {w.lower() for w in _words(value) if re.search(r"([a-z])\1", w.lower())}
        return len(distinct) >= self._num_words


class ParagraphWordRange(Instruction):
    def build_description(self, *, low: Optional[int] = None, high: Optional[int] = None) -> str:
        self._low = low if low is not None else random.choice([30, 40, 50, 60])
        self._high = high if high is not None else self._low + random.choice([20, 30, 40])
        return (
            "Write at least two paragraphs, separated by a blank line, and make every paragraph "
            f"between {self._low} and {self._high} words long."
        )

    def get_instruction_args(self):
        return {"low": self._low, "high": self._high}

    def check_following(self, value):
        paragraphs = _paragraphs(value)
        return len(paragraphs) >= 2 and all(self._low <= len(_words(p)) <= self._high for p in paragraphs)


class SentenceSameInitials(Instruction):
    def build_description(self) -> str:
        return "In every sentence, the first word and the last word must start with the same letter."

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        sentences = _sentences(value)
        return bool(sentences) and all(_words(s)[0][0].lower() == _words(s)[-1][0].lower() for s in sentences)


class NoRepeatInSentence(Instruction):
    def build_description(self) -> str:
        return "No word may appear more than once within the same sentence (ignoring case)."

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        sentences = _sentences(value)
        for s in sentences:
            words = [w.lower() for w in _words(s)]
            if len(words) != len(set(words)):
                return False
        return bool(sentences)


class LetterShareCap(Instruction):
    # English letter frequencies: e ~12.7%, t ~9.1%, a ~8.2%; the caps sit well below them.
    _CAPS = {"e": [8, 9], "t": [5, 6], "a": [5, 6]}

    def build_description(self, *, letter: Optional[str] = None, percent: Optional[int] = None) -> str:
        self._letter = letter if letter is not None else random.choice(sorted(self._CAPS))
        self._percent = percent if percent is not None else random.choice(self._CAPS[self._letter])
        return (
            f"The letter '{self._letter}' must make up less than {self._percent}% of all the letters in your response."
        )

    def get_instruction_args(self):
        return {"letter": self._letter, "percent": self._percent}

    def check_following(self, value):
        letters = [c.lower() for c in value if c.isascii() and c.isalpha()]
        return bool(letters) and 100 * letters.count(self._letter) < self._percent * len(letters)


class AlternatingWordLength(Instruction):
    def build_description(self) -> str:
        return (
            "Your first sentence must have at least 6 words, alternating between short words "
            "(4 letters or fewer) and long words (5 letters or more), starting with a short word."
        )

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        sentences = _sentences(value)
        if not sentences:
            return False
        lengths = [len(_letters(w)) for w in _words(sentences[0])]
        return len(lengths) >= 6 and all((n <= 4) == (i % 2 == 0) for i, n in enumerate(lengths))


class NondecreasingWordLength(Instruction):
    def build_description(self) -> str:
        return "Your first sentence must have at least 6 words, each one at least as long as the word before it."

    def get_instruction_args(self):
        return None

    def check_following(self, value):
        sentences = _sentences(value)
        if not sentences:
            return False
        lengths = [len(_letters(w)) for w in _words(sentences[0])]
        return len(lengths) >= 6 and all(a <= b for a, b in zip(lengths, lengths[1:]))


INSTRUCTION_DICT = {
    "ifx:max_word_length": MaxWordLength,
    "ifx:long_word_share": LongWordShare,
    "ifx:capital_letter_range": CapitalLetterRange,
    "ifx:letter_words": LetterWords,
    "ifx:vowel_start_share": VowelStartShare,
    "ifx:sentence_word_cap": SentenceWordCap,
    "ifx:sentence_word_floor": SentenceWordFloor,
    "ifx:acrostic_sentences": AcrosticSentences,
    "ifx:sentence_final_letter": SentenceFinalLetter,
    "ifx:one_comma_per_sentence": OneCommaPerSentence,
    "ifx:digit_sum": DigitSum,
    "ifx:double_letter_words": DoubleLetterWords,
    "ifx:paragraph_word_range": ParagraphWordRange,
    "ifx:sentence_same_initials": SentenceSameInitials,
    "ifx:no_repeat_in_sentence": NoRepeatInSentence,
    "ifx:letter_share_cap": LetterShareCap,
    "ifx:alternating_word_length": AlternatingWordLength,
    "ifx:nondecreasing_word_length": NondecreasingWordLength,
}
