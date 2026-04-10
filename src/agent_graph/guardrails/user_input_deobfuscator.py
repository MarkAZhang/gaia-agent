import base64
import binascii
import codecs
import re
import unicodedata
from dataclasses import dataclass

from nltk.corpus import stopwords

ENGLISH_STOPWORDS: set[str] = set(stopwords.words("english"))

# Minimum number of English stopwords that must appear in the text to consider
# it already readable.  Stopwords are very common (the, is, a, of, ...), so
# a threshold of 3 catches even short sentences while avoiding false positives
# on random character sequences.
_MIN_ENGLISH_WORDS = 3

# Leetspeak mapping: digit/symbol -> possible letter replacements
_LEET_MAP: dict[str, str] = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "8": "b",
    "@": "a",
    "$": "s",
    "!": "i",
    "|": "i",
    "+": "t",
}


@dataclass
class DeobfuscationResult:
    """Result of attempting to deobfuscate user input."""

    text: str
    technique: str


def _count_english_stopwords(text: str) -> int:
    words = re.findall(r"[a-zA-Z]{2,}", text.lower())
    return sum(1 for w in words if w in ENGLISH_STOPWORDS)


def _has_enough_english(text: str) -> bool:
    return _count_english_stopwords(text) >= _MIN_ENGLISH_WORDS


# ---------------------------------------------------------------------------
# Decoding techniques
# ---------------------------------------------------------------------------


def _try_reverse(text: str) -> str:
    return text[::-1]


def _try_base64(text: str) -> str | None:
    stripped = text.strip()
    try:
        decoded = base64.b64decode(stripped, validate=True).decode("utf-8")
        if decoded.isprintable() or "\n" in decoded:
            return decoded
    except (binascii.Error, UnicodeDecodeError, ValueError):
        pass
    return None


def _try_hex(text: str) -> str | None:
    stripped = text.strip()
    try:
        decoded = bytes.fromhex(stripped).decode("utf-8")
        if decoded.isprintable() or "\n" in decoded:
            return decoded
    except (ValueError, UnicodeDecodeError):
        pass
    return None


def _try_rot13(text: str) -> str:
    return codecs.decode(text, "rot_13")


def _try_caesar(text: str) -> list[tuple[str, int]]:
    """Try all 25 non-identity Caesar shifts."""
    results: list[tuple[str, int]] = []
    for shift in range(1, 26):
        if shift == 13:
            continue  # handled by rot13
        chars = []
        for ch in text:
            if ch.isalpha():
                base = ord("A") if ch.isupper() else ord("a")
                chars.append(chr((ord(ch) - base + shift) % 26 + base))
            else:
                chars.append(ch)
        results.append(("".join(chars), shift))
    return results


def _try_leetspeak(text: str) -> str:
    return "".join(_LEET_MAP.get(ch, ch) for ch in text)


def _try_remove_zero_width(text: str) -> str:
    """Remove zero-width and invisible Unicode characters."""
    zero_width_chars = {
        "\u200b",  # zero-width space
        "\u200c",  # zero-width non-joiner
        "\u200d",  # zero-width joiner
        "\u200e",  # left-to-right mark
        "\u200f",  # right-to-left mark
        "\ufeff",  # zero-width no-break space / BOM
        "\u2060",  # word joiner
        "\u2061",  # function application
        "\u2062",  # invisible times
        "\u2063",  # invisible separator
        "\u2064",  # invisible plus
    }
    return "".join(ch for ch in text if ch not in zero_width_chars)


def _try_remove_delimiters(text: str) -> str | None:
    """Detect and remove delimiter injection patterns like T.h.i.s or T-h-i-s."""
    for delimiter in [".", "-", "_", "/", "|", "*"]:
        esc = re.escape(delimiter)
        # Pattern: at least 3 single-chars separated by the delimiter
        pattern = rf"(?:[A-Za-z0-9]{esc}){{2,}}[A-Za-z0-9]"
        matches = re.findall(pattern, text)
        if not matches:
            continue
        # Total characters covered by delimiter-separated sequences
        matched_len = sum(len(m) for m in matches)
        # If most of the non-space text matches the pattern, strip delimiters
        non_space_len = len(text.replace(" ", ""))
        if non_space_len > 0 and matched_len / non_space_len > 0.5:
            return text.replace(delimiter, "")
    return None


def _try_unicode_normalize(text: str) -> str:
    """Normalize Unicode homoglyphs and fullwidth characters to ASCII."""
    # NFKD decomposes compatibility characters (e.g., fullwidth A -> A)
    normalized = unicodedata.normalize("NFKD", text)
    # Strip combining marks left after decomposition
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _try_atbash(text: str) -> str:
    """Atbash cipher: A<->Z, B<->Y, etc."""
    chars = []
    for ch in text:
        if ch.isalpha():
            base = ord("A") if ch.isupper() else ord("a")
            chars.append(chr(base + 25 - (ord(ch) - base)))
        else:
            chars.append(ch)
    return "".join(chars)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

# Each technique entry: (name, callable that returns candidate string(s))
# Callables return either a single str, str | None, or a list of (str, shift).
_TECHNIQUES: list[tuple[str, str]] = [
    ("zero_width_removal", "zero_width"),
    ("unicode_normalization", "unicode"),
    ("delimiter_removal", "delimiter"),
    ("string_reversal", "reverse"),
    ("base64_decoding", "base64"),
    ("hex_decoding", "hex"),
    ("rot13", "rot13"),
    ("caesar_cipher", "caesar"),
    ("atbash_cipher", "atbash"),
    ("leetspeak", "leetspeak"),
]


def deobfuscate_user_input(input_str: str) -> DeobfuscationResult:
    """Attempt to deobfuscate user input using common encoding techniques.

    If the input already contains enough English stopwords, returns it unchanged
    with technique="none".  Otherwise, tries a series of decoding strategies
    and returns the one that produces the most English stopwords.
    """
    if _has_enough_english(input_str):
        return DeobfuscationResult(text=input_str, technique="none")

    best_text = input_str
    best_score = _count_english_stopwords(input_str)
    best_technique = "none"

    candidates: list[tuple[str, str]] = []

    # Collect all candidates from each technique
    for technique_name, technique_key in _TECHNIQUES:
        if technique_key == "zero_width":
            result = _try_remove_zero_width(input_str)
            if result != input_str:
                candidates.append((result, technique_name))
        elif technique_key == "unicode":
            result = _try_unicode_normalize(input_str)
            if result != input_str:
                candidates.append((result, technique_name))
        elif technique_key == "delimiter":
            result = _try_remove_delimiters(input_str)
            if result is not None:
                candidates.append((result, technique_name))
        elif technique_key == "reverse":
            candidates.append((_try_reverse(input_str), technique_name))
        elif technique_key == "base64":
            result = _try_base64(input_str)
            if result is not None:
                candidates.append((result, technique_name))
        elif technique_key == "hex":
            result = _try_hex(input_str)
            if result is not None:
                candidates.append((result, technique_name))
        elif technique_key == "rot13":
            candidates.append((_try_rot13(input_str), technique_name))
        elif technique_key == "caesar":
            for shifted_text, shift in _try_caesar(input_str):
                candidates.append(
                    (shifted_text, f"caesar_cipher_shift_{shift}")
                )
        elif technique_key == "atbash":
            candidates.append((_try_atbash(input_str), technique_name))
        elif technique_key == "leetspeak":
            candidates.append((_try_leetspeak(input_str), technique_name))

    for candidate_text, technique_name in candidates:
        score = _count_english_stopwords(candidate_text)
        if score > best_score:
            best_score = score
            best_text = candidate_text
            best_technique = technique_name

    return DeobfuscationResult(text=best_text, technique=best_technique)
