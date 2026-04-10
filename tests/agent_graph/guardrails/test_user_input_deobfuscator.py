import base64

from agent_graph.guardrails.user_input_deobfuscator import (
    DeobfuscationResult,
    _count_english_stopwords,
    _has_enough_english,
    _try_atbash,
    _try_base64,
    _try_caesar,
    _try_hex,
    _try_leetspeak,
    _try_remove_delimiters,
    _try_remove_zero_width,
    _try_reverse,
    _try_rot13,
    _try_unicode_normalize,
    deobfuscate_user_input,
)


class TestCountEnglishStopwords:
    def test_empty_string(self):
        assert _count_english_stopwords("") == 0

    def test_no_english_words(self):
        assert _count_english_stopwords("xyz qwerty zxcvb") == 0

    def test_counts_stopwords(self):
        assert _count_english_stopwords("this is not the test") >= 3

    def test_case_insensitive(self):
        assert _count_english_stopwords("This Is Not The Test") >= 3


class TestHasEnoughEnglish:
    def test_plain_english_sentence(self):
        assert _has_enough_english("This is a simple test of the system")

    def test_gibberish(self):
        assert not _has_enough_english("xkcd qwerty asdfgh")

    def test_short_sentence_with_stopwords(self):
        assert _has_enough_english("I am the one who is here")

    def test_empty_string(self):
        assert not _has_enough_english("")


class TestTryReverse:
    def test_reverses_string(self):
        assert _try_reverse("dlrow olleh") == "hello world"

    def test_empty_string(self):
        assert _try_reverse("") == ""

    def test_palindrome(self):
        assert _try_reverse("racecar") == "racecar"


class TestTryBase64:
    def test_valid_base64(self):
        original = "This is a test message"
        encoded = base64.b64encode(original.encode()).decode()
        assert _try_base64(encoded) == original

    def test_invalid_base64(self):
        assert _try_base64("not valid base64!!!") is None

    def test_non_utf8_base64(self):
        # Base64 of non-UTF-8 bytes
        encoded = base64.b64encode(b"\xff\xfe\xfd").decode()
        assert _try_base64(encoded) is None

    def test_strips_whitespace(self):
        original = "hello world"
        encoded = "  " + base64.b64encode(original.encode()).decode() + "  "
        assert _try_base64(encoded) == original


class TestTryHex:
    def test_valid_hex(self):
        original = "hello"
        hex_str = original.encode().hex()
        assert _try_hex(hex_str) == original

    def test_invalid_hex(self):
        assert _try_hex("not hex at all") is None

    def test_odd_length_hex(self):
        assert _try_hex("abc") is None


class TestTryRot13:
    def test_rot13_encode(self):
        assert _try_rot13("Guvf vf n grfg") == "This is a test"

    def test_rot13_roundtrip(self):
        original = "Hello World"
        assert _try_rot13(_try_rot13(original)) == original

    def test_non_alpha_unchanged(self):
        assert _try_rot13("123!@#") == "123!@#"


class TestTryCaesar:
    def test_returns_24_shifts(self):
        results = _try_caesar("hello")
        assert len(results) == 24  # 25 shifts minus rot13

    def test_shift_1_produces_correct_output(self):
        results = _try_caesar("a")
        shifts = {shift: text for text, shift in results}
        assert shifts[1] == "b"
        assert shifts[25] == "z"

    def test_preserves_non_alpha(self):
        results = _try_caesar("a 1!")
        for text, _ in results:
            assert text.endswith(" 1!")


class TestTryLeetspeak:
    def test_leet_translation(self):
        result = _try_leetspeak("7h15 15 l337")
        assert result == "this is leet"

    def test_no_leet_chars(self):
        assert _try_leetspeak("hello") == "hello"

    def test_mixed_leet(self):
        result = _try_leetspeak("h3ll0")
        assert result == "hello"


class TestTryRemoveZeroWidth:
    def test_removes_zero_width_spaces(self):
        text = "h\u200be\u200bl\u200bl\u200bo"
        assert _try_remove_zero_width(text) == "hello"

    def test_removes_multiple_types(self):
        text = "a\u200b\u200c\u200db"
        assert _try_remove_zero_width(text) == "ab"

    def test_no_zero_width_chars(self):
        assert _try_remove_zero_width("hello") == "hello"

    def test_removes_bom(self):
        text = "\ufeffhello"
        assert _try_remove_zero_width(text) == "hello"


class TestTryRemoveDelimiters:
    def test_dot_delimiter(self):
        result = _try_remove_delimiters("T.h.i.s. .i.s. .t.e.x.t")
        assert result is not None
        assert "." not in result

    def test_dash_delimiter(self):
        result = _try_remove_delimiters("h-e-l-l-o- -w-o-r-l-d")
        assert result is not None
        assert "-" not in result

    def test_no_delimiter_pattern(self):
        result = _try_remove_delimiters("hello world")
        assert result is None

    def test_normal_sentence_with_period(self):
        result = _try_remove_delimiters("This is a sentence.")
        assert result is None


class TestTryUnicodeNormalize:
    def test_fullwidth_characters(self):
        # Fullwidth "Hello" -> ASCII "Hello"
        fullwidth = "\uff28\uff45\uff4c\uff4c\uff4f"
        result = _try_unicode_normalize(fullwidth)
        assert result == "Hello"

    def test_plain_ascii_unchanged(self):
        assert _try_unicode_normalize("hello") == "hello"


class TestTryAtbash:
    def test_atbash_cipher(self):
        # A->Z, B->Y, etc.
        assert _try_atbash("Zyxw") == "Abcd"

    def test_atbash_roundtrip(self):
        original = "Hello World"
        assert _try_atbash(_try_atbash(original)) == original

    def test_non_alpha_unchanged(self):
        assert _try_atbash("123!") == "123!"


class TestDeobfuscateUserInput:
    def test_plain_english_returns_unchanged(self):
        text = "What is the capital of France and how do I get there?"
        result = deobfuscate_user_input(text)
        assert result.text == text
        assert result.technique == "none"

    def test_returns_deobfuscation_result(self):
        result = deobfuscate_user_input("hello")
        assert isinstance(result, DeobfuscationResult)

    def test_reversed_text(self):
        original = "What is the answer to this question?"
        reversed_text = original[::-1]
        result = deobfuscate_user_input(reversed_text)
        assert result.text == original
        assert result.technique == "string_reversal"

    def test_base64_encoded(self):
        original = "This is a test message with many words"
        encoded = base64.b64encode(original.encode()).decode()
        result = deobfuscate_user_input(encoded)
        assert result.text == original
        assert result.technique == "base64_decoding"

    def test_rot13_encoded(self):
        # "This is a test" rot13 encoded
        encoded = "Guvf vf n grfg bs gur flfgrz"
        result = deobfuscate_user_input(encoded)
        assert "This is a test of the system" == result.text
        assert result.technique == "rot13"

    def test_leetspeak(self):
        leet = "7h15 15 4 7357 0f 7h3 5y573m"
        result = deobfuscate_user_input(leet)
        assert result.technique == "leetspeak"
        assert "this" in result.text.lower()

    def test_zero_width_characters(self):
        original = "What is the answer to this question?"
        obfuscated = "\u200b".join(original)
        result = deobfuscate_user_input(obfuscated)
        assert result.text == original
        assert result.technique == "zero_width_removal"

    def test_hex_encoded(self):
        original = "What is the answer to this question about hex"
        hex_str = original.encode().hex()
        result = deobfuscate_user_input(hex_str)
        assert result.text == original
        assert result.technique == "hex_decoding"

    def test_no_technique_when_nothing_improves(self):
        gibberish = "xkqz brvw"
        result = deobfuscate_user_input(gibberish)
        assert result.technique == "none"

    def test_selects_best_technique(self):
        # Base64 of a long English sentence should beat random noise from
        # other decodings
        original = "The quick brown fox jumps over the lazy dog and then some"
        encoded = base64.b64encode(original.encode()).decode()
        result = deobfuscate_user_input(encoded)
        assert result.text == original

    def test_atbash_cipher(self):
        # "this is a test" in atbash
        original = "this is a test of the system"
        atbash = _try_atbash(original)
        result = deobfuscate_user_input(atbash)
        assert result.text == original
        assert result.technique == "atbash_cipher"

    def test_delimiter_injection(self):
        obfuscated = "T.h.i.s. .i.s. .a. .t.e.s.t. .o.f. .t.h.e. .s.y.s.t.e.m"
        result = deobfuscate_user_input(obfuscated)
        assert result.technique == "delimiter_removal"
        assert "this" in result.text.lower() or "This" in result.text
