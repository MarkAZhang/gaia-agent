from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def test_transcribe_audio_returns_transcript():
    segment1 = SimpleNamespace(text=" Hello world. ")
    segment2 = SimpleNamespace(text=" How are you? ")
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([segment1, segment2], None)

    with patch("tools.audio_transcriber.model", mock_model):
        from tools.audio_transcriber import transcribe_audio

        result = transcribe_audio.invoke({"file_path": "2023/validation/audio.mp3"})

    mock_model.transcribe.assert_called_once_with(
        ".gaia-questions/files/2023/validation/audio.mp3"
    )
    assert result == "Hello world. How are you?"


def test_transcribe_audio_resolves_absolute_path():
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([], None)

    with patch("tools.audio_transcriber.model", mock_model):
        from tools.audio_transcriber import transcribe_audio

        transcribe_audio.invoke({"file_path": "/tmp/absolute/audio.wav"})

    mock_model.transcribe.assert_called_once_with("/tmp/absolute/audio.wav")


def test_transcribe_audio_resolves_already_rooted_path():
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([], None)

    with patch("tools.audio_transcriber.model", mock_model):
        from tools.audio_transcriber import transcribe_audio

        transcribe_audio.invoke(
            {"file_path": ".gaia-questions/files/2023/validation/audio.mp3"}
        )

    mock_model.transcribe.assert_called_once_with(
        ".gaia-questions/files/2023/validation/audio.mp3"
    )


def test_transcribe_audio_returns_empty_string_for_silence():
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([], None)

    with patch("tools.audio_transcriber.model", mock_model):
        from tools.audio_transcriber import transcribe_audio

        result = transcribe_audio.invoke({"file_path": "silence.wav"})

    assert result == ""


def test_transcribe_audio_strips_whitespace_from_segments():
    segment = SimpleNamespace(text="  padded text  ")
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([segment], None)

    with patch("tools.audio_transcriber.model", mock_model):
        from tools.audio_transcriber import transcribe_audio

        result = transcribe_audio.invoke({"file_path": "audio.mp3"})

    assert result == "padded text"
