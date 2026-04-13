import os
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from agent_graph.nodes.output_formatter import output_formatter, _find_user_question

OPENAI_ENV = {"OPENAI_API_KEY": "test-key"}


def _mock_openai_response(content: str) -> MagicMock:
    mock_message = MagicMock()
    mock_message.content = content
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


@patch.dict(os.environ, OPENAI_ENV)
@patch("agent_graph.nodes.output_formatter.OpenAI")
def test_returns_formatted_answer(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_openai_response("42")

    state = {
        "messages": [
            HumanMessage(content="What is 6*7?"),
            AIMessage(content="42."),
        ]
    }

    result = output_formatter(state)
    assert result["messages"][0]["content"] == "42"
    assert result["messages"][0]["role"] == "ai"


@patch.dict(os.environ, OPENAI_ENV)
@patch("agent_graph.nodes.output_formatter.OpenAI")
def test_passes_user_question_and_answer_to_openai(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_openai_response("Paris")

    state = {
        "messages": [
            HumanMessage(content="What is the capital of France?"),
            AIMessage(content="Paris."),
        ]
    }

    output_formatter(state)

    call_args = mock_client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "What is the capital of France?" in messages[1]["content"]
    assert "Paris." in messages[1]["content"]


@patch.dict(os.environ, OPENAI_ENV)
@patch("agent_graph.nodes.output_formatter.OpenAI")
def test_returns_original_answer_when_openai_returns_none(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_openai_response(None)
    # Simulate None content
    mock_client.chat.completions.create.return_value.choices[0].message.content = None

    state = {
        "messages": [
            HumanMessage(content="Question?"),
            AIMessage(content="original answer"),
        ]
    }

    result = output_formatter(state)
    assert result["messages"][0]["content"] == "original answer"


@patch.dict(os.environ, OPENAI_ENV)
@patch("agent_graph.nodes.output_formatter.OpenAI")
def test_uses_last_message_as_final_answer(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_openai_response("second")

    state = {
        "messages": [
            HumanMessage(content="Q?"),
            AIMessage(content="first"),
            AIMessage(content="second answer"),
        ]
    }

    output_formatter(state)

    call_args = mock_client.chat.completions.create.call_args
    user_msg = call_args.kwargs["messages"][1]["content"]
    assert "second answer" in user_msg


@patch.dict(os.environ, OPENAI_ENV)
@patch("agent_graph.nodes.output_formatter.OpenAI")
def test_uses_gpt_4o_mini_model(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_openai_response("42")

    state = {
        "messages": [
            HumanMessage(content="Q?"),
            AIMessage(content="42"),
        ]
    }

    output_formatter(state)

    call_args = mock_client.chat.completions.create.call_args
    assert call_args.kwargs["model"] == "gpt-4o-mini"


def test_find_user_question_returns_first_human_message():
    messages = [
        HumanMessage(content="First question"),
        AIMessage(content="Answer"),
        HumanMessage(content="Second question"),
    ]
    assert _find_user_question(messages) == "First question"


def test_find_user_question_returns_empty_when_no_human_message():
    messages = [AIMessage(content="Answer")]
    assert _find_user_question(messages) == ""
