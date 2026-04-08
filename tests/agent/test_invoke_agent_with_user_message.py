from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, ToolMessage

from agent.agent_response import AgentResponse, AgentRunMetrics
from agent.invoke_agent_with_user_message import (
    _compute_metrics,
    build_system_prompt,
    invoke_agent_with_user_message,
)
from tools.execute_code import execute_code_file, execute_code_snippet


class TestComputeMetrics:
    def test_empty_messages(self):
        input_tokens, output_tokens, total_turns = _compute_metrics([])
        assert input_tokens == 0
        assert output_tokens == 0
        assert total_turns == 0

    def test_single_ai_message_with_usage(self):
        msg = AIMessage(
            content="hello",
            response_metadata={"usage": {"input_tokens": 10, "output_tokens": 20}},
        )
        input_tokens, output_tokens, total_turns = _compute_metrics([msg])
        assert input_tokens == 10
        assert output_tokens == 20
        assert total_turns == 1

    def test_multiple_ai_messages_sums_tokens(self):
        msg1 = AIMessage(
            content="first",
            response_metadata={"usage": {"input_tokens": 10, "output_tokens": 5}},
        )
        msg2 = AIMessage(
            content="second",
            response_metadata={"usage": {"input_tokens": 30, "output_tokens": 15}},
        )
        input_tokens, output_tokens, total_turns = _compute_metrics([msg1, msg2])
        assert input_tokens == 40
        assert output_tokens == 20
        assert total_turns == 2

    def test_tool_messages_count_as_turns(self):
        ai_msg = AIMessage(
            content="call tool",
            response_metadata={"usage": {"input_tokens": 10, "output_tokens": 5}},
        )
        tool_msg = ToolMessage(content="tool result", tool_call_id="1")
        input_tokens, output_tokens, total_turns = _compute_metrics(
            [ai_msg, tool_msg]
        )
        assert input_tokens == 10
        assert output_tokens == 5
        assert total_turns == 2

    def test_ai_message_without_usage_metadata(self):
        msg = AIMessage(content="hello", response_metadata={})
        input_tokens, output_tokens, total_turns = _compute_metrics([msg])
        assert input_tokens == 0
        assert output_tokens == 0
        assert total_turns == 1

    def test_mixed_message_types(self):
        from langchain_core.messages import HumanMessage

        human = HumanMessage(content="question")
        ai1 = AIMessage(
            content="thinking",
            response_metadata={"usage": {"input_tokens": 100, "output_tokens": 50}},
        )
        tool = ToolMessage(content="result", tool_call_id="1")
        ai2 = AIMessage(
            content="answer",
            response_metadata={"usage": {"input_tokens": 200, "output_tokens": 30}},
        )
        input_tokens, output_tokens, total_turns = _compute_metrics(
            [human, ai1, tool, ai2]
        )
        assert input_tokens == 300
        assert output_tokens == 80
        assert total_turns == 3  # 2 AI + 1 Tool; Human not counted


class TestInvokeAgentWithUserMessage:
    @patch("agent.invoke_agent_with_user_message.build_graph")
    @patch("agent.invoke_agent_with_user_message.ChatAnthropic")
    @patch("agent.invoke_agent_with_user_message.create_web_search")
    def test_returns_agent_response(
        self, mock_create_web_search, mock_chat_anthropic, mock_build_graph
    ):
        mock_create_web_search.return_value = MagicMock()
        mock_chat_anthropic.return_value.bind_tools.return_value = MagicMock()

        mock_graph = MagicMock()
        mock_build_graph.return_value = mock_graph
        mock_graph.invoke.return_value = {
            "messages": [
                AIMessage(
                    content="Hello world",
                    response_metadata={
                        "usage": {"input_tokens": 10, "output_tokens": 20}
                    },
                )
            ]
        }

        result = invoke_agent_with_user_message("say hello", langfuse_handler=None)

        assert isinstance(result, AgentResponse)
        assert result.answer == "Hello world"
        assert result.metrics.input_tokens == 10
        assert result.metrics.output_tokens == 20
        assert result.metrics.total_turns == 1
        assert result.metrics.latency_seconds >= 0

    @patch("agent.invoke_agent_with_user_message.build_graph")
    @patch("agent.invoke_agent_with_user_message.ChatAnthropic")
    @patch("agent.invoke_agent_with_user_message.create_web_search")
    def test_returns_no_answer_when_messages_empty(
        self, mock_create_web_search, mock_chat_anthropic, mock_build_graph
    ):
        mock_create_web_search.return_value = MagicMock()
        mock_chat_anthropic.return_value.bind_tools.return_value = MagicMock()

        mock_graph = MagicMock()
        mock_build_graph.return_value = mock_graph
        mock_graph.invoke.return_value = {"messages": []}

        result = invoke_agent_with_user_message("say hello", langfuse_handler=None)

        assert result.answer == "No answer found"
        assert result.metrics.input_tokens == 0
        assert result.metrics.output_tokens == 0
        assert result.metrics.total_turns == 0

    @patch("agent.invoke_agent_with_user_message.build_graph")
    @patch("agent.invoke_agent_with_user_message.ChatAnthropic")
    @patch("agent.invoke_agent_with_user_message.create_web_search")
    def test_passes_user_message_to_graph(
        self, mock_create_web_search, mock_chat_anthropic, mock_build_graph
    ):
        mock_create_web_search.return_value = MagicMock()
        mock_chat_anthropic.return_value.bind_tools.return_value = MagicMock()

        mock_graph = MagicMock()
        mock_build_graph.return_value = mock_graph
        mock_graph.invoke.return_value = {
            "messages": [AIMessage(content="response")]
        }

        invoke_agent_with_user_message("test question", langfuse_handler=None)

        call_args = mock_graph.invoke.call_args
        messages = call_args[0][0]["messages"]
        assert messages[0]["role"] == "system"
        assert "Provided file path:" not in messages[0]["content"]
        assert messages[-1] == {"role": "user", "content": "test question"}

    @patch("agent.invoke_agent_with_user_message.build_graph")
    @patch("agent.invoke_agent_with_user_message.ChatAnthropic")
    @patch("agent.invoke_agent_with_user_message.create_web_search")
    def test_includes_langfuse_handler_in_callbacks(
        self, mock_create_web_search, mock_chat_anthropic, mock_build_graph
    ):
        mock_create_web_search.return_value = MagicMock()
        mock_chat_anthropic.return_value.bind_tools.return_value = MagicMock()

        mock_graph = MagicMock()
        mock_build_graph.return_value = mock_graph
        mock_graph.invoke.return_value = {
            "messages": [AIMessage(content="response")]
        }

        mock_handler = MagicMock()
        invoke_agent_with_user_message("question", langfuse_handler=mock_handler)

        call_args = mock_graph.invoke.call_args
        config = call_args[1]["config"]
        assert mock_handler in config["callbacks"]

    @patch("agent.invoke_agent_with_user_message.build_graph")
    @patch("agent.invoke_agent_with_user_message.ChatAnthropic")
    @patch("agent.invoke_agent_with_user_message.create_web_search")
    def test_no_callbacks_when_langfuse_handler_is_none(
        self, mock_create_web_search, mock_chat_anthropic, mock_build_graph
    ):
        mock_create_web_search.return_value = MagicMock()
        mock_chat_anthropic.return_value.bind_tools.return_value = MagicMock()

        mock_graph = MagicMock()
        mock_build_graph.return_value = mock_graph
        mock_graph.invoke.return_value = {
            "messages": [AIMessage(content="response")]
        }

        invoke_agent_with_user_message("question", langfuse_handler=None)

        call_args = mock_graph.invoke.call_args
        config = call_args[1]["config"]
        assert config["callbacks"] == []

    @patch("agent.invoke_agent_with_user_message.build_graph")
    @patch("agent.invoke_agent_with_user_message.ChatAnthropic")
    @patch("agent.invoke_agent_with_user_message.create_web_search")
    def test_builds_graph_with_tools(
        self, mock_create_web_search, mock_chat_anthropic, mock_build_graph
    ):
        mock_tool = MagicMock()
        mock_create_web_search.return_value = mock_tool
        mock_chat_anthropic.return_value.bind_tools.return_value = MagicMock()

        mock_graph = MagicMock()
        mock_build_graph.return_value = mock_graph
        mock_graph.invoke.return_value = {
            "messages": [AIMessage(content="response")]
        }

        invoke_agent_with_user_message("question", langfuse_handler=None)

        mock_build_graph.assert_called_once_with(
            tools=[mock_tool, execute_code_snippet, execute_code_file]
        )

    @patch("agent.invoke_agent_with_user_message.build_graph")
    @patch("agent.invoke_agent_with_user_message.ChatAnthropic")
    @patch("agent.invoke_agent_with_user_message.create_web_search")
    def test_binds_tools_to_llm(
        self, mock_create_web_search, mock_chat_anthropic, mock_build_graph
    ):
        mock_tool = MagicMock()
        mock_create_web_search.return_value = mock_tool
        mock_llm_instance = MagicMock()
        mock_chat_anthropic.return_value = mock_llm_instance

        mock_graph = MagicMock()
        mock_build_graph.return_value = mock_graph
        mock_graph.invoke.return_value = {
            "messages": [AIMessage(content="response")]
        }

        invoke_agent_with_user_message("question", langfuse_handler=None)

        mock_llm_instance.bind_tools.assert_called_once_with(
            [mock_tool, execute_code_snippet, execute_code_file]
        )

    @patch("agent.invoke_agent_with_user_message.build_graph")
    @patch("agent.invoke_agent_with_user_message.ChatAnthropic")
    @patch("agent.invoke_agent_with_user_message.create_web_search")
    def test_latency_is_positive(
        self, mock_create_web_search, mock_chat_anthropic, mock_build_graph
    ):
        mock_create_web_search.return_value = MagicMock()
        mock_chat_anthropic.return_value.bind_tools.return_value = MagicMock()

        mock_graph = MagicMock()
        mock_build_graph.return_value = mock_graph
        mock_graph.invoke.return_value = {
            "messages": [AIMessage(content="response")]
        }

        result = invoke_agent_with_user_message("question", langfuse_handler=None)

        assert result.metrics.latency_seconds >= 0

    @patch("agent.invoke_agent_with_user_message.build_graph")
    @patch("agent.invoke_agent_with_user_message.ChatAnthropic")
    @patch("agent.invoke_agent_with_user_message.create_web_search")
    def test_file_path_added_to_system_prompt(
        self, mock_create_web_search, mock_chat_anthropic, mock_build_graph
    ):
        mock_create_web_search.return_value = MagicMock()
        mock_chat_anthropic.return_value.bind_tools.return_value = MagicMock()
        mock_graph = MagicMock()
        mock_build_graph.return_value = mock_graph
        mock_graph.invoke.return_value = {
            "messages": [AIMessage(content="response")]
        }

        invoke_agent_with_user_message(
            "question",
            langfuse_handler=None,
            available_file_path="2023/validation/abc.png",
        )

        messages = mock_graph.invoke.call_args[0][0]["messages"]
        system_content = messages[0]["content"]
        assert "Provided file path:" in system_content
        assert "- 2023/validation/abc.png" in system_content
        file_path_idx = system_content.index("Provided file path:")
        # The conclusion copy appears multiple times (once in the
        # tool_not_available examples), so anchor on the final one.
        conclusion_idx = system_content.rindex(
            "Now please answer the following question:"
        )
        assert file_path_idx < conclusion_idx


class TestBuildSystemPrompt:
    def test_no_file_path_has_files_section_but_no_listing(self):
        prompt = build_system_prompt()
        assert "<files>" in prompt
        assert "Provided file path:" not in prompt
        assert "Now please answer the following question:" in prompt

    def test_empty_file_path_has_no_listing(self):
        prompt = build_system_prompt(None)
        assert "Provided file path:" not in prompt

    def test_file_path_appears_right_before_conclusion_question(self):
        prompt = build_system_prompt("foo/bar.png")
        assert "Provided file path:\n- foo/bar.png" in prompt
        # The listing must appear inside the conclusion section, directly
        # before the question line.
        tail = prompt.split("Provided file path:")[1]
        assert tail.lstrip().startswith("- foo/bar.png")
        assert "Now please answer the following question:" in tail
