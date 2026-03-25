from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from agent.invoke_agent_with_user_message import invoke_agent_with_user_message


class TestInvokeAgentWithUserMessage:
    @patch("agent.invoke_agent_with_user_message.build_graph")
    @patch("agent.invoke_agent_with_user_message.ChatAnthropic")
    @patch("agent.invoke_agent_with_user_message.create_web_search")
    def test_returns_final_message_content(
        self, mock_create_web_search, mock_chat_anthropic, mock_build_graph
    ):
        mock_tool = MagicMock()
        mock_create_web_search.return_value = mock_tool

        mock_llm = MagicMock()
        mock_chat_anthropic.return_value.bind_tools.return_value = mock_llm

        mock_graph = MagicMock()
        mock_build_graph.return_value = mock_graph
        mock_graph.invoke.return_value = {
            "messages": [AIMessage(content="Hello world")]
        }

        result = invoke_agent_with_user_message("say hello", langfuse_handler=None)

        assert result == "Hello world"

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

        assert result == "No answer found"

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
        assert call_args[0][0] == {
            "messages": [{"role": "user", "content": "test question"}]
        }

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

        mock_build_graph.assert_called_once_with(tools=[mock_tool])

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

        mock_llm_instance.bind_tools.assert_called_once_with([mock_tool])
