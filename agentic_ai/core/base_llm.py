"""
Abstract base class for all Language Model (LLM) chat providers.

This module defines the interface that every LLM provider must implement.
By inheriting from BaseLLM and implementing the abstract methods, new
providers can be added without modifying any existing agent or framework code.

The design follows the Strategy pattern: agents depend on the BaseLLM
interface, and concrete implementations (OpenAI, Gemini, Claude, Groq)
are injected at runtime. This enables easy testing with mock LLMs and
seamless switching between providers.

Supported operations:
    - chat():           Generate a response given a conversation history.
    - chat_with_tools(): Generate a response with tool-calling capabilities.
    - stream():         Stream a response token-by-token (optional override).

Example:
    >>> from agentic_ai.llms.openai import OpenAIChatModel
    >>> llm = OpenAIChatModel(api_key="sk-...", model="gpt-4")
    >>> response = llm.chat([Message(role=Role.USER, content="Hello!")])
    >>> print(response.content)
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from agentic_ai.core.models import Message


@dataclass
class LLMConfig:
    """
    Configuration parameters shared across all LLM providers.

    This dataclass centralizes common configuration so that each provider
    doesn't need to re-define the same parameters. Provider-specific
    parameters can be added via the `extra` dictionary.

    Attributes:
        model:        The model identifier string (e.g., "gpt-4", "claude-3-opus").
        api_key:      The API key for authenticating with the provider.
        temperature:  Controls randomness in generation. 0.0 = deterministic,
                      1.0 = maximum creativity. Default is 0.7 for balanced output.
        max_tokens:   Maximum number of tokens to generate in the response.
                      None means use the provider's default limit.
        top_p:        Nucleus sampling parameter. Only tokens with cumulative
                      probability <= top_p are considered. Default is 1.0 (disabled).
        stop:         A list of stop sequences. Generation halts when any of
                      these strings are encountered in the output.
        base_url:     Optional custom API endpoint URL for self-hosted or proxy
                      deployments (e.g., Azure OpenAI, local vLLM servers).
        timeout:      Request timeout in seconds. Default is 30 seconds.
        max_retries:  Number of times to retry failed API calls. Default is 3.
        extra:        Provider-specific configuration that doesn't fit the
                      common parameters (e.g., Azure deployment name, region).
    """

    model: str = "gpt-4"
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int | None = None
    top_p: float = 1.0
    stop: list[str] | None = None
    base_url: str | None = None
    timeout: int = 30
    max_retries: int = 3
    extra: dict[str, Any] = field(default_factory=dict)


class BaseLLM(ABC):
    """
    Abstract base class that all LLM chat providers must inherit from.

    Subclasses must implement the `chat()` and `chat_with_tools()` methods.
    The `stream()` method has a default implementation that falls back to
    `chat()`, but providers are encouraged to override it for true streaming.

    Attributes:
        config: An LLMConfig instance containing provider configuration.

    Methods:
        chat(messages):               Generate a single response message.
        chat_with_tools(messages, tools): Generate a response that may include
                                          tool invocation requests.
        stream(messages):             Asynchronously yield response tokens.
        count_tokens(messages):       Estimate the token count of messages.
    """

    def __init__(self, config: LLMConfig) -> None:
        """
        Initialize the LLM provider with the given configuration.

        Args:
            config: An LLMConfig instance containing all necessary parameters
                    for connecting to and configuring the LLM provider.
        """
        # Store the configuration for use by subclass methods.
        self.config = config

    @abstractmethod
    def chat(self, messages: list[Message]) -> Message:
        """
        Generate a response given a list of conversation messages.

        This is the primary method for interacting with the LLM. It sends
        the full conversation history and returns the assistant's response.

        Args:
            messages: A list of Message objects representing the conversation
                      history, ordered from oldest to newest.

        Returns:
            A Message object with role=ASSISTANT containing the LLM's response.

        Raises:
            ConnectionError: If the API call fails after all retries.
            ValueError: If the messages list is empty or malformed.
        """
        ...

    @abstractmethod
    def chat_with_tools(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> Message:
        """
        Generate a response that may include tool invocation requests.

        This method is used when the LLM has access to tools (functions) that
        it can choose to call. The LLM examines the conversation and the
        available tools, then either responds directly or emits ToolCall
        objects requesting specific tool invocations.

        Args:
            messages: A list of Message objects representing the conversation.
            tools:    A list of tool schema dictionaries describing available
                      tools. Each dictionary follows the OpenAI function-calling
                      schema format with 'name', 'description', and 'parameters'.

        Returns:
            A Message object that may contain tool_calls if the LLM decided
            to invoke one or more tools, or plain content if it responded directly.

        Raises:
            ConnectionError: If the API call fails after all retries.
            ValueError: If the tool schemas are malformed.
        """
        ...

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        """
        Asynchronously stream the response token by token.

        This default implementation falls back to the synchronous `chat()`
        method and yields the entire response as a single chunk. Providers
        that support native streaming should override this method for
        real-time token delivery.

        Args:
            messages: A list of Message objects representing the conversation.

        Yields:
            String chunks of the response, typically individual tokens or
            small groups of tokens.

        Example:
            >>> async for token in llm.stream(messages):
            ...     print(token, end="", flush=True)
        """
        # Fallback: call the synchronous chat method and yield the full response.
        response = self.chat(messages)
        yield response.content

    def count_tokens(self, messages: list[Message]) -> int:
        """
        Estimate the total number of tokens in the given messages.

        This default implementation uses a rough heuristic of ~4 characters
        per token. Providers with access to their own tokenizer (e.g., OpenAI's
        tiktoken) should override this for accurate counts.

        Args:
            messages: A list of Message objects to count tokens for.

        Returns:
            An integer estimate of the total token count across all messages.
        """
        # Sum up characters in all message contents and divide by 4 as a rough estimate.
        total_chars = sum(len(msg.content) for msg in messages)
        return total_chars // 4
