"""
OpenAI LLM provider implementation.

This module provides integration with OpenAI's chat completion API, supporting
models like GPT-4, GPT-4o, GPT-4o-mini, and GPT-3.5-turbo. It handles message
formatting, tool/function calling, streaming, and token counting using OpenAI's
native tokenizer (tiktoken).

The implementation supports both the official OpenAI API and any OpenAI-compatible
endpoint (e.g., Azure OpenAI, vLLM, Ollama with OpenAI compatibility mode) via
the `base_url` configuration parameter.

Requirements:
    pip install openai tiktoken

Example:
    >>> from agentic_ai.llms.openai_llm import OpenAIChatModel
    >>> llm = OpenAIChatModel(api_key="sk-...", model="gpt-4o")
    >>> response = llm.chat([Message(role=Role.USER, content="Explain RAG.")])
    >>> print(response.content)
"""

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from agentic_ai.core.base_llm import BaseLLM, LLMConfig
from agentic_ai.core.models import Message, Role, ToolCall

# Configure module-level logger for debugging API interactions.
logger = logging.getLogger(__name__)


class OpenAIChatModel(BaseLLM):
    """
    OpenAI chat completion provider.

    Wraps the OpenAI Python SDK to provide chat completions with optional
    tool/function calling support. Supports all OpenAI chat models and
    any OpenAI-compatible API endpoint.

    Attributes:
        client: The OpenAI client instance used for API calls.
        async_client: The async OpenAI client for streaming operations.

    Example:
        >>> llm = OpenAIChatModel(api_key="sk-...", model="gpt-4o")
        >>> msgs = [Message(role=Role.USER, content="Hello!")]
        >>> response = llm.chat(msgs)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the OpenAI chat model provider.

        Creates both synchronous and asynchronous OpenAI clients for
        flexibility in different execution contexts.

        Args:
            api_key:     OpenAI API key for authentication.
            model:       Model identifier (e.g., "gpt-4o", "gpt-4o-mini").
            temperature: Sampling temperature (0.0 to 2.0). Default is 0.7.
            max_tokens:  Maximum tokens in the response. None uses model default.
            base_url:    Custom API endpoint for OpenAI-compatible services.
            **kwargs:    Additional parameters passed to LLMConfig.extra.
        """
        # Build the configuration object with all provided parameters.
        config = LLMConfig(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            extra=kwargs,
        )
        # Initialize the base class with the configuration.
        super().__init__(config)

        # Import openai here to make it an optional dependency.
        # This allows the rest of the framework to work without openai installed.
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "The 'openai' package is required for OpenAIChatModel. "
                "Install it with: pip install openai"
            ) from e

        # Create the synchronous client for standard chat operations.
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        # Create the asynchronous client for streaming operations.
        self.async_client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        logger.info("Initialized OpenAI chat model: %s", model)

    def _format_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """
        Convert framework Message objects to OpenAI API format.

        OpenAI expects messages as dictionaries with 'role' and 'content' keys.
        This method handles the conversion, including special formatting for
        tool calls and tool results.

        Args:
            messages: A list of framework Message objects.

        Returns:
            A list of dictionaries formatted for the OpenAI API.
        """
        formatted = []
        for msg in messages:
            # Build the base message dictionary with role and content.
            message_dict: dict[str, Any] = {
                "role": msg.role.value,
                "content": msg.content,
            }

            # Add the name field if present (used for tool results).
            if msg.name:
                message_dict["name"] = msg.name

            # Add tool calls if the assistant is requesting tool invocations.
            if msg.tool_calls:
                message_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]

            # Add the tool_call_id for tool result messages.
            if msg.tool_call_id:
                message_dict["tool_call_id"] = msg.tool_call_id

            formatted.append(message_dict)

        return formatted

    def chat(self, messages: list[Message]) -> Message:
        """
        Generate a chat completion response from OpenAI.

        Sends the conversation history to OpenAI's API and returns the
        assistant's response as a Message object.

        Args:
            messages: The conversation history as a list of Message objects.

        Returns:
            A Message with role=ASSISTANT containing the model's response.

        Raises:
            openai.APIError: If the API call fails after retries.
        """
        # Format messages for the OpenAI API.
        formatted_messages = self._format_messages(messages)

        # Build the API call parameters.
        params: dict[str, Any] = {
            "model": self.config.model,
            "messages": formatted_messages,
            "temperature": self.config.temperature,
        }

        # Add optional parameters if configured.
        if self.config.max_tokens:
            params["max_tokens"] = self.config.max_tokens
        if self.config.stop:
            params["stop"] = self.config.stop

        logger.debug("Sending chat request to OpenAI with %d messages", len(messages))

        # Make the API call to OpenAI.
        response = self.client.chat.completions.create(**params)

        # Extract the first choice from the response.
        choice = response.choices[0]

        # Build and return the response Message.
        return Message(
            role=Role.ASSISTANT,
            content=choice.message.content or "",
            metadata={
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                "finish_reason": choice.finish_reason,
            },
        )

    def chat_with_tools(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> Message:
        """
        Generate a response that may include tool/function calls.

        Sends the conversation along with tool schemas to OpenAI. The model
        may choose to call one or more tools instead of (or in addition to)
        generating a text response.

        Args:
            messages: The conversation history as Message objects.
            tools:    Tool schemas in OpenAI function-calling format.

        Returns:
            A Message that may contain tool_calls if the model chose to
            invoke tools, or plain content for a direct response.
        """
        # Format messages for the OpenAI API.
        formatted_messages = self._format_messages(messages)

        # Build the API call parameters with tools included.
        params: dict[str, Any] = {
            "model": self.config.model,
            "messages": formatted_messages,
            "temperature": self.config.temperature,
            "tools": tools,
        }

        if self.config.max_tokens:
            params["max_tokens"] = self.config.max_tokens

        logger.debug(
            "Sending tool-enabled chat request with %d tools", len(tools)
        )

        # Make the API call with tools.
        response = self.client.chat.completions.create(**params)
        choice = response.choices[0]

        # Parse any tool calls from the response.
        tool_calls: list[ToolCall] = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                # Parse the JSON arguments string into a dictionary.
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {"raw": tc.function.arguments}

                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments,
                    )
                )

        return Message(
            role=Role.ASSISTANT,
            content=choice.message.content or "",
            tool_calls=tool_calls,
            metadata={
                "model": response.model,
                "finish_reason": choice.finish_reason,
            },
        )

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        """
        Stream the response token by token using OpenAI's streaming API.

        Uses server-sent events (SSE) to receive tokens as they are generated,
        enabling real-time display of the response.

        Args:
            messages: The conversation history as Message objects.

        Yields:
            Individual tokens or small text chunks as they are generated.
        """
        # Format messages for the OpenAI API.
        formatted_messages = self._format_messages(messages)

        logger.debug("Starting streaming chat request to OpenAI")

        # Create a streaming completion request.
        stream = await self.async_client.chat.completions.create(
            model=self.config.model,
            messages=formatted_messages,
            temperature=self.config.temperature,
            stream=True,
        )

        # Yield each token as it arrives from the stream.
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def count_tokens(self, messages: list[Message]) -> int:
        """
        Count tokens accurately using OpenAI's tiktoken tokenizer.

        Falls back to the base class heuristic if tiktoken is not available.

        Args:
            messages: The messages to count tokens for.

        Returns:
            The total number of tokens across all messages.
        """
        try:
            import tiktoken

            # Get the encoding for the current model.
            encoding = tiktoken.encoding_for_model(self.config.model)

            total = 0
            for msg in messages:
                # Each message has overhead tokens for role formatting.
                total += 4  # <|im_start|>{role}\n ... \n<|im_end|>
                total += len(encoding.encode(msg.content))
                if msg.name:
                    total += len(encoding.encode(msg.name))
            total += 2  # Every reply is primed with <|im_start|>assistant

            return total
        except ImportError:
            # Fall back to the base class heuristic if tiktoken is not installed.
            return super().count_tokens(messages)
