"""
Groq LLM provider implementation.

This module provides integration with Groq's ultra-fast inference API,
supporting models like LLaMA 3, Mixtral, and Gemma. Groq's custom LPU
(Language Processing Unit) hardware delivers exceptionally low latency,
making it ideal for real-time agent applications.

Groq's API is OpenAI-compatible, so this implementation leverages the
OpenAI SDK with a custom base URL pointing to Groq's endpoint.

Requirements:
    pip install groq

Example:
    >>> from agentic_ai.llms.groq_llm import GroqChatModel
    >>> llm = GroqChatModel(api_key="gsk_...", model="llama3-70b-8192")
    >>> response = llm.chat([Message(role=Role.USER, content="Hello!")])
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


class GroqChatModel(BaseLLM):
    """
    Groq fast inference chat completion provider.

    Wraps the Groq Python SDK to provide ultra-low-latency chat completions.
    Supports tool/function calling for models that support it, and streaming
    for real-time response delivery.

    Groq's LPU hardware provides significantly faster inference than
    traditional GPU-based providers, making it excellent for latency-sensitive
    agent applications.

    Attributes:
        client: The Groq client instance for API calls.
        async_client: The async Groq client for streaming operations.

    Example:
        >>> llm = GroqChatModel(api_key="gsk_...", model="llama3-70b-8192")
        >>> response = llm.chat([Message(role=Role.USER, content="Hi!")])
    """

    def __init__(
        self,
        api_key: str,
        model: str = "llama3-70b-8192",
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Groq chat model provider.

        Args:
            api_key:     Groq API key for authentication (starts with "gsk_").
            model:       Model identifier (e.g., "llama3-70b-8192", "mixtral-8x7b-32768").
            temperature: Sampling temperature (0.0 to 2.0). Default is 0.7.
            max_tokens:  Maximum tokens in the response. None uses model default.
            **kwargs:    Additional parameters stored in LLMConfig.extra.
        """
        config = LLMConfig(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            extra=kwargs,
        )
        super().__init__(config)

        # Import groq here to make it an optional dependency.
        try:
            import groq
        except ImportError as e:
            raise ImportError(
                "The 'groq' package is required for GroqChatModel. "
                "Install it with: pip install groq"
            ) from e

        # Create the Groq client for API calls.
        self.client = groq.Groq(api_key=api_key)
        self.async_client = groq.AsyncGroq(api_key=api_key)

        logger.info("Initialized Groq chat model: %s", model)

    def _format_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """
        Convert framework Message objects to Groq API format.

        Groq uses an OpenAI-compatible message format, so the conversion
        is straightforward with role and content fields.

        Args:
            messages: Framework Message objects to convert.

        Returns:
            A list of dictionaries formatted for the Groq API.
        """
        formatted = []
        for msg in messages:
            message_dict: dict[str, Any] = {
                "role": msg.role.value,
                "content": msg.content,
            }

            # Add tool calls for assistant messages that invoke tools.
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

            # Add tool_call_id for tool result messages.
            if msg.tool_call_id:
                message_dict["tool_call_id"] = msg.tool_call_id

            formatted.append(message_dict)

        return formatted

    def chat(self, messages: list[Message]) -> Message:
        """
        Generate a chat completion response from Groq.

        Sends the conversation to Groq's API and returns the response.
        Groq typically responds in milliseconds due to its LPU hardware.

        Args:
            messages: The conversation history as Message objects.

        Returns:
            A Message with role=ASSISTANT containing the model's response.

        Raises:
            groq.APIError: If the API call fails.
        """
        formatted_messages = self._format_messages(messages)

        params: dict[str, Any] = {
            "model": self.config.model,
            "messages": formatted_messages,
            "temperature": self.config.temperature,
        }

        if self.config.max_tokens:
            params["max_tokens"] = self.config.max_tokens
        if self.config.stop:
            params["stop"] = self.config.stop

        logger.debug("Sending chat request to Groq with %d messages", len(messages))

        # Make the API call to Groq.
        response = self.client.chat.completions.create(**params)
        choice = response.choices[0]

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
                # Groq provides speed metrics in its responses.
                "x_groq": getattr(response, "x_groq", None),
            },
        )

    def chat_with_tools(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> Message:
        """
        Generate a response with tool/function calling via Groq.

        Groq supports OpenAI-compatible function calling for models that
        have tool-use capabilities (e.g., LLaMA 3 with function calling).

        Args:
            messages: The conversation history as Message objects.
            tools:    Tool schemas in OpenAI function-calling format.

        Returns:
            A Message that may contain tool_calls if the model chose to
            invoke tools.
        """
        formatted_messages = self._format_messages(messages)

        params: dict[str, Any] = {
            "model": self.config.model,
            "messages": formatted_messages,
            "temperature": self.config.temperature,
            "tools": tools,
        }

        if self.config.max_tokens:
            params["max_tokens"] = self.config.max_tokens

        logger.debug("Sending tool-enabled chat request to Groq with %d tools", len(tools))

        response = self.client.chat.completions.create(**params)
        choice = response.choices[0]

        # Parse any tool calls from the response.
        tool_calls: list[ToolCall] = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
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
        Stream the response token by token from Groq.

        Groq's streaming is exceptionally fast due to its LPU hardware,
        often delivering the full response faster than other providers
        can deliver the first token.

        Args:
            messages: The conversation history as Message objects.

        Yields:
            Text chunks as they are generated by the model.
        """
        formatted_messages = self._format_messages(messages)

        logger.debug("Starting streaming chat request to Groq")

        stream = await self.async_client.chat.completions.create(
            model=self.config.model,
            messages=formatted_messages,
            temperature=self.config.temperature,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
