"""
Anthropic Claude LLM provider implementation.

This module provides integration with Anthropic's Messages API, supporting
Claude 3 Opus, Sonnet, Haiku, and newer Claude models. It handles message
formatting for Anthropic's conversation format (which separates system
prompts from message history), tool use via Anthropic's native tool_use
blocks, and streaming responses.

Requirements:
    pip install anthropic

Example:
    >>> from agentic_ai.llms.claude_llm import ClaudeChatModel
    >>> llm = ClaudeChatModel(api_key="sk-ant-...", model="claude-3-opus-20240229")
    >>> response = llm.chat([Message(role=Role.USER, content="Hello!")])
    >>> print(response.content)
"""

import logging
from collections.abc import AsyncIterator
from typing import Any

from agentic_ai.core.base_llm import BaseLLM, LLMConfig
from agentic_ai.core.models import Message, Role, ToolCall

# Configure module-level logger for debugging API interactions.
logger = logging.getLogger(__name__)


class ClaudeChatModel(BaseLLM):
    """
    Anthropic Claude chat completion provider.

    Wraps the Anthropic Python SDK to provide chat completions using Claude
    models. Handles the conversion between the framework's message format
    and Anthropic's format, where system prompts are a top-level parameter
    rather than part of the message array.

    Attributes:
        client: The Anthropic client instance for synchronous API calls.
        async_client: The async Anthropic client for streaming operations.

    Example:
        >>> llm = ClaudeChatModel(api_key="sk-ant-...", model="claude-3-sonnet-20240229")
        >>> response = llm.chat([Message(role=Role.USER, content="Hi!")])
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-sonnet-20240229",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Claude chat model provider.

        Args:
            api_key:     Anthropic API key for authentication.
            model:       Claude model identifier (e.g., "claude-3-opus-20240229").
            temperature: Sampling temperature (0.0 to 1.0). Default is 0.7.
            max_tokens:  Maximum tokens in the response. Default is 4096.
                         Note: Anthropic requires max_tokens to be explicitly set.
            base_url:    Custom API endpoint for Anthropic-compatible services.
            **kwargs:    Additional parameters stored in LLMConfig.extra.
        """
        config = LLMConfig(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            extra=kwargs,
        )
        super().__init__(config)

        # Import anthropic here to make it an optional dependency.
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "The 'anthropic' package is required for ClaudeChatModel. "
                "Install it with: pip install anthropic"
            ) from e

        # Build client kwargs, only including base_url if provided.
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        # Create synchronous and asynchronous clients.
        self.client = anthropic.Anthropic(**client_kwargs)
        self.async_client = anthropic.AsyncAnthropic(**client_kwargs)

        logger.info("Initialized Claude chat model: %s", model)

    def _extract_system_and_messages(
        self, messages: list[Message]
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Separate system prompt from conversation messages.

        Anthropic's API requires the system prompt as a top-level parameter,
        not as part of the messages array. This method extracts system messages
        and formats the remaining messages for Anthropic's API.

        Args:
            messages: Framework Message objects to process.

        Returns:
            A tuple of (system_prompt, formatted_messages) where system_prompt
            is the concatenated system instructions, and formatted_messages is
            the conversation in Anthropic's format.
        """
        system_parts: list[str] = []
        formatted: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                # Collect all system messages into a single system prompt.
                system_parts.append(msg.content)
            elif msg.role == Role.USER:
                formatted.append({
                    "role": "user",
                    "content": msg.content,
                })
            elif msg.role == Role.ASSISTANT:
                # Handle assistant messages that may contain tool use blocks.
                if msg.tool_calls:
                    # Format as tool_use content blocks.
                    content_blocks: list[dict[str, Any]] = []
                    if msg.content:
                        content_blocks.append({"type": "text", "text": msg.content})
                    for tc in msg.tool_calls:
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        })
                    formatted.append({"role": "assistant", "content": content_blocks})
                else:
                    formatted.append({
                        "role": "assistant",
                        "content": msg.content,
                    })
            elif msg.role == Role.TOOL:
                # Tool results in Anthropic's format use tool_result content blocks.
                formatted.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id or "",
                            "content": msg.content,
                        }
                    ],
                })

        # Join all system message parts into a single string.
        system_prompt = "\n\n".join(system_parts) if system_parts else ""

        return system_prompt, formatted

    def chat(self, messages: list[Message]) -> Message:
        """
        Generate a chat completion response from Claude.

        Sends the conversation to Anthropic's Messages API and returns
        the response as a framework Message.

        Args:
            messages: The conversation history as Message objects.

        Returns:
            A Message with role=ASSISTANT containing Claude's response.

        Raises:
            anthropic.APIError: If the API call fails.
        """
        # Separate system prompt from conversation messages.
        system_prompt, formatted_messages = self._extract_system_and_messages(messages)

        # Build the API call parameters.
        params: dict[str, Any] = {
            "model": self.config.model,
            "messages": formatted_messages,
            "max_tokens": self.config.max_tokens or 4096,
            "temperature": self.config.temperature,
        }

        # Add system prompt if present.
        if system_prompt:
            params["system"] = system_prompt

        logger.debug("Sending chat request to Claude with %d messages", len(messages))

        # Make the API call.
        response = self.client.messages.create(**params)

        # Extract text content from the response content blocks.
        response_text = ""
        for block in response.content:
            if block.type == "text":
                response_text += block.text

        return Message(
            role=Role.ASSISTANT,
            content=response_text,
            metadata={
                "model": response.model,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                "stop_reason": response.stop_reason,
            },
        )

    def chat_with_tools(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> Message:
        """
        Generate a response with tool use support via Anthropic's native format.

        Converts OpenAI-format tool schemas to Anthropic's tool format,
        sends the request, and parses any tool_use blocks from the response.

        Args:
            messages: The conversation history as Message objects.
            tools:    Tool schemas in OpenAI function-calling format.

        Returns:
            A Message that may contain tool_calls if Claude chose to use tools.
        """
        # Separate system prompt from messages.
        system_prompt, formatted_messages = self._extract_system_and_messages(messages)

        # Convert OpenAI tool schemas to Anthropic's format.
        anthropic_tools = []
        for tool in tools:
            func = tool.get("function", {})
            anthropic_tools.append({
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
            })

        # Build the API call parameters with tools.
        params: dict[str, Any] = {
            "model": self.config.model,
            "messages": formatted_messages,
            "max_tokens": self.config.max_tokens or 4096,
            "temperature": self.config.temperature,
            "tools": anthropic_tools,
        }

        if system_prompt:
            params["system"] = system_prompt

        logger.debug("Sending tool-enabled chat request to Claude with %d tools", len(tools))

        # Make the API call with tools.
        response = self.client.messages.create(**params)

        # Parse the response content blocks for text and tool_use.
        response_text = ""
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                response_text += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    )
                )

        return Message(
            role=Role.ASSISTANT,
            content=response_text,
            tool_calls=tool_calls,
            metadata={
                "model": response.model,
                "stop_reason": response.stop_reason,
            },
        )

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        """
        Stream the response token by token using Anthropic's streaming API.

        Uses Anthropic's message streaming to yield text deltas as they
        are generated.

        Args:
            messages: The conversation history as Message objects.

        Yields:
            Text chunks as they are generated by Claude.
        """
        system_prompt, formatted_messages = self._extract_system_and_messages(messages)

        params: dict[str, Any] = {
            "model": self.config.model,
            "messages": formatted_messages,
            "max_tokens": self.config.max_tokens or 4096,
            "temperature": self.config.temperature,
        }

        if system_prompt:
            params["system"] = system_prompt

        logger.debug("Starting streaming chat request to Claude")

        # Use the async client's streaming API.
        async with self.async_client.messages.stream(**params) as stream:
            async for text in stream.text_stream:
                yield text
