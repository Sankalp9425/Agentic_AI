"""
Google Gemini LLM provider implementation.

This module provides integration with Google's Generative AI API, supporting
Gemini Pro, Gemini Ultra, and other Google AI models. It handles message
formatting for Google's conversation format, tool/function calling using
Google's function declaration schema, and streaming responses.

Requirements:
    pip install google-generativeai

Example:
    >>> from agentic_ai.llms.gemini_llm import GeminiChatModel
    >>> llm = GeminiChatModel(api_key="AIza...", model="gemini-pro")
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


class GeminiChatModel(BaseLLM):
    """
    Google Gemini chat completion provider.

    Wraps the Google Generative AI SDK to provide chat completions using
    Gemini models. Handles the conversion between the framework's message
    format and Google's content/parts format.

    Attributes:
        model_instance: The Google GenerativeModel instance for API calls.

    Example:
        >>> llm = GeminiChatModel(api_key="AIza...", model="gemini-pro")
        >>> response = llm.chat([Message(role=Role.USER, content="Hi!")])
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-pro",
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Gemini chat model provider.

        Configures the Google AI SDK with the provided API key and creates
        a model instance for generating responses.

        Args:
            api_key:     Google AI API key for authentication.
            model:       Gemini model identifier (e.g., "gemini-pro", "gemini-ultra").
            temperature: Sampling temperature (0.0 to 1.0). Default is 0.7.
            max_tokens:  Maximum tokens in the response. None uses model default.
            **kwargs:    Additional parameters stored in LLMConfig.extra.
        """
        # Build the configuration object.
        config = LLMConfig(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            extra=kwargs,
        )
        super().__init__(config)

        # Import google.generativeai here to make it an optional dependency.
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ImportError(
                "The 'google-generativeai' package is required for GeminiChatModel. "
                "Install it with: pip install google-generativeai"
            ) from e

        # Configure the Google AI SDK with the API key.
        genai.configure(api_key=api_key)

        # Create the model instance for generating responses.
        self.model_instance = genai.GenerativeModel(model)

        # Store the genai module reference for use in other methods.
        self._genai = genai

        logger.info("Initialized Gemini chat model: %s", model)

    def _format_messages_for_gemini(
        self, messages: list[Message]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """
        Convert framework messages to Google's conversation format.

        Google's API uses a different message format with 'role' values of
        'user' and 'model' (not 'assistant'), and 'parts' instead of 'content'.
        System messages are handled separately as system instructions.

        Args:
            messages: Framework Message objects to convert.

        Returns:
            A tuple of (system_instruction, history) where system_instruction
            is the system prompt text (or None), and history is a list of
            message dictionaries in Google's format.
        """
        system_instruction = None
        history: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                # Google handles system messages as a separate parameter.
                system_instruction = msg.content
            elif msg.role == Role.USER:
                # User messages map directly.
                history.append({
                    "role": "user",
                    "parts": [{"text": msg.content}],
                })
            elif msg.role == Role.ASSISTANT:
                # Assistant messages use "model" as the role in Google's format.
                history.append({
                    "role": "model",
                    "parts": [{"text": msg.content}],
                })
            elif msg.role == Role.TOOL:
                # Tool results are formatted as function responses.
                history.append({
                    "role": "user",
                    "parts": [{"text": f"[Tool Result]: {msg.content}"}],
                })

        return system_instruction, history

    def chat(self, messages: list[Message]) -> Message:
        """
        Generate a chat completion response from Gemini.

        Converts messages to Google's format, sends the request, and
        parses the response back into a framework Message.

        Args:
            messages: The conversation history as Message objects.

        Returns:
            A Message with role=ASSISTANT containing Gemini's response.

        Raises:
            google.api_core.exceptions.GoogleAPIError: If the API call fails.
        """
        # Convert messages to Google's format.
        system_instruction, history = self._format_messages_for_gemini(messages)

        # Build generation configuration from our settings.
        generation_config = {
            "temperature": self.config.temperature,
        }
        if self.config.max_tokens:
            generation_config["max_output_tokens"] = self.config.max_tokens

        logger.debug("Sending chat request to Gemini with %d messages", len(messages))

        # If there's a system instruction, create a new model instance with it.
        if system_instruction:
            model = self._genai.GenerativeModel(
                self.config.model,
                system_instruction=system_instruction,
            )
        else:
            model = self.model_instance

        # Start a chat session with the history (excluding the last user message).
        # The last user message is sent via send_message().
        chat_history = history[:-1] if len(history) > 1 else []
        last_message = history[-1]["parts"][0]["text"] if history else ""

        chat = model.start_chat(history=chat_history)
        response = chat.send_message(
            last_message,
            generation_config=generation_config,
        )

        # Extract the response text.
        response_text = response.text if response.text else ""

        return Message(
            role=Role.ASSISTANT,
            content=response_text,
            metadata={
                "model": self.config.model,
                "finish_reason": "stop",
            },
        )

    def chat_with_tools(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> Message:
        """
        Generate a response with function calling support.

        Converts OpenAI-format tool schemas to Google's function declaration
        format, sends the request, and parses any function calls from the
        response.

        Args:
            messages: The conversation history as Message objects.
            tools:    Tool schemas in OpenAI function-calling format.

        Returns:
            A Message that may contain tool_calls if Gemini chose to
            invoke functions.
        """
        from google.generativeai.types import content_types

        # Convert OpenAI tool schemas to Google's function declaration format.
        function_declarations = []
        for tool in tools:
            func = tool.get("function", {})
            function_declarations.append({
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
            })

        # Convert messages and set up the model with tools.
        system_instruction, history = self._format_messages_for_gemini(messages)

        # Create a model with function calling enabled.
        gemini_tools = content_types.to_tool(function_declarations)

        if system_instruction:
            model = self._genai.GenerativeModel(
                self.config.model,
                system_instruction=system_instruction,
                tools=[gemini_tools],
            )
        else:
            model = self._genai.GenerativeModel(
                self.config.model,
                tools=[gemini_tools],
            )

        # Send the message through a chat session.
        chat_history = history[:-1] if len(history) > 1 else []
        last_message = history[-1]["parts"][0]["text"] if history else ""

        chat = model.start_chat(history=chat_history)
        response = chat.send_message(last_message)

        # Parse function calls from the response.
        tool_calls: list[ToolCall] = []
        response_text = ""

        for part in response.parts:
            if hasattr(part, "function_call") and part.function_call:
                # Extract the function call details.
                fc = part.function_call
                tool_calls.append(
                    ToolCall(
                        id=f"call_{fc.name}",
                        name=fc.name,
                        arguments=dict(fc.args) if fc.args else {},
                    )
                )
            elif hasattr(part, "text") and part.text:
                response_text += part.text

        return Message(
            role=Role.ASSISTANT,
            content=response_text,
            tool_calls=tool_calls,
            metadata={"model": self.config.model},
        )

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        """
        Stream the response token by token from Gemini.

        Uses Gemini's streaming API to yield response chunks as they
        are generated by the model.

        Args:
            messages: The conversation history as Message objects.

        Yields:
            Text chunks as they are generated by Gemini.
        """
        system_instruction, history = self._format_messages_for_gemini(messages)

        if system_instruction:
            model = self._genai.GenerativeModel(
                self.config.model,
                system_instruction=system_instruction,
            )
        else:
            model = self.model_instance

        chat_history = history[:-1] if len(history) > 1 else []
        last_message = history[-1]["parts"][0]["text"] if history else ""

        chat = model.start_chat(history=chat_history)

        # Use stream=True for streaming responses.
        response = chat.send_message(last_message, stream=True)

        for chunk in response:
            if chunk.text:
                yield chunk.text
