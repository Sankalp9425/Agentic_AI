"""
LLM provider implementations for chat completions.

This module contains concrete implementations of the BaseLLM interface for
all supported language model providers. Each provider handles API authentication,
request formatting, response parsing, and error handling specific to its platform.

Supported Providers:
    - OpenAI:  GPT-4, GPT-4o, GPT-3.5-turbo and compatible models.
    - Google:  Gemini Pro, Gemini Ultra and other Google AI models.
    - Anthropic: Claude 3 Opus, Sonnet, Haiku and other Claude models.
    - Groq:    LLaMA, Mixtral, and other models via Groq's fast inference API.

Example:
    >>> from agentic_ai.llms.openai import OpenAIChatModel
    >>> llm = OpenAIChatModel(api_key="sk-...", model="gpt-4")
    >>> response = llm.chat([Message(role=Role.USER, content="Hello!")])
"""
