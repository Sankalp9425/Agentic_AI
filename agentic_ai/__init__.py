"""
Agentic AI - A Comprehensive Python Framework for Building Advanced AI Agents.

This package provides a modular, extensible framework for building sophisticated
AI agent systems. It includes support for multiple agentic paradigms (RAG, ReAct,
Hierarchical, Planning), integrations with popular LLM providers (OpenAI, Gemini,
Claude, Groq), vector database connectors (Chroma, Pinecone, PGVector, FAISS),
and MCP (Model Context Protocol) connectors for tasks like email and web search.

Architecture Overview:
    - core/        : Abstract base classes and shared data models (agents, LLMs,
                     embeddings, vector stores, tools, memory, messages).
    - llms/        : Concrete LLM provider implementations for chat completions.
    - embeddings/  : Concrete embedding provider implementations for text vectorization.
    - vectorstores/: Concrete vector database connectors for similarity search.
    - agents/      : High-level agentic framework implementations (RAG, ReAct, etc.).
    - mcp/         : MCP-compatible tool connectors (email, search, etc.).
    - utils/       : Shared utilities (logging, retry logic, configuration helpers).

Example Usage:
    >>> from agentic_ai.llms.openai import OpenAIChatModel
    >>> from agentic_ai.agents.react import ReActAgent
    >>> llm = OpenAIChatModel(api_key="sk-...", model="gpt-4")
    >>> agent = ReActAgent(llm=llm, tools=[...])
    >>> response = agent.run("What is the weather in Tokyo?")
"""

# Package version following semantic versioning (major.minor.patch).
__version__ = "0.1.0"

# Package metadata for identification and attribution.
__author__ = "Agentic AI Contributors"
__license__ = "MIT"
