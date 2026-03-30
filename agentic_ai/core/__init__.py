"""
Core module - Abstract base classes and shared data models.

This module defines the foundational abstractions that all concrete implementations
must inherit from. By programming against these interfaces, the framework achieves
loose coupling between components, making it easy to swap LLM providers, vector
stores, or agent strategies without changing application code.

Exports:
    - BaseLLM: Abstract base class for all chat/language model providers.
    - BaseEmbedding: Abstract base class for all text embedding providers.
    - BaseVectorStore: Abstract base class for all vector database connectors.
    - BaseAgent: Abstract base class for all agent implementations.
    - BaseTool: Abstract base class for all tools that agents can invoke.
    - Message: Data class representing a single message in a conversation.
    - ToolCall: Data class representing a tool invocation request.
    - ToolResult: Data class representing the result of a tool invocation.
    - AgentState: Data class representing the internal state of an agent.
    - Document: Data class representing a text document with metadata.
    - MemoryStore: Abstract base class for agent memory backends.
"""

from agentic_ai.core.base_agent import BaseAgent
from agentic_ai.core.base_embedding import BaseEmbedding
from agentic_ai.core.base_llm import BaseLLM
from agentic_ai.core.base_tool import BaseTool
from agentic_ai.core.base_vectorstore import BaseVectorStore
from agentic_ai.core.memory import MemoryStore
from agentic_ai.core.models import (
    AgentState,
    Document,
    Message,
    ToolCall,
    ToolResult,
)

__all__ = [
    "BaseLLM",
    "BaseEmbedding",
    "BaseVectorStore",
    "BaseAgent",
    "BaseTool",
    "Message",
    "ToolCall",
    "ToolResult",
    "AgentState",
    "Document",
    "MemoryStore",
]
