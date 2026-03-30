"""
Abstract base class and implementations for agent memory systems.

Memory allows agents to persist and retrieve information across interactions.
This is essential for agents that need to maintain context over long
conversations, learn from past interactions, or access previously
gathered information.

The module provides:
    - MemoryStore: Abstract base class defining the memory interface.
    - InMemoryStore: A simple dictionary-based implementation for development
                     and testing (data is lost when the process exits).
    - ConversationBufferMemory: Stores the last N conversation turns.

Example:
    >>> memory = InMemoryStore()
    >>> memory.store("What is Python?", "Python is a programming language.")
    >>> memory.retrieve("Tell me about Python")
    'Q: What is Python?\\nA: Python is a programming language.'
"""

from abc import ABC, abstractmethod
from collections import deque


class MemoryStore(ABC):
    """
    Abstract base class for agent memory backends.

    Memory stores provide a simple key-value-like interface for agents
    to persist information. The `store()` method saves a query-response
    pair, and `retrieve()` fetches relevant past interactions.

    Different implementations may use different storage backends (in-memory,
    Redis, vector databases) and different retrieval strategies (exact match,
    recency, semantic similarity).
    """

    @abstractmethod
    def store(self, query: str, response: str) -> None:
        """
        Store a query-response pair in memory.

        Args:
            query:    The user's input query or instruction.
            response: The agent's response to the query.
        """
        ...

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> str:
        """
        Retrieve relevant past interactions based on a query.

        Args:
            query: The current query to find relevant context for.
            k:     The maximum number of past interactions to retrieve.

        Returns:
            A string containing relevant past interactions, formatted
            as Q&A pairs. Returns an empty string if no relevant
            interactions are found.
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """
        Clear all stored memories.

        This is a destructive operation that removes all past interactions.
        Use with caution in production environments.
        """
        ...


class InMemoryStore(MemoryStore):
    """
    A simple in-memory implementation of MemoryStore.

    Stores query-response pairs in a Python list. Retrieval is based on
    simple substring matching - if any word from the current query appears
    in a stored query, that interaction is considered relevant.

    This implementation is suitable for development, testing, and short-lived
    agents. For production use, consider a persistent backend.

    Attributes:
        _history: A list of (query, response) tuples stored in chronological order.
    """

    def __init__(self) -> None:
        """Initialize an empty in-memory store."""
        # Internal storage: a list of (query, response) tuples.
        self._history: list[tuple[str, str]] = []

    def store(self, query: str, response: str) -> None:
        """
        Append a query-response pair to the in-memory history.

        Args:
            query:    The user's input query.
            response: The agent's response to the query.
        """
        self._history.append((query, response))

    def retrieve(self, query: str, k: int = 5) -> str:
        """
        Retrieve relevant past interactions using substring matching.

        Searches stored queries for any word overlap with the current query.
        Returns the most recent matching interactions, up to `k` results.

        Args:
            query: The current query to find relevant context for.
            k:     Maximum number of results to return. Default is 5.

        Returns:
            A formatted string of matching Q&A pairs, or empty string if none found.
        """
        # Tokenize the query into lowercase words for matching.
        query_words = set(query.lower().split())

        # Find all stored interactions where any query word appears.
        matches: list[tuple[str, str]] = []
        for stored_query, stored_response in self._history:
            stored_words = set(stored_query.lower().split())
            # Check if there's any word overlap between current and stored query.
            if query_words & stored_words:
                matches.append((stored_query, stored_response))

        # Return the most recent k matches, formatted as Q&A pairs.
        recent_matches = matches[-k:]
        if not recent_matches:
            return ""

        # Format each match as a readable Q&A pair.
        formatted = []
        for q, a in recent_matches:
            formatted.append(f"Q: {q}\nA: {a}")

        return "\n\n".join(formatted)

    def clear(self) -> None:
        """Clear all stored interactions from memory."""
        self._history.clear()


class ConversationBufferMemory(MemoryStore):
    """
    A sliding-window memory that stores the last N conversation turns.

    Unlike InMemoryStore, which performs keyword matching, this memory
    simply returns the most recent interactions regardless of relevance.
    This is useful for agents that need recent conversational context
    but don't need sophisticated retrieval.

    Attributes:
        _buffer: A deque of (query, response) tuples with a fixed max length.
        max_turns: The maximum number of conversation turns to remember.
    """

    def __init__(self, max_turns: int = 10) -> None:
        """
        Initialize the conversation buffer with a maximum capacity.

        Args:
            max_turns: The maximum number of conversation turns to store.
                       Oldest turns are automatically discarded when the
                       buffer is full. Default is 10.
        """
        # Store the maximum number of turns to keep.
        self.max_turns = max_turns
        # Use a deque with maxlen for automatic eviction of old entries.
        self._buffer: deque[tuple[str, str]] = deque(maxlen=max_turns)

    def store(self, query: str, response: str) -> None:
        """
        Add a conversation turn to the buffer.

        If the buffer is full, the oldest turn is automatically removed.

        Args:
            query:    The user's input query.
            response: The agent's response.
        """
        self._buffer.append((query, response))

    def retrieve(self, query: str, k: int = 5) -> str:
        """
        Retrieve the most recent conversation turns.

        Ignores the query parameter and simply returns the last `k` turns
        from the buffer. This provides recent context regardless of topic.

        Args:
            query: Ignored - included for interface compatibility.
            k:     Maximum number of recent turns to return. Default is 5.

        Returns:
            A formatted string of recent Q&A pairs, or empty string if
            the buffer is empty.
        """
        # Take the last k entries from the buffer.
        recent = list(self._buffer)[-k:]
        if not recent:
            return ""

        # Format as readable Q&A pairs.
        formatted = []
        for q, a in recent:
            formatted.append(f"Q: {q}\nA: {a}")

        return "\n\n".join(formatted)

    def clear(self) -> None:
        """Clear all conversation turns from the buffer."""
        self._buffer.clear()
