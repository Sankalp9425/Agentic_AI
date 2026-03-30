"""
Abstract base class for all text embedding providers.

Embeddings convert text into dense numerical vectors that capture semantic
meaning. These vectors are used for similarity search in vector databases,
enabling retrieval-augmented generation (RAG) and other information retrieval
tasks. Different providers (OpenAI, Gemini, Groq, etc.) offer embedding
models with varying dimensions, performance characteristics, and costs.

The design follows the Strategy pattern, allowing seamless swapping of
embedding providers without changing application code.

Example:
    >>> from agentic_ai.embeddings.openai import OpenAIEmbedding
    >>> embedder = OpenAIEmbedding(api_key="sk-...", model="text-embedding-3-small")
    >>> vectors = embedder.embed_documents(["Hello world", "Goodbye world"])
    >>> len(vectors[0])  # embedding dimension
    1536
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EmbeddingConfig:
    """
    Configuration parameters shared across all embedding providers.

    Attributes:
        model:      The embedding model identifier (e.g., "text-embedding-3-small").
        api_key:    The API key for authenticating with the provider.
        dimensions: The desired output dimensionality. Some providers support
                    configurable dimensions (e.g., OpenAI's Matryoshka embeddings).
                    None means use the model's default dimension.
        batch_size: Maximum number of texts to embed in a single API call.
                    Larger batches are more efficient but use more memory.
        base_url:   Optional custom API endpoint URL for self-hosted deployments.
        timeout:    Request timeout in seconds. Default is 30 seconds.
        extra:      Provider-specific configuration parameters.
    """

    model: str = "text-embedding-3-small"
    api_key: str = ""
    dimensions: int | None = None
    batch_size: int = 100
    base_url: str | None = None
    timeout: int = 30
    extra: dict[str, Any] = field(default_factory=dict)


class BaseEmbedding(ABC):
    """
    Abstract base class that all embedding providers must inherit from.

    Subclasses must implement `embed_documents()` for batch embedding and
    `embed_query()` for single-query embedding. Some providers optimize
    these differently (e.g., query embeddings may use a different model
    variant for better retrieval performance).

    Attributes:
        config: An EmbeddingConfig instance containing provider configuration.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        """
        Initialize the embedding provider with the given configuration.

        Args:
            config: An EmbeddingConfig instance with all necessary parameters
                    for connecting to the embedding service.
        """
        # Store the configuration for use by subclass methods.
        self.config = config

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embedding vectors for a batch of documents.

        This method is optimized for bulk embedding operations, such as
        indexing a corpus of documents into a vector database. Implementations
        should handle batching internally based on `config.batch_size`.

        Args:
            texts: A list of text strings to embed. Each string is treated
                   as a separate document.

        Returns:
            A list of embedding vectors, one per input text. Each vector
            is a list of floats with length equal to the model's dimension.

        Raises:
            ConnectionError: If the API call fails.
            ValueError: If the texts list is empty.
        """
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """
        Generate an embedding vector for a single query string.

        This method is optimized for query-time embedding, which may use
        a different model variant or preprocessing than document embedding
        for improved retrieval accuracy.

        Args:
            text: The query string to embed.

        Returns:
            A list of floats representing the query's embedding vector.

        Raises:
            ConnectionError: If the API call fails.
            ValueError: If the text is empty.
        """
        ...

    @property
    def dimension(self) -> int | None:
        """
        Return the output dimensionality of the embedding model.

        Returns:
            The number of dimensions in the output vectors, or None if
            the dimension is unknown until the first embedding call.
        """
        # Return the configured dimension if explicitly set.
        return self.config.dimensions
