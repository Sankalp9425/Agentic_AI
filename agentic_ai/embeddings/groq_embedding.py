"""
Groq embedding provider implementation.

This module provides integration with embedding models accessible via
OpenAI-compatible APIs. Groq's fast inference can also be leveraged for
embedding generation when compatible models are available. This implementation
uses the OpenAI SDK with a configurable base URL, making it compatible with
any OpenAI-API-compatible embedding service.

Requirements:
    pip install openai

Example:
    >>> from agentic_ai.embeddings.groq_embedding import GroqEmbedding
    >>> embedder = GroqEmbedding(
    ...     api_key="gsk_...",
    ...     model="nomic-embed-text-v1.5",
    ...     base_url="https://api.groq.com/openai/v1",
    ... )
    >>> vector = embedder.embed_query("What is deep learning?")
"""

import logging
from typing import Any

from agentic_ai.core.base_embedding import BaseEmbedding, EmbeddingConfig

# Configure module-level logger.
logger = logging.getLogger(__name__)


class GroqEmbedding(BaseEmbedding):
    """
    Groq/OpenAI-compatible embedding provider.

    Uses the OpenAI SDK with a custom base URL to connect to Groq's
    embedding API or any OpenAI-compatible embedding endpoint. This allows
    flexibility in choosing different embedding services while maintaining
    a consistent interface.

    Attributes:
        client: The OpenAI client configured to point to Groq's API.

    Example:
        >>> embedder = GroqEmbedding(api_key="gsk_...", model="nomic-embed-text-v1.5")
        >>> vectors = embedder.embed_documents(["Hello world"])
    """

    def __init__(
        self,
        api_key: str,
        model: str = "nomic-embed-text-v1.5",
        base_url: str = "https://api.groq.com/openai/v1",
        batch_size: int = 100,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Groq embedding provider.

        Args:
            api_key:   Groq API key for authentication.
            model:     Embedding model identifier. Default is "nomic-embed-text-v1.5".
            base_url:  API endpoint URL. Default points to Groq's OpenAI-compatible API.
            batch_size: Maximum texts per API call. Default is 100.
            **kwargs:  Additional parameters stored in EmbeddingConfig.extra.
        """
        config = EmbeddingConfig(
            model=model,
            api_key=api_key,
            batch_size=batch_size,
            base_url=base_url,
            extra=kwargs,
        )
        super().__init__(config)

        # Import openai to use its client with Groq's endpoint.
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "The 'openai' package is required for GroqEmbedding. "
                "Install it with: pip install openai"
            ) from e

        # Create an OpenAI client configured for Groq's API endpoint.
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

        logger.info("Initialized Groq embedding model: %s", model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a batch of documents via Groq's API.

        Processes texts in configurable batches for efficient API usage.

        Args:
            texts: A list of text strings to embed.

        Returns:
            A list of embedding vectors, one per input text.
        """
        if not texts:
            raise ValueError("Cannot embed an empty list of texts.")

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]

            logger.debug(
                "Embedding batch %d-%d of %d texts via Groq",
                i, i + len(batch), len(texts),
            )

            # Use the OpenAI-compatible embeddings endpoint.
            response = self.client.embeddings.create(
                model=self.config.model,
                input=batch,
            )

            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """
        Generate an embedding for a single query string.

        Args:
            text: The query string to embed.

        Returns:
            A list of floats representing the query embedding vector.
        """
        result = self.embed_documents([text])
        return result[0]
