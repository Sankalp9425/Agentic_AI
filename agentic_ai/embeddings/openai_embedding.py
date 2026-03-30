"""
OpenAI embedding provider implementation.

This module provides integration with OpenAI's Embeddings API, supporting
models like text-embedding-3-small, text-embedding-3-large, and the legacy
text-embedding-ada-002. OpenAI's embedding models are widely used for RAG
applications due to their strong performance on retrieval benchmarks.

The text-embedding-3 family supports configurable dimensions via Matryoshka
Representation Learning (MRL), allowing you to trade off between embedding
quality and storage/compute costs.

Requirements:
    pip install openai

Example:
    >>> from agentic_ai.embeddings.openai_embedding import OpenAIEmbedding
    >>> embedder = OpenAIEmbedding(api_key="sk-...", model="text-embedding-3-small")
    >>> vectors = embedder.embed_documents(["Hello", "World"])
    >>> len(vectors[0])
    1536
"""

import logging
from typing import Any

from agentic_ai.core.base_embedding import BaseEmbedding, EmbeddingConfig

# Configure module-level logger.
logger = logging.getLogger(__name__)


class OpenAIEmbedding(BaseEmbedding):
    """
    OpenAI text embedding provider.

    Wraps the OpenAI Python SDK to generate text embeddings using OpenAI's
    embedding models. Handles batching for efficient processing of large
    document sets and supports configurable dimensions for the v3 models.

    Attributes:
        client: The OpenAI client instance for API calls.

    Example:
        >>> embedder = OpenAIEmbedding(api_key="sk-...", model="text-embedding-3-small")
        >>> query_vector = embedder.embed_query("What is Python?")
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
        batch_size: int = 100,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the OpenAI embedding provider.

        Args:
            api_key:    OpenAI API key for authentication.
            model:      Embedding model identifier. Default is "text-embedding-3-small"
                        which offers a good balance of quality and cost.
            dimensions: Target output dimensionality for v3 models. None uses the
                        model's native dimension (1536 for small, 3072 for large).
            batch_size: Maximum texts per API call. Default is 100 (OpenAI's limit).
            base_url:   Custom API endpoint for OpenAI-compatible services.
            **kwargs:   Additional parameters stored in EmbeddingConfig.extra.
        """
        config = EmbeddingConfig(
            model=model,
            api_key=api_key,
            dimensions=dimensions,
            batch_size=batch_size,
            base_url=base_url,
            extra=kwargs,
        )
        super().__init__(config)

        # Import openai to create the client.
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "The 'openai' package is required for OpenAIEmbedding. "
                "Install it with: pip install openai"
            ) from e

        # Create the OpenAI client for embedding API calls.
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

        logger.info("Initialized OpenAI embedding model: %s", model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a batch of documents.

        Automatically splits large batches into smaller chunks based on
        the configured batch_size to respect API limits.

        Args:
            texts: A list of text strings to embed.

        Returns:
            A list of embedding vectors, one per input text.

        Raises:
            openai.APIError: If the API call fails.
            ValueError: If the texts list is empty.
        """
        if not texts:
            raise ValueError("Cannot embed an empty list of texts.")

        all_embeddings: list[list[float]] = []

        # Process texts in batches to respect API limits.
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]

            logger.debug(
                "Embedding batch %d-%d of %d texts",
                i, i + len(batch), len(texts),
            )

            # Build the API call parameters.
            params: dict[str, Any] = {
                "model": self.config.model,
                "input": batch,
            }

            # Add dimensions parameter for v3 models if configured.
            if self.config.dimensions:
                params["dimensions"] = self.config.dimensions

            # Make the API call.
            response = self.client.embeddings.create(**params)

            # Extract embeddings from the response, maintaining input order.
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """
        Generate an embedding for a single query string.

        This is a convenience method that wraps embed_documents for
        single-text embedding. OpenAI doesn't differentiate between
        query and document embeddings.

        Args:
            text: The query string to embed.

        Returns:
            A list of floats representing the query's embedding vector.
        """
        # Delegate to embed_documents and return the single result.
        result = self.embed_documents([text])
        return result[0]
