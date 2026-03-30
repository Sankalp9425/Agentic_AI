"""
Anthropic/Voyage AI embedding provider implementation.

Anthropic recommends Voyage AI as their partner for text embeddings. This
module provides integration with the Voyage AI API, which offers embedding
models specifically optimized for use with Claude and other LLMs.

Voyage AI models support different input types (query vs. document) for
optimized retrieval performance, similar to Google's task-type system.

Requirements:
    pip install voyageai

Example:
    >>> from agentic_ai.embeddings.claude_embedding import VoyageEmbedding
    >>> embedder = VoyageEmbedding(api_key="pa-...", model="voyage-large-2")
    >>> vector = embedder.embed_query("What is machine learning?")
"""

import logging
from typing import Any

from agentic_ai.core.base_embedding import BaseEmbedding, EmbeddingConfig

# Configure module-level logger.
logger = logging.getLogger(__name__)


class VoyageEmbedding(BaseEmbedding):
    """
    Voyage AI text embedding provider (recommended by Anthropic).

    Wraps the Voyage AI SDK to generate high-quality text embeddings.
    Voyage models are designed to work well with Claude for RAG applications,
    offering strong performance on retrieval benchmarks.

    Attributes:
        client: The Voyage AI client instance for API calls.

    Example:
        >>> embedder = VoyageEmbedding(api_key="pa-...", model="voyage-large-2")
        >>> vectors = embedder.embed_documents(["Hello world"])
    """

    def __init__(
        self,
        api_key: str,
        model: str = "voyage-large-2",
        batch_size: int = 128,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Voyage AI embedding provider.

        Args:
            api_key:    Voyage AI API key for authentication (starts with "pa-").
            model:      Voyage model identifier (e.g., "voyage-large-2", "voyage-code-2").
            batch_size: Maximum texts per API call. Default is 128.
            **kwargs:   Additional parameters stored in EmbeddingConfig.extra.
        """
        config = EmbeddingConfig(
            model=model,
            api_key=api_key,
            batch_size=batch_size,
            extra=kwargs,
        )
        super().__init__(config)

        # Import voyageai to create the client.
        try:
            import voyageai
        except ImportError as e:
            raise ImportError(
                "The 'voyageai' package is required for VoyageEmbedding. "
                "Install it with: pip install voyageai"
            ) from e

        # Create the Voyage AI client.
        self.client = voyageai.Client(api_key=api_key)

        logger.info("Initialized Voyage AI embedding model: %s", model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a batch of documents using Voyage AI.

        Uses input_type="document" to optimize embeddings for indexing.

        Args:
            texts: A list of text strings to embed.

        Returns:
            A list of embedding vectors, one per input text.
        """
        if not texts:
            raise ValueError("Cannot embed an empty list of texts.")

        all_embeddings: list[list[float]] = []

        # Process in batches to respect API limits.
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]

            logger.debug(
                "Embedding batch %d-%d of %d texts with Voyage AI",
                i, i + len(batch), len(texts),
            )

            # Use input_type="document" for document embeddings.
            result = self.client.embed(
                texts=batch,
                model=self.config.model,
                input_type="document",
            )
            all_embeddings.extend(result.embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """
        Generate an embedding for a single query using Voyage AI.

        Uses input_type="query" to optimize the embedding for retrieval
        against document embeddings.

        Args:
            text: The query string to embed.

        Returns:
            A list of floats representing the query embedding vector.
        """
        # Use input_type="query" for query embeddings.
        result = self.client.embed(
            texts=[text],
            model=self.config.model,
            input_type="query",
        )
        return result.embeddings[0]
