"""
Google Gemini embedding provider implementation.

This module provides integration with Google's Generative AI embedding API,
supporting models like embedding-001 and text-embedding-004. Google's embedding
models are optimized for different task types (retrieval, clustering, classification),
allowing you to specify the intended use case for better quality embeddings.

Requirements:
    pip install google-generativeai

Example:
    >>> from agentic_ai.embeddings.gemini_embedding import GeminiEmbedding
    >>> embedder = GeminiEmbedding(api_key="AIza...", model="models/text-embedding-004")
    >>> vector = embedder.embed_query("What is Python?")
"""

import logging
from typing import Any

from agentic_ai.core.base_embedding import BaseEmbedding, EmbeddingConfig

# Configure module-level logger.
logger = logging.getLogger(__name__)


class GeminiEmbedding(BaseEmbedding):
    """
    Google Gemini text embedding provider.

    Wraps the Google Generative AI SDK to generate text embeddings. Supports
    task-type-aware embedding where you can specify whether the text is a
    query (for retrieval) or a document (for indexing), which can improve
    retrieval quality.

    Attributes:
        _genai: Reference to the google.generativeai module.
        _task_type: The task type to use for embedding generation.

    Example:
        >>> embedder = GeminiEmbedding(api_key="AIza...", model="models/text-embedding-004")
        >>> vectors = embedder.embed_documents(["Hello world"])
    """

    def __init__(
        self,
        api_key: str,
        model: str = "models/text-embedding-004",
        dimensions: int | None = None,
        task_type: str = "retrieval_document",
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Gemini embedding provider.

        Args:
            api_key:   Google AI API key for authentication.
            model:     Embedding model identifier. Default is "models/text-embedding-004".
            dimensions: Target output dimensionality. None uses model's default.
            task_type: The intended use for the embeddings. Options include:
                       "retrieval_query", "retrieval_document", "semantic_similarity",
                       "classification", "clustering". Default is "retrieval_document".
            **kwargs:  Additional parameters stored in EmbeddingConfig.extra.
        """
        config = EmbeddingConfig(
            model=model,
            api_key=api_key,
            dimensions=dimensions,
            extra=kwargs,
        )
        super().__init__(config)

        # Import google.generativeai to configure the API.
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ImportError(
                "The 'google-generativeai' package is required for GeminiEmbedding. "
                "Install it with: pip install google-generativeai"
            ) from e

        # Configure the API with the provided key.
        genai.configure(api_key=api_key)

        # Store the module reference and task type for use in embedding methods.
        self._genai = genai
        self._task_type = task_type

        logger.info("Initialized Gemini embedding model: %s", model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a batch of documents using Gemini.

        Uses the "retrieval_document" task type for document embeddings
        to optimize for indexing and retrieval use cases.

        Args:
            texts: A list of text strings to embed.

        Returns:
            A list of embedding vectors, one per input text.

        Raises:
            google.api_core.exceptions.GoogleAPIError: If the API call fails.
        """
        if not texts:
            raise ValueError("Cannot embed an empty list of texts.")

        embeddings: list[list[float]] = []

        # Gemini's embed_content can handle batches, but we process
        # individually for reliability and consistent error handling.
        for text in texts:
            params: dict[str, Any] = {
                "model": self.config.model,
                "content": text,
                "task_type": self._task_type,
            }

            # Add output dimensionality if configured.
            if self.config.dimensions:
                params["output_dimensionality"] = self.config.dimensions

            result = self._genai.embed_content(**params)
            embeddings.append(result["embedding"])

        logger.debug("Embedded %d documents with Gemini", len(texts))
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """
        Generate an embedding for a single query using Gemini.

        Uses the "retrieval_query" task type, which is optimized for
        queries that will be matched against document embeddings.

        Args:
            text: The query string to embed.

        Returns:
            A list of floats representing the query embedding vector.
        """
        params: dict[str, Any] = {
            "model": self.config.model,
            "content": text,
            # Use retrieval_query for queries to match against documents.
            "task_type": "retrieval_query",
        }

        if self.config.dimensions:
            params["output_dimensionality"] = self.config.dimensions

        result = self._genai.embed_content(**params)
        return result["embedding"]
