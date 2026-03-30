"""
Abstract base class for all vector database connectors.

Vector stores are databases optimized for storing and querying high-dimensional
vectors (embeddings). They are a critical component in RAG (Retrieval-Augmented
Generation) pipelines, where they enable fast similarity search to find documents
relevant to a user's query.

This module defines a unified interface that abstracts away the differences
between vector database implementations (Chroma, Pinecone, PGVector, FAISS),
allowing the rest of the framework to work with any backend interchangeably.

Example:
    >>> from agentic_ai.vectorstores.chroma import ChromaVectorStore
    >>> from agentic_ai.embeddings.openai import OpenAIEmbedding
    >>> embedder = OpenAIEmbedding(api_key="sk-...")
    >>> store = ChromaVectorStore(embedding=embedder, collection_name="docs")
    >>> store.add_documents([Document(content="Hello world")])
    >>> results = store.similarity_search("greeting", k=5)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from agentic_ai.core.base_embedding import BaseEmbedding
from agentic_ai.core.models import Document


@dataclass
class VectorStoreConfig:
    """
    Configuration parameters shared across all vector store implementations.

    Attributes:
        collection_name:  The name of the collection/index to use within the
                          vector database. Collections are logical groupings of
                          related vectors (similar to tables in SQL databases).
        persist_directory: Local directory path for file-based vector stores
                           (e.g., Chroma, FAISS) to persist data between runs.
                           None means data is stored in memory only.
        host:             Hostname for network-based vector databases
                          (e.g., Pinecone, PGVector). None for local stores.
        port:             Port number for network-based vector databases.
        api_key:          API key for cloud-hosted vector databases (e.g., Pinecone).
        extra:            Provider-specific configuration parameters (e.g.,
                          Pinecone environment, PGVector connection string).
    """

    collection_name: str = "default"
    persist_directory: str | None = None
    host: str | None = None
    port: int | None = None
    api_key: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class BaseVectorStore(ABC):
    """
    Abstract base class that all vector store connectors must inherit from.

    Subclasses must implement methods for adding, searching, and deleting
    documents. The embedding provider is injected via the constructor,
    maintaining separation of concerns between vector storage and embedding.

    Attributes:
        embedding: The embedding provider used to vectorize text documents
                   and queries before storage/retrieval.
        config:    A VectorStoreConfig instance containing backend configuration.
    """

    def __init__(self, embedding: BaseEmbedding, config: VectorStoreConfig) -> None:
        """
        Initialize the vector store with an embedding provider and configuration.

        Args:
            embedding: A BaseEmbedding instance for converting text to vectors.
            config:    A VectorStoreConfig with backend-specific settings.
        """
        # Store the embedding provider for vectorizing documents and queries.
        self.embedding = embedding
        # Store the backend configuration.
        self.config = config

    @abstractmethod
    def add_documents(self, documents: list[Document]) -> list[str]:
        """
        Add documents to the vector store.

        Each document's content is embedded using the configured embedding
        provider, and the resulting vector is stored alongside the document's
        metadata. If a document has a pre-computed embedding, it is used directly.

        Args:
            documents: A list of Document objects to add to the store.

        Returns:
            A list of string IDs assigned to each document by the store.

        Raises:
            ConnectionError: If the vector database is unreachable.
        """
        ...

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Search for documents similar to the given query text.

        The query is embedded using the configured embedding provider, and
        the resulting vector is compared against stored vectors using the
        database's similarity metric (typically cosine similarity or L2 distance).

        Args:
            query:   The text query to search for.
            k:       The maximum number of results to return. Default is 5.
            filters: Optional metadata filters to narrow the search space.
                     Filter syntax varies by backend (e.g., {"source": "wiki"}).

        Returns:
            A list of Document objects ranked by similarity (most similar first),
            with up to `k` results.

        Raises:
            ConnectionError: If the vector database is unreachable.
        """
        ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """
        Delete documents from the vector store by their IDs.

        Args:
            ids: A list of document IDs to remove from the store.

        Raises:
            ConnectionError: If the vector database is unreachable.
            KeyError: If any of the IDs do not exist in the store.
        """
        ...

    def similarity_search_with_scores(
        self,
        query: str,
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Search for similar documents and return them with similarity scores.

        This default implementation calls `similarity_search()` and assigns
        a score of 0.0 to each result. Backends that support native scoring
        should override this method to return actual similarity scores.

        Args:
            query:   The text query to search for.
            k:       The maximum number of results to return.
            filters: Optional metadata filters to narrow the search.

        Returns:
            A list of (Document, score) tuples. Higher scores indicate
            greater similarity. The scale depends on the backend.
        """
        # Fallback: perform search without scores and assign 0.0 to each result.
        documents = self.similarity_search(query, k=k, filters=filters)
        return [(doc, 0.0) for doc in documents]
