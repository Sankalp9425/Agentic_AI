"""
ChromaDB vector store connector implementation.

Chroma is an open-source embedding database designed for AI applications.
It supports both in-memory and persistent storage modes, making it excellent
for local development, prototyping, and small-to-medium production workloads.

Key features:
    - Simple API with automatic embedding generation.
    - Persistent storage via SQLite + DuckDB backend.
    - Metadata filtering for refined searches.
    - Built-in support for multiple distance metrics.

Requirements:
    pip install chromadb

Example:
    >>> from agentic_ai.vectorstores.chroma_store import ChromaVectorStore
    >>> store = ChromaVectorStore(embedding=embedder, collection_name="docs")
    >>> store.add_documents([Document(content="Python is great.")])
    >>> results = store.similarity_search("programming languages", k=3)
"""

import logging
import uuid
from typing import Any

from agentic_ai.core.base_embedding import BaseEmbedding
from agentic_ai.core.base_vectorstore import BaseVectorStore, VectorStoreConfig
from agentic_ai.core.models import Document

# Configure module-level logger.
logger = logging.getLogger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB vector store connector.

    Provides document storage and similarity search using ChromaDB. Supports
    both ephemeral (in-memory) and persistent (file-based) storage modes.
    Documents are automatically embedded using the configured embedding
    provider before storage.

    Attributes:
        _client: The ChromaDB client instance.
        _collection: The ChromaDB collection for storing documents.

    Example:
        >>> store = ChromaVectorStore(
        ...     embedding=embedder,
        ...     collection_name="my_docs",
        ...     persist_directory="./chroma_data",
        ... )
    """

    def __init__(
        self,
        embedding: BaseEmbedding,
        collection_name: str = "default",
        persist_directory: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the ChromaDB vector store.

        Creates a ChromaDB client and gets (or creates) the specified
        collection. If persist_directory is provided, data will be saved
        to disk for persistence between sessions.

        Args:
            embedding:         The embedding provider for vectorizing documents.
            collection_name:   Name of the collection to use. Default is "default".
            persist_directory: Directory for persistent storage. None for in-memory.
            **kwargs:          Additional parameters passed to VectorStoreConfig.extra.
        """
        config = VectorStoreConfig(
            collection_name=collection_name,
            persist_directory=persist_directory,
            extra=kwargs,
        )
        super().__init__(embedding=embedding, config=config)

        # Import chromadb to create the client.
        try:
            import chromadb
        except ImportError as e:
            raise ImportError(
                "The 'chromadb' package is required for ChromaVectorStore. "
                "Install it with: pip install chromadb"
            ) from e

        # Create the appropriate client based on storage mode.
        if persist_directory:
            # Persistent client stores data to the specified directory.
            self._client = chromadb.PersistentClient(path=persist_directory)
            logger.info(
                "Initialized persistent ChromaDB at: %s", persist_directory
            )
        else:
            # Ephemeral client stores data in memory only.
            self._client = chromadb.EphemeralClient()
            logger.info("Initialized ephemeral (in-memory) ChromaDB")

        # Get or create the collection for storing documents.
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity.
        )

        logger.info("Using ChromaDB collection: %s", collection_name)

    def add_documents(self, documents: list[Document]) -> list[str]:
        """
        Add documents to the ChromaDB collection.

        Each document is embedded and stored with its metadata. If a document
        has a pre-computed embedding, it is used directly. Otherwise, the
        configured embedding provider generates the embedding.

        Args:
            documents: A list of Document objects to add.

        Returns:
            A list of string IDs assigned to each document.
        """
        # Generate IDs for documents that don't have one.
        ids = [doc.id or str(uuid.uuid4()) for doc in documents]

        # Extract text content for embedding.
        texts = [doc.content for doc in documents]

        # Get embeddings - use pre-computed ones or generate new ones.
        embeddings: list[list[float]] = []
        texts_to_embed: list[str] = []
        embed_indices: list[int] = []

        for i, doc in enumerate(documents):
            if doc.embedding:
                embeddings.append(doc.embedding)
            else:
                texts_to_embed.append(doc.content)
                embed_indices.append(i)
                embeddings.append([])  # Placeholder to maintain order.

        # Batch-embed texts that don't have pre-computed embeddings.
        if texts_to_embed:
            new_embeddings = self.embedding.embed_documents(texts_to_embed)
            for idx, emb in zip(embed_indices, new_embeddings, strict=True):
                embeddings[idx] = emb

        # Extract metadata dictionaries, ensuring they are JSON-serializable.
        metadatas = [doc.metadata or {} for doc in documents]

        # Add documents to the ChromaDB collection.
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        logger.info("Added %d documents to ChromaDB collection", len(documents))
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Search for similar documents in the ChromaDB collection.

        Embeds the query and performs approximate nearest neighbor search
        using ChromaDB's HNSW index with cosine similarity.

        Args:
            query:   The text query to search for.
            k:       Maximum number of results to return. Default is 5.
            filters: Optional metadata filters in ChromaDB's where clause format
                     (e.g., {"source": "wiki"}).

        Returns:
            A list of Document objects ranked by similarity.
        """
        # Embed the query text.
        query_embedding = self.embedding.embed_query(query)

        # Build the query parameters.
        query_params: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": k,
        }

        # Add metadata filter if provided.
        if filters:
            query_params["where"] = filters

        # Execute the similarity search.
        results = self._collection.query(**query_params)

        # Convert ChromaDB results to Document objects.
        documents: list[Document] = []
        if results["documents"] and results["documents"][0]:
            for i, content in enumerate(results["documents"][0]):
                metadata = (
                    results["metadatas"][0][i]
                    if results["metadatas"] and results["metadatas"][0]
                    else {}
                )
                doc_id = (
                    results["ids"][0][i]
                    if results["ids"] and results["ids"][0]
                    else None
                )
                documents.append(
                    Document(content=content, metadata=metadata, id=doc_id)
                )

        logger.debug("Found %d similar documents in ChromaDB", len(documents))
        return documents

    def similarity_search_with_scores(
        self,
        query: str,
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Search with similarity scores from ChromaDB.

        Returns documents along with their distance scores. Lower distances
        indicate higher similarity when using cosine distance.

        Args:
            query:   The text query to search for.
            k:       Maximum number of results. Default is 5.
            filters: Optional metadata filters.

        Returns:
            A list of (Document, score) tuples sorted by similarity.
        """
        query_embedding = self.embedding.embed_query(query)

        query_params: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }

        if filters:
            query_params["where"] = filters

        results = self._collection.query(**query_params)

        scored_documents: list[tuple[Document, float]] = []
        if results["documents"] and results["documents"][0]:
            for i, content in enumerate(results["documents"][0]):
                metadata = (
                    results["metadatas"][0][i]
                    if results["metadatas"] and results["metadatas"][0]
                    else {}
                )
                doc_id = (
                    results["ids"][0][i]
                    if results["ids"] and results["ids"][0]
                    else None
                )
                distance = (
                    results["distances"][0][i]
                    if results["distances"] and results["distances"][0]
                    else 0.0
                )
                doc = Document(content=content, metadata=metadata, id=doc_id)
                # Convert cosine distance to similarity score (1 - distance).
                similarity_score = 1.0 - distance
                scored_documents.append((doc, similarity_score))

        return scored_documents

    def delete(self, ids: list[str]) -> None:
        """
        Delete documents from the ChromaDB collection by their IDs.

        Args:
            ids: A list of document IDs to remove.
        """
        self._collection.delete(ids=ids)
        logger.info("Deleted %d documents from ChromaDB collection", len(ids))
