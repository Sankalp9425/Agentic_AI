"""
Pinecone vector store connector implementation.

Pinecone is a fully managed, cloud-native vector database designed for
production AI applications at scale. It handles infrastructure management,
scaling, and performance optimization automatically, making it suitable
for large-scale deployments with millions or billions of vectors.

Key features:
    - Fully managed infrastructure with auto-scaling.
    - Namespace support for multi-tenant applications.
    - Metadata filtering with rich query expressions.
    - Real-time index updates with low-latency queries.

Requirements:
    pip install pinecone-client

Example:
    >>> from agentic_ai.vectorstores.pinecone_store import PineconeVectorStore
    >>> store = PineconeVectorStore(
    ...     embedding=embedder,
    ...     api_key="pc-...",
    ...     index_name="my-index",
    ... )
    >>> store.add_documents([Document(content="Hello world")])
"""

import logging
import uuid
from typing import Any

from agentic_ai.core.base_embedding import BaseEmbedding
from agentic_ai.core.base_vectorstore import BaseVectorStore, VectorStoreConfig
from agentic_ai.core.models import Document

# Configure module-level logger.
logger = logging.getLogger(__name__)


class PineconeVectorStore(BaseVectorStore):
    """
    Pinecone managed vector database connector.

    Provides document storage and similarity search using Pinecone's
    cloud-hosted vector database. Supports namespaces for logical
    separation of data within a single index.

    Attributes:
        _index: The Pinecone index instance for vector operations.
        _namespace: The namespace within the index for data isolation.

    Example:
        >>> store = PineconeVectorStore(
        ...     embedding=embedder,
        ...     api_key="pc-...",
        ...     index_name="my-index",
        ...     namespace="production",
        ... )
    """

    def __init__(
        self,
        embedding: BaseEmbedding,
        api_key: str,
        index_name: str,
        namespace: str = "",
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Pinecone vector store.

        Connects to the specified Pinecone index using the provided API key.
        The index must already exist in your Pinecone project.

        Args:
            embedding:  The embedding provider for vectorizing documents.
            api_key:    Pinecone API key for authentication.
            index_name: Name of the Pinecone index to use.
            namespace:  Namespace within the index for data isolation.
                        Default is "" (default namespace).
            **kwargs:   Additional parameters (e.g., environment, region).
        """
        config = VectorStoreConfig(
            collection_name=index_name,
            api_key=api_key,
            extra=kwargs,
        )
        super().__init__(embedding=embedding, config=config)

        # Import pinecone to create the client.
        try:
            from pinecone import Pinecone
        except ImportError as e:
            raise ImportError(
                "The 'pinecone-client' package is required for PineconeVectorStore. "
                "Install it with: pip install pinecone-client"
            ) from e

        # Create the Pinecone client and connect to the index.
        pc = Pinecone(api_key=api_key)
        self._index = pc.Index(index_name)

        # Store the namespace for data isolation.
        self._namespace = namespace

        logger.info(
            "Initialized Pinecone vector store - index: %s, namespace: %s",
            index_name,
            namespace or "(default)",
        )

    def add_documents(self, documents: list[Document]) -> list[str]:
        """
        Add documents to the Pinecone index.

        Each document is embedded and upserted (insert or update) into the
        index. Pinecone supports batch upserts for efficient ingestion.

        Args:
            documents: A list of Document objects to add.

        Returns:
            A list of string IDs assigned to each document.
        """
        # Generate IDs for documents that don't have one.
        ids = [doc.id or str(uuid.uuid4()) for doc in documents]

        # Generate embeddings for all documents.
        embeddings = []
        for doc in documents:
            if doc.embedding:
                embeddings.append(doc.embedding)
            else:
                embeddings.append(self.embedding.embed_query(doc.content))

        # Build the upsert vectors in Pinecone's format.
        vectors = []
        for doc_id, emb, doc in zip(ids, embeddings, documents, strict=True):
            # Pinecone metadata must be flat key-value pairs.
            metadata = dict(doc.metadata) if doc.metadata else {}
            # Store the original text content in metadata for retrieval.
            metadata["_content"] = doc.content

            vectors.append({
                "id": doc_id,
                "values": emb,
                "metadata": metadata,
            })

        # Upsert vectors in batches of 100 (Pinecone's recommended batch size).
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self._index.upsert(
                vectors=batch,
                namespace=self._namespace,
            )

        logger.info("Upserted %d documents to Pinecone index", len(documents))
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Search for similar documents in the Pinecone index.

        Embeds the query and performs approximate nearest neighbor search
        using Pinecone's optimized index.

        Args:
            query:   The text query to search for.
            k:       Maximum number of results. Default is 5.
            filters: Optional metadata filters in Pinecone's filter format
                     (e.g., {"genre": {"$eq": "comedy"}}).

        Returns:
            A list of Document objects ranked by similarity.
        """
        # Embed the query.
        query_embedding = self.embedding.embed_query(query)

        # Build query parameters.
        query_params: dict[str, Any] = {
            "vector": query_embedding,
            "top_k": k,
            "include_metadata": True,
            "namespace": self._namespace,
        }

        if filters:
            query_params["filter"] = filters

        # Execute the query.
        results = self._index.query(**query_params)

        # Convert Pinecone results to Document objects.
        documents: list[Document] = []
        for match in results.get("matches", []):
            metadata = dict(match.get("metadata", {}))
            # Extract the original content from metadata.
            content = metadata.pop("_content", "")
            documents.append(
                Document(content=content, metadata=metadata, id=match["id"])
            )

        logger.debug("Found %d similar documents in Pinecone", len(documents))
        return documents

    def similarity_search_with_scores(
        self,
        query: str,
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Search with similarity scores from Pinecone.

        Returns documents along with their similarity scores. Pinecone
        returns scores in [0, 1] range where 1 is most similar (for
        cosine similarity).

        Args:
            query:   The text query to search for.
            k:       Maximum number of results. Default is 5.
            filters: Optional metadata filters.

        Returns:
            A list of (Document, score) tuples sorted by similarity.
        """
        query_embedding = self.embedding.embed_query(query)

        query_params: dict[str, Any] = {
            "vector": query_embedding,
            "top_k": k,
            "include_metadata": True,
            "namespace": self._namespace,
        }

        if filters:
            query_params["filter"] = filters

        results = self._index.query(**query_params)

        scored_documents: list[tuple[Document, float]] = []
        for match in results.get("matches", []):
            metadata = dict(match.get("metadata", {}))
            content = metadata.pop("_content", "")
            doc = Document(content=content, metadata=metadata, id=match["id"])
            score = match.get("score", 0.0)
            scored_documents.append((doc, score))

        return scored_documents

    def delete(self, ids: list[str]) -> None:
        """
        Delete vectors from the Pinecone index by their IDs.

        Args:
            ids: A list of vector IDs to remove.
        """
        self._index.delete(ids=ids, namespace=self._namespace)
        logger.info("Deleted %d vectors from Pinecone index", len(ids))
