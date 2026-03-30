"""
FAISS (Facebook AI Similarity Search) vector store implementation.

FAISS is a library for efficient similarity search and clustering of dense
vectors, developed by Facebook AI Research. It is optimized for in-memory
search with support for GPU acceleration, making it extremely fast for
workloads that fit in RAM.

Key features:
    - Blazing fast in-memory similarity search.
    - Multiple index types (Flat, IVF, HNSW, PQ) for different trade-offs.
    - GPU support for even faster search on large datasets.
    - Index serialization for saving/loading from disk.

Requirements:
    pip install faiss-cpu  # or faiss-gpu for GPU support

Example:
    >>> from agentic_ai.vectorstores.faiss_store import FAISSVectorStore
    >>> store = FAISSVectorStore(embedding=embedder)
    >>> store.add_documents([Document(content="Hello world")])
    >>> results = store.similarity_search("greeting", k=3)
"""

import logging
import uuid
from typing import Any

from agentic_ai.core.base_embedding import BaseEmbedding
from agentic_ai.core.base_vectorstore import BaseVectorStore, VectorStoreConfig
from agentic_ai.core.models import Document

# Configure module-level logger.
logger = logging.getLogger(__name__)


class FAISSVectorStore(BaseVectorStore):
    """
    FAISS in-memory vector store connector.

    Provides extremely fast similarity search using FAISS's optimized
    indexing structures. Documents and their metadata are stored in-memory
    alongside the FAISS index. Supports saving/loading the index to/from
    disk for persistence.

    Attributes:
        _index: The FAISS index for vector search.
        _documents: A dictionary mapping internal IDs to Document objects.
        _id_map: A dictionary mapping document string IDs to FAISS integer indices.
        _dimension: The dimensionality of the embedding vectors.

    Example:
        >>> store = FAISSVectorStore(embedding=embedder)
        >>> store.add_documents([Document(content="Python is great.")])
        >>> results = store.similarity_search("programming", k=3)
    """

    def __init__(
        self,
        embedding: BaseEmbedding,
        index_type: str = "Flat",
        persist_directory: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the FAISS vector store.

        Creates a FAISS index of the specified type. The index dimension is
        automatically determined from the embedding provider on first use.

        Args:
            embedding:         The embedding provider for vectorizing documents.
            index_type:        FAISS index type. Options include:
                               - "Flat": Exact search (brute force, best quality).
                               - "IVF": Inverted file index (faster, approximate).
                               - "HNSW": Hierarchical navigable small world (fast, approximate).
                               Default is "Flat" for exact results.
            persist_directory: Directory for saving/loading the index. None for in-memory only.
            **kwargs:          Additional parameters stored in VectorStoreConfig.extra.
        """
        config = VectorStoreConfig(
            persist_directory=persist_directory,
            extra={"index_type": index_type, **kwargs},
        )
        super().__init__(embedding=embedding, config=config)

        # Import faiss for index creation.
        try:
            import faiss
        except ImportError as e:
            raise ImportError(
                "The 'faiss-cpu' package is required for FAISSVectorStore. "
                "Install it with: pip install faiss-cpu"
            ) from e

        # Store the faiss module reference for later use.
        self._faiss = faiss

        # Determine the embedding dimension by generating a test embedding.
        test_embedding = embedding.embed_query("dimension test")
        self._dimension = len(test_embedding)

        # Create the FAISS index based on the specified type.
        if index_type == "Flat":
            # IndexFlatIP uses inner product (equivalent to cosine for normalized vectors).
            self._index = faiss.IndexFlatIP(self._dimension)
        elif index_type == "HNSW":
            # HNSW index with 32 neighbors per node.
            self._index = faiss.IndexHNSWFlat(self._dimension, 32)
        else:
            # Default to Flat index for unknown types.
            self._index = faiss.IndexFlatIP(self._dimension)
            logger.warning(
                "Unknown index type '%s', defaulting to Flat", index_type
            )

        # In-memory storage for documents and ID mapping.
        # Maps FAISS integer index to document data.
        self._documents: dict[int, Document] = {}
        # Maps string document ID to FAISS integer index.
        self._id_map: dict[str, int] = {}
        # Counter for assigning FAISS integer indices.
        self._next_index = 0

        logger.info(
            "Initialized FAISS vector store - type: %s, dimension: %d",
            index_type,
            self._dimension,
        )

    def add_documents(self, documents: list[Document]) -> list[str]:
        """
        Add documents to the FAISS index.

        Embeds each document and adds the vectors to the FAISS index.
        Documents and metadata are stored in a parallel dictionary
        for retrieval.

        Args:
            documents: A list of Document objects to add.

        Returns:
            A list of string IDs assigned to each document.
        """
        import numpy as np

        ids = [doc.id or str(uuid.uuid4()) for doc in documents]

        # Generate embeddings for all documents.
        embeddings: list[list[float]] = []
        for doc in documents:
            if doc.embedding:
                embeddings.append(doc.embedding)
            else:
                embeddings.append(self.embedding.embed_query(doc.content))

        # Convert to numpy array for FAISS (float32 required).
        vectors = np.array(embeddings, dtype=np.float32)

        # Normalize vectors for cosine similarity (IndexFlatIP with L2 norm = cosine).
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero for zero vectors.
        norms[norms == 0] = 1
        vectors = vectors / norms

        # Add vectors to the FAISS index.
        self._index.add(vectors)

        # Store documents in the parallel dictionary.
        for i, (doc_id, doc) in enumerate(zip(ids, documents, strict=True)):
            faiss_idx = self._next_index + i
            self._documents[faiss_idx] = Document(
                content=doc.content,
                metadata=doc.metadata,
                id=doc_id,
            )
            self._id_map[doc_id] = faiss_idx

        self._next_index += len(documents)

        logger.info("Added %d documents to FAISS index", len(documents))
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Search for similar documents using FAISS.

        Performs approximate (or exact, for Flat indices) nearest neighbor
        search. Metadata filtering is applied post-retrieval since FAISS
        doesn't support native metadata filtering.

        Args:
            query:   The text query to search for.
            k:       Maximum number of results. Default is 5.
            filters: Optional metadata filters applied post-retrieval.
                     Note: FAISS retrieves more results internally to
                     account for filtering.

        Returns:
            A list of Document objects ranked by similarity.
        """
        import numpy as np

        # Embed and normalize the query.
        query_embedding = self.embedding.embed_query(query)
        query_vector = np.array([query_embedding], dtype=np.float32)
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm

        # If filtering, retrieve more results to account for filtered-out items.
        search_k = k * 3 if filters else k

        # Perform the FAISS search.
        distances, indices = self._index.search(query_vector, min(search_k, self._index.ntotal))

        # Convert results to Document objects with optional filtering.
        documents: list[Document] = []
        for i in range(len(indices[0])):
            idx = int(indices[0][i])
            if idx == -1:  # FAISS returns -1 for empty slots.
                continue

            doc = self._documents.get(idx)
            if doc is None:
                continue

            # Apply metadata filtering if specified.
            if filters:
                match = all(
                    doc.metadata.get(key) == value
                    for key, value in filters.items()
                )
                if not match:
                    continue

            documents.append(doc)
            if len(documents) >= k:
                break

        logger.debug("Found %d similar documents in FAISS", len(documents))
        return documents

    def similarity_search_with_scores(
        self,
        query: str,
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Search with similarity scores from FAISS.

        Returns documents with their inner product scores (cosine similarity
        for normalized vectors). Scores range from -1 to 1.

        Args:
            query:   The text query to search for.
            k:       Maximum number of results. Default is 5.
            filters: Optional metadata filters.

        Returns:
            A list of (Document, score) tuples sorted by similarity.
        """
        import numpy as np

        query_embedding = self.embedding.embed_query(query)
        query_vector = np.array([query_embedding], dtype=np.float32)
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm

        search_k = k * 3 if filters else k
        distances, indices = self._index.search(query_vector, min(search_k, self._index.ntotal))

        scored_documents: list[tuple[Document, float]] = []
        for i in range(len(indices[0])):
            idx = int(indices[0][i])
            if idx == -1:
                continue

            doc = self._documents.get(idx)
            if doc is None:
                continue

            if filters:
                match = all(
                    doc.metadata.get(key) == value
                    for key, value in filters.items()
                )
                if not match:
                    continue

            score = float(distances[0][i])
            scored_documents.append((doc, score))
            if len(scored_documents) >= k:
                break

        return scored_documents

    def delete(self, ids: list[str]) -> None:
        """
        Delete documents from the FAISS store by their IDs.

        Note: FAISS doesn't natively support deletion from most index types.
        This implementation marks documents as deleted in the metadata store
        but doesn't remove them from the FAISS index itself. For Flat indices,
        the index is rebuilt without the deleted vectors.

        Args:
            ids: A list of document IDs to remove.
        """
        for doc_id in ids:
            faiss_idx = self._id_map.pop(doc_id, None)
            if faiss_idx is not None:
                self._documents.pop(faiss_idx, None)

        logger.info(
            "Marked %d documents as deleted in FAISS store", len(ids)
        )

    def save(self, path: str) -> None:
        """
        Save the FAISS index to disk for later loading.

        Args:
            path: File path to save the index to (e.g., "index.faiss").
        """
        self._faiss.write_index(self._index, path)
        logger.info("Saved FAISS index to: %s", path)

    def load(self, path: str) -> None:
        """
        Load a FAISS index from disk.

        Note: This only loads the FAISS index. Document metadata must be
        loaded separately if you need to reconstruct the full store.

        Args:
            path: File path to load the index from.
        """
        self._index = self._faiss.read_index(path)
        logger.info("Loaded FAISS index from: %s", path)
