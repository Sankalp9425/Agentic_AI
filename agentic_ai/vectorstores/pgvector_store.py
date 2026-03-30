"""
PGVector (PostgreSQL) vector store connector implementation.

PGVector is a PostgreSQL extension that adds support for vector similarity
search directly within your existing PostgreSQL database. This is ideal for
applications that already use PostgreSQL and want to add vector search
without introducing a separate database system.

Key features:
    - Runs within your existing PostgreSQL infrastructure.
    - Supports exact and approximate nearest neighbor search (IVFFlat, HNSW).
    - Full SQL query capabilities alongside vector search.
    - ACID transactions for data consistency.

Requirements:
    pip install psycopg2-binary pgvector

Example:
    >>> from agentic_ai.vectorstores.pgvector_store import PGVectorStore
    >>> store = PGVectorStore(
    ...     embedding=embedder,
    ...     connection_string="postgresql://user:pass@localhost:5432/mydb",
    ...     collection_name="documents",
    ... )
"""

import json
import logging
import uuid
from typing import Any

from agentic_ai.core.base_embedding import BaseEmbedding
from agentic_ai.core.base_vectorstore import BaseVectorStore, VectorStoreConfig
from agentic_ai.core.models import Document

# Configure module-level logger.
logger = logging.getLogger(__name__)


class PGVectorStore(BaseVectorStore):
    """
    PostgreSQL + pgvector vector store connector.

    Stores document embeddings in a PostgreSQL table with a pgvector column,
    enabling vector similarity search using SQL queries. Automatically
    creates the required table and extension if they don't exist.

    Attributes:
        _connection_string: The PostgreSQL connection URI.
        _conn: The active database connection.
        _table_name: The name of the table storing vectors.

    Example:
        >>> store = PGVectorStore(
        ...     embedding=embedder,
        ...     connection_string="postgresql://localhost/mydb",
        ...     collection_name="docs",
        ... )
    """

    def __init__(
        self,
        embedding: BaseEmbedding,
        connection_string: str,
        collection_name: str = "documents",
        **kwargs: Any,
    ) -> None:
        """
        Initialize the PGVector store and create required database objects.

        Connects to PostgreSQL, enables the pgvector extension, and creates
        the storage table if it doesn't already exist.

        Args:
            embedding:          The embedding provider for vectorizing documents.
            connection_string:  PostgreSQL connection URI
                                (e.g., "postgresql://user:pass@host:5432/db").
            collection_name:    Name of the table to store vectors. Default is "documents".
            **kwargs:           Additional parameters stored in VectorStoreConfig.extra.
        """
        config = VectorStoreConfig(
            collection_name=collection_name,
            extra={"connection_string": connection_string, **kwargs},
        )
        super().__init__(embedding=embedding, config=config)

        # Import required database libraries.
        try:
            import psycopg2
        except ImportError as e:
            raise ImportError(
                "The 'psycopg2-binary' package is required for PGVectorStore. "
                "Install it with: pip install psycopg2-binary"
            ) from e

        # Store the connection string and table name.
        self._connection_string = connection_string
        self._table_name = collection_name

        # Establish the database connection.
        self._conn = psycopg2.connect(connection_string)
        self._conn.autocommit = True

        # Initialize the database schema (create extension and table).
        self._initialize_schema()

        logger.info(
            "Initialized PGVector store with table: %s", collection_name
        )

    def _initialize_schema(self) -> None:
        """
        Create the pgvector extension and storage table if they don't exist.

        This method is called during initialization to ensure the database
        has the required schema. It is idempotent (safe to call multiple times).
        """
        cursor = self._conn.cursor()

        # Enable the pgvector extension in the database.
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Determine the embedding dimension for the vector column.
        # Use a test embedding to detect the dimension automatically.
        test_embedding = self.embedding.embed_query("test")
        dimension = len(test_embedding)

        # Create the table for storing documents and their embeddings.
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._table_name} (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata JSONB DEFAULT '{{}}'::jsonb,
                embedding vector({dimension})
            );
        """)

        # Create an HNSW index for fast approximate nearest neighbor search.
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS {self._table_name}_embedding_idx
            ON {self._table_name}
            USING hnsw (embedding vector_cosine_ops);
        """)

        cursor.close()
        logger.info(
            "PGVector schema initialized with dimension %d", dimension
        )

    def add_documents(self, documents: list[Document]) -> list[str]:
        """
        Add documents to the PostgreSQL table.

        Inserts each document with its embedding and metadata. Uses
        ON CONFLICT to update existing documents with the same ID.

        Args:
            documents: A list of Document objects to add.

        Returns:
            A list of string IDs assigned to each document.
        """
        ids = [doc.id or str(uuid.uuid4()) for doc in documents]
        cursor = self._conn.cursor()

        for doc_id, doc in zip(ids, documents, strict=True):
            # Get or generate the embedding.
            embedding = doc.embedding or self.embedding.embed_query(doc.content)

            # Serialize metadata as JSON.
            metadata_json = json.dumps(doc.metadata or {})

            # Insert or update the document.
            cursor.execute(
                f"""
                INSERT INTO {self._table_name} (id, content, metadata, embedding)
                VALUES (%s, %s, %s::jsonb, %s::vector)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding;
                """,
                (doc_id, doc.content, metadata_json, str(embedding)),
            )

        cursor.close()
        logger.info("Added %d documents to PGVector table", len(documents))
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Search for similar documents using pgvector's cosine distance.

        Uses the <=> operator for cosine distance, which is optimized
        by the HNSW index for fast approximate nearest neighbor search.

        Args:
            query:   The text query to search for.
            k:       Maximum number of results. Default is 5.
            filters: Optional metadata filters as key-value pairs. Translated
                     to JSONB containment queries (e.g., {"source": "wiki"}).

        Returns:
            A list of Document objects ranked by similarity.
        """
        # Embed the query.
        query_embedding = self.embedding.embed_query(query)
        cursor = self._conn.cursor()

        # Build the SQL query with optional metadata filtering.
        sql = f"""
            SELECT id, content, metadata, embedding <=> %s::vector AS distance
            FROM {self._table_name}
        """

        params: list[Any] = [str(query_embedding)]

        # Add metadata filters using JSONB containment (@>).
        if filters:
            filter_json = json.dumps(filters)
            sql += " WHERE metadata @> %s::jsonb"
            params.append(filter_json)

        sql += " ORDER BY distance ASC LIMIT %s;"
        params.append(k)

        cursor.execute(sql, params)
        rows = cursor.fetchall()
        cursor.close()

        # Convert rows to Document objects.
        documents: list[Document] = []
        for row in rows:
            documents.append(
                Document(
                    content=row[1],
                    metadata=row[2] if row[2] else {},
                    id=row[0],
                )
            )

        logger.debug("Found %d similar documents in PGVector", len(documents))
        return documents

    def similarity_search_with_scores(
        self,
        query: str,
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Search with cosine similarity scores from PGVector.

        Args:
            query:   The text query to search for.
            k:       Maximum number of results. Default is 5.
            filters: Optional metadata filters.

        Returns:
            A list of (Document, score) tuples. Score is cosine similarity (0 to 1).
        """
        query_embedding = self.embedding.embed_query(query)
        cursor = self._conn.cursor()

        sql = f"""
            SELECT id, content, metadata, 1 - (embedding <=> %s::vector) AS similarity
            FROM {self._table_name}
        """

        params: list[Any] = [str(query_embedding)]

        if filters:
            filter_json = json.dumps(filters)
            sql += " WHERE metadata @> %s::jsonb"
            params.append(filter_json)

        sql += " ORDER BY similarity DESC LIMIT %s;"
        params.append(k)

        cursor.execute(sql, params)
        rows = cursor.fetchall()
        cursor.close()

        scored_documents: list[tuple[Document, float]] = []
        for row in rows:
            doc = Document(
                content=row[1],
                metadata=row[2] if row[2] else {},
                id=row[0],
            )
            scored_documents.append((doc, float(row[3])))

        return scored_documents

    def delete(self, ids: list[str]) -> None:
        """
        Delete documents from the PGVector table by their IDs.

        Args:
            ids: A list of document IDs to remove.
        """
        cursor = self._conn.cursor()
        # Use ANY for efficient batch deletion.
        cursor.execute(
            f"DELETE FROM {self._table_name} WHERE id = ANY(%s);",
            (ids,),
        )
        cursor.close()
        logger.info("Deleted %d documents from PGVector table", len(ids))
