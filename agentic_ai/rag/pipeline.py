"""
Complete RAG pipeline orchestrator.

Provides a high-level, configurable RAG pipeline that connects all stages:
ingestion → chunking → indexing → retrieval → generation → (evaluation).

The pipeline is fully configurable via a PipelineConfig dataclass and
supports swapping any component (chunker, retriever, LLM, vector store)
without changing the pipeline code.

Example:
    >>> from agentic_ai.rag.pipeline import RAGPipeline, PipelineConfig
    >>> config = PipelineConfig(
    ...     chunker_type="semantic",
    ...     retriever_type="mmr",
    ...     top_k=5,
    ... )
    >>> pipeline = RAGPipeline(
    ...     llm=my_llm,
    ...     embedding=my_embedder,
    ...     vector_store=my_store,
    ...     config=config,
    ... )
    >>> pipeline.ingest(["docs/manual.pdf", "docs/faq.txt"])
    >>> answer = pipeline.query("How do I reset my password?")
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from agentic_ai.core.base_embedding import BaseEmbedding
from agentic_ai.core.base_llm import BaseLLM
from agentic_ai.core.base_vectorstore import BaseVectorStore
from agentic_ai.core.models import Document, Message, Role
from agentic_ai.rag.chunking import (
    BaseChunker,
    FixedSizeChunker,
    RecursiveChunker,
    SemanticChunker,
    SentenceChunker,
)
from agentic_ai.rag.ingestion import DocumentIngestionPipeline
from agentic_ai.rag.retrieval import (
    BaseRetriever,
    HybridRetriever,
    MMRRetriever,
    QueryExpansionRetriever,
    ReRankingRetriever,
    SimpleRetriever,
)

# Configure module-level logger.
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """
    Configuration for the RAG pipeline.

    Controls the behavior of each pipeline stage including chunking
    strategy, retrieval method, and generation parameters.

    Attributes:
        chunker_type:     Chunking strategy ("fixed", "recursive", "sentence", "semantic").
        chunk_size:       Maximum characters per chunk (for fixed/recursive).
        chunk_overlap:    Overlap between chunks (for fixed/recursive).
        sentences_per_chunk: Sentences per chunk (for sentence chunker).
        semantic_threshold: Similarity threshold (for semantic chunker).
        retriever_type:   Retrieval strategy ("simple", "mmr", "hybrid", "expansion", "rerank").
        top_k:            Number of documents to retrieve.
        mmr_lambda:       Lambda parameter for MMR retriever.
        hybrid_alpha:     Alpha parameter for hybrid retriever.
        num_expansions:   Number of query expansions.
        rerank_initial_k: Initial retrieval count for re-ranking.
        extract_images:   Whether to extract images from PDFs.
        extract_tables:   Whether to extract tables from PDFs.
        system_prompt:    Custom system prompt template for generation.
        extra:            Additional configuration parameters.
    """

    chunker_type: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    sentences_per_chunk: int = 5
    semantic_threshold: float = 0.5
    retriever_type: str = "simple"
    top_k: int = 5
    mmr_lambda: float = 0.7
    hybrid_alpha: float = 0.7
    num_expansions: int = 3
    rerank_initial_k: int = 20
    extract_images: bool = True
    extract_tables: bool = True
    system_prompt: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class RAGPipeline:
    """
    Complete RAG pipeline orchestrating all stages.

    Connects ingestion, chunking, indexing, retrieval, and generation
    into a single, easy-to-use interface. Fully configurable via
    PipelineConfig.

    Attributes:
        llm:           The LLM for generation and (optionally) image captioning.
        embedding:     The embedding provider for vectorization.
        vector_store:  The vector store for document storage and retrieval.
        config:        Pipeline configuration.
        chunker:       The document chunker instance.
        retriever:     The document retriever instance.
        ingestion:     The document ingestion pipeline.
        all_documents: All ingested and chunked documents (for hybrid search).

    Example:
        >>> pipeline = RAGPipeline(llm=llm, embedding=emb, vector_store=store)
        >>> pipeline.ingest(["paper.pdf"])
        >>> answer = pipeline.query("What is the main finding?")
        >>> print(answer)
    """

    def __init__(
        self,
        llm: BaseLLM,
        embedding: BaseEmbedding,
        vector_store: BaseVectorStore,
        config: PipelineConfig | None = None,
    ) -> None:
        """
        Initialize the RAG pipeline.

        Sets up the chunker, retriever, and ingestion pipeline based
        on the provided configuration.

        Args:
            llm:          The LLM for generation.
            embedding:    The embedding provider.
            vector_store: The vector store backend.
            config:       Optional pipeline configuration. Uses defaults if None.
        """
        self.llm = llm
        self.embedding = embedding
        self.vector_store = vector_store
        self.config = config or PipelineConfig()

        # All chunked documents (kept for hybrid retriever).
        self.all_documents: list[Document] = []

        # Initialize components based on config.
        self.chunker = self._create_chunker()
        self.ingestion = DocumentIngestionPipeline(
            llm=llm,
            extract_images=self.config.extract_images,
            extract_tables=self.config.extract_tables,
        )
        # Retriever is created lazily (needs documents for hybrid).
        self._retriever: BaseRetriever | None = None

        logger.info(
            "RAG pipeline initialized — chunker: %s, retriever: %s, top_k: %d",
            self.config.chunker_type,
            self.config.retriever_type,
            self.config.top_k,
        )

    @property
    def retriever(self) -> BaseRetriever:
        """Get or create the retriever instance."""
        if self._retriever is None:
            self._retriever = self._create_retriever()
        return self._retriever

    def ingest(self, file_paths: list[str], **kwargs: Any) -> list[str]:
        """
        Ingest documents into the pipeline.

        Parses files, chunks them, and indexes the chunks in the vector store.

        Args:
            file_paths: List of file paths to ingest.
            **kwargs:   Additional arguments for the ingestion pipeline.

        Returns:
            A list of document IDs for the indexed chunks.
        """
        # Stage 1: Parse documents from files.
        raw_documents = self.ingestion.ingest(file_paths, **kwargs)
        logger.info("Ingestion: parsed %d raw documents", len(raw_documents))

        # Stage 2: Chunk documents.
        chunks = self.chunker.chunk_many(raw_documents)
        logger.info("Chunking: produced %d chunks", len(chunks))

        # Store all chunks for hybrid retriever.
        self.all_documents.extend(chunks)

        # Reset retriever so it picks up new documents.
        self._retriever = None

        # Stage 3: Index chunks in the vector store.
        ids = self.vector_store.add_documents(chunks)
        logger.info("Indexing: stored %d chunks in vector store", len(ids))

        return ids

    def ingest_documents(self, documents: list[Document]) -> list[str]:
        """
        Ingest pre-parsed Document objects (skip file parsing).

        Useful when you already have Document objects from a custom
        source. Still applies chunking and indexing.

        Args:
            documents: Pre-parsed Document objects.

        Returns:
            A list of document IDs for the indexed chunks.
        """
        # Chunk and index.
        chunks = self.chunker.chunk_many(documents)
        self.all_documents.extend(chunks)
        self._retriever = None
        return self.vector_store.add_documents(chunks)

    def query(
        self,
        question: str,
        return_contexts: bool = False,
        **kwargs: Any,
    ) -> str | tuple[str, list[Document]]:
        """
        Query the RAG pipeline.

        Retrieves relevant documents and generates an answer using the LLM.

        Args:
            question:        The user's question.
            return_contexts: If True, also return the retrieved documents.
            **kwargs:        Additional retriever parameters.

        Returns:
            The generated answer string. If return_contexts is True,
            returns a tuple of (answer, retrieved_documents).
        """
        # Stage 4: Retrieve relevant documents.
        retrieved_docs = self.retriever.retrieve(
            question, k=self.config.top_k, **kwargs
        )

        if not retrieved_docs:
            answer = "I couldn't find any relevant information to answer your question."
            if return_contexts:
                return answer, []
            return answer

        # Build the context string from retrieved documents.
        context = "\n\n---\n\n".join(doc.content for doc in retrieved_docs)

        # Stage 5: Generate the answer using the LLM.
        from agentic_ai.prompts.templates import RAG_SYSTEM_PROMPT

        system_prompt = self.config.system_prompt or RAG_SYSTEM_PROMPT
        prompt = system_prompt.format(context=context, question=question)

        messages = [Message(role=Role.USER, content=prompt)]
        response = self.llm.chat(messages)
        answer = response.content

        logger.info(
            "Generated answer using %d context documents", len(retrieved_docs)
        )

        if return_contexts:
            return answer, retrieved_docs
        return answer

    def _create_chunker(self) -> BaseChunker:
        """
        Create a chunker instance based on the pipeline config.

        Returns:
            A BaseChunker implementation.
        """
        chunker_type = self.config.chunker_type.lower()

        if chunker_type == "fixed":
            return FixedSizeChunker(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
        elif chunker_type == "recursive":
            return RecursiveChunker(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
        elif chunker_type == "sentence":
            return SentenceChunker(
                sentences_per_chunk=self.config.sentences_per_chunk,
            )
        elif chunker_type == "semantic":
            return SemanticChunker(
                embedding=self.embedding,
                threshold=self.config.semantic_threshold,
            )
        else:
            logger.warning(
                "Unknown chunker type '%s', defaulting to recursive",
                chunker_type,
            )
            return RecursiveChunker(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )

    def _create_retriever(self) -> BaseRetriever:
        """
        Create a retriever instance based on the pipeline config.

        Returns:
            A BaseRetriever implementation.
        """
        retriever_type = self.config.retriever_type.lower()

        if retriever_type == "simple":
            return SimpleRetriever(vector_store=self.vector_store)

        elif retriever_type == "mmr":
            return MMRRetriever(
                vector_store=self.vector_store,
                embedding=self.embedding,
                lambda_mult=self.config.mmr_lambda,
            )

        elif retriever_type == "hybrid":
            return HybridRetriever(
                vector_store=self.vector_store,
                documents=self.all_documents,
                alpha=self.config.hybrid_alpha,
            )

        elif retriever_type == "expansion":
            return QueryExpansionRetriever(
                vector_store=self.vector_store,
                llm=self.llm,
                num_expansions=self.config.num_expansions,
            )

        elif retriever_type == "rerank":
            return ReRankingRetriever(
                vector_store=self.vector_store,
                llm=self.llm,
                initial_k=self.config.rerank_initial_k,
            )

        else:
            logger.warning(
                "Unknown retriever type '%s', defaulting to simple",
                retriever_type,
            )
            return SimpleRetriever(vector_store=self.vector_store)
