"""
Advanced retrieval strategies for the RAG pipeline.

Provides multiple retrieval techniques beyond basic similarity search
to improve the quality and diversity of retrieved documents:

    - **MMRRetriever**: Maximal Marginal Relevance — balances relevance
      to the query with diversity among results to reduce redundancy.

    - **HybridRetriever**: Combines keyword-based (BM25/TF-IDF) search
      with semantic (embedding) search for better coverage.

    - **QueryExpansionRetriever**: Generates multiple reformulations of
      the query using an LLM, retrieves for each, and merges results.

    - **ReRankingRetriever**: Retrieves an initial broad set and then
      re-ranks using an LLM or cross-encoder for precision.

All retrievers implement the BaseRetriever interface and work with
any BaseVectorStore implementation.

Example:
    >>> from agentic_ai.rag.retrieval import MMRRetriever
    >>> retriever = MMRRetriever(vector_store=store, lambda_mult=0.7)
    >>> results = retriever.retrieve("What is RAG?", k=5)
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from agentic_ai.core.base_embedding import BaseEmbedding
from agentic_ai.core.base_llm import BaseLLM
from agentic_ai.core.base_vectorstore import BaseVectorStore
from agentic_ai.core.models import Document, Message, Role

# Configure module-level logger.
logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """
    Abstract base class for document retrievers.

    All retrieval strategies implement the ``retrieve`` method that takes
    a query string and returns a ranked list of Document objects.
    """

    @abstractmethod
    def retrieve(
        self, query: str, k: int = 5, **kwargs: Any
    ) -> list[Document]:
        """
        Retrieve relevant documents for a query.

        Args:
            query:   The search query string.
            k:       Maximum number of documents to return.
            **kwargs: Strategy-specific parameters.

        Returns:
            A ranked list of Document objects.
        """
        ...


class SimpleRetriever(BaseRetriever):
    """
    Basic similarity search retriever.

    Wraps a vector store's similarity_search method as a retriever
    for use in retrieval pipelines.

    Example:
        >>> retriever = SimpleRetriever(vector_store=store)
        >>> docs = retriever.retrieve("What is RAG?", k=5)
    """

    def __init__(self, vector_store: BaseVectorStore) -> None:
        """
        Initialize the simple retriever.

        Args:
            vector_store: The vector store to search.
        """
        self.vector_store = vector_store

    def retrieve(
        self, query: str, k: int = 5, **kwargs: Any
    ) -> list[Document]:
        """
        Retrieve documents using basic similarity search.

        Args:
            query:   The search query.
            k:       Number of results. Default is 5.
            **kwargs: Passed to vector_store.similarity_search.

        Returns:
            A list of similar documents.
        """
        filters = kwargs.get("filters")
        return self.vector_store.similarity_search(query, k=k, filters=filters)


class MMRRetriever(BaseRetriever):
    """
    Maximal Marginal Relevance (MMR) retriever.

    Balances relevance to the query with diversity among results.
    Iteratively selects documents that are both similar to the query
    and dissimilar to already-selected documents.

    The MMR formula:
        MMR = argmax_d [lambda * sim(d, q) - (1 - lambda) * max_s sim(d, s)]

    Where:
        - d is a candidate document
        - q is the query
        - s is a document already selected
        - lambda controls the relevance-diversity trade-off

    Attributes:
        vector_store: The vector store for initial retrieval.
        embedding:    The embedding provider for computing similarity.
        lambda_mult:  Trade-off parameter (0=max diversity, 1=max relevance).
        fetch_k:      Number of initial candidates to retrieve before MMR.

    Example:
        >>> retriever = MMRRetriever(
        ...     vector_store=store,
        ...     embedding=embedder,
        ...     lambda_mult=0.7,
        ... )
        >>> docs = retriever.retrieve("machine learning", k=5)
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding: BaseEmbedding,
        lambda_mult: float = 0.7,
        fetch_k: int = 20,
    ) -> None:
        """
        Initialize the MMR retriever.

        Args:
            vector_store: The vector store for candidate retrieval.
            embedding:    Embedding provider for similarity computation.
            lambda_mult:  Relevance-diversity balance (0.0 to 1.0). Default is 0.7.
            fetch_k:      Number of candidates to fetch before MMR selection.
                          Should be larger than k. Default is 20.
        """
        self.vector_store = vector_store
        self.embedding = embedding
        self.lambda_mult = lambda_mult
        self.fetch_k = fetch_k

    def retrieve(
        self, query: str, k: int = 5, **kwargs: Any
    ) -> list[Document]:
        """
        Retrieve documents using Maximal Marginal Relevance.

        1. Fetch ``fetch_k`` candidates via similarity search.
        2. Embed the query and all candidates.
        3. Iteratively select documents using the MMR criterion.

        Args:
            query: The search query.
            k:     Number of final results. Default is 5.

        Returns:
            A diverse, relevant list of documents.
        """
        # Step 1: Fetch initial candidates with scores.
        candidates_with_scores = self.vector_store.similarity_search_with_scores(
            query, k=self.fetch_k
        )

        if not candidates_with_scores:
            return []

        # Step 2: Embed the query.
        query_embedding = self.embedding.embed_query(query)

        # Step 3: Embed all candidates.
        candidates = [doc for doc, _ in candidates_with_scores]
        candidate_embeddings = self.embedding.embed_documents(
            [doc.content for doc in candidates]
        )

        # Step 4: Apply MMR selection.
        selected_indices: list[int] = []
        remaining_indices = list(range(len(candidates)))

        for _ in range(min(k, len(candidates))):
            best_idx = -1
            best_score = float("-inf")

            for idx in remaining_indices:
                # Compute relevance to query.
                relevance = self._cosine_similarity(
                    query_embedding, candidate_embeddings[idx]
                )

                # Compute maximum similarity to already-selected documents.
                max_sim_to_selected = 0.0
                for sel_idx in selected_indices:
                    sim = self._cosine_similarity(
                        candidate_embeddings[idx],
                        candidate_embeddings[sel_idx],
                    )
                    max_sim_to_selected = max(max_sim_to_selected, sim)

                # MMR score: balance relevance and diversity.
                mmr_score = (
                    self.lambda_mult * relevance
                    - (1 - self.lambda_mult) * max_sim_to_selected
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx >= 0:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)

        result = [candidates[i] for i in selected_indices]
        logger.debug("MMR retriever selected %d documents", len(result))
        return result

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec_a: First vector.
            vec_b: Second vector.

        Returns:
            Cosine similarity score (-1.0 to 1.0).
        """
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5

        if norm_a > 0 and norm_b > 0:
            return dot_product / (norm_a * norm_b)
        return 0.0


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining keyword and semantic search.

    Performs both keyword-based search (using simple TF-IDF scoring)
    and embedding-based semantic search, then fuses the results using
    Reciprocal Rank Fusion (RRF).

    RRF formula:
        score(d) = sum(1 / (k + rank_i(d))) for each ranker i

    This approach captures both exact keyword matches and semantic
    similarity, providing better coverage than either method alone.

    Attributes:
        vector_store: The vector store for semantic search.
        documents:    The document corpus for keyword search.
        alpha:        Weight for semantic vs keyword (0=keyword only, 1=semantic only).
        rrf_k:        RRF constant (default 60, per the original RRF paper).

    Example:
        >>> retriever = HybridRetriever(
        ...     vector_store=store,
        ...     documents=all_docs,
        ...     alpha=0.7,
        ... )
        >>> docs = retriever.retrieve("machine learning tutorial", k=5)
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        documents: list[Document],
        alpha: float = 0.7,
        rrf_k: int = 60,
    ) -> None:
        """
        Initialize the hybrid retriever.

        Args:
            vector_store: Vector store for semantic search.
            documents:    Full document corpus for keyword search.
            alpha:        Semantic weight (0.0 to 1.0). Default is 0.7.
            rrf_k:        RRF smoothing constant. Default is 60.
        """
        self.vector_store = vector_store
        self.documents = documents
        self.alpha = alpha
        self.rrf_k = rrf_k

    def retrieve(
        self, query: str, k: int = 5, **kwargs: Any
    ) -> list[Document]:
        """
        Retrieve documents using hybrid keyword + semantic search.

        1. Perform semantic search via the vector store.
        2. Perform keyword search using TF-IDF scoring.
        3. Fuse results using Reciprocal Rank Fusion.

        Args:
            query: The search query.
            k:     Number of results. Default is 5.

        Returns:
            A fused list of documents ranked by combined relevance.
        """
        fetch_k = k * 3  # Fetch more candidates for fusion.

        # Semantic search results.
        semantic_results = self.vector_store.similarity_search(query, k=fetch_k)

        # Keyword search results (simple TF-IDF-like scoring).
        keyword_results = self._keyword_search(query, k=fetch_k)

        # Fuse results using Reciprocal Rank Fusion (RRF).
        fused = self._reciprocal_rank_fusion(
            [semantic_results, keyword_results],
            weights=[self.alpha, 1 - self.alpha],
        )

        result = fused[:k]
        logger.debug("Hybrid retriever returned %d documents", len(result))
        return result

    def _keyword_search(self, query: str, k: int = 10) -> list[Document]:
        """
        Perform simple keyword-based search using term frequency.

        Scores documents by the number of query terms they contain,
        normalized by document length.

        Args:
            query: The search query.
            k:     Number of results.

        Returns:
            Documents ranked by keyword relevance.
        """
        query_terms = set(query.lower().split())

        scored_docs: list[tuple[Document, float]] = []
        for doc in self.documents:
            doc_text_lower = doc.content.lower()
            # Count how many query terms appear in the document.
            term_count = sum(1 for term in query_terms if term in doc_text_lower)
            # Normalize by document length to avoid bias toward long documents.
            doc_length = max(len(doc.content.split()), 1)
            score = term_count / (doc_length ** 0.5)
            if score > 0:
                scored_docs.append((doc, score))

        # Sort by score descending.
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:k]]

    def _reciprocal_rank_fusion(
        self,
        result_lists: list[list[Document]],
        weights: list[float],
    ) -> list[Document]:
        """
        Fuse multiple ranked lists using Reciprocal Rank Fusion.

        Each document gets a score based on its rank in each list:
            score = weight * (1 / (rrf_k + rank))

        Args:
            result_lists: A list of ranked document lists.
            weights:      Weight for each list.

        Returns:
            A single fused list of documents ranked by combined score.
        """
        # Map document content to (Document, cumulative_score).
        doc_scores: dict[str, tuple[Document, float]] = {}

        for results, weight in zip(result_lists, weights, strict=True):
            for rank, doc in enumerate(results):
                rrf_score = weight * (1.0 / (self.rrf_k + rank + 1))
                key = doc.content[:200]  # Use content prefix as key.

                if key in doc_scores:
                    existing_doc, existing_score = doc_scores[key]
                    doc_scores[key] = (existing_doc, existing_score + rrf_score)
                else:
                    doc_scores[key] = (doc, rrf_score)

        # Sort by fused score descending.
        sorted_docs = sorted(
            doc_scores.values(), key=lambda x: x[1], reverse=True
        )
        return [doc for doc, _ in sorted_docs]


class QueryExpansionRetriever(BaseRetriever):
    """
    Retriever that expands the query into multiple reformulations.

    Uses an LLM to generate alternative phrasings of the query,
    retrieves for each variant, and merges results. This helps
    capture different aspects of the user's information need.

    Attributes:
        vector_store: The vector store for retrieval.
        llm:          The LLM for generating query expansions.
        num_expansions: Number of query variants to generate.

    Example:
        >>> retriever = QueryExpansionRetriever(
        ...     vector_store=store,
        ...     llm=llm,
        ...     num_expansions=3,
        ... )
        >>> docs = retriever.retrieve("How does RAG work?", k=5)
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        llm: BaseLLM,
        num_expansions: int = 3,
    ) -> None:
        """
        Initialize the query expansion retriever.

        Args:
            vector_store:   The vector store for retrieval.
            llm:            The LLM for generating query expansions.
            num_expansions: Number of query variants. Default is 3.
        """
        self.vector_store = vector_store
        self.llm = llm
        self.num_expansions = num_expansions

    def retrieve(
        self, query: str, k: int = 5, **kwargs: Any
    ) -> list[Document]:
        """
        Retrieve documents using query expansion.

        1. Generate alternative query formulations using the LLM.
        2. Retrieve documents for the original query and each expansion.
        3. Deduplicate and merge results.

        Args:
            query: The original search query.
            k:     Number of final results. Default is 5.

        Returns:
            A merged list of documents from all query variants.
        """
        # Generate expanded queries.
        expanded_queries = self._expand_query(query)
        all_queries = [query] + expanded_queries

        logger.info(
            "Query expansion: %d variants for '%s'", len(all_queries), query
        )

        # Retrieve for each query variant.
        all_results: list[list[Document]] = []
        for q in all_queries:
            results = self.vector_store.similarity_search(q, k=k)
            all_results.append(results)

        # Fuse results using simple deduplication and round-robin.
        fused = self._merge_results(all_results, k)
        logger.debug(
            "Query expansion retriever returned %d documents", len(fused)
        )
        return fused

    def _expand_query(self, query: str) -> list[str]:
        """
        Generate alternative formulations of a query using the LLM.

        Args:
            query: The original query.

        Returns:
            A list of alternative query strings.
        """
        from agentic_ai.prompts.templates import QUERY_EXPANSION_PROMPT

        prompt = QUERY_EXPANSION_PROMPT.format(
            query=query,
            num_expansions=self.num_expansions,
        )

        messages = [Message(role=Role.USER, content=prompt)]
        response = self.llm.chat(messages)

        # Parse the numbered list from the response.
        expansions: list[str] = []
        for line in response.content.strip().split("\n"):
            # Remove numbering (e.g., "1. ", "2. ").
            cleaned = line.strip().lstrip("0123456789.-) ").strip()
            if cleaned and cleaned != query:
                expansions.append(cleaned)

        return expansions[:self.num_expansions]

    @staticmethod
    def _merge_results(
        result_lists: list[list[Document]], k: int
    ) -> list[Document]:
        """
        Merge multiple result lists with deduplication.

        Uses round-robin selection to ensure diversity across queries.

        Args:
            result_lists: Lists of documents from different queries.
            k:            Maximum number of results.

        Returns:
            A deduplicated, merged list of documents.
        """
        seen_content: set[str] = set()
        merged: list[Document] = []

        # Round-robin through result lists.
        max_len = max(len(r) for r in result_lists) if result_lists else 0

        for i in range(max_len):
            for results in result_lists:
                if i < len(results):
                    doc = results[i]
                    content_key = doc.content[:200]
                    if content_key not in seen_content:
                        seen_content.add(content_key)
                        merged.append(doc)
                        if len(merged) >= k:
                            return merged

        return merged


class ReRankingRetriever(BaseRetriever):
    """
    Retriever that re-ranks initial results using an LLM.

    Performs a broad initial retrieval, then uses an LLM to score
    each candidate's relevance to the query. The LLM acts as a
    cross-encoder-like re-ranker, providing more accurate relevance
    judgments than embedding similarity alone.

    Attributes:
        vector_store: The vector store for initial retrieval.
        llm:          The LLM for re-ranking.
        initial_k:    Number of candidates for initial retrieval.

    Example:
        >>> retriever = ReRankingRetriever(
        ...     vector_store=store,
        ...     llm=llm,
        ...     initial_k=20,
        ... )
        >>> docs = retriever.retrieve("quantum computing basics", k=5)
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        llm: BaseLLM,
        initial_k: int = 20,
    ) -> None:
        """
        Initialize the re-ranking retriever.

        Args:
            vector_store: The vector store for initial retrieval.
            llm:          The LLM for scoring document relevance.
            initial_k:    Number of initial candidates. Default is 20.
        """
        self.vector_store = vector_store
        self.llm = llm
        self.initial_k = initial_k

    def retrieve(
        self, query: str, k: int = 5, **kwargs: Any
    ) -> list[Document]:
        """
        Retrieve documents with LLM-based re-ranking.

        1. Retrieve ``initial_k`` candidates via similarity search.
        2. Score each candidate's relevance using the LLM.
        3. Return the top-k documents by LLM relevance score.

        Args:
            query: The search query.
            k:     Number of final results. Default is 5.

        Returns:
            Documents re-ranked by LLM-assessed relevance.
        """
        # Step 1: Broad initial retrieval.
        candidates = self.vector_store.similarity_search(
            query, k=self.initial_k
        )

        if not candidates:
            return []

        # Step 2: Score each candidate with the LLM.
        scored_candidates: list[tuple[Document, float]] = []
        for doc in candidates:
            score = self._score_relevance(query, doc)
            scored_candidates.append((doc, score))

        # Step 3: Sort by score descending and return top-k.
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        result = [doc for doc, _ in scored_candidates[:k]]

        logger.debug("Re-ranking retriever returned %d documents", len(result))
        return result

    def _score_relevance(self, query: str, document: Document) -> float:
        """
        Score a document's relevance to a query using the LLM.

        Asks the LLM to rate the document's relevance on a 0-10 scale.
        Parses the numeric score from the response.

        Args:
            query:    The search query.
            document: The candidate document to score.

        Returns:
            A relevance score (0.0 to 10.0).
        """
        from agentic_ai.prompts.templates import RERANKING_PROMPT

        prompt = RERANKING_PROMPT.format(
            query=query,
            document=document.content[:1000],  # Truncate long documents.
        )

        messages = [Message(role=Role.USER, content=prompt)]

        try:
            response = self.llm.chat(messages)
            # Extract the numeric score from the response.
            score_text = response.content.strip()
            # Try to find a number in the response.
            import re
            numbers = re.findall(r'\d+(?:\.\d+)?', score_text)
            if numbers:
                score = float(numbers[0])
                return min(max(score, 0.0), 10.0)

        except Exception as e:
            logger.warning("Failed to score document: %s", e)

        return 5.0  # Default score on failure.
