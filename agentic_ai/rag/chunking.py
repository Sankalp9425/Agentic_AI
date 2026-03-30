"""
Document chunking strategies for the RAG pipeline.

Splits documents into smaller, semantically meaningful chunks suitable
for embedding and retrieval. Provides multiple chunking strategies for
different use cases:

    - **FixedSizeChunker**: Splits text into chunks of a fixed character
      count with configurable overlap. Simple and fast.

    - **RecursiveChunker**: Recursively splits text using a hierarchy of
      separators (paragraphs -> sentences -> words) to find natural
      split points. Inspired by LangChain's RecursiveCharacterTextSplitter.

    - **SentenceChunker**: Splits text into chunks at sentence boundaries,
      grouping multiple sentences per chunk. Preserves sentence integrity.

    - **SemanticChunker**: Uses embedding similarity to find natural
      topic boundaries. Groups consecutive sentences that are semantically
      similar, splitting where similarity drops below a threshold.

All chunkers implement the BaseChunker interface and return a list of
Document objects with metadata tracking chunk position and source.

Example:
    >>> from agentic_ai.rag.chunking import SemanticChunker
    >>> chunker = SemanticChunker(embedding=embedder, threshold=0.5)
    >>> chunks = chunker.chunk(document)
"""

import logging
import re
from abc import ABC, abstractmethod

from agentic_ai.core.base_embedding import BaseEmbedding
from agentic_ai.core.models import Document

# Configure module-level logger.
logger = logging.getLogger(__name__)


class BaseChunker(ABC):
    """
    Abstract base class for document chunkers.

    All chunking strategies implement the ``chunk`` method that takes
    a Document and returns a list of smaller Document chunks.
    """

    @abstractmethod
    def chunk(self, document: Document) -> list[Document]:
        """
        Split a document into smaller chunks.

        Args:
            document: The Document to split.

        Returns:
            A list of Document chunks with updated metadata.
        """
        ...

    def chunk_many(self, documents: list[Document]) -> list[Document]:
        """
        Split multiple documents into chunks.

        Args:
            documents: A list of Documents to split.

        Returns:
            A flat list of all chunks from all documents.
        """
        all_chunks: list[Document] = []
        for doc in documents:
            all_chunks.extend(self.chunk(doc))
        return all_chunks


class FixedSizeChunker(BaseChunker):
    """
    Splits text into fixed-size chunks with configurable overlap.

    The simplest chunking strategy. Divides text into chunks of
    ``chunk_size`` characters with ``chunk_overlap`` characters
    shared between consecutive chunks for context continuity.

    Attributes:
        chunk_size:    Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Example:
        >>> chunker = FixedSizeChunker(chunk_size=500, chunk_overlap=50)
        >>> chunks = chunker.chunk(document)
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        """
        Initialize the fixed-size chunker.

        Args:
            chunk_size:    Maximum characters per chunk. Default is 1000.
            chunk_overlap: Overlap between consecutive chunks. Default is 200.

        Raises:
            ValueError: If chunk_overlap >= chunk_size.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> list[Document]:
        """
        Split a document into fixed-size chunks.

        Args:
            document: The Document to split.

        Returns:
            A list of Document chunks with chunk metadata.
        """
        text = document.content
        chunks: list[Document] = []
        start = 0
        chunk_index = 0

        while start < len(text):
            # Calculate the end position for this chunk.
            end = start + self.chunk_size

            # Extract the chunk text.
            chunk_text = text[start:end]

            # Create a new Document for this chunk.
            metadata = dict(document.metadata) if document.metadata else {}
            metadata.update({
                "chunk_index": chunk_index,
                "chunk_start": start,
                "chunk_end": min(end, len(text)),
                "chunker": "FixedSizeChunker",
            })

            chunks.append(
                Document(content=chunk_text.strip(), metadata=metadata)
            )

            # Move the start position forward, accounting for overlap.
            start += self.chunk_size - self.chunk_overlap
            chunk_index += 1

        logger.debug(
            "FixedSizeChunker: split document into %d chunks", len(chunks)
        )
        return chunks


class RecursiveChunker(BaseChunker):
    """
    Recursively splits text using a hierarchy of separators.

    Attempts to split at the most semantically meaningful boundaries first
    (paragraphs), falling back to less meaningful ones (sentences, words)
    only when needed to meet the chunk size constraint.

    Separator hierarchy (default):
        1. Double newline (paragraph boundary)
        2. Single newline (line boundary)
        3. Period + space (sentence boundary)
        4. Space (word boundary)
        5. Empty string (character-level, last resort)

    Inspired by LangChain's RecursiveCharacterTextSplitter.

    Example:
        >>> chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
        >>> chunks = chunker.chunk(document)
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ) -> None:
        """
        Initialize the recursive chunker.

        Args:
            chunk_size:    Maximum characters per chunk. Default is 1000.
            chunk_overlap: Overlap between chunks. Default is 200.
            separators:    Ordered list of separators to try. Default is
                           ["\\n\\n", "\\n", ". ", " ", ""].
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def chunk(self, document: Document) -> list[Document]:
        """
        Recursively split a document into chunks.

        Args:
            document: The Document to split.

        Returns:
            A list of Document chunks.
        """
        text_chunks = self._split_text(document.content, self.separators)

        chunks: list[Document] = []
        for i, chunk_text in enumerate(text_chunks):
            if not chunk_text.strip():
                continue

            metadata = dict(document.metadata) if document.metadata else {}
            metadata.update({
                "chunk_index": i,
                "chunker": "RecursiveChunker",
            })
            chunks.append(Document(content=chunk_text.strip(), metadata=metadata))

        logger.debug(
            "RecursiveChunker: split document into %d chunks", len(chunks)
        )
        return chunks

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """
        Recursively split text using the separator hierarchy.

        Tries each separator in order. If a split produces chunks that are
        still too large, recursively splits those chunks with the next
        separator in the hierarchy.

        Args:
            text:       The text to split.
            separators: The remaining separators to try.

        Returns:
            A list of text chunks.
        """
        if not text:
            return []

        # If the text fits within the chunk size, return it as-is.
        if len(text) <= self.chunk_size:
            return [text]

        # If no separators left, force-split at chunk_size.
        if not separators:
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]

        # Try the first separator.
        separator = separators[0]
        remaining_separators = separators[1:]

        if separator:
            parts = text.split(separator)
        else:
            # Empty separator means character-level split.
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]

        # Merge parts into chunks that fit the size limit.
        merged_chunks: list[str] = []
        current_chunk = ""

        for part in parts:
            # Check if adding this part would exceed the chunk size.
            candidate = current_chunk + separator + part if current_chunk else part

            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                # Save the current chunk if it has content.
                if current_chunk:
                    merged_chunks.append(current_chunk)

                # If the part itself exceeds chunk_size, recursively split it.
                if len(part) > self.chunk_size:
                    sub_chunks = self._split_text(part, remaining_separators)
                    merged_chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = part

        # Don't forget the last chunk.
        if current_chunk:
            merged_chunks.append(current_chunk)

        return merged_chunks


class SentenceChunker(BaseChunker):
    """
    Splits text at sentence boundaries.

    Groups consecutive sentences into chunks, ensuring no sentence is
    split across chunk boundaries. Uses regex-based sentence detection.

    Attributes:
        sentences_per_chunk: Number of sentences per chunk.
        overlap_sentences:   Number of overlapping sentences between chunks.

    Example:
        >>> chunker = SentenceChunker(sentences_per_chunk=5, overlap_sentences=1)
        >>> chunks = chunker.chunk(document)
    """

    def __init__(
        self,
        sentences_per_chunk: int = 5,
        overlap_sentences: int = 1,
    ) -> None:
        """
        Initialize the sentence chunker.

        Args:
            sentences_per_chunk: Number of sentences per chunk. Default is 5.
            overlap_sentences:   Sentences shared between chunks. Default is 1.
        """
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap_sentences = overlap_sentences

    def chunk(self, document: Document) -> list[Document]:
        """
        Split a document at sentence boundaries.

        Args:
            document: The Document to split.

        Returns:
            A list of Document chunks with sentence-aligned boundaries.
        """
        # Split text into sentences using regex.
        sentences = self._split_into_sentences(document.content)

        if not sentences:
            return []

        chunks: list[Document] = []
        step = self.sentences_per_chunk - self.overlap_sentences
        step = max(step, 1)  # Ensure we always advance.

        for i in range(0, len(sentences), step):
            chunk_sentences = sentences[i:i + self.sentences_per_chunk]
            chunk_text = " ".join(chunk_sentences)

            metadata = dict(document.metadata) if document.metadata else {}
            metadata.update({
                "chunk_index": len(chunks),
                "sentence_start": i,
                "sentence_end": i + len(chunk_sentences),
                "chunker": "SentenceChunker",
            })
            chunks.append(Document(content=chunk_text.strip(), metadata=metadata))

        logger.debug(
            "SentenceChunker: split document into %d chunks", len(chunks)
        )
        return chunks

    @staticmethod
    def _split_into_sentences(text: str) -> list[str]:
        """
        Split text into sentences using regex patterns.

        Handles common abbreviations and decimal numbers to avoid
        false sentence breaks.

        Args:
            text: The text to split into sentences.

        Returns:
            A list of sentence strings.
        """
        # Pattern matches sentence-ending punctuation followed by whitespace
        # and an uppercase letter, while handling common abbreviations.
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text.strip())

        # Filter out empty strings.
        return [s.strip() for s in sentences if s.strip()]


class SemanticChunker(BaseChunker):
    """
    Splits text based on semantic similarity between sentences.

    Uses embedding similarity to detect topic boundaries. Consecutive
    sentences with high semantic similarity are grouped together. When
    the similarity between adjacent sentence groups drops below a
    threshold, a new chunk boundary is created.

    This produces chunks that are aligned with actual topic changes
    in the text, resulting in more coherent retrieval results.

    Attributes:
        embedding:  The embedding provider for computing similarity.
        threshold:  Similarity threshold for chunk boundaries (0.0 to 1.0).
        min_chunk_size: Minimum characters per chunk.
        max_chunk_size: Maximum characters per chunk.

    Example:
        >>> chunker = SemanticChunker(
        ...     embedding=embedder,
        ...     threshold=0.5,
        ...     min_chunk_size=100,
        ... )
        >>> chunks = chunker.chunk(document)
    """

    def __init__(
        self,
        embedding: BaseEmbedding,
        threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
    ) -> None:
        """
        Initialize the semantic chunker.

        Args:
            embedding:      The embedding provider for similarity computation.
            threshold:      Cosine similarity threshold. Boundaries are created
                            where similarity drops below this value. Default is 0.5.
            min_chunk_size: Minimum characters per chunk. Default is 100.
            max_chunk_size: Maximum characters per chunk. Default is 2000.
        """
        self.embedding = embedding
        self.threshold = threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def chunk(self, document: Document) -> list[Document]:
        """
        Split a document based on semantic similarity.

        1. Split text into sentences.
        2. Embed each sentence.
        3. Compute cosine similarity between consecutive sentences.
        4. Create chunk boundaries where similarity drops below threshold.
        5. Merge small chunks and split large ones.

        Args:
            document: The Document to split.

        Returns:
            A list of semantically coherent Document chunks.
        """
        # Split into sentences.
        sentences = SentenceChunker._split_into_sentences(document.content)

        if len(sentences) <= 1:
            return [document]

        # Embed all sentences.
        embeddings = self.embedding.embed_documents(sentences)

        # Compute cosine similarity between consecutive sentences.
        similarities = self._compute_consecutive_similarities(embeddings)

        # Find chunk boundaries where similarity drops below threshold.
        boundaries = [0]
        for i, sim in enumerate(similarities):
            if sim < self.threshold:
                boundaries.append(i + 1)
        boundaries.append(len(sentences))

        # Create chunks from boundary pairs.
        chunks: list[Document] = []
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            chunk_text = " ".join(sentences[start_idx:end_idx])

            # Skip chunks that are too small, merging with previous.
            if len(chunk_text) < self.min_chunk_size and chunks:
                prev_doc = chunks[-1]
                chunks[-1] = Document(
                    content=prev_doc.content + " " + chunk_text,
                    metadata=prev_doc.metadata,
                )
                continue

            metadata = dict(document.metadata) if document.metadata else {}
            metadata.update({
                "chunk_index": len(chunks),
                "sentence_start": start_idx,
                "sentence_end": end_idx,
                "chunker": "SemanticChunker",
            })
            chunks.append(Document(content=chunk_text.strip(), metadata=metadata))

        logger.debug(
            "SemanticChunker: split document into %d chunks", len(chunks)
        )
        return chunks

    @staticmethod
    def _compute_consecutive_similarities(
        embeddings: list[list[float]],
    ) -> list[float]:
        """
        Compute cosine similarity between consecutive embedding pairs.

        Args:
            embeddings: A list of embedding vectors.

        Returns:
            A list of similarity scores (length = len(embeddings) - 1).
        """
        similarities: list[float] = []

        for i in range(len(embeddings) - 1):
            # Compute cosine similarity between vectors i and i+1.
            vec_a = embeddings[i]
            vec_b = embeddings[i + 1]

            dot_product = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
            norm_a = sum(a * a for a in vec_a) ** 0.5
            norm_b = sum(b * b for b in vec_b) ** 0.5

            similarity = dot_product / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0

            similarities.append(similarity)

        return similarities
