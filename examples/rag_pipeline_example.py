"""
Advanced RAG Pipeline Examples.

Demonstrates the full RAG pipeline with various configurations:
1. Basic RAG with recursive chunking and simple retrieval.
2. Semantic chunking with MMR retrieval for diversity.
3. Hybrid search combining keyword and semantic retrieval.
4. Query expansion for broader coverage.
5. Re-ranking for precision.
6. Full pipeline with PDF ingestion, image captioning, and evaluation.
7. Output parsing with Pydantic models.

Requirements:
    pip install -e ".[openai,chroma]"  # Minimum for these examples

Usage:
    export OPENAI_API_KEY="sk-..."
    python examples/rag_pipeline_example.py
"""

import os
import sys

# Add the parent directory to the path so we can import agentic_ai.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def example_basic_rag() -> None:
    """
    Example 1: Basic RAG pipeline.

    Uses recursive chunking and simple similarity search retrieval.
    This is the simplest configuration for getting started.
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic RAG Pipeline")
    print("=" * 60)

    from agentic_ai.embeddings.openai_embedding import OpenAIEmbedding
    from agentic_ai.llms.openai_llm import OpenAIChatModel
    from agentic_ai.rag.pipeline import PipelineConfig, RAGPipeline
    from agentic_ai.vectorstores.chroma_store import ChromaVectorStore

    # Initialize components.
    api_key = os.getenv("OPENAI_API_KEY", "")
    llm = OpenAIChatModel(api_key=api_key, model="gpt-4o-mini")
    embedding = OpenAIEmbedding(api_key=api_key)
    vector_store = ChromaVectorStore(embedding=embedding, collection_name="basic_rag")

    # Configure the pipeline with defaults.
    config = PipelineConfig(
        chunker_type="recursive",
        chunk_size=500,
        chunk_overlap=50,
        retriever_type="simple",
        top_k=3,
    )

    # Create the pipeline.
    pipeline = RAGPipeline(
        llm=llm,
        embedding=embedding,
        vector_store=vector_store,
        config=config,
    )

    # Ingest text documents.
    # In a real scenario, you would pass file paths:
    # pipeline.ingest(["docs/manual.pdf", "docs/faq.txt"])

    # For this example, ingest documents directly.
    from agentic_ai.core.models import Document

    documents = [
        Document(
            content=(
                "Python is a high-level, general-purpose programming language. "
                "Its design philosophy emphasizes code readability with the use of "
                "significant indentation. Python is dynamically typed and garbage-collected. "
                "It supports multiple programming paradigms, including structured, "
                "object-oriented, and functional programming."
            ),
            metadata={"source": "wiki", "topic": "python"},
        ),
        Document(
            content=(
                "Machine learning is a subset of artificial intelligence that focuses "
                "on building systems that learn from data. Unlike traditional programming "
                "where rules are explicitly coded, ML algorithms identify patterns in data "
                "and make predictions. Common types include supervised learning, unsupervised "
                "learning, and reinforcement learning."
            ),
            metadata={"source": "textbook", "topic": "ml"},
        ),
        Document(
            content=(
                "RAG (Retrieval-Augmented Generation) is a technique that combines "
                "information retrieval with text generation. It first retrieves relevant "
                "documents from a knowledge base, then uses them as context for an LLM "
                "to generate accurate, grounded answers. This reduces hallucination and "
                "allows the model to access up-to-date information."
            ),
            metadata={"source": "paper", "topic": "rag"},
        ),
    ]

    pipeline.ingest_documents(documents)
    print("Ingested 3 documents into the pipeline.")

    # Query the pipeline.
    answer = pipeline.query("What is RAG and how does it work?")
    print("\nQuestion: What is RAG and how does it work?")
    print(f"Answer: {answer}")


def example_semantic_chunking_mmr() -> None:
    """
    Example 2: Semantic chunking with MMR retrieval.

    Uses embedding-based semantic chunking to split documents at
    topic boundaries, and MMR retrieval for diverse results.
    """
    print("\n" + "=" * 60)
    print("Example 2: Semantic Chunking + MMR Retrieval")
    print("=" * 60)

    from agentic_ai.embeddings.openai_embedding import OpenAIEmbedding
    from agentic_ai.llms.openai_llm import OpenAIChatModel
    from agentic_ai.rag.pipeline import PipelineConfig, RAGPipeline
    from agentic_ai.vectorstores.chroma_store import ChromaVectorStore

    api_key = os.getenv("OPENAI_API_KEY", "")
    llm = OpenAIChatModel(api_key=api_key, model="gpt-4o-mini")
    embedding = OpenAIEmbedding(api_key=api_key)
    vector_store = ChromaVectorStore(embedding=embedding, collection_name="semantic_mmr")

    config = PipelineConfig(
        chunker_type="semantic",
        semantic_threshold=0.5,
        retriever_type="mmr",
        mmr_lambda=0.7,
        top_k=3,
    )

    _pipeline = RAGPipeline(
        llm=llm,
        embedding=embedding,
        vector_store=vector_store,
        config=config,
    )

    print("Pipeline configured with semantic chunking and MMR retrieval.")
    print("Ready for document ingestion via pipeline.ingest(['file.pdf'])")


def example_hybrid_search() -> None:
    """
    Example 3: Hybrid search combining keyword and semantic retrieval.

    Uses Reciprocal Rank Fusion (RRF) to combine TF-IDF keyword
    matching with embedding-based semantic search.
    """
    print("\n" + "=" * 60)
    print("Example 3: Hybrid Search (Keyword + Semantic)")
    print("=" * 60)

    from agentic_ai.embeddings.openai_embedding import OpenAIEmbedding
    from agentic_ai.llms.openai_llm import OpenAIChatModel
    from agentic_ai.rag.pipeline import PipelineConfig, RAGPipeline
    from agentic_ai.vectorstores.chroma_store import ChromaVectorStore

    api_key = os.getenv("OPENAI_API_KEY", "")
    llm = OpenAIChatModel(api_key=api_key, model="gpt-4o-mini")
    embedding = OpenAIEmbedding(api_key=api_key)
    vector_store = ChromaVectorStore(embedding=embedding, collection_name="hybrid")

    config = PipelineConfig(
        chunker_type="recursive",
        retriever_type="hybrid",
        hybrid_alpha=0.7,  # 70% semantic, 30% keyword.
        top_k=5,
    )

    _pipeline = RAGPipeline(
        llm=llm,
        embedding=embedding,
        vector_store=vector_store,
        config=config,
    )

    print("Pipeline configured with hybrid search (alpha=0.7).")
    print("Hybrid search combines keyword matching with semantic similarity.")


def example_query_expansion() -> None:
    """
    Example 4: Query expansion for broader coverage.

    Uses an LLM to generate alternative query formulations,
    retrieves for each, and merges results.
    """
    print("\n" + "=" * 60)
    print("Example 4: Query Expansion Retrieval")
    print("=" * 60)

    from agentic_ai.embeddings.openai_embedding import OpenAIEmbedding
    from agentic_ai.llms.openai_llm import OpenAIChatModel
    from agentic_ai.rag.pipeline import PipelineConfig, RAGPipeline
    from agentic_ai.vectorstores.chroma_store import ChromaVectorStore

    api_key = os.getenv("OPENAI_API_KEY", "")
    llm = OpenAIChatModel(api_key=api_key, model="gpt-4o-mini")
    embedding = OpenAIEmbedding(api_key=api_key)
    vector_store = ChromaVectorStore(embedding=embedding, collection_name="expansion")

    config = PipelineConfig(
        chunker_type="sentence",
        sentences_per_chunk=5,
        retriever_type="expansion",
        num_expansions=3,
        top_k=5,
    )

    _pipeline = RAGPipeline(
        llm=llm,
        embedding=embedding,
        vector_store=vector_store,
        config=config,
    )

    print("Pipeline configured with query expansion (3 variants).")
    print("Each query is expanded into 3 alternative formulations.")


def example_reranking() -> None:
    """
    Example 5: Re-ranking for precision.

    Retrieves a broad set of candidates, then uses an LLM
    to re-rank them by relevance for higher precision.
    """
    print("\n" + "=" * 60)
    print("Example 5: LLM Re-ranking Retrieval")
    print("=" * 60)

    from agentic_ai.embeddings.openai_embedding import OpenAIEmbedding
    from agentic_ai.llms.openai_llm import OpenAIChatModel
    from agentic_ai.rag.pipeline import PipelineConfig, RAGPipeline
    from agentic_ai.vectorstores.chroma_store import ChromaVectorStore

    api_key = os.getenv("OPENAI_API_KEY", "")
    llm = OpenAIChatModel(api_key=api_key, model="gpt-4o-mini")
    embedding = OpenAIEmbedding(api_key=api_key)
    vector_store = ChromaVectorStore(embedding=embedding, collection_name="rerank")

    config = PipelineConfig(
        chunker_type="recursive",
        retriever_type="rerank",
        rerank_initial_k=20,
        top_k=5,
    )

    _pipeline = RAGPipeline(
        llm=llm,
        embedding=embedding,
        vector_store=vector_store,
        config=config,
    )

    print("Pipeline configured with LLM re-ranking (initial_k=20, top_k=5).")
    print("Retrieves 20 candidates and re-ranks to top 5 using LLM scoring.")


def example_output_parsing() -> None:
    """
    Example 6: Pydantic output parsing.

    Demonstrates parsing LLM output into structured Pydantic models.
    """
    print("\n" + "=" * 60)
    print("Example 6: Pydantic Output Parsing")
    print("=" * 60)

    from pydantic import BaseModel, Field

    from agentic_ai.rag.output_parser import (
        JSONOutputParser,
        ListOutputParser,
        PydanticOutputParser,
    )

    # Define a Pydantic model for structured output.
    class ResearchSummary(BaseModel):
        """A structured research summary."""
        title: str = Field(description="Title of the research")
        key_findings: list[str] = Field(description="List of key findings")
        confidence: float = Field(description="Confidence score 0-1")
        methodology: str = Field(description="Research methodology used")

    # Create the parser and get format instructions.
    parser = PydanticOutputParser(model=ResearchSummary)
    instructions = parser.get_format_instructions()
    print(f"Format instructions for LLM:\n{instructions}\n")

    # Simulate an LLM response and parse it.
    simulated_response = '''```json
{
    "title": "Effects of RAG on LLM Accuracy",
    "key_findings": [
        "RAG reduces hallucination by 40%",
        "Context window size affects retrieval quality",
        "Hybrid search outperforms pure semantic search"
    ],
    "confidence": 0.85,
    "methodology": "Controlled experiment with 1000 queries"
}
```'''

    result = parser.parse(simulated_response)
    print(f"Parsed result: {result}")
    print(f"Title: {result.title}")
    print(f"Confidence: {result.confidence}")

    # List output parser example.
    list_parser = ListOutputParser()
    items = list_parser.parse("1. First item\n2. Second item\n3. Third item")
    print(f"\nList parser result: {items}")

    # JSON output parser example.
    json_parser = JSONOutputParser()
    data = json_parser.parse('{"name": "test", "value": 42}')
    print(f"JSON parser result: {data}")


def example_evaluation() -> None:
    """
    Example 7: RAG evaluation metrics.

    Shows how to evaluate a RAG pipeline's output quality
    using the LLM-as-judge approach.
    """
    print("\n" + "=" * 60)
    print("Example 7: RAG Evaluation")
    print("=" * 60)

    from agentic_ai.rag.evaluation import EvaluationResult

    # Demonstrate the EvaluationResult data structure.
    result = EvaluationResult(
        faithfulness=0.9,
        answer_relevance=0.85,
        context_relevance=0.8,
        context_precision=0.75,
        answer_correctness=0.88,
    )

    print(f"Faithfulness:      {result.faithfulness}")
    print(f"Answer Relevance:  {result.answer_relevance}")
    print(f"Context Relevance: {result.context_relevance}")
    print(f"Context Precision: {result.context_precision}")
    print(f"Answer Correctness:{result.answer_correctness}")
    print(f"Overall Score:     {result.overall_score():.2f}")
    print(f"\nAs dict: {result.to_dict()}")

    # Note: To actually run evaluation, you need an LLM:
    # evaluator = RAGEvaluator(llm=judge_llm)
    # scores = evaluator.evaluate(
    #     question="What is RAG?",
    #     answer="RAG combines retrieval with generation...",
    #     contexts=["RAG is a technique that..."],
    #     reference="RAG retrieves documents and uses them...",
    # )


def example_chunking_strategies() -> None:
    """
    Example 8: Comparing different chunking strategies.

    Demonstrates how the same document is split differently
    by each chunking strategy.
    """
    print("\n" + "=" * 60)
    print("Example 8: Chunking Strategy Comparison")
    print("=" * 60)

    from agentic_ai.core.models import Document
    from agentic_ai.rag.chunking import (
        FixedSizeChunker,
        RecursiveChunker,
        SentenceChunker,
    )

    sample_text = (
        "Machine learning is transforming industries worldwide. "
        "Healthcare uses ML for diagnosis and drug discovery. "
        "Finance applies it for fraud detection and trading. "
        "Manufacturing leverages ML for quality control.\n\n"
        "Deep learning, a subset of ML, uses neural networks with many layers. "
        "CNNs excel at image recognition tasks. "
        "RNNs and Transformers handle sequential data like text. "
        "GPT and BERT are popular transformer-based models.\n\n"
        "The future of AI includes more efficient models, better interpretability, "
        "and stronger safety measures. Researchers are working on reducing "
        "computational costs while maintaining model quality."
    )

    doc = Document(content=sample_text, metadata={"source": "example"})

    # Fixed-size chunking.
    fixed = FixedSizeChunker(chunk_size=200, chunk_overlap=30)
    fixed_chunks = fixed.chunk(doc)
    print(f"\nFixed-size chunker: {len(fixed_chunks)} chunks")
    for i, chunk in enumerate(fixed_chunks):
        print(f"  Chunk {i}: {chunk.content[:60]}...")

    # Recursive chunking.
    recursive = RecursiveChunker(chunk_size=200, chunk_overlap=30)
    recursive_chunks = recursive.chunk(doc)
    print(f"\nRecursive chunker: {len(recursive_chunks)} chunks")
    for i, chunk in enumerate(recursive_chunks):
        print(f"  Chunk {i}: {chunk.content[:60]}...")

    # Sentence chunking.
    sentence = SentenceChunker(sentences_per_chunk=3, overlap_sentences=1)
    sentence_chunks = sentence.chunk(doc)
    print(f"\nSentence chunker: {len(sentence_chunks)} chunks")
    for i, chunk in enumerate(sentence_chunks):
        print(f"  Chunk {i}: {chunk.content[:60]}...")


if __name__ == "__main__":
    print("=" * 60)
    print("  Agentic AI — RAG Pipeline Examples")
    print("=" * 60)

    # These examples work without API keys.
    example_output_parsing()
    example_evaluation()
    example_chunking_strategies()

    # These examples require an OPENAI_API_KEY.
    if os.getenv("OPENAI_API_KEY"):
        example_basic_rag()
        example_semantic_chunking_mmr()
        example_hybrid_search()
        example_query_expansion()
        example_reranking()
    else:
        print("\n" + "=" * 60)
        print("Set OPENAI_API_KEY to run the LLM-powered examples.")
        print("=" * 60)
