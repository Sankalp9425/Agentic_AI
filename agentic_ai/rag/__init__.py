"""
Advanced RAG (Retrieval-Augmented Generation) pipeline module.

Provides a complete, production-ready RAG pipeline with modular components
for each stage of the retrieval-augmented generation process:

Stages:
    1. **Ingestion** — Parse documents from various formats (PDF, text, HTML)
       including advanced PDF parsing with text extraction, image extraction
       and captioning via multimodal LLMs, and table parsing to Markdown.

    2. **Chunking** — Split documents into semantically meaningful chunks
       using multiple strategies: fixed-size, recursive, sentence-based,
       and semantic chunking (embedding-similarity-based splitting).

    3. **Retrieval** — Retrieve relevant chunks using advanced techniques:
       MMR (Maximal Marginal Relevance), hybrid search (keyword + semantic),
       query expansion, and re-ranking with cross-encoders or LLMs.

    4. **Output Parsing** — Parse and validate LLM outputs into structured
       formats using Pydantic models, JSON schemas, or custom parsers.

    5. **Evaluation** — Evaluate RAG pipeline quality with metrics for
       faithfulness, relevance, context precision, and answer correctness.

    6. **Pipeline** — High-level orchestrator that connects all stages
       into a single, configurable pipeline.

Usage:
    >>> from agentic_ai.rag.pipeline import RAGPipeline, PipelineConfig
    >>> from agentic_ai.rag.ingestion import PDFParser, TextFileParser
    >>> from agentic_ai.rag.chunking import SemanticChunker
    >>> from agentic_ai.rag.retrieval import MMRRetriever
    >>> from agentic_ai.rag.output_parser import PydanticOutputParser
    >>> from agentic_ai.rag.evaluation import RAGEvaluator
"""
