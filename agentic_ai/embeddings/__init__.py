"""
Embedding provider implementations for text vectorization.

This module contains concrete implementations of the BaseEmbedding interface
for all supported embedding providers. Embeddings convert text into dense
numerical vectors that capture semantic meaning, enabling similarity search
and retrieval-augmented generation (RAG).

Supported Providers:
    - OpenAI:    text-embedding-3-small, text-embedding-3-large, ada-002.
    - Google:    Gemini embedding models (embedding-001, text-embedding-004).
    - Anthropic: Voyage AI embeddings (via Anthropic's recommended partner).
    - Groq:      Embedding models available through Groq's API.

Example:
    >>> from agentic_ai.embeddings.openai_embedding import OpenAIEmbedding
    >>> embedder = OpenAIEmbedding(api_key="sk-...", model="text-embedding-3-small")
    >>> vector = embedder.embed_query("What is machine learning?")
    >>> len(vector)  # 1536 dimensions
    1536
"""
