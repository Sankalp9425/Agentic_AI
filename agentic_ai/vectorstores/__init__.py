"""
Vector database connector implementations.

This module contains concrete implementations of the BaseVectorStore interface
for all supported vector databases. Vector stores enable fast similarity search
over document embeddings, which is the foundation of RAG pipelines.

Supported Backends:
    - Chroma:    Open-source embedding database, great for local development.
    - Pinecone:  Cloud-native vector database with managed infrastructure.
    - PGVector:  PostgreSQL extension for vector similarity search.
    - FAISS:     Facebook AI Similarity Search, optimized for in-memory search.

Example:
    >>> from agentic_ai.vectorstores.chroma_store import ChromaVectorStore
    >>> from agentic_ai.embeddings.openai_embedding import OpenAIEmbedding
    >>> embedder = OpenAIEmbedding(api_key="sk-...")
    >>> store = ChromaVectorStore(embedding=embedder, collection_name="my_docs")
    >>> store.add_documents([Document(content="Hello world")])
    >>> results = store.similarity_search("greeting", k=3)
"""
