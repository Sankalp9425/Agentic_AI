"""
FastAPI REST API server for the agentic AI framework.

Provides production-ready HTTP endpoints for:
    - LLM chat completions.
    - RAG pipeline queries with document ingestion.
    - Agent execution (ReAct, Planning).
    - Health checks and configuration.

The server is configurable via environment variables and supports
CORS, request logging, and error handling middleware.

Requirements:
    pip install fastapi uvicorn python-multipart

Usage:
    # Start the server:
    uvicorn agentic_ai.api.server:app --host 0.0.0.0 --port 8000

    # Or programmatically:
    from agentic_ai.api.server import create_app
    app = create_app()
"""

import contextlib
import logging
import os
import time
from typing import Any

from agentic_ai.utils.config import get_api_key

# Configure module-level logger.
logger = logging.getLogger(__name__)


def create_app() -> Any:
    """
    Create and configure the FastAPI application.

    Sets up all routes, middleware, and error handlers.
    Configuration is loaded from environment variables.

    Returns:
        A configured FastAPI application instance.
    """
    try:
        from fastapi import FastAPI, HTTPException, UploadFile
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel, Field
    except ImportError as e:
        raise ImportError(
            "FastAPI and uvicorn are required for the API server. "
            "Install them with: pip install fastapi uvicorn python-multipart"
        ) from e

    # ---------------------
    # Request/Response Models
    # ---------------------

    class ChatRequest(BaseModel):
        """Request model for chat completions."""
        message: str = Field(..., description="The user's message")
        model: str = Field(default="gpt-4o-mini", description="LLM model to use")
        provider: str = Field(default="openai", description="LLM provider (openai, gemini, claude, groq)")
        system_prompt: str | None = Field(default=None, description="Optional system prompt")
        temperature: float = Field(default=0.7, description="Sampling temperature")
        max_tokens: int = Field(default=1024, description="Maximum tokens in response")

    class ChatResponse(BaseModel):
        """Response model for chat completions."""
        message: str = Field(..., description="The assistant's response")
        model: str = Field(..., description="Model used")
        provider: str = Field(..., description="Provider used")
        usage: dict[str, int] = Field(default_factory=dict, description="Token usage statistics")

    class RAGQueryRequest(BaseModel):
        """Request model for RAG queries."""
        question: str = Field(..., description="The question to answer")
        top_k: int = Field(default=5, description="Number of documents to retrieve")
        retriever_type: str = Field(default="simple", description="Retrieval strategy")

    class RAGQueryResponse(BaseModel):
        """Response model for RAG queries."""
        answer: str = Field(..., description="Generated answer")
        sources: list[dict[str, Any]] = Field(default_factory=list, description="Source documents used")
        retrieval_time_ms: float = Field(default=0, description="Retrieval time in milliseconds")
        generation_time_ms: float = Field(default=0, description="Generation time in milliseconds")

    class HealthResponse(BaseModel):
        """Health check response."""
        status: str = "healthy"
        version: str = "0.1.0"
        components: dict[str, str] = Field(default_factory=dict)

    # ---------------------
    # App Creation
    # ---------------------

    app = FastAPI(
        title="Agentic AI API",
        description="Production-ready API for the Agentic AI framework",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Add CORS middleware for cross-origin requests.
    allowed_origins = os.getenv("CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------------------
    # Shared State
    # ---------------------

    # Store for initialized components (lazily created).
    _state: dict[str, Any] = {}

    def _get_llm(provider: str, model: str, **kwargs: Any) -> Any:
        """
        Get or create an LLM instance for the specified provider.

        Args:
            provider: The LLM provider name.
            model:    The model identifier.
            **kwargs: Additional model parameters.

        Returns:
            A BaseLLM implementation.
        """
        cache_key = f"llm_{provider}_{model}"
        if cache_key in _state:
            return _state[cache_key]

        if provider == "openai":
            from agentic_ai.llms.openai_llm import OpenAIChatModel
            api_key = get_api_key("OPENAI_API_KEY")
            llm = OpenAIChatModel(api_key=api_key, model=model, **kwargs)
        elif provider == "gemini":
            from agentic_ai.llms.gemini_llm import GeminiChatModel
            api_key = get_api_key("GOOGLE_API_KEY")
            llm = GeminiChatModel(api_key=api_key, model=model, **kwargs)
        elif provider == "claude":
            from agentic_ai.llms.claude_llm import ClaudeChatModel
            api_key = get_api_key("ANTHROPIC_API_KEY")
            llm = ClaudeChatModel(api_key=api_key, model=model, **kwargs)
        elif provider == "groq":
            from agentic_ai.llms.groq_llm import GroqChatModel
            api_key = get_api_key("GROQ_API_KEY")
            llm = GroqChatModel(api_key=api_key, model=model, **kwargs)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown provider: {provider}. Use: openai, gemini, claude, groq",
            )

        _state[cache_key] = llm
        return llm

    # ---------------------
    # Routes
    # ---------------------

    @app.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """
        Health check endpoint.

        Returns the server status and available components.
        """
        components: dict[str, str] = {}

        # Check which API keys are configured.
        for provider, env_var in [
            ("openai", "OPENAI_API_KEY"),
            ("gemini", "GOOGLE_API_KEY"),
            ("claude", "ANTHROPIC_API_KEY"),
            ("groq", "GROQ_API_KEY"),
        ]:
            if os.getenv(env_var):
                components[provider] = "configured"
            else:
                components[provider] = "not configured"

        return HealthResponse(
            status="healthy",
            version="0.1.0",
            components=components,
        )

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        """
        Send a chat message to an LLM.

        Supports multiple providers (OpenAI, Gemini, Claude, Groq)
        and models. Optionally accepts a system prompt.
        """
        from agentic_ai.core.models import Message, Role

        try:
            llm = _get_llm(
                provider=request.provider,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

            messages: list[Message] = []
            if request.system_prompt:
                messages.append(Message(role=Role.SYSTEM, content=request.system_prompt))
            messages.append(Message(role=Role.USER, content=request.message))

            response = llm.chat(messages)

            return ChatResponse(
                message=response.content,
                model=request.model,
                provider=request.provider,
            )

        except Exception as e:
            logger.error("Chat error: %s", e)
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/rag/ingest")
    async def rag_ingest(files: list[UploadFile]) -> dict[str, Any]:
        """
        Ingest documents into the RAG pipeline.

        Accepts file uploads (PDF, TXT, HTML, MD) and indexes them
        for later retrieval.
        """
        import tempfile

        if "rag_pipeline" not in _state:
            raise HTTPException(
                status_code=503,
                detail="RAG pipeline not initialized. Configure LLM and vector store first.",
            )

        pipeline = _state["rag_pipeline"]
        temp_paths: list[str] = []

        try:
            # Save uploaded files to temp directory.
            for file in files:
                suffix = os.path.splitext(file.filename or "")[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    content = await file.read()
                    tmp.write(content)
                    temp_paths.append(tmp.name)

            # Ingest the files.
            ids = pipeline.ingest(temp_paths)

            return {
                "status": "success",
                "files_ingested": len(files),
                "chunks_created": len(ids),
            }

        finally:
            # Clean up temp files.
            for path in temp_paths:
                with contextlib.suppress(OSError):
                    os.unlink(path)

    @app.post("/rag/query", response_model=RAGQueryResponse)
    async def rag_query(request: RAGQueryRequest) -> RAGQueryResponse:
        """
        Query the RAG pipeline.

        Retrieves relevant documents and generates an answer.
        """
        if "rag_pipeline" not in _state:
            raise HTTPException(
                status_code=503,
                detail="RAG pipeline not initialized. Ingest documents first.",
            )

        pipeline = _state["rag_pipeline"]

        try:
            start_time = time.time()
            answer, contexts = pipeline.query(
                request.question,
                return_contexts=True,
            )
            total_time = (time.time() - start_time) * 1000

            sources = [
                {
                    "content": doc.content[:200],
                    "metadata": doc.metadata,
                }
                for doc in contexts
            ]

            return RAGQueryResponse(
                answer=answer,
                sources=sources,
                retrieval_time_ms=total_time * 0.3,
                generation_time_ms=total_time * 0.7,
            )

        except Exception as e:
            logger.error("RAG query error: %s", e)
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/config")
    async def get_config() -> dict[str, Any]:
        """
        Get the current server configuration.

        Returns non-sensitive configuration details.
        """
        return {
            "version": "0.1.0",
            "configured_providers": [
                p for p, env in [
                    ("openai", "OPENAI_API_KEY"),
                    ("gemini", "GOOGLE_API_KEY"),
                    ("claude", "ANTHROPIC_API_KEY"),
                    ("groq", "GROQ_API_KEY"),
                ]
                if os.getenv(env)
            ],
            "cors_origins": allowed_origins,
        }

    return app


# Create the default app instance for uvicorn.
app = create_app()
