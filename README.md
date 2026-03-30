# Agentic AI

A comprehensive, modular Python 3.11+ framework for building advanced AI agent systems with support for RAG pipelines, ReAct agents, hierarchical multi-agent systems, and planning agents.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration (.env)](#configuration)
- [RAG Pipeline](#rag-pipeline)
- [Agents](#agents)
- [LLM Providers](#llm-providers)
- [Embedding Providers](#embedding-providers)
- [Vector Stores](#vector-stores)
- [MCP Connectors (Tools)](#mcp-connectors-tools)
- [Prompts](#prompts)
- [API Server](#api-server)
- [Docker](#docker)
- [Examples](#examples)
- [Architecture](#architecture)
- [Design Principles](#design-principles)
- [License](#license)

---

## Features

### Agentic Frameworks
| Framework | Description |
|-----------|-------------|
| **RAG** | Retrieval-Augmented Generation — grounds LLM responses in retrieved documents to reduce hallucinations |
| **ReAct** | Reasoning + Acting — iterative think → act → observe loop for multi-step problem solving with tools |
| **Hierarchical** | Manager agent delegates sub-tasks to specialized worker agents and synthesizes results |
| **Planning** | Creates an explicit step-by-step plan before executing, with dynamic re-planning on failure |

### Advanced RAG Pipeline
| Stage | Capabilities |
|-------|-------------|
| **Ingestion** | PDF (text + image extraction + LLM captioning + table-to-Markdown), HTML, plain text (.txt, .md, .csv, .log, .rst) |
| **Chunking** | Fixed-size, recursive (hierarchical separators), sentence-aligned, semantic (embedding-similarity boundaries) |
| **Retrieval** | Simple similarity, MMR (Maximal Marginal Relevance), hybrid (keyword + semantic with RRF), query expansion, LLM re-ranking |
| **Output Parsing** | Pydantic model parser, JSON parser, list parser, regex parser — each with format instruction generation |
| **Evaluation** | Faithfulness, answer relevance, context relevance, context precision, answer correctness (LLM-as-judge) |

### LLM Providers (Chat + Embeddings)
| Provider | Chat Model | Embedding Model |
|----------|-----------|-----------------|
| **OpenAI** | GPT-4, GPT-4o, GPT-3.5 | text-embedding-3-small/large |
| **Google Gemini** | Gemini Pro, Ultra | text-embedding-004 |
| **Anthropic Claude** | Claude 3 Opus, Sonnet, Haiku | Voyage AI (partner) |
| **Groq** | LLaMA 3, Mixtral | nomic-embed-text |

### Vector Database Connectors
| Database | Type | Best For |
|----------|------|----------|
| **ChromaDB** | Embedded / Local | Development, prototyping |
| **Pinecone** | Cloud-managed | Production at scale |
| **PGVector** | PostgreSQL extension | Existing Postgres infrastructure |
| **FAISS** | In-memory | Maximum speed, GPU support |

### MCP Connectors (Tools)
- **Email** — Send emails via SMTP (Gmail, Outlook, etc.)
- **Google Search** — Web search via Custom Search API
- **Web Scraper** — Extract content from web pages
- **File Reader/Writer** — Sandboxed local file operations
- **HTTP Request** — General-purpose REST API integration

### Production-Ready Features
- FastAPI REST API server with CORS, health checks, file upload
- Dockerfile with multi-stage builds and non-root user
- Centralized, configurable prompt templates
- Environment-based configuration management
- Retry with exponential backoff for all API calls
- Structured logging with file and console handlers

---

## Installation

**Requirements:** Python 3.11+

```bash
# Clone the repository
git clone https://github.com/Sankalp9425/Agentic_AI.git
cd Agentic_AI

# Core framework (minimal — only requests + pydantic)
pip install -e .

# With specific LLM providers
pip install -e ".[openai]"          # OpenAI GPT + embeddings
pip install -e ".[gemini]"          # Google Gemini
pip install -e ".[claude]"          # Anthropic Claude
pip install -e ".[groq]"            # Groq (LLaMA, Mixtral)

# With specific vector stores
pip install -e ".[chroma]"          # ChromaDB (recommended for local dev)
pip install -e ".[pinecone]"        # Pinecone (cloud)
pip install -e ".[pgvector]"        # PostgreSQL + pgvector
pip install -e ".[faiss]"           # FAISS (in-memory)

# RAG pipeline dependencies (PDF parsing, HTML parsing)
pip install -e ".[rag]"

# API server
pip install -e ".[api]"

# Multiple extras at once
pip install -e ".[openai,chroma,rag,api]"

# Everything
pip install -e ".[all]"

# Development tools (ruff, mypy, pytest)
pip install -e ".[dev]"
```

---

## Quick Start

### 1. Set up your environment

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your API key(s)
# At minimum, you need ONE LLM provider key (e.g., OPENAI_API_KEY)
```

### 2. Basic chat with an LLM

```python
from agentic_ai.llms.openai_llm import OpenAIChatModel
from agentic_ai.core.models import Message, Role

llm = OpenAIChatModel(api_key="sk-...", model="gpt-4o-mini")

messages = [Message(role=Role.USER, content="Explain RAG in 2 sentences.")]
response = llm.chat(messages)
print(response.content)
```

### 3. RAG in 10 lines

```python
from agentic_ai.llms.openai_llm import OpenAIChatModel
from agentic_ai.embeddings.openai_embedding import OpenAIEmbedding
from agentic_ai.vectorstores.chroma_store import ChromaVectorStore
from agentic_ai.rag.pipeline import RAGPipeline, PipelineConfig

llm = OpenAIChatModel(api_key="sk-...", model="gpt-4o-mini")
embedding = OpenAIEmbedding(api_key="sk-...")
store = ChromaVectorStore(embedding=embedding, collection_name="my_docs")

pipeline = RAGPipeline(llm=llm, embedding=embedding, vector_store=store)
pipeline.ingest(["docs/manual.pdf", "docs/faq.txt"])  # Ingest files
answer = pipeline.query("How do I reset my password?")
print(answer)
```

### 4. ReAct agent with tools

```python
from agentic_ai.llms.openai_llm import OpenAIChatModel
from agentic_ai.agents.react import ReActAgent
from agentic_ai.mcp.google_search import GoogleSearchTool

llm = OpenAIChatModel(api_key="sk-...", model="gpt-4o")
search = GoogleSearchTool(api_key="AIza...", search_engine_id="...")

agent = ReActAgent(llm=llm, tools=[search], max_steps=5)
result = agent.run("What are the latest developments in quantum computing?")
print(result)
```

---

## Configuration

Copy `.env.example` to `.env` and fill in the values you need. You only need to configure the providers you plan to use.

### Required (at least one LLM provider)

```bash
# Pick ONE or more:
OPENAI_API_KEY=sk-...                    # https://platform.openai.com/api-keys
GOOGLE_API_KEY=AIza...                   # https://aistudio.google.com/apikey
ANTHROPIC_API_KEY=sk-ant-...             # https://console.anthropic.com/settings/keys
GROQ_API_KEY=gsk_...                     # https://console.groq.com/keys
```

### Embedding providers (optional, if different from chat key)

```bash
# Voyage AI (recommended for Anthropic/Claude users)
VOYAGE_API_KEY=pa-...                    # https://dash.voyageai.com/
```

### Vector stores (pick one)

```bash
# ChromaDB (default — works out of the box, no config needed)
CHROMA_PERSIST_DIRECTORY=./chroma_data

# Pinecone
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=my-index
PINECONE_NAMESPACE=

# PostgreSQL + pgvector
PGVECTOR_CONNECTION_STRING=postgresql://user:pass@localhost:5432/vectordb

# FAISS (no config needed — in-memory)
FAISS_INDEX_PATH=./faiss_index
```

### RAG pipeline defaults

```bash
RAG_CHUNK_SIZE=1000
RAG_CHUNK_OVERLAP=200
RAG_CHUNKER_TYPE=recursive          # fixed | recursive | sentence | semantic
RAG_RETRIEVER_TYPE=simple           # simple | mmr | hybrid | expansion | rerank
RAG_TOP_K=5
RAG_EXTRACT_IMAGES=true
RAG_EXTRACT_TABLES=true
```

### API server

```bash
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1
CORS_ORIGINS=*
LOG_LEVEL=INFO
```

### MCP connectors

```bash
# Email (SMTP)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=you@gmail.com
SMTP_PASSWORD=your-app-password

# Google Search
GOOGLE_SEARCH_API_KEY=AIza...
GOOGLE_SEARCH_ENGINE_ID=...
```

### General settings

```bash
MAX_RETRIES=3
REQUEST_TIMEOUT=30
DEBUG=false
```

See [`.env.example`](.env.example) for the complete list with documentation links for each API key.

---

## RAG Pipeline

The RAG pipeline provides a complete ingestion → chunking → retrieval → generation workflow.

### Pipeline Configuration

```python
from agentic_ai.rag.pipeline import RAGPipeline, PipelineConfig

config = PipelineConfig(
    chunker_type="semantic",      # Use semantic chunking
    retriever_type="mmr",         # Use MMR for diverse results
    chunk_size=1000,              # Characters per chunk
    chunk_overlap=200,            # Overlap between chunks
    top_k=5,                      # Retrieve top 5 documents
    mmr_lambda=0.7,               # Relevance vs diversity trade-off
    extract_images=True,          # Extract images from PDFs
    extract_tables=True,          # Extract tables from PDFs
)

pipeline = RAGPipeline(
    llm=llm, embedding=embedding, vector_store=store, config=config
)
```

### Ingestion

Supports multiple document formats with advanced PDF parsing:

```python
# Ingest from files (PDF, TXT, HTML, MD, CSV, etc.)
pipeline.ingest(["report.pdf", "notes.txt", "page.html"])

# Or ingest pre-parsed Document objects
from agentic_ai.core.models import Document
docs = [Document(content="...", metadata={"source": "manual"})]
pipeline.ingest_documents(docs)
```

**PDF capabilities:**
- Text extraction via PyMuPDF
- Image extraction with base64 encoding and LLM captioning (multimodal)
- Table detection and extraction to Markdown format via multimodal LLM

**Using individual parsers:**

```python
from agentic_ai.rag.ingestion import PDFParser, TextFileParser, HTMLParser

# PDF with image and table extraction
pdf_parser = PDFParser(
    llm=llm,                     # LLM for image captioning and table parsing
    extract_images=True,
    extract_tables=True,
)
documents = pdf_parser.parse("research_paper.pdf")

# HTML with BeautifulSoup
html_parser = HTMLParser()
documents = html_parser.parse("page.html")

# Plain text
text_parser = TextFileParser()
documents = text_parser.parse("notes.txt")
```

### Chunking Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `fixed` | Fixed character count with overlap | Simple, predictable chunks |
| `recursive` | Hierarchical separators (`\n\n` → `\n` → `. ` → ` `) | General purpose (default) |
| `sentence` | Sentence-boundary-aligned chunks | Natural language documents |
| `semantic` | Split at embedding-similarity drops | Topic-coherent chunks |

```python
from agentic_ai.rag.chunking import (
    FixedSizeChunker,
    RecursiveChunker,
    SentenceChunker,
    SemanticChunker,
)

# Fixed-size chunks
chunker = FixedSizeChunker(chunk_size=500, chunk_overlap=100)
chunks = chunker.chunk(document)

# Recursive (hierarchical separators)
chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=200)
chunks = chunker.chunk(document)

# Sentence-aligned
chunker = SentenceChunker(max_sentences=10, overlap_sentences=2)
chunks = chunker.chunk(document)

# Semantic (requires embedding model)
chunker = SemanticChunker(embedding=embedding, threshold=0.5)
chunks = chunker.chunk(document)
```

### Retrieval Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `simple` | Basic similarity search | Fast, straightforward retrieval |
| `mmr` | Maximal Marginal Relevance | Diverse results, reducing redundancy |
| `hybrid` | Keyword + semantic with RRF fusion | Best coverage (exact + semantic match) |
| `expansion` | LLM-generated query reformulations | Broad coverage across query aspects |
| `rerank` | LLM-scored re-ranking of candidates | Maximum precision |

```python
from agentic_ai.rag.retrieval import (
    SimpleRetriever,
    MMRRetriever,
    HybridRetriever,
    QueryExpansionRetriever,
    ReRankingRetriever,
)

# Simple similarity search
retriever = SimpleRetriever(vector_store=store)
results = retriever.retrieve("machine learning", k=5)

# MMR — balance relevance and diversity
retriever = MMRRetriever(vector_store=store, lambda_mult=0.7, fetch_k=20)
results = retriever.retrieve("machine learning", k=5)

# Hybrid — combine keyword + semantic search
retriever = HybridRetriever(
    vector_store=store,
    documents=all_docs,     # Full document corpus for keyword search
    alpha=0.7,              # 70% semantic, 30% keyword
)
results = retriever.retrieve("machine learning", k=5)

# Query expansion — generate multiple query variants
retriever = QueryExpansionRetriever(
    vector_store=store, llm=llm, num_expansions=3
)
results = retriever.retrieve("machine learning", k=5)

# Re-ranking — LLM re-scores initial candidates
retriever = ReRankingRetriever(
    vector_store=store, llm=llm, initial_k=20
)
results = retriever.retrieve("machine learning", k=5)
```

### Output Parsing

Parse LLM responses into structured data:

```python
from pydantic import BaseModel, Field
from agentic_ai.rag.output_parser import PydanticOutputParser

class Answer(BaseModel):
    answer: str = Field(description="The answer")
    confidence: float = Field(description="Confidence 0-1")
    sources: list[str] = Field(description="Source documents")

parser = PydanticOutputParser(model=Answer)

# Get format instructions to include in your prompt
instructions = parser.get_format_instructions()
print(instructions)

# Parse the LLM response
result = parser.parse(llm_response_text)
print(result.answer, result.confidence)
```

**Other parsers:**

```python
from agentic_ai.rag.output_parser import (
    JSONOutputParser,
    ListOutputParser,
    RegexOutputParser,
)

# JSON with schema validation
json_parser = JSONOutputParser(schema={"type": "object", "properties": {...}})
data = json_parser.parse(response)

# List parsing (numbered or bulleted)
list_parser = ListOutputParser()
items = list_parser.parse("1. First\n2. Second\n3. Third")  # ["First", "Second", "Third"]

# Regex-based extraction
regex_parser = RegexOutputParser(
    pattern=r"Name: (?P<name>.+)\nAge: (?P<age>\d+)"
)
match = regex_parser.parse("Name: Alice\nAge: 30")  # {"name": "Alice", "age": "30"}
```

### Evaluation

Evaluate RAG pipeline quality using LLM-as-judge:

```python
from agentic_ai.rag.evaluation import RAGEvaluator

evaluator = RAGEvaluator(llm=judge_llm)
scores = evaluator.evaluate(
    question="What is RAG?",
    answer="RAG combines retrieval with generation...",
    contexts=["RAG is a technique that..."],
    reference="RAG retrieves documents and uses them...",  # optional
)

print(f"Faithfulness:       {scores.faithfulness:.2f}")
print(f"Answer Relevance:   {scores.answer_relevance:.2f}")
print(f"Context Relevance:  {scores.context_relevance:.2f}")
print(f"Context Precision:  {scores.context_precision:.2f}")
print(f"Answer Correctness: {scores.answer_correctness:.2f}")
print(f"Overall Score:      {scores.overall_score():.2f}")
```

**Metrics explained:**
| Metric | What it measures |
|--------|-----------------|
| Faithfulness | Is the answer grounded in the provided context? |
| Answer Relevance | Does the answer address the question? |
| Context Relevance | Are the retrieved contexts relevant to the question? |
| Context Precision | What fraction of retrieved contexts are useful? |
| Answer Correctness | Does the answer match a reference answer? |

---

## Agents

### RAG Agent

```python
from agentic_ai.agents.rag import RAGAgent
from agentic_ai.core.models import Document

agent = RAGAgent(llm=llm, vector_store=store, top_k=5)

# Add documents
docs = [Document(content="...", metadata={"source": "docs"})]
store.add_documents(docs)

# Query
answer = agent.run("What does the documentation say about authentication?")
```

### ReAct Agent

```python
from agentic_ai.agents.react import ReActAgent

agent = ReActAgent(
    llm=llm,
    tools=[search_tool, scraper_tool],
    max_steps=10,
)
result = agent.run("Find the current weather in Tokyo and compare it to New York")
```

### Hierarchical Agent

```python
from agentic_ai.agents.hierarchical import HierarchicalAgent, WorkerAgent

researcher = WorkerAgent(
    llm=llm,
    name="Researcher",
    specialization="Finding information from the web",
    tools=[search_tool],
)
writer = WorkerAgent(
    llm=llm,
    name="Writer",
    specialization="Writing clear, structured content",
    tools=[],
)

manager = HierarchicalAgent(llm=llm, workers=[researcher, writer])
result = manager.run("Research and write a summary about Mars rovers")
```

### Planning Agent

```python
from agentic_ai.agents.planning import PlanningAgent

agent = PlanningAgent(llm=llm, tools=[scraper_tool], max_steps=15)
result = agent.run("Compare Python, Rust, and Go for backend development")
```

### Agent Memory

```python
from agentic_ai.core.memory import ConversationBufferMemory

memory = ConversationBufferMemory(max_turns=20)
agent = ReActAgent(llm=llm, tools=[...], memory=memory)

agent.run("My name is Alice and I work on ML projects")
agent.run("What kind of projects do I work on?")  # Remembers context
```

---

## LLM Providers

All providers implement the same `BaseLLM` interface with `chat()`, `chat_with_tools()`, and `stream()`.

```python
# OpenAI
from agentic_ai.llms.openai_llm import OpenAIChatModel
llm = OpenAIChatModel(api_key="sk-...", model="gpt-4o-mini")

# Google Gemini
from agentic_ai.llms.gemini_llm import GeminiChatModel
llm = GeminiChatModel(api_key="AIza...", model="gemini-pro")

# Anthropic Claude
from agentic_ai.llms.claude_llm import ClaudeChatModel
llm = ClaudeChatModel(api_key="sk-ant-...", model="claude-3-sonnet-20240229")

# Groq
from agentic_ai.llms.groq_llm import GroqChatModel
llm = GroqChatModel(api_key="gsk_...", model="llama3-70b-8192")
```

**Common methods:**
```python
from agentic_ai.core.models import Message, Role

messages = [Message(role=Role.USER, content="Hello!")]

# Standard chat
response = llm.chat(messages)
print(response.content)

# Chat with tool calling
response = llm.chat_with_tools(messages, tools=[search_tool])

# Streaming
for chunk in llm.stream(messages):
    print(chunk, end="")
```

---

## Embedding Providers

All providers implement `BaseEmbedding` with `embed_query()` and `embed_documents()`.

```python
# OpenAI
from agentic_ai.embeddings.openai_embedding import OpenAIEmbedding
emb = OpenAIEmbedding(api_key="sk-...", model="text-embedding-3-small")

# Google Gemini
from agentic_ai.embeddings.gemini_embedding import GeminiEmbedding
emb = GeminiEmbedding(api_key="AIza...", model="text-embedding-004")

# Voyage AI (for Claude users)
from agentic_ai.embeddings.claude_embedding import VoyageEmbedding
emb = VoyageEmbedding(api_key="pa-...", model="voyage-2")

# Groq-compatible
from agentic_ai.embeddings.groq_embedding import GroqEmbedding
emb = GroqEmbedding(api_key="gsk_...", model="nomic-embed-text-v1.5")
```

**Usage:**
```python
# Embed a single query
vector = emb.embed_query("What is machine learning?")
print(len(vector))  # e.g., 1536 for OpenAI

# Embed multiple documents
vectors = emb.embed_documents(["Document 1 text...", "Document 2 text..."])
print(len(vectors))  # 2
```

---

## Vector Stores

All stores implement `BaseVectorStore` with `add_documents()`, `similarity_search()`, and `delete()`.

```python
from agentic_ai.core.models import Document

# ChromaDB (recommended for development)
from agentic_ai.vectorstores.chroma_store import ChromaVectorStore
store = ChromaVectorStore(embedding=emb, collection_name="my_docs")

# Pinecone (cloud-managed)
from agentic_ai.vectorstores.pinecone_store import PineconeVectorStore
store = PineconeVectorStore(
    embedding=emb, api_key="...", index_name="my-index"
)

# PostgreSQL + pgvector
from agentic_ai.vectorstores.pgvector_store import PGVectorStore
store = PGVectorStore(
    embedding=emb,
    connection_string="postgresql://user:pass@localhost:5432/vectordb",
)

# FAISS (maximum speed)
from agentic_ai.vectorstores.faiss_store import FAISSVectorStore
store = FAISSVectorStore(embedding=emb, dimension=1536)
```

**Usage:**
```python
# Add documents
docs = [
    Document(content="RAG combines retrieval...", metadata={"source": "paper"}),
    Document(content="Vector databases store...", metadata={"source": "docs"}),
]
store.add_documents(docs)

# Search
results = store.similarity_search("What is RAG?", k=3)
for doc in results:
    print(doc.content, doc.metadata)

# Delete
store.delete(ids=["doc-id-1", "doc-id-2"])
```

---

## MCP Connectors (Tools)

Tools implement `BaseTool` and can be passed to any agent.

```python
# Google Search
from agentic_ai.mcp.google_search import GoogleSearchTool
search = GoogleSearchTool(api_key="AIza...", search_engine_id="...")

# Web Scraper
from agentic_ai.mcp.web_scraper import WebScraperTool
scraper = WebScraperTool()

# Email
from agentic_ai.mcp.email_tool import EmailTool
email = EmailTool(
    smtp_host="smtp.gmail.com", smtp_port=587,
    username="you@gmail.com", password="app-password",
)
result = email.execute(
    to="colleague@example.com",
    subject="Weekly Report",
    body="Here are the key findings...",
)

# HTTP requests
from agentic_ai.mcp.http_tool import HTTPRequestTool
http = HTTPRequestTool()

# File operations
from agentic_ai.mcp.file_tools import FileReaderTool, FileWriterTool
reader = FileReaderTool(allowed_directories=["./data"])
writer = FileWriterTool(allowed_directories=["./output"])
```

---

## Prompts

All prompt templates are centralized in `agentic_ai/prompts/templates.py` for easy customization:

```python
from agentic_ai.prompts.templates import (
    # RAG prompts
    RAG_SYSTEM_PROMPT,
    RAG_SYSTEM_PROMPT_WITH_SOURCES,
    RAG_CONVERSATIONAL_PROMPT,
    # Agent prompts
    REACT_SYSTEM_PROMPT,
    REACT_STEP_PROMPT,
    PLANNING_SYSTEM_PROMPT,
    PLANNING_CREATE_PLAN_PROMPT,
    PLANNING_EXECUTE_STEP_PROMPT,
    HIERARCHICAL_MANAGER_PROMPT,
    HIERARCHICAL_WORKER_PROMPT,
    HIERARCHICAL_SYNTHESIZE_PROMPT,
    # Retrieval prompts
    QUERY_EXPANSION_PROMPT,
    RERANKING_PROMPT,
    HYDE_PROMPT,
    # Evaluation prompts
    FAITHFULNESS_EVAL_PROMPT,
    ANSWER_RELEVANCE_EVAL_PROMPT,
    CONTEXT_RELEVANCE_EVAL_PROMPT,
    ANSWER_CORRECTNESS_EVAL_PROMPT,
)

# All templates use str.format() with named placeholders
prompt = RAG_SYSTEM_PROMPT.format(context="...", question="...")

# Override in PipelineConfig
config = PipelineConfig(
    system_prompt="Your custom prompt with {context} and {question} placeholders"
)
```

---

## API Server

A FastAPI-based REST API for accessing all framework capabilities over HTTP.

### Start the server

```bash
# Install API dependencies
pip install -e ".[api]"

# Start with uvicorn
uvicorn agentic_ai.api.server:app --host 0.0.0.0 --port 8000

# Or with auto-reload for development
uvicorn agentic_ai.api.server:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check with configured providers |
| `POST` | `/chat` | Chat completion with any LLM provider |
| `POST` | `/rag/ingest` | Upload and ingest documents |
| `POST` | `/rag/query` | Query the RAG pipeline |
| `GET` | `/config` | View current configuration |
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/redoc` | ReDoc API documentation |

### Example API calls

```bash
# Health check
curl http://localhost:8000/health

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "provider": "openai", "model": "gpt-4o-mini"}'

# Ingest documents
curl -X POST http://localhost:8000/rag/ingest \
  -F "files=@document.pdf" \
  -F "files=@notes.txt"

# RAG query
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "top_k": 5}'

# View config
curl http://localhost:8000/config
```

---

## Docker

```bash
# Build the image (installs all extras by default)
docker build -t agentic-ai .

# Build with specific extras only
docker build --build-arg EXTRAS="openai,chroma,rag,api" -t agentic-ai .

# Run with your .env file
docker run -p 8000:8000 --env-file .env agentic-ai

# The container starts the FastAPI server on port 8000
# Access Swagger docs at http://localhost:8000/docs
```

---

## Examples

Run the examples to see the framework in action:

```bash
# Install with extras for examples
pip install -e ".[openai,chroma,rag,api,dev]"

# --- Examples that work WITHOUT API keys (demo structure) ---

# RAG pipeline — chunking comparison, output parsing, evaluation setup
python examples/rag_pipeline_example.py

# ReAct agent — custom calculator tool demo
python examples/react_agent_example.py

# Hierarchical agent — standalone worker with word count tool
python examples/hierarchical_agent_example.py

# Planning agent — plan structure demonstration
python examples/planning_agent_example.py

# Agentic RAG — self-correcting RAG, structured output setup
python examples/agentic_rag_example.py

# --- Examples that REQUIRE API keys (set in .env) ---

# Quickstart — end-to-end chat, RAG, ReAct, hierarchical, memory
# Requires: OPENAI_API_KEY
python examples/quickstart.py
```

**What each example covers:**

| Example | Topics |
|---------|--------|
| `rag_pipeline_example.py` | Basic RAG, semantic chunking + MMR, hybrid search, query expansion, re-ranking, output parsing (Pydantic/JSON/list), evaluation, chunking strategy comparison |
| `react_agent_example.py` | Google Search agent, multi-tool agent (search + scraper + HTTP), custom calculator tool |
| `hierarchical_agent_example.py` | Basic manager-worker, research team (3 workers), standalone worker with custom tool |
| `planning_agent_example.py` | Basic planning, planning with HTTP tool, plan structure inspection |
| `agentic_rag_example.py` | Basic agentic RAG, self-correcting RAG with evaluation loop, structured output with Pydantic |
| `quickstart.py` | End-to-end examples with real API calls (OpenAI) |

---

## Architecture

```
agentic_ai/
├── core/                       # Abstract base classes & data models
│   ├── base_agent.py           # BaseAgent — common agent interface
│   ├── base_llm.py             # BaseLLM — chat model interface
│   ├── base_embedding.py       # BaseEmbedding — embedding interface
│   ├── base_vectorstore.py     # BaseVectorStore — vector DB interface
│   ├── base_tool.py            # BaseTool — tool/function interface
│   ├── memory.py               # ConversationBufferMemory, SlidingWindowMemory
│   └── models.py               # Message, ToolCall, Document, Role, etc.
├── llms/                       # LLM provider implementations
│   ├── openai_llm.py           # OpenAI (GPT-4, GPT-4o, GPT-3.5)
│   ├── gemini_llm.py           # Google Gemini
│   ├── claude_llm.py           # Anthropic Claude
│   └── groq_llm.py             # Groq (LLaMA, Mixtral)
├── embeddings/                 # Embedding provider implementations
│   ├── openai_embedding.py     # OpenAI text-embedding-3
│   ├── gemini_embedding.py     # Google text-embedding-004
│   ├── claude_embedding.py     # Voyage AI (Anthropic partner)
│   └── groq_embedding.py       # Groq/nomic embeddings
├── vectorstores/               # Vector database connectors
│   ├── chroma_store.py         # ChromaDB
│   ├── pinecone_store.py       # Pinecone
│   ├── pgvector_store.py       # PostgreSQL + pgvector
│   └── faiss_store.py          # FAISS
├── agents/                     # Agentic framework implementations
│   ├── rag.py                  # RAG agent
│   ├── react.py                # ReAct agent
│   ├── hierarchical.py         # Hierarchical multi-agent system
│   └── planning.py             # Plan-then-execute agent
├── rag/                        # Advanced RAG pipeline
│   ├── ingestion.py            # Document parsers (PDF, HTML, text)
│   ├── chunking.py             # Chunking strategies (4 types)
│   ├── retrieval.py            # Retrieval strategies (5 types)
│   ├── output_parser.py        # Output parsers (Pydantic, JSON, list, regex)
│   ├── evaluation.py           # RAG quality evaluation (5 metrics)
│   └── pipeline.py             # RAGPipeline orchestrator
├── prompts/                    # Centralized prompt templates
│   └── templates.py            # All prompts (RAG, agents, retrieval, eval)
├── api/                        # FastAPI REST API
│   └── server.py               # Endpoints: /health, /chat, /rag/*
├── mcp/                        # MCP tool connectors
│   ├── email_tool.py           # Email via SMTP
│   ├── google_search.py        # Google Custom Search
│   ├── web_scraper.py          # Web page scraper
│   ├── file_tools.py           # File read/write
│   └── http_tool.py            # HTTP requests
└── utils/                      # Shared utilities
    ├── logger.py               # Logging configuration
    ├── retry.py                # Retry with exponential backoff
    └── config.py               # Configuration management
```

---

## Design Principles

1. **Modular & Pluggable** — Swap any component (LLM, vector store, chunker, retriever, agent) without changing application code.
2. **Provider Agnostic** — Unified interfaces across all LLM and embedding providers.
3. **Minimal Dependencies** — Core requires only `requests` + `pydantic`. Everything else is optional extras.
4. **Type-Safe** — Full type annotations with Python 3.11+ syntax (`str | None`, `list[str]`).
5. **Well-Documented** — Every class, method, and parameter has detailed docstrings.
6. **Configurable** — All settings controllable via `.env`, `PipelineConfig`, or direct parameters.
7. **Production-Ready** — Built-in retry logic, error handling, logging, API server, Docker, and evaluation metrics.

---

## License

MIT
