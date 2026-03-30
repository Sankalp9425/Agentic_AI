# Agentic AI

A comprehensive, modular Python framework for building advanced AI agent systems.

## Features

### Agentic Frameworks
- **RAG** (Retrieval-Augmented Generation) — Grounds LLM responses in retrieved documents to reduce hallucinations.
- **ReAct** (Reasoning + Acting) — Iterative think-then-act loop for multi-step problem solving with tools.
- **Hierarchical Agents** — Manager agent delegates sub-tasks to specialized worker agents.
- **Planning Agent** — Creates an explicit step-by-step plan before executing actions.

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

### MCP Connectors
- **Email** — Send emails via SMTP (Gmail, Outlook, etc.).
- **Google Search** — Web search via Custom Search API.
- **Web Scraper** — Extract content from web pages.
- **File Reader/Writer** — Sandboxed local file operations.
- **HTTP Request** — General-purpose REST API integration.

## Installation

```bash
# Core framework (minimal dependencies)
pip install agentic-ai

# With specific providers
pip install agentic-ai[openai]           # OpenAI chat + embeddings
pip install agentic-ai[gemini]           # Google Gemini
pip install agentic-ai[claude]           # Anthropic Claude
pip install agentic-ai[groq]             # Groq fast inference

# With specific vector stores
pip install agentic-ai[chroma]           # ChromaDB
pip install agentic-ai[pinecone]         # Pinecone
pip install agentic-ai[pgvector]         # PostgreSQL + pgvector
pip install agentic-ai[faiss]            # FAISS

# Install everything
pip install agentic-ai[all]

# Development
pip install agentic-ai[dev]
```

## Quick Start

### ReAct Agent with Tools

```python
from agentic_ai.llms.openai_llm import OpenAIChatModel
from agentic_ai.agents.react import ReActAgent
from agentic_ai.mcp.google_search import GoogleSearchTool

# Initialize the LLM
llm = OpenAIChatModel(api_key="sk-...", model="gpt-4o")

# Create tools
search = GoogleSearchTool(api_key="AIza...", search_engine_id="...")

# Build and run the agent
agent = ReActAgent(llm=llm, tools=[search], max_steps=5)
result = agent.run("What are the latest developments in quantum computing?")
print(result)
```

### RAG Pipeline

```python
from agentic_ai.llms.openai_llm import OpenAIChatModel
from agentic_ai.embeddings.openai_embedding import OpenAIEmbedding
from agentic_ai.vectorstores.chroma_store import ChromaVectorStore
from agentic_ai.agents.rag import RAGAgent
from agentic_ai.core.models import Document

# Set up components
llm = OpenAIChatModel(api_key="sk-...", model="gpt-4o")
embedder = OpenAIEmbedding(api_key="sk-...", model="text-embedding-3-small")
store = ChromaVectorStore(embedding=embedder, collection_name="knowledge_base")

# Index documents
docs = [
    Document(content="RAG combines retrieval with generation...", metadata={"source": "paper"}),
    Document(content="Vector databases store embeddings...", metadata={"source": "docs"}),
]
store.add_documents(docs)

# Query with RAG
agent = RAGAgent(llm=llm, vector_store=store, top_k=3)
answer = agent.run("How does RAG work?")
print(answer)
```

### Hierarchical Multi-Agent System

```python
from agentic_ai.llms.openai_llm import OpenAIChatModel
from agentic_ai.agents.hierarchical import HierarchicalAgent, WorkerAgent
from agentic_ai.mcp.google_search import GoogleSearchTool

llm = OpenAIChatModel(api_key="sk-...", model="gpt-4o")

# Create specialized workers
researcher = WorkerAgent(
    name="researcher",
    description="Expert at finding and analyzing information from the web.",
    llm=llm,
    tools=[GoogleSearchTool(api_key="...", search_engine_id="...")],
)
writer = WorkerAgent(
    name="writer",
    description="Expert at writing clear, well-structured content.",
    llm=llm,
)

# Create the manager
manager = HierarchicalAgent(llm=llm, workers=[researcher, writer])
result = manager.run("Research and write a summary about the Mars rover missions")
print(result)
```

### Planning Agent

```python
from agentic_ai.llms.claude_llm import ClaudeChatModel
from agentic_ai.agents.planning import PlanningAgent
from agentic_ai.mcp.web_scraper import WebScraperTool

llm = ClaudeChatModel(api_key="sk-ant-...", model="claude-3-sonnet-20240229")
scraper = WebScraperTool()

agent = PlanningAgent(llm=llm, tools=[scraper], max_steps=15)
result = agent.run("Compare Python, Rust, and Go for backend development")
print(result)
```

### Using Memory

```python
from agentic_ai.core.memory import ConversationBufferMemory
from agentic_ai.agents.react import ReActAgent

memory = ConversationBufferMemory(max_turns=20)
agent = ReActAgent(llm=llm, tools=[...], memory=memory)

# First interaction
agent.run("My name is Alice and I work on ML projects")

# Second interaction - agent remembers context
agent.run("What kind of projects do I work on?")
```

### Sending Emails

```python
from agentic_ai.mcp.email_tool import EmailTool

email = EmailTool(
    smtp_host="smtp.gmail.com",
    smtp_port=587,
    username="your-email@gmail.com",
    password="your-app-password",  # Use App Password, not account password
)

result = email.execute(
    to="colleague@example.com",
    subject="Weekly Report",
    body="Here are the key findings from this week...",
)
```

### Configuring Logging

```python
from agentic_ai.utils.logger import setup_logging

# Enable debug logging to see agent reasoning
setup_logging(level="DEBUG", log_file="agent.log")
```

## Architecture

```
agentic_ai/
├── core/                    # Abstract base classes & data models
│   ├── base_agent.py        # BaseAgent - common agent interface
│   ├── base_llm.py          # BaseLLM - chat model interface
│   ├── base_embedding.py    # BaseEmbedding - embedding interface
│   ├── base_vectorstore.py  # BaseVectorStore - vector DB interface
│   ├── base_tool.py         # BaseTool - tool/function interface
│   ├── memory.py            # MemoryStore - agent memory
│   └── models.py            # Message, ToolCall, Document, etc.
├── llms/                    # LLM provider implementations
│   ├── openai_llm.py        # OpenAI (GPT-4, GPT-4o, etc.)
│   ├── gemini_llm.py        # Google Gemini
│   ├── claude_llm.py        # Anthropic Claude
│   └── groq_llm.py          # Groq (LLaMA, Mixtral)
├── embeddings/              # Embedding provider implementations
│   ├── openai_embedding.py  # OpenAI embeddings
│   ├── gemini_embedding.py  # Google Gemini embeddings
│   ├── claude_embedding.py  # Voyage AI (Anthropic partner)
│   └── groq_embedding.py    # Groq/OpenAI-compatible embeddings
├── vectorstores/            # Vector database connectors
│   ├── chroma_store.py      # ChromaDB
│   ├── pinecone_store.py    # Pinecone
│   ├── pgvector_store.py    # PostgreSQL + pgvector
│   └── faiss_store.py       # FAISS
├── agents/                  # Agentic framework implementations
│   ├── rag.py               # RAG (Retrieval-Augmented Generation)
│   ├── react.py             # ReAct (Reasoning + Acting)
│   ├── hierarchical.py      # Hierarchical multi-agent system
│   └── planning.py          # Plan-then-execute agent
├── mcp/                     # MCP tool connectors
│   ├── email_tool.py        # Email via SMTP
│   ├── google_search.py     # Google Custom Search
│   ├── web_scraper.py       # Web page scraper
│   ├── file_tools.py        # File read/write
│   └── http_tool.py         # HTTP requests
└── utils/                   # Shared utilities
    ├── logger.py            # Logging configuration
    ├── retry.py             # Retry with exponential backoff
    └── config.py            # Configuration management
```

## Design Principles

1. **Modular & Pluggable**: Swap any component (LLM, vector store, agent type) without changing application code.
2. **Provider Agnostic**: Unified interfaces across all LLM and embedding providers.
3. **Minimal Dependencies**: Core framework requires only `requests`. Provider-specific dependencies are optional extras.
4. **Type-Safe**: Full type annotations with Python 3.11+ syntax.
5. **Well-Documented**: Every class, method, and parameter has detailed docstrings.
6. **Production-Ready Patterns**: Built-in retry logic, error handling, logging, and configuration management.

## License

MIT
