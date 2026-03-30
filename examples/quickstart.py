"""
Agentic AI Quick Start Examples.

This file demonstrates the core capabilities of the Agentic AI framework
with runnable examples for each major feature. Each example is wrapped in
a function that can be run independently.

Before running, set the required environment variables:
    export OPENAI_API_KEY="sk-..."
    export GOOGLE_API_KEY="AIza..."
    export ANTHROPIC_API_KEY="sk-ant-..."
    export GROQ_API_KEY="gsk_..."

Usage:
    python examples/quickstart.py
"""

from agentic_ai.core.memory import ConversationBufferMemory
from agentic_ai.core.models import Document, Message, Role
from agentic_ai.utils.config import get_api_key
from agentic_ai.utils.logger import setup_logging


def example_react_agent() -> None:
    """
    Demonstrate a ReAct agent with web search capability.

    The agent reasons step-by-step, using Google Search to gather information
    before providing a final answer.
    """
    print("\n=== ReAct Agent Example ===\n")

    from agentic_ai.agents.react import ReActAgent
    from agentic_ai.llms.openai_llm import OpenAIChatModel
    from agentic_ai.mcp.google_search import GoogleSearchTool

    # Initialize components.
    llm = OpenAIChatModel(
        api_key=get_api_key("OPENAI_API_KEY"),
        model="gpt-4o",
        temperature=0.3,
    )

    search = GoogleSearchTool(
        api_key=get_api_key("GOOGLE_API_KEY"),
        search_engine_id=get_api_key("GOOGLE_CSE_ID"),
    )

    # Create and run the agent.
    agent = ReActAgent(llm=llm, tools=[search], max_steps=5)
    result = agent.run("What are the top 3 Python web frameworks in 2024?")
    print(result)


def example_rag_pipeline() -> None:
    """
    Demonstrate a RAG (Retrieval-Augmented Generation) pipeline.

    Documents are indexed in ChromaDB, then the RAG agent retrieves
    relevant context before generating an answer.
    """
    print("\n=== RAG Pipeline Example ===\n")

    from agentic_ai.agents.rag import RAGAgent
    from agentic_ai.embeddings.openai_embedding import OpenAIEmbedding
    from agentic_ai.llms.openai_llm import OpenAIChatModel
    from agentic_ai.vectorstores.chroma_store import ChromaVectorStore

    api_key = get_api_key("OPENAI_API_KEY")

    # Initialize components.
    llm = OpenAIChatModel(api_key=api_key, model="gpt-4o")
    embedder = OpenAIEmbedding(api_key=api_key, model="text-embedding-3-small")
    store = ChromaVectorStore(embedding=embedder, collection_name="demo_docs")

    # Index some sample documents.
    documents = [
        Document(
            content=(
                "Retrieval-Augmented Generation (RAG) combines information retrieval "
                "with text generation. It first retrieves relevant documents from a "
                "knowledge base, then uses them as context for the LLM to generate "
                "a grounded, factual response."
            ),
            metadata={"source": "ai_handbook", "topic": "rag"},
        ),
        Document(
            content=(
                "Vector databases store high-dimensional vectors (embeddings) and "
                "enable fast similarity search. Popular options include ChromaDB, "
                "Pinecone, FAISS, and PGVector. They are essential for RAG pipelines."
            ),
            metadata={"source": "db_guide", "topic": "vector_databases"},
        ),
        Document(
            content=(
                "The ReAct framework interleaves reasoning (thinking) with acting "
                "(tool use). The agent thinks about what to do, uses a tool to "
                "gather information, observes the result, and repeats until it "
                "can provide a final answer."
            ),
            metadata={"source": "ai_handbook", "topic": "react"},
        ),
    ]
    store.add_documents(documents)

    # Create and run the RAG agent.
    agent = RAGAgent(llm=llm, vector_store=store, top_k=2)
    answer = agent.run("What is RAG and how does it use vector databases?")
    print(answer)


def example_hierarchical_agents() -> None:
    """
    Demonstrate a hierarchical multi-agent system.

    A manager agent coordinates a researcher and a writer to produce
    a comprehensive response.
    """
    print("\n=== Hierarchical Agent Example ===\n")

    from agentic_ai.agents.hierarchical import HierarchicalAgent, WorkerAgent
    from agentic_ai.llms.openai_llm import OpenAIChatModel

    llm = OpenAIChatModel(
        api_key=get_api_key("OPENAI_API_KEY"),
        model="gpt-4o",
    )

    # Create specialized workers.
    researcher = WorkerAgent(
        name="researcher",
        description="Expert at analyzing technical topics and providing detailed facts.",
        llm=llm,
    )
    writer = WorkerAgent(
        name="writer",
        description="Expert at writing clear, engaging, well-structured content.",
        llm=llm,
    )

    # Create the manager and run.
    manager = HierarchicalAgent(llm=llm, workers=[researcher, writer])
    result = manager.run("Explain the differences between RAG and fine-tuning for LLMs")
    print(result)


def example_planning_agent() -> None:
    """
    Demonstrate a planning agent that creates a plan before executing.

    The agent first generates a step-by-step plan, then executes each step
    sequentially, and finally synthesizes the results.
    """
    print("\n=== Planning Agent Example ===\n")

    from agentic_ai.agents.planning import PlanningAgent
    from agentic_ai.llms.openai_llm import OpenAIChatModel

    llm = OpenAIChatModel(
        api_key=get_api_key("OPENAI_API_KEY"),
        model="gpt-4o",
    )

    agent = PlanningAgent(llm=llm, max_steps=10)
    result = agent.run("Compare Python, JavaScript, and Rust for building CLI tools")
    print(result)


def example_memory() -> None:
    """
    Demonstrate agent memory for context persistence.

    The agent remembers information from previous interactions using
    a conversation buffer memory.
    """
    print("\n=== Memory Example ===\n")

    # Create a conversation buffer that remembers the last 10 turns.
    memory = ConversationBufferMemory(max_turns=10)

    # Simulate storing interactions.
    memory.store(
        "What is my project about?",
        "Your project is about building an AI-powered document analyzer.",
    )
    memory.store(
        "What tech stack am I using?",
        "You're using Python, FastAPI, and ChromaDB.",
    )

    # Retrieve context for a new query.
    context = memory.retrieve("Tell me about my project setup")
    print("Retrieved context:")
    print(context)


def example_direct_llm_usage() -> None:
    """
    Demonstrate using LLM providers directly without agents.

    Shows how to use OpenAI's chat completion API directly for simple
    question-answering without the agent framework.
    """
    print("\n=== Direct LLM Usage Example ===\n")

    from agentic_ai.llms.openai_llm import OpenAIChatModel

    llm = OpenAIChatModel(
        api_key=get_api_key("OPENAI_API_KEY"),
        model="gpt-4o",
        temperature=0.5,
    )

    messages = [
        Message(role=Role.SYSTEM, content="You are a Python expert. Be concise."),
        Message(role=Role.USER, content="What are the key features of Python 3.12?"),
    ]

    response = llm.chat(messages)
    print(f"Response: {response.content}")
    print(f"Metadata: {response.metadata}")


if __name__ == "__main__":
    # Configure logging to see framework internals.
    setup_logging(level="INFO")

    # Run the memory example (no API keys needed).
    example_memory()

    # The following examples require API keys.
    # Uncomment the examples you want to run after setting up API keys.

    # example_direct_llm_usage()
    # example_react_agent()
    # example_rag_pipeline()
    # example_hierarchical_agents()
    # example_planning_agent()

    print("\nDone! Uncomment other examples in __main__ to try them.")
