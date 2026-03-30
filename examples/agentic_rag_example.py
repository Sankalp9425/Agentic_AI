"""
Agentic RAG Examples.

Demonstrates RAG combined with agentic capabilities:
1. RAG Agent that retrieves and reasons before answering.
2. Agentic RAG with tool augmentation (retrieve + search + reason).
3. Self-correcting RAG with evaluation feedback loop.

Agentic RAG goes beyond simple retrieve-then-generate by adding
reasoning, tool use, and self-correction loops.

Requirements:
    pip install -e ".[openai,chroma]"

Usage:
    export OPENAI_API_KEY="sk-..."
    python examples/agentic_rag_example.py
"""

import os
import sys

# Add the parent directory to the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def example_basic_agentic_rag() -> None:
    """
    Example 1: Basic Agentic RAG.

    Uses the RAG agent that retrieves documents and reasons
    about them before generating an answer.
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic Agentic RAG")
    print("=" * 60)

    from agentic_ai.agents.rag import RAGAgent
    from agentic_ai.core.models import Document
    from agentic_ai.embeddings.openai_embedding import OpenAIEmbedding
    from agentic_ai.llms.openai_llm import OpenAIChatModel
    from agentic_ai.vectorstores.chroma_store import ChromaVectorStore

    api_key = os.getenv("OPENAI_API_KEY", "")
    llm = OpenAIChatModel(api_key=api_key, model="gpt-4o-mini")
    embedding = OpenAIEmbedding(api_key=api_key)
    store = ChromaVectorStore(embedding=embedding, collection_name="agentic_rag")

    # Add some documents to the knowledge base.
    docs = [
        Document(
            content=(
                "Transformers are a neural network architecture introduced in 2017 "
                "by Vaswani et al. in the paper 'Attention Is All You Need'. They "
                "use self-attention mechanisms to process sequences in parallel, "
                "making them much faster to train than RNNs and LSTMs."
            ),
            metadata={"source": "ml_textbook", "chapter": "transformers"},
        ),
        Document(
            content=(
                "GPT (Generative Pre-trained Transformer) models are a family of "
                "large language models developed by OpenAI. GPT-4, released in 2023, "
                "is a multimodal model that can process both text and images. "
                "It demonstrates human-level performance on many professional benchmarks."
            ),
            metadata={"source": "ai_report", "year": "2023"},
        ),
        Document(
            content=(
                "Fine-tuning is the process of taking a pre-trained model and further "
                "training it on a specific dataset for a particular task. This is more "
                "efficient than training from scratch because the model has already "
                "learned general language patterns during pre-training."
            ),
            metadata={"source": "ml_textbook", "chapter": "fine_tuning"},
        ),
    ]

    store.add_documents(docs)

    # Create the RAG agent.
    rag_agent = RAGAgent(
        llm=llm,
        vector_store=store,
        top_k=3,
    )

    # Ask a question.
    result = rag_agent.run("What are Transformers and how are they used in GPT?")
    print(f"\nAnswer: {result}")


def example_self_correcting_rag() -> None:
    """
    Example 2: Self-correcting RAG with evaluation.

    Demonstrates a RAG pipeline that evaluates its own output
    and retries with different retrieval if quality is low.
    """
    print("\n" + "=" * 60)
    print("Example 2: Self-Correcting RAG Concept")
    print("=" * 60)

    from agentic_ai.rag.evaluation import EvaluationResult

    # Simulate the self-correction loop.
    print("\nSelf-correcting RAG workflow:")
    print("1. User asks a question")
    print("2. Retrieve relevant documents")
    print("3. Generate an answer")
    print("4. Evaluate the answer quality")

    # First attempt — low faithfulness.
    attempt1 = EvaluationResult(
        faithfulness=0.4,
        answer_relevance=0.8,
        context_relevance=0.5,
    )
    print(f"\nAttempt 1 — Overall: {attempt1.overall_score():.2f}")
    print(f"  Faithfulness: {attempt1.faithfulness} (LOW — re-retrieve)")

    # Second attempt — better retrieval, higher quality.
    attempt2 = EvaluationResult(
        faithfulness=0.9,
        answer_relevance=0.85,
        context_relevance=0.88,
    )
    print(f"\nAttempt 2 — Overall: {attempt2.overall_score():.2f}")
    print(f"  Faithfulness: {attempt2.faithfulness} (GOOD — accept)")

    print("\nThe self-correcting loop:")
    print("  if score < threshold:")
    print("    → expand query, retrieve more docs, regenerate")
    print("  else:")
    print("    → return answer to user")


def example_rag_with_output_parsing() -> None:
    """
    Example 3: RAG with structured output parsing.

    Combines RAG retrieval with Pydantic output parsing to
    produce structured answers.
    """
    print("\n" + "=" * 60)
    print("Example 3: RAG + Structured Output")
    print("=" * 60)

    from pydantic import BaseModel, Field

    from agentic_ai.rag.output_parser import PydanticOutputParser

    # Define the structured output format.
    class TechnicalAnswer(BaseModel):
        """Structured technical answer from RAG."""
        answer: str = Field(description="The main answer to the question")
        confidence: float = Field(description="Confidence score from 0 to 1")
        sources: list[str] = Field(description="List of source documents used")
        caveats: list[str] = Field(
            default_factory=list,
            description="Any caveats or limitations of the answer",
        )

    # Create the parser.
    parser = PydanticOutputParser(model=TechnicalAnswer)

    # Get format instructions to include in the RAG prompt.
    instructions = parser.get_format_instructions()
    print("Format instructions for the LLM prompt:")
    print(instructions[:200] + "...")

    # Simulate parsing a structured LLM response.
    simulated_response = '''```json
{
    "answer": "Transformers use self-attention mechanisms to process sequences in parallel, making them significantly faster to train than sequential models like RNNs.",
    "confidence": 0.92,
    "sources": ["ml_textbook/chapter/transformers", "attention_is_all_you_need_paper"],
    "caveats": ["Training large transformers still requires significant computational resources"]
}
```'''

    result = parser.parse(simulated_response)
    print(f"\nParsed answer: {result.answer[:80]}...")
    print(f"Confidence: {result.confidence}")
    print(f"Sources: {result.sources}")
    print(f"Caveats: {result.caveats}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Agentic AI — Agentic RAG Examples")
    print("=" * 60)

    # These examples work without API keys.
    example_self_correcting_rag()
    example_rag_with_output_parsing()

    # This example requires an OPENAI_API_KEY.
    if os.getenv("OPENAI_API_KEY"):
        example_basic_agentic_rag()
    else:
        print("\nSet OPENAI_API_KEY to run the LLM-powered examples.")
