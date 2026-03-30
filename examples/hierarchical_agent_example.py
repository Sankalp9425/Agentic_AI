"""
Hierarchical Agent Examples.

Demonstrates the hierarchical (manager-worker) agent pattern:
1. Basic hierarchical agent with specialized workers.
2. Research team simulation with manager delegation.
3. Multi-level hierarchy with sub-managers.

The hierarchical pattern:
    Manager Agent → delegates subtasks → Worker Agents → results → Manager synthesizes

Requirements:
    pip install -e ".[openai]"

Usage:
    export OPENAI_API_KEY="sk-..."
    python examples/hierarchical_agent_example.py
"""

import os
import sys

# Add the parent directory to the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def example_basic_hierarchical() -> None:
    """
    Example 1: Basic hierarchical agent setup.

    Creates a manager with two specialized worker agents.
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic Hierarchical Agent")
    print("=" * 60)

    from agentic_ai.agents.hierarchical import HierarchicalAgent, WorkerAgent
    from agentic_ai.llms.openai_llm import OpenAIChatModel

    api_key = os.getenv("OPENAI_API_KEY", "")
    llm = OpenAIChatModel(api_key=api_key, model="gpt-4o-mini")

    # Create specialized worker agents.
    researcher = WorkerAgent(
        llm=llm,
        name="Researcher",
        specialization="Finding and summarizing information from various sources",
        tools=[],
    )

    writer = WorkerAgent(
        llm=llm,
        name="Writer",
        specialization="Writing clear, well-structured content based on research",
        tools=[],
    )

    # Create the manager that delegates to workers.
    manager = HierarchicalAgent(
        llm=llm,
        workers=[researcher, writer],
    )

    # Run a task.
    result = manager.run(
        "Research the benefits of microservices architecture and write a brief summary."
    )
    print(f"\nResult:\n{result}")


def example_research_team() -> None:
    """
    Example 2: Research team simulation.

    Simulates a research team with a project manager, data analyst,
    and report writer.
    """
    print("\n" + "=" * 60)
    print("Example 2: Research Team")
    print("=" * 60)

    from agentic_ai.agents.hierarchical import HierarchicalAgent, WorkerAgent
    from agentic_ai.llms.openai_llm import OpenAIChatModel

    api_key = os.getenv("OPENAI_API_KEY", "")
    llm = OpenAIChatModel(api_key=api_key, model="gpt-4o-mini")

    # Specialized team members.
    data_analyst = WorkerAgent(
        llm=llm,
        name="DataAnalyst",
        specialization=(
            "Analyzing data, identifying trends, and providing statistical insights. "
            "Expert in data visualization and interpretation."
        ),
        tools=[],
    )

    domain_expert = WorkerAgent(
        llm=llm,
        name="DomainExpert",
        specialization=(
            "Providing deep domain knowledge and context. "
            "Expert in industry trends and best practices."
        ),
        tools=[],
    )

    report_writer = WorkerAgent(
        llm=llm,
        name="ReportWriter",
        specialization=(
            "Writing professional reports, executive summaries, and presentations. "
            "Expert in clear communication and document structure."
        ),
        tools=[],
    )

    # Create the project manager.
    _manager = HierarchicalAgent(
        llm=llm,
        workers=[data_analyst, domain_expert, report_writer],
    )

    print("Research team created:")
    print("  - Manager (project lead)")
    print("  - DataAnalyst (data & trends)")
    print("  - DomainExpert (industry knowledge)")
    print("  - ReportWriter (documentation)")
    print("\nThe manager delegates subtasks to the most appropriate worker.")


def example_worker_agent_standalone() -> None:
    """
    Example 3: Using a WorkerAgent standalone.

    Worker agents can also be used independently without
    a manager for simple, specialized tasks.
    """
    print("\n" + "=" * 60)
    print("Example 3: Standalone Worker Agent")
    print("=" * 60)

    from agentic_ai.core.base_tool import BaseTool
    from agentic_ai.core.models import ToolResult

    # Create a simple tool for the worker.
    class WordCountTool(BaseTool):
        """Counts words in a text."""

        @property
        def name(self) -> str:
            return "word_counter"

        @property
        def description(self) -> str:
            return "Counts the number of words in the given text."

        @property
        def parameters_schema(self) -> dict:
            return {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to count words in"}
                },
                "required": ["text"],
            }

        def execute(self, **kwargs: object) -> ToolResult:
            text = str(kwargs.get("text", ""))
            count = len(text.split())
            return ToolResult(
                tool_name=self.name,
                output=f"Word count: {count}",
                success=True,
            )

    # Test the tool directly.
    tool = WordCountTool()
    result = tool.execute(text="Hello world this is a test")
    print(f"Tool test: {result.output}")
    print(f"Tool schema: {tool.parameters_schema}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Agentic AI — Hierarchical Agent Examples")
    print("=" * 60)

    # This example works without API keys.
    example_worker_agent_standalone()

    # These examples require an API key.
    if os.getenv("OPENAI_API_KEY"):
        example_basic_hierarchical()
        example_research_team()
    else:
        print("\nSet OPENAI_API_KEY to run the LLM-powered examples.")
