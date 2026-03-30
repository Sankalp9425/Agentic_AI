"""
Planning Agent Examples.

Demonstrates the Planning Agent (plan-then-execute) pattern:
1. Basic planning agent that creates and follows a plan.
2. Planning agent with tool integration.
3. Dynamic re-planning when steps fail.

The planning pattern:
    Task → Create Plan → Execute Step 1 → Execute Step 2 → ... → Synthesize

Requirements:
    pip install -e ".[openai]"

Usage:
    export OPENAI_API_KEY="sk-..."
    python examples/planning_agent_example.py
"""

import os
import sys

# Add the parent directory to the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def example_basic_planning() -> None:
    """
    Example 1: Basic planning agent.

    Creates a step-by-step plan and executes each step.
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic Planning Agent")
    print("=" * 60)

    from agentic_ai.agents.planning import PlanningAgent
    from agentic_ai.llms.openai_llm import OpenAIChatModel

    api_key = os.getenv("OPENAI_API_KEY", "")
    llm = OpenAIChatModel(api_key=api_key, model="gpt-4o-mini")

    # Create the planning agent.
    agent = PlanningAgent(
        llm=llm,
        tools=[],
        max_steps=10,
    )

    # Run a complex task that benefits from planning.
    result = agent.run(
        "Create a comprehensive comparison of Python, JavaScript, and Rust "
        "for backend web development, covering performance, ecosystem, "
        "learning curve, and use cases."
    )
    print(f"\nResult:\n{result}")


def example_planning_with_tools() -> None:
    """
    Example 2: Planning agent with tools.

    Uses tools to gather information during plan execution.
    """
    print("\n" + "=" * 60)
    print("Example 2: Planning Agent with Tools")
    print("=" * 60)

    from agentic_ai.agents.planning import PlanningAgent
    from agentic_ai.llms.openai_llm import OpenAIChatModel
    from agentic_ai.mcp.http_tool import HTTPRequestTool

    api_key = os.getenv("OPENAI_API_KEY", "")
    llm = OpenAIChatModel(api_key=api_key, model="gpt-4o-mini")

    # Create the agent with an HTTP tool.
    _agent = PlanningAgent(
        llm=llm,
        tools=[HTTPRequestTool()],
        max_steps=8,
    )

    print("Planning agent created with HTTP request tool.")
    print("The agent creates a plan first, then executes each step.")
    print("During execution, it can make HTTP API calls as needed.")


def example_plan_structure() -> None:
    """
    Example 3: Understanding plan structure.

    Demonstrates the internal plan data structure used by
    the planning agent.
    """
    print("\n" + "=" * 60)
    print("Example 3: Plan Structure")
    print("=" * 60)

    # Simulated plan structure.
    plan_steps = [
        {"step": 1, "action": "Research Python backend frameworks (Django, FastAPI, Flask)"},
        {"step": 2, "action": "Research JavaScript backend frameworks (Express, Nest, Fastify)"},
        {"step": 3, "action": "Research Rust backend frameworks (Actix, Axum, Rocket)"},
        {"step": 4, "action": "Compare performance benchmarks across languages"},
        {"step": 5, "action": "Evaluate ecosystem maturity and package availability"},
        {"step": 6, "action": "Assess learning curve for each language"},
        {"step": 7, "action": "Synthesize findings into a structured comparison"},
    ]

    print("Planning agent creates a structured plan like this:\n")
    for step in plan_steps:
        print(f"  Step {step['step']}: {step['action']}")

    print(f"\nTotal steps: {len(plan_steps)}")
    print("Each step is executed sequentially, with results passed to subsequent steps.")
    print("If a step fails, the agent can re-plan the remaining steps.")


if __name__ == "__main__":
    print("=" * 60)
    print("  Agentic AI — Planning Agent Examples")
    print("=" * 60)

    # This example works without API keys.
    example_plan_structure()

    # These examples require an API key.
    if os.getenv("OPENAI_API_KEY"):
        example_basic_planning()
        example_planning_with_tools()
    else:
        print("\nSet OPENAI_API_KEY to run the LLM-powered examples.")
