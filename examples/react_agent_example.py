"""
ReAct Agent Examples.

Demonstrates the ReAct (Reasoning + Acting) agent framework with
various tool configurations:
1. Basic ReAct agent with Google Search.
2. ReAct agent with multiple tools (search + web scraper).
3. ReAct agent with custom tools.

The ReAct pattern follows the cycle:
    Thought → Action → Observation → Thought → ... → Final Answer

Requirements:
    pip install -e ".[openai]"

Usage:
    export OPENAI_API_KEY="sk-..."
    export GOOGLE_SEARCH_API_KEY="..."
    export GOOGLE_SEARCH_ENGINE_ID="..."
    python examples/react_agent_example.py
"""

import os
import sys

# Add the parent directory to the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def example_react_with_search() -> None:
    """
    Example 1: ReAct agent with Google Search tool.

    The agent can search the web to find information before
    providing an answer.
    """
    print("\n" + "=" * 60)
    print("Example 1: ReAct Agent with Google Search")
    print("=" * 60)

    from agentic_ai.agents.react import ReActAgent
    from agentic_ai.llms.openai_llm import OpenAIChatModel
    from agentic_ai.mcp.google_search import GoogleSearchTool

    api_key = os.getenv("OPENAI_API_KEY", "")
    search_api_key = os.getenv("GOOGLE_SEARCH_API_KEY", "")
    search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID", "")

    llm = OpenAIChatModel(api_key=api_key, model="gpt-4o-mini")

    # Create tools.
    search_tool = GoogleSearchTool(
        api_key=search_api_key,
        search_engine_id=search_engine_id,
    )

    # Create the ReAct agent.
    agent = ReActAgent(
        llm=llm,
        tools=[search_tool],
        max_iterations=5,
    )

    # Run the agent.
    result = agent.run("What are the latest developments in quantum computing?")
    print(f"\nFinal Answer: {result}")


def example_react_multi_tool() -> None:
    """
    Example 2: ReAct agent with multiple tools.

    Combines Google Search with a Web Scraper so the agent can
    search for information and then scrape detailed content.
    """
    print("\n" + "=" * 60)
    print("Example 2: ReAct Agent with Multiple Tools")
    print("=" * 60)

    from agentic_ai.agents.react import ReActAgent
    from agentic_ai.llms.openai_llm import OpenAIChatModel
    from agentic_ai.mcp.google_search import GoogleSearchTool
    from agentic_ai.mcp.http_tool import HTTPRequestTool
    from agentic_ai.mcp.web_scraper import WebScraperTool

    api_key = os.getenv("OPENAI_API_KEY", "")
    search_api_key = os.getenv("GOOGLE_SEARCH_API_KEY", "")
    search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID", "")

    llm = OpenAIChatModel(api_key=api_key, model="gpt-4o-mini")

    # Create multiple tools.
    tools = [
        GoogleSearchTool(api_key=search_api_key, search_engine_id=search_engine_id),
        WebScraperTool(),
        HTTPRequestTool(),
    ]

    # Create the agent with a higher iteration limit.
    _agent = ReActAgent(
        llm=llm,
        tools=tools,
        max_iterations=8,
    )

    print("ReAct agent created with 3 tools: Search, WebScraper, HTTP")
    print("Agent can search, scrape pages, and make API calls.")


def example_react_custom_tool() -> None:
    """
    Example 3: ReAct agent with a custom tool.

    Demonstrates creating a custom tool for the ReAct agent.
    """
    print("\n" + "=" * 60)
    print("Example 3: ReAct Agent with Custom Tool")
    print("=" * 60)

    from agentic_ai.core.base_tool import BaseTool
    from agentic_ai.core.models import ToolResult

    # Define a custom calculator tool.
    class CalculatorTool(BaseTool):
        """A simple calculator tool for arithmetic operations."""

        @property
        def name(self) -> str:
            return "calculator"

        @property
        def description(self) -> str:
            return (
                "Performs arithmetic calculations. "
                "Input should be a mathematical expression like '2 + 3 * 4'."
            )

        @property
        def parameters_schema(self) -> dict:
            return {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate",
                    }
                },
                "required": ["expression"],
            }

        def execute(self, **kwargs: object) -> ToolResult:
            """Evaluate a math expression safely."""
            expression = str(kwargs.get("expression", ""))
            try:
                # Use a restricted eval for safety.
                allowed_chars = set("0123456789+-*/().% ")
                if not all(c in allowed_chars for c in expression):
                    return ToolResult(
                        tool_name=self.name,
                        output=f"Invalid expression: {expression}",
                        success=False,
                    )
                result = eval(expression)  # noqa: S307
                return ToolResult(
                    tool_name=self.name,
                    output=str(result),
                    success=True,
                )
            except Exception as e:
                return ToolResult(
                    tool_name=self.name,
                    output=f"Error: {e}",
                    success=False,
                )

    # Create the tool.
    calc_tool = CalculatorTool()

    print(f"Custom tool created: {calc_tool.name}")
    print(f"Description: {calc_tool.description}")

    # Test the tool directly.
    result = calc_tool.execute(expression="2 + 3 * 4")
    print(f"Calculator test: 2 + 3 * 4 = {result.output}")

    result = calc_tool.execute(expression="(100 - 37) / 9")
    print(f"Calculator test: (100 - 37) / 9 = {result.output}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Agentic AI — ReAct Agent Examples")
    print("=" * 60)

    # This example works without API keys.
    example_react_custom_tool()

    # These examples require API keys.
    if os.getenv("OPENAI_API_KEY") and os.getenv("GOOGLE_SEARCH_API_KEY"):
        example_react_with_search()
        example_react_multi_tool()
    else:
        print("\nSet OPENAI_API_KEY and GOOGLE_SEARCH_API_KEY for full examples.")
