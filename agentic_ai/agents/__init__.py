"""
Agent framework implementations.

This module contains concrete implementations of different agentic paradigms,
each offering a distinct approach to autonomous task completion. All agents
inherit from BaseAgent and can be used interchangeably.

Supported Paradigms:
    - RAG (Retrieval-Augmented Generation): Retrieves relevant context from
      a vector store before generating responses. Best for knowledge-intensive tasks.
    - ReAct (Reasoning + Acting): Alternates between thinking and tool use in
      an iterative loop. Best for multi-step reasoning with external tools.
    - Hierarchical: A manager agent delegates sub-tasks to specialized worker
      agents. Best for complex tasks requiring diverse expertise.
    - Planning: Creates an explicit step-by-step plan before execution, then
      follows the plan. Best for tasks requiring structured approaches.

Example:
    >>> from agentic_ai.agents.react import ReActAgent
    >>> agent = ReActAgent(llm=my_llm, tools=[search, calculator])
    >>> result = agent.run("What is the GDP of Japan divided by its population?")
"""
