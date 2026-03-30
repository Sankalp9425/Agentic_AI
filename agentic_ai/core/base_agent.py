"""
Abstract base class for all agent implementations.

Agents are the central orchestrators in the Agentic AI framework. They combine
an LLM (for reasoning), tools (for acting), and memory (for context) into an
autonomous system that can accomplish complex tasks through iterative reasoning
and action cycles.

Different agent architectures implement different reasoning strategies:
    - ReAct agents alternate between thinking and acting.
    - RAG agents retrieve relevant context before responding.
    - Hierarchical agents delegate sub-tasks to child agents.
    - Planning agents create explicit plans before execution.

All agents share a common interface defined by this base class, enabling
uniform orchestration and monitoring regardless of the underlying strategy.

Example:
    >>> from agentic_ai.agents.react import ReActAgent
    >>> agent = ReActAgent(llm=my_llm, tools=[search_tool, calc_tool])
    >>> result = agent.run("What is the population of Tokyo divided by 2?")
"""

from abc import ABC, abstractmethod
from typing import Any

from agentic_ai.core.base_llm import BaseLLM
from agentic_ai.core.base_tool import BaseTool
from agentic_ai.core.memory import MemoryStore
from agentic_ai.core.models import AgentState, AgentStatus, Message, Role


class BaseAgent(ABC):
    """
    Abstract base class that all agent implementations must inherit from.

    Provides the common infrastructure for agent execution, including
    conversation management, tool registration, state tracking, and the
    main execution loop. Subclasses implement `step()` to define their
    specific reasoning strategy.

    Attributes:
        llm:     The language model used for reasoning and decision-making.
        tools:   A dictionary mapping tool names to BaseTool instances.
        memory:  An optional memory store for persisting context across runs.
        state:   The current AgentState tracking execution progress.
        system_prompt: The system message that defines the agent's persona
                       and instructions.

    Methods:
        run(query):     Execute the agent on a user query (main entry point).
        step(state):    Perform one reasoning/action cycle (abstract).
        add_tool(tool): Register a new tool with the agent.
        reset():        Reset the agent's state for a new query.
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools: list[BaseTool] | None = None,
        memory: MemoryStore | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        max_steps: int = 10,
    ) -> None:
        """
        Initialize the agent with an LLM, optional tools, and configuration.

        Args:
            llm:           The language model provider to use for reasoning.
            tools:         An optional list of tools the agent can invoke.
                           Each tool must have a unique name.
            memory:        An optional memory store for persisting context
                           (e.g., conversation history, learned facts).
            system_prompt: The system message that sets the agent's behavior
                           and persona. Default is a generic helpful assistant.
            max_steps:     The maximum number of reasoning/action steps before
                           the agent stops to prevent infinite loops. Default is 10.
        """
        # Store the LLM provider for reasoning.
        self.llm = llm

        # Build a name-to-tool mapping for quick lookup during execution.
        self.tools: dict[str, BaseTool] = {}
        if tools:
            for tool in tools:
                self.tools[tool.name] = tool

        # Store the optional memory backend.
        self.memory = memory

        # Store the system prompt that defines the agent's behavior.
        self.system_prompt = system_prompt

        # Initialize the agent's execution state.
        self.state = AgentState(max_steps=max_steps)

    def run(self, query: str) -> str:
        """
        Execute the agent on a user query and return the final response.

        This is the main entry point for interacting with the agent. It
        initializes the conversation, runs the step loop until completion
        or the step limit is reached, and returns the final response.

        Args:
            query: The user's input query or instruction.

        Returns:
            A string containing the agent's final response.

        Raises:
            RuntimeError: If the agent fails during execution.
        """
        # Reset state for a fresh execution run.
        self.reset()

        # Add the system prompt as the first message in the conversation.
        self.state.messages.append(
            Message(role=Role.SYSTEM, content=self.system_prompt)
        )

        # Add the user's query as the second message.
        self.state.messages.append(Message(role=Role.USER, content=query))

        # Load any relevant context from memory, if a memory store is configured.
        if self.memory:
            context = self.memory.retrieve(query)
            if context:
                # Insert retrieved context as a system message before the user query.
                context_message = Message(
                    role=Role.SYSTEM,
                    content=f"Relevant context from memory:\n{context}",
                )
                self.state.messages.insert(1, context_message)

        # Execute the step loop until completion or step limit.
        while (
            self.state.current_step < self.state.max_steps
            and self.state.status not in (AgentStatus.COMPLETED, AgentStatus.FAILED)
        ):
            # Update status to indicate the agent is reasoning.
            self.state.status = AgentStatus.THINKING

            # Execute one reasoning/action cycle (implemented by subclasses).
            self.step(self.state)

            # Increment the step counter.
            self.state.current_step += 1

        # If the agent hit the step limit without completing, mark it as completed
        # and use the last assistant message as the response.
        if self.state.status not in (AgentStatus.COMPLETED, AgentStatus.FAILED):
            self.state.status = AgentStatus.COMPLETED

        # Extract the final response from the last assistant message.
        for message in reversed(self.state.messages):
            if message.role == Role.ASSISTANT and message.content:
                # Save the interaction to memory if configured.
                if self.memory:
                    self.memory.store(query, message.content)
                return message.content

        return "I was unable to generate a response."

    @abstractmethod
    def step(self, state: AgentState) -> AgentState:
        """
        Perform a single reasoning/action cycle.

        This is the core method that defines the agent's behavior. Each
        agent architecture implements this differently:
        - ReAct: Think, then optionally act with a tool.
        - RAG: Retrieve context, then generate a response.
        - Planning: Update the plan, then execute the next step.

        Args:
            state: The current AgentState containing conversation history,
                   step count, and any intermediate results.

        Returns:
            The updated AgentState after this step.
        """
        ...

    def add_tool(self, tool: BaseTool) -> None:
        """
        Register a new tool with the agent.

        The tool becomes available for the agent to invoke during execution.
        If a tool with the same name already exists, it is replaced.

        Args:
            tool: The BaseTool instance to register.
        """
        self.tools[tool.name] = tool

    def reset(self) -> None:
        """
        Reset the agent's state for a new execution run.

        Clears the conversation history, resets the step counter, and sets
        the status back to IDLE. This method is called automatically at
        the beginning of each `run()` call.
        """
        self.state = AgentState(max_steps=self.state.max_steps)

    def _get_tool_schemas(self) -> list[dict[str, Any]]:
        """
        Generate JSON schemas for all registered tools.

        This method collects the schemas from all tools and returns them
        in the format expected by LLM function-calling APIs. It is called
        internally before each LLM call that supports tool use.

        Returns:
            A list of tool schema dictionaries in OpenAI function-calling format.
        """
        return [tool.to_schema() for tool in self.tools.values()]

    def _execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """
        Execute a registered tool by name with the given arguments.

        This method looks up the tool, validates the arguments, executes it,
        and returns the result. If the tool is not found or execution fails,
        an error message is returned instead of raising an exception.

        Args:
            tool_name:  The name of the tool to execute.
            arguments:  A dictionary of keyword arguments to pass to the tool.

        Returns:
            A string containing the tool's output or an error message.
        """
        # Look up the tool by name.
        tool = self.tools.get(tool_name)
        if not tool:
            return f"Error: Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}"

        try:
            # Validate arguments against the tool's schema.
            tool.validate_args(**arguments)
            # Execute the tool and return its output.
            return tool.execute(**arguments)
        except Exception as e:
            # Return the error message so the LLM can adjust its approach.
            return f"Error executing tool '{tool_name}': {e!s}"
