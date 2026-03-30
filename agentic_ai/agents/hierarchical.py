"""
Hierarchical multi-agent system implementation.

In a hierarchical agent architecture, a manager agent coordinates multiple
specialized worker agents. The manager receives a complex task, breaks it
into sub-tasks, delegates each sub-task to the most appropriate worker,
collects results, and synthesizes a final response.

This approach mirrors how human teams operate: a project manager distributes
work to specialists (researcher, writer, analyst, etc.), each contributing
their expertise to the overall goal.

Architecture:
    Manager Agent
    ├── Worker 1 (e.g., Research specialist)
    ├── Worker 2 (e.g., Writing specialist)
    ├── Worker 3 (e.g., Analysis specialist)
    └── Worker N (e.g., Code specialist)

The manager has access to a special "delegate" tool that routes sub-tasks
to worker agents. Workers can have their own tools and system prompts
optimized for their specialty.

Example:
    >>> from agentic_ai.agents.hierarchical import HierarchicalAgent
    >>> researcher = WorkerAgent(name="researcher", llm=llm, tools=[search])
    >>> writer = WorkerAgent(name="writer", llm=llm)
    >>> manager = HierarchicalAgent(llm=llm, workers=[researcher, writer])
    >>> result = manager.run("Research and write a summary about quantum computing")
"""

import logging
from typing import Any

from agentic_ai.core.base_agent import BaseAgent
from agentic_ai.core.base_llm import BaseLLM
from agentic_ai.core.base_tool import BaseTool
from agentic_ai.core.memory import MemoryStore
from agentic_ai.core.models import AgentState, AgentStatus, Message, Role

# Configure module-level logger.
logger = logging.getLogger(__name__)

# System prompt for the manager agent.
MANAGER_SYSTEM_PROMPT = """You are a manager AI that coordinates a team of specialist \
workers to accomplish complex tasks. Your team members are:

{worker_descriptions}

For each task:
1. Analyze the task and break it into sub-tasks.
2. Delegate each sub-task to the most appropriate worker using the delegate tool.
3. Review worker outputs and provide feedback or re-delegate if needed.
4. Synthesize all worker outputs into a comprehensive final response.

Guidelines:
- Always explain your delegation reasoning.
- Delegate to the most specialized worker for each sub-task.
- You can delegate to the same worker multiple times.
- Synthesize results don't just concatenate worker outputs."""


class WorkerAgent:
    """
    A specialized worker agent within a hierarchical system.

    Workers are lightweight agents with a specific role and expertise.
    They receive sub-tasks from the manager, process them using their
    LLM and tools, and return results. Workers don't manage their own
    execution loop - the manager handles orchestration.

    Attributes:
        name:        A unique identifier for this worker (e.g., "researcher").
        description: A brief description of the worker's expertise, shown to
                     the manager to help it decide which worker to delegate to.
        llm:         The language model this worker uses for reasoning.
        tools:       Tools available to this worker.
        system_prompt: The system prompt defining this worker's persona.

    Example:
        >>> researcher = WorkerAgent(
        ...     name="researcher",
        ...     description="Expert at finding and analyzing information.",
        ...     llm=my_llm,
        ...     tools=[web_search],
        ... )
    """

    def __init__(
        self,
        name: str,
        description: str,
        llm: BaseLLM,
        tools: list[BaseTool] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """
        Initialize a worker agent with its role and capabilities.

        Args:
            name:          Unique name identifying this worker.
            description:   Brief description of the worker's specialty.
            llm:           Language model for the worker's reasoning.
            tools:         Optional tools the worker can use.
            system_prompt: Optional custom system prompt. If not provided,
                           a default prompt is generated from the name and description.
        """
        self.name = name
        self.description = description
        self.llm = llm
        self.tools: dict[str, BaseTool] = {}
        if tools:
            for tool in tools:
                self.tools[tool.name] = tool

        # Generate a default system prompt if none provided.
        self.system_prompt = system_prompt or (
            f"You are a {name} specialist. {description} "
            f"Complete the assigned task thoroughly and return your results."
        )

    def execute_task(self, task: str) -> str:
        """
        Execute a sub-task assigned by the manager.

        Creates a conversation with the task, optionally uses tools, and
        returns the result. This is a simplified execution without the
        full agent loop, suitable for focused sub-tasks.

        Args:
            task: The sub-task description to complete.

        Returns:
            A string containing the worker's response/output.
        """
        logger.info("Worker '%s' executing task: %s", self.name, task[:100])

        # Build the conversation for this sub-task.
        messages = [
            Message(role=Role.SYSTEM, content=self.system_prompt),
            Message(role=Role.USER, content=task),
        ]

        # If the worker has tools, use tool-enabled chat.
        if self.tools:
            tool_schemas = [tool.to_schema() for tool in self.tools.values()]
            response = self.llm.chat_with_tools(messages, tool_schemas)

            # If the LLM wants to use tools, execute them.
            if response.tool_calls:
                messages.append(response)
                for tool_call in response.tool_calls:
                    tool = self.tools.get(tool_call.name)
                    if tool:
                        try:
                            result = tool.execute(**tool_call.arguments)
                        except Exception as e:
                            result = f"Error: {e!s}"
                    else:
                        result = f"Tool '{tool_call.name}' not found."

                    messages.append(
                        Message(
                            role=Role.TOOL,
                            content=result,
                            tool_call_id=tool_call.id,
                        )
                    )

                # Get the final response after tool execution.
                response = self.llm.chat(messages)
        else:
            # No tools - just get a direct response.
            response = self.llm.chat(messages)

        logger.info(
            "Worker '%s' completed task with %d chars",
            self.name,
            len(response.content),
        )
        return response.content


class DelegateTool(BaseTool):
    """
    Internal tool that allows the manager to delegate tasks to workers.

    This tool is automatically created by the HierarchicalAgent and added
    to the manager's tool set. When the manager calls this tool, it routes
    the specified task to the named worker agent.

    Attributes:
        _workers: Dictionary mapping worker names to WorkerAgent instances.
    """

    name = "delegate"
    description = "Delegate a sub-task to a specialist worker agent."
    parameters = {
        "worker_name": {
            "type": "string",
            "description": "The name of the worker to delegate to.",
            "required": True,
        },
        "task": {
            "type": "string",
            "description": "The sub-task description to assign to the worker.",
            "required": True,
        },
    }

    def __init__(self, workers: dict[str, WorkerAgent]) -> None:
        """
        Initialize the delegate tool with available workers.

        Args:
            workers: A dictionary mapping worker names to WorkerAgent instances.
        """
        self._workers = workers

    def execute(self, **kwargs: Any) -> str:
        """
        Delegate a task to the specified worker and return the result.

        Args:
            **kwargs: Must include 'worker_name' and 'task'.

        Returns:
            The worker's output as a string, or an error message.
        """
        worker_name = kwargs.get("worker_name", "")
        task = kwargs.get("task", "")

        # Look up the requested worker.
        worker = self._workers.get(worker_name)
        if not worker:
            available = list(self._workers.keys())
            return (
                f"Error: Worker '{worker_name}' not found. "
                f"Available workers: {available}"
            )

        # Execute the task using the worker.
        try:
            result = worker.execute_task(task)
            return f"[{worker_name}]: {result}"
        except Exception as e:
            return f"Error from worker '{worker_name}': {e!s}"


class HierarchicalAgent(BaseAgent):
    """
    Hierarchical multi-agent system with a manager and specialized workers.

    The manager agent receives complex tasks, decomposes them, delegates
    sub-tasks to appropriate worker agents, and synthesizes the results
    into a comprehensive response.

    Attributes:
        workers: Dictionary mapping worker names to WorkerAgent instances.

    Example:
        >>> researcher = WorkerAgent("researcher", "Finds information", llm)
        >>> writer = WorkerAgent("writer", "Writes content", llm)
        >>> manager = HierarchicalAgent(llm=llm, workers=[researcher, writer])
        >>> result = manager.run("Write a report on AI trends")
    """

    def __init__(
        self,
        llm: BaseLLM,
        workers: list[WorkerAgent],
        tools: list[BaseTool] | None = None,
        memory: MemoryStore | None = None,
        max_steps: int = 15,
    ) -> None:
        """
        Initialize the hierarchical agent with workers.

        Creates the delegate tool and configures the manager's system prompt
        with descriptions of all available workers.

        Args:
            llm:       The language model for the manager's reasoning.
            workers:   List of WorkerAgent instances that can receive tasks.
            tools:     Optional additional tools for the manager (beyond delegate).
            memory:    Optional memory store for context persistence.
            max_steps: Maximum delegation cycles. Default is 15.
        """
        # Build the workers dictionary from the list.
        self.workers: dict[str, WorkerAgent] = {w.name: w for w in workers}

        # Create the delegate tool with access to all workers.
        delegate_tool = DelegateTool(self.workers)

        # Combine the delegate tool with any additional tools.
        all_tools = [delegate_tool]
        if tools:
            all_tools.extend(tools)

        # Build worker descriptions for the manager's system prompt.
        worker_descriptions = "\n".join(
            f"- **{w.name}**: {w.description}" for w in workers
        )

        # Format the manager's system prompt with worker info.
        system_prompt = MANAGER_SYSTEM_PROMPT.format(
            worker_descriptions=worker_descriptions
        )

        super().__init__(
            llm=llm,
            tools=all_tools,
            memory=memory,
            system_prompt=system_prompt,
            max_steps=max_steps,
        )

    def step(self, state: AgentState) -> AgentState:
        """
        Execute one manager reasoning/delegation cycle.

        The manager reasons about the task, delegates to workers via the
        delegate tool, reviews results, and either delegates more work
        or provides a final synthesized answer.

        Args:
            state: The current agent state.

        Returns:
            The updated agent state.
        """
        logger.info(
            "Hierarchical agent step %d/%d",
            state.current_step + 1,
            state.max_steps,
        )

        # Get tool schemas (includes the delegate tool).
        tool_schemas = self._get_tool_schemas()

        # Ask the manager to reason and decide on next actions.
        response = self.llm.chat_with_tools(state.messages, tool_schemas)
        state.messages.append(response)

        if response.tool_calls:
            # Manager chose to delegate to workers or use other tools.
            for tool_call in response.tool_calls:
                result_str = self._execute_tool(
                    tool_call.name, tool_call.arguments
                )

                # Add the tool result as an observation.
                observation = Message(
                    role=Role.TOOL,
                    content=result_str,
                    name=tool_call.name,
                    tool_call_id=tool_call.id,
                )
                state.messages.append(observation)

                # Track delegation results.
                state.intermediate_results.append({
                    "step": state.current_step,
                    "tool": tool_call.name,
                    "arguments": tool_call.arguments,
                    "result": result_str[:500],
                })

                logger.info(
                    "Manager used tool '%s': %s",
                    tool_call.name,
                    result_str[:200],
                )

            state.status = AgentStatus.ACTING

        else:
            # Manager provided a final synthesized answer.
            logger.info("Manager provided final synthesized answer")
            state.status = AgentStatus.COMPLETED

        return state
