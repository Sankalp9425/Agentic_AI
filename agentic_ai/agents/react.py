"""
ReAct (Reasoning + Acting) agent implementation.

ReAct is a paradigm where the agent alternates between reasoning (thinking
about what to do next) and acting (invoking tools to gather information or
perform actions). This iterative loop continues until the agent has enough
information to provide a final answer or reaches the maximum step limit.

The ReAct loop:
    1. Thought:      Agent reasons about the current state and what to do next.
    2. Action:       Agent chooses a tool to invoke with specific arguments.
    3. Observation:  The tool's output is fed back to the agent.
    4. Repeat:       Steps 1-3 repeat until the agent can give a final answer.

This approach is particularly effective for complex questions that require
multiple information-gathering steps and intermediate reasoning.

Reference:
    Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models"
    https://arxiv.org/abs/2210.03629

Example:
    >>> from agentic_ai.agents.react import ReActAgent
    >>> agent = ReActAgent(llm=my_llm, tools=[search_tool, calculator])
    >>> result = agent.run("What is the population of France times 2?")
"""

import logging

from agentic_ai.core.base_agent import BaseAgent
from agentic_ai.core.base_llm import BaseLLM
from agentic_ai.core.base_tool import BaseTool
from agentic_ai.core.memory import MemoryStore
from agentic_ai.core.models import AgentState, AgentStatus, Message, Role, ToolResult

# Configure module-level logger.
logger = logging.getLogger(__name__)

# System prompt that instructs the LLM to follow the ReAct reasoning pattern.
REACT_SYSTEM_PROMPT = """You are an AI assistant that solves problems step-by-step using \
available tools. For each step, follow this pattern:

1. **Thought**: Analyze the current state and decide what to do next.
2. **Action**: Choose a tool to use and provide the required arguments.
3. **Observation**: Review the tool's output and incorporate it into your reasoning.

When you have enough information to provide a final answer, respond directly \
without using any tools. Be thorough but efficient - use the minimum number of \
tool calls needed to answer the question accurately.

Important guidelines:
- Always explain your reasoning before using a tool.
- If a tool returns an error, try a different approach.
- Synthesize information from multiple tool calls when needed.
- Provide a clear, complete final answer."""


class ReActAgent(BaseAgent):
    """
    ReAct (Reasoning + Acting) agent.

    Implements the ReAct paradigm where the agent iteratively reasons about
    the problem and takes actions (tool calls) until it can provide a
    final answer. Each step involves sending the conversation to the LLM
    with available tools, then processing any tool calls or final responses.

    Attributes:
        All inherited from BaseAgent.

    Example:
        >>> agent = ReActAgent(
        ...     llm=my_llm,
        ...     tools=[web_search, calculator],
        ...     max_steps=10,
        ... )
        >>> result = agent.run("What is 15% of the US GDP?")
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools: list[BaseTool] | None = None,
        memory: MemoryStore | None = None,
        system_prompt: str = REACT_SYSTEM_PROMPT,
        max_steps: int = 10,
    ) -> None:
        """
        Initialize the ReAct agent.

        Args:
            llm:           The language model for reasoning and decision-making.
            tools:         Tools the agent can invoke during reasoning. At least
                           one tool should be provided for the ReAct pattern to
                           be useful.
            memory:        Optional memory store for context persistence.
            system_prompt: System prompt optimized for the ReAct pattern.
            max_steps:     Maximum reasoning/acting cycles. Default is 10.
        """
        super().__init__(
            llm=llm,
            tools=tools,
            memory=memory,
            system_prompt=system_prompt,
            max_steps=max_steps,
        )

    def step(self, state: AgentState) -> AgentState:
        """
        Execute one Thought-Action-Observation cycle.

        Sends the current conversation (including previous observations) to
        the LLM with tool schemas. If the LLM responds with tool calls,
        those tools are executed and the results are added as observations.
        If the LLM responds with plain text (no tool calls), it's treated
        as the final answer.

        Args:
            state: The current agent state with full conversation history.

        Returns:
            The updated agent state. Status is set to COMPLETED if the
            LLM gave a final answer without tool calls.
        """
        logger.info(
            "ReAct step %d/%d", state.current_step + 1, state.max_steps
        )

        # Get tool schemas for the LLM.
        tool_schemas = self._get_tool_schemas()

        if tool_schemas:
            # Ask the LLM to reason and optionally call tools.
            response = self.llm.chat_with_tools(state.messages, tool_schemas)
        else:
            # No tools available - just get a regular chat response.
            response = self.llm.chat(state.messages)

        # Add the LLM's response to the conversation history.
        state.messages.append(response)

        if response.tool_calls:
            # The LLM chose to invoke one or more tools (Action step).
            logger.info(
                "Agent invoked %d tool(s): %s",
                len(response.tool_calls),
                [tc.name for tc in response.tool_calls],
            )

            # Execute each tool call and collect results.
            for tool_call in response.tool_calls:
                # Execute the tool using the base class helper.
                result_str = self._execute_tool(
                    tool_call.name, tool_call.arguments
                )

                # Create a ToolResult to track the outcome.
                tool_result = ToolResult(
                    tool_call_id=tool_call.id,
                    output=result_str,
                    success="Error" not in result_str,
                )

                # Add the observation to the conversation as a TOOL message.
                observation = Message(
                    role=Role.TOOL,
                    content=tool_result.output,
                    name=tool_call.name,
                    tool_call_id=tool_call.id,
                )
                state.messages.append(observation)

                # Store the tool result in intermediate results for tracking.
                state.intermediate_results.append({
                    "step": state.current_step,
                    "tool": tool_call.name,
                    "arguments": tool_call.arguments,
                    "result": result_str,
                    "success": tool_result.success,
                })

                logger.debug(
                    "Tool '%s' returned: %s",
                    tool_call.name,
                    result_str[:200],
                )

            # Set status to ACTING since we're in the middle of tool execution.
            state.status = AgentStatus.ACTING

        else:
            # The LLM responded with text only (no tool calls) - this is the final answer.
            logger.info("Agent provided final answer (no tool calls)")
            state.status = AgentStatus.COMPLETED

        return state
