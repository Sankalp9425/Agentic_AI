"""
Planning agent implementation.

The Planning agent creates an explicit step-by-step plan before executing
any actions. This "plan-then-execute" approach is particularly effective
for complex tasks that require a structured methodology, as it ensures
the agent thinks through the entire approach before diving into execution.

Pipeline:
    1. Planning Phase: Agent analyzes the task and creates a numbered plan.
    2. Execution Phase: Agent executes each plan step sequentially, using
       tools as needed for each step.
    3. Synthesis Phase: Agent reviews all step results and provides a
       comprehensive final answer.

This approach offers better transparency and debuggability compared to
purely reactive agents, as the plan can be inspected and modified before
execution begins.

Example:
    >>> from agentic_ai.agents.planning import PlanningAgent
    >>> agent = PlanningAgent(llm=my_llm, tools=[search, calculator])
    >>> result = agent.run("Compare Python and Rust for web development")
"""

import logging
import re

from agentic_ai.core.base_agent import BaseAgent
from agentic_ai.core.base_llm import BaseLLM
from agentic_ai.core.base_tool import BaseTool
from agentic_ai.core.memory import MemoryStore
from agentic_ai.core.models import AgentState, AgentStatus, Message, Role

# Configure module-level logger.
logger = logging.getLogger(__name__)

# System prompt for the planning phase.
PLANNING_SYSTEM_PROMPT = """You are an AI assistant that approaches complex tasks \
methodically by creating and following plans.

When given a task, you will:
1. First, create a clear step-by-step plan.
2. Then, execute each step of the plan using available tools.
3. Finally, synthesize all results into a comprehensive answer.

When creating a plan, format it as a numbered list:
1. [Step description]
2. [Step description]
...

When executing steps, clearly indicate which step you're working on.
After completing all steps, provide a final synthesized answer.

Be thorough and systematic. If a step reveals new information that requires
adjusting the plan, explain the adjustment."""


class PlanningAgent(BaseAgent):
    """
    Planning-based agent that creates and follows explicit plans.

    Operates in three phases:
    1. Plan Creation: LLM generates a numbered step-by-step plan.
    2. Step Execution: Each plan step is executed sequentially with tools.
    3. Synthesis: Results from all steps are synthesized into a final answer.

    The plan is stored in the agent state's metadata, making it inspectable
    and potentially modifiable between steps.

    Attributes:
        plan_steps: The parsed plan steps after the planning phase.

    Example:
        >>> agent = PlanningAgent(llm=llm, tools=[search, calc], max_steps=15)
        >>> result = agent.run("Analyze the top 3 Python web frameworks")
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools: list[BaseTool] | None = None,
        memory: MemoryStore | None = None,
        system_prompt: str = PLANNING_SYSTEM_PROMPT,
        max_steps: int = 15,
    ) -> None:
        """
        Initialize the Planning agent.

        Args:
            llm:           The language model for planning and execution.
            tools:         Tools available during the execution phase.
            memory:        Optional memory store for context persistence.
            system_prompt: System prompt optimized for planning behavior.
            max_steps:     Maximum steps (includes planning + execution). Default 15.
        """
        super().__init__(
            llm=llm,
            tools=tools,
            memory=memory,
            system_prompt=system_prompt,
            max_steps=max_steps,
        )

    def _parse_plan(self, plan_text: str) -> list[str]:
        """
        Parse a numbered plan from the LLM's response text.

        Extracts lines that start with a number followed by a period or
        parenthesis, treating each as a plan step.

        Args:
            plan_text: The raw text response containing the plan.

        Returns:
            A list of plan step strings, in order.
        """
        # Match lines starting with "1.", "2.", "1)", "2)", etc.
        pattern = r"^\s*\d+[.)]\s*(.+)$"
        steps = re.findall(pattern, plan_text, re.MULTILINE)

        if not steps:
            # If no numbered steps found, split by newlines as fallback.
            lines = [
                line.strip()
                for line in plan_text.split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]
            steps = lines

        logger.info("Parsed %d plan steps", len(steps))
        return steps

    def step(self, state: AgentState) -> AgentState:
        """
        Execute one step of the plan-then-execute pipeline.

        Step 0: Create the plan by asking the LLM to generate it.
        Steps 1-N: Execute each plan step, using tools as needed.
        Final step: Synthesize results into a comprehensive answer.

        Args:
            state: The current agent state.

        Returns:
            The updated agent state.
        """
        if state.current_step == 0:
            # === PLANNING PHASE ===
            # Ask the LLM to create a plan for the task.
            planning_prompt = Message(
                role=Role.SYSTEM,
                content=(
                    "First, create a step-by-step plan to accomplish the user's task. "
                    "Output ONLY the numbered plan, nothing else. Each step should be "
                    "a clear, actionable item."
                ),
            )
            # Insert the planning instruction before the user's message.
            planning_messages = state.messages.copy()
            planning_messages.insert(-1, planning_prompt)

            # Generate the plan.
            plan_response = self.llm.chat(planning_messages)
            state.messages.append(plan_response)

            # Parse the plan into individual steps.
            plan_steps = self._parse_plan(plan_response.content)

            # Store the plan in the agent's metadata for reference.
            state.metadata["plan_steps"] = plan_steps
            state.metadata["current_plan_step"] = 0
            state.metadata["step_results"] = []

            logger.info(
                "Created plan with %d steps: %s", len(plan_steps), plan_steps
            )

        else:
            # === EXECUTION PHASE ===
            plan_steps = state.metadata.get("plan_steps", [])
            current_plan_step = state.metadata.get("current_plan_step", 0)
            step_results: list[str] = state.metadata.get("step_results", [])

            if current_plan_step < len(plan_steps):
                # Execute the current plan step.
                step_description = plan_steps[current_plan_step]

                logger.info(
                    "Executing plan step %d/%d: %s",
                    current_plan_step + 1,
                    len(plan_steps),
                    step_description,
                )

                # Create a focused prompt for this specific step.
                step_prompt = Message(
                    role=Role.USER,
                    content=(
                        f"Execute step {current_plan_step + 1} of the plan: "
                        f"{step_description}\n\n"
                        f"Previous step results:\n"
                        + "\n".join(
                            f"Step {i + 1}: {r}" for i, r in enumerate(step_results)
                        )
                    ),
                )
                state.messages.append(step_prompt)

                # Use tools if available for this step.
                tool_schemas = self._get_tool_schemas()
                if tool_schemas:
                    response = self.llm.chat_with_tools(
                        state.messages, tool_schemas
                    )
                    state.messages.append(response)

                    # Execute any tool calls.
                    if response.tool_calls:
                        for tool_call in response.tool_calls:
                            result_str = self._execute_tool(
                                tool_call.name, tool_call.arguments
                            )
                            observation = Message(
                                role=Role.TOOL,
                                content=result_str,
                                name=tool_call.name,
                                tool_call_id=tool_call.id,
                            )
                            state.messages.append(observation)

                        # Get the step summary after tool execution.
                        summary_response = self.llm.chat(state.messages)
                        state.messages.append(summary_response)
                        step_results.append(summary_response.content)
                    else:
                        step_results.append(response.content)
                else:
                    response = self.llm.chat(state.messages)
                    state.messages.append(response)
                    step_results.append(response.content)

                # Advance to the next plan step.
                state.metadata["current_plan_step"] = current_plan_step + 1
                state.metadata["step_results"] = step_results

                # Store intermediate results for tracking.
                state.intermediate_results.append({
                    "plan_step": current_plan_step + 1,
                    "description": step_description,
                    "result": step_results[-1][:500],
                })

                state.status = AgentStatus.ACTING

            else:
                # === SYNTHESIS PHASE ===
                # All plan steps are complete - synthesize the final answer.
                logger.info("All plan steps complete, synthesizing final answer")

                synthesis_prompt = Message(
                    role=Role.USER,
                    content=(
                        "All plan steps are now complete. Here are the results:\n\n"
                        + "\n\n".join(
                            f"**Step {i + 1}** ({plan_steps[i] if i < len(plan_steps) else 'N/A'}):\n{r}"
                            for i, r in enumerate(step_results)
                        )
                        + "\n\nPlease synthesize these results into a comprehensive, "
                        "well-structured final answer."
                    ),
                )
                state.messages.append(synthesis_prompt)

                final_response = self.llm.chat(state.messages)
                state.messages.append(final_response)

                state.status = AgentStatus.COMPLETED

        return state
