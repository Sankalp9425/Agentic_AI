"""
Centralized prompt templates for the agentic AI framework.

All prompts used by agents, retrieval strategies, and evaluation metrics
are defined here as module-level constants. This allows:

    1. Easy customization by modifying a single file.
    2. Consistent prompt formatting across the framework.
    3. Version control of prompt changes.

Each template uses Python str.format() syntax with named placeholders
(e.g., ``{question}``, ``{context}``). See each template's docstring
for the required variables.

Categories:
    - RAG prompts: For retrieval-augmented generation.
    - Agent prompts: For ReAct, planning, and hierarchical agents.
    - Retrieval prompts: For query expansion and re-ranking.
    - Evaluation prompts: For RAG quality assessment.

Example:
    >>> from agentic_ai.prompts.templates import RAG_SYSTEM_PROMPT
    >>> prompt = RAG_SYSTEM_PROMPT.format(context="Python is...", question="What is Python?")
"""

# =============================================================================
# RAG Prompts
# =============================================================================

RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on the provided context. "
    "Use ONLY the information from the context below to answer the question. "
    "If the context doesn't contain enough information to answer the question, "
    "say so clearly. Do not make up information.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)
"""
RAG system prompt for context-grounded question answering.

Variables:
    context: The retrieved context documents concatenated together.
    question: The user's question.
"""

RAG_SYSTEM_PROMPT_WITH_SOURCES = (
    "You are a helpful assistant that answers questions based on the provided context. "
    "Use ONLY the information from the context below to answer the question. "
    "After your answer, list the source documents you used.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Provide your answer followed by a list of sources used."
)
"""
RAG prompt that asks the model to cite its sources.

Variables:
    context: The retrieved context documents with source metadata.
    question: The user's question.
"""

RAG_CONVERSATIONAL_PROMPT = (
    "You are a helpful assistant engaged in a conversation. "
    "Use the provided context to answer the user's latest question. "
    "You may also use the conversation history for continuity, but prioritize "
    "the context for factual information.\n\n"
    "Context:\n{context}\n\n"
    "Conversation History:\n{history}\n\n"
    "User: {question}\n\n"
    "Assistant:"
)
"""
RAG prompt for multi-turn conversational QA.

Variables:
    context: The retrieved context documents.
    history: Previous conversation turns.
    question: The user's latest question.
"""


# =============================================================================
# Agent Prompts — ReAct
# =============================================================================

REACT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant that can use tools to accomplish tasks. "
    "You follow the ReAct (Reasoning + Acting) framework:\n\n"
    "1. **Thought**: Reason about what you need to do next.\n"
    "2. **Action**: Choose a tool to use and specify the input.\n"
    "3. **Observation**: Review the result of the action.\n"
    "4. Repeat until you can provide a final answer.\n\n"
    "Available tools:\n{tools}\n\n"
    "Always think step-by-step. When you have enough information to answer, "
    "provide your final answer clearly."
)
"""
ReAct agent system prompt.

Variables:
    tools: Formatted list of available tools with descriptions.
"""

REACT_STEP_PROMPT = (
    "Given the task and previous steps, decide what to do next.\n\n"
    "Task: {task}\n\n"
    "Previous steps:\n{previous_steps}\n\n"
    "Think about what information you still need and which tool to use. "
    "Respond with your Thought and Action."
)
"""
ReAct agent step prompt for deciding the next action.

Variables:
    task: The original user task.
    previous_steps: Formatted history of previous thought-action-observation triples.
"""


# =============================================================================
# Agent Prompts — Planning
# =============================================================================

PLANNING_SYSTEM_PROMPT = (
    "You are a planning agent that breaks down complex tasks into steps. "
    "First, create a detailed plan, then execute each step using available tools.\n\n"
    "Available tools:\n{tools}\n\n"
    "Start by analyzing the task and creating a numbered plan."
)
"""
Planning agent system prompt.

Variables:
    tools: Formatted list of available tools.
"""

PLANNING_CREATE_PLAN_PROMPT = (
    "Create a step-by-step plan to accomplish the following task.\n\n"
    "Task: {task}\n\n"
    "Break the task down into clear, actionable steps. Each step should be "
    "specific enough to execute with a single tool call or reasoning step.\n\n"
    "Respond with a numbered list of steps."
)
"""
Prompt for generating an execution plan.

Variables:
    task: The user's task to plan for.
"""

PLANNING_EXECUTE_STEP_PROMPT = (
    "Execute the following step from the plan.\n\n"
    "Overall task: {task}\n\n"
    "Current step ({step_number}/{total_steps}): {step}\n\n"
    "Previous results:\n{previous_results}\n\n"
    "Execute this step and provide the result."
)
"""
Prompt for executing a single plan step.

Variables:
    task: The overall task.
    step_number: Current step number.
    total_steps: Total number of steps.
    step: The current step description.
    previous_results: Results from previous steps.
"""


# =============================================================================
# Agent Prompts — Hierarchical
# =============================================================================

HIERARCHICAL_MANAGER_PROMPT = (
    "You are a manager agent that delegates tasks to specialized worker agents. "
    "Analyze the task, break it down, and assign subtasks to the most appropriate "
    "worker based on their capabilities.\n\n"
    "Available workers:\n{workers}\n\n"
    "Task: {task}\n\n"
    "Decide which worker should handle which part of the task. "
    "Respond with your delegation plan."
)
"""
Hierarchical agent manager prompt for task delegation.

Variables:
    workers: Descriptions of available worker agents.
    task: The task to delegate.
"""

HIERARCHICAL_WORKER_PROMPT = (
    "You are a specialized worker agent. Complete the assigned subtask "
    "using your available tools.\n\n"
    "Available tools:\n{tools}\n\n"
    "Your specialization: {specialization}\n\n"
    "Assigned subtask: {subtask}\n\n"
    "Complete the subtask and provide your result."
)
"""
Hierarchical agent worker prompt.

Variables:
    tools: Available tools for this worker.
    specialization: The worker's area of expertise.
    subtask: The assigned subtask.
"""

HIERARCHICAL_SYNTHESIZE_PROMPT = (
    "You are a manager agent synthesizing results from your workers.\n\n"
    "Original task: {task}\n\n"
    "Worker results:\n{results}\n\n"
    "Synthesize all worker results into a coherent final answer."
)
"""
Prompt for synthesizing worker results into a final answer.

Variables:
    task: The original task.
    results: Formatted results from all workers.
"""


# =============================================================================
# Retrieval Prompts
# =============================================================================

QUERY_EXPANSION_PROMPT = (
    "Generate {num_expansions} alternative phrasings for the following search query. "
    "Each alternative should capture a different aspect or perspective of the "
    "same information need. Make the alternatives diverse but relevant.\n\n"
    "Original query: {query}\n\n"
    "Provide {num_expansions} alternative queries, one per line, numbered:\n"
    "1. "
)
"""
Prompt for generating query expansions.

Variables:
    query: The original search query.
    num_expansions: Number of alternative queries to generate.
"""

RERANKING_PROMPT = (
    "Rate how relevant the following document is to the given query.\n\n"
    "Query: {query}\n\n"
    "Document: {document}\n\n"
    "Rate the relevance on a scale of 0 to 10, where:\n"
    "- 0: Completely irrelevant\n"
    "- 5: Somewhat relevant\n"
    "- 10: Perfectly relevant\n\n"
    "Respond with ONLY a number from 0 to 10."
)
"""
Prompt for LLM-based document re-ranking.

Variables:
    query: The search query.
    document: The document content to evaluate.
"""

HYDE_PROMPT = (
    "Given the following question, write a hypothetical passage that would "
    "perfectly answer this question. The passage should be detailed and "
    "informative, as if it were from a high-quality reference document.\n\n"
    "Question: {question}\n\n"
    "Hypothetical passage:"
)
"""
HyDE (Hypothetical Document Embeddings) prompt.

Variables:
    question: The user's question.
"""


# =============================================================================
# Evaluation Prompts
# =============================================================================

FAITHFULNESS_EVAL_PROMPT = (
    "Evaluate the faithfulness of the following answer based on the given context.\n\n"
    "Question: {question}\n\n"
    "Context:\n{context}\n\n"
    "Answer: {answer}\n\n"
    "Faithfulness means every claim in the answer can be traced back to the context. "
    "Rate the faithfulness on a scale of 0 to 1, where:\n"
    "- 0.0: The answer contains significant information not in the context (hallucination)\n"
    "- 0.5: The answer is partially grounded in the context\n"
    "- 1.0: Every claim in the answer is supported by the context\n\n"
    "Respond with ONLY a number between 0 and 1."
)
"""
Prompt for evaluating answer faithfulness to context.

Variables:
    question: The user's question.
    context: The retrieved context documents.
    answer: The generated answer.
"""

ANSWER_RELEVANCE_EVAL_PROMPT = (
    "Evaluate how relevant the following answer is to the question.\n\n"
    "Question: {question}\n\n"
    "Answer: {answer}\n\n"
    "Rate the relevance on a scale of 0 to 1, where:\n"
    "- 0.0: The answer is completely off-topic\n"
    "- 0.5: The answer partially addresses the question\n"
    "- 1.0: The answer directly and fully addresses the question\n\n"
    "Respond with ONLY a number between 0 and 1."
)
"""
Prompt for evaluating answer relevance to the question.

Variables:
    question: The user's question.
    answer: The generated answer.
"""

CONTEXT_RELEVANCE_EVAL_PROMPT = (
    "Evaluate how relevant the following retrieved contexts are to the question.\n\n"
    "Question: {question}\n\n"
    "Retrieved Contexts:\n{context}\n\n"
    "Rate the overall context relevance on a scale of 0 to 1, where:\n"
    "- 0.0: None of the contexts are relevant to the question\n"
    "- 0.5: Some contexts are relevant but many are not\n"
    "- 1.0: All contexts are highly relevant to the question\n\n"
    "Respond with ONLY a number between 0 and 1."
)
"""
Prompt for evaluating retrieved context relevance.

Variables:
    question: The user's question.
    context: The retrieved context documents.
"""

ANSWER_CORRECTNESS_EVAL_PROMPT = (
    "Compare the following answer to the reference answer and rate its correctness.\n\n"
    "Question: {question}\n\n"
    "Generated Answer: {answer}\n\n"
    "Reference Answer: {reference}\n\n"
    "Rate the correctness on a scale of 0 to 1, where:\n"
    "- 0.0: The answer contradicts the reference or is completely wrong\n"
    "- 0.5: The answer is partially correct\n"
    "- 1.0: The answer is fully correct and matches the reference\n\n"
    "Respond with ONLY a number between 0 and 1."
)
"""
Prompt for evaluating answer correctness against a reference.

Variables:
    question: The user's question.
    answer: The generated answer.
    reference: The ground-truth reference answer.
"""
