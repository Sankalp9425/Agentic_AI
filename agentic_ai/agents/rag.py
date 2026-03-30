"""
RAG (Retrieval-Augmented Generation) agent implementation.

RAG is a paradigm that combines information retrieval with language generation.
When a user asks a question, the agent first retrieves relevant documents from
a vector store, then uses those documents as context for the LLM to generate
a grounded, factual response.

This approach dramatically reduces hallucinations by anchoring the LLM's
response in actual source documents, and enables the LLM to answer questions
about information that wasn't in its training data.

Pipeline:
    1. User submits a query.
    2. Agent embeds the query and searches the vector store for relevant documents.
    3. Retrieved documents are injected into the prompt as context.
    4. LLM generates a response grounded in the retrieved context.
    5. (Optional) Source documents are cited in the response.

Requirements:
    - A configured vector store with indexed documents.
    - An embedding provider (used by the vector store).
    - An LLM provider for response generation.

Example:
    >>> from agentic_ai.agents.rag import RAGAgent
    >>> agent = RAGAgent(
    ...     llm=my_llm,
    ...     vector_store=my_store,
    ...     top_k=5,
    ... )
    >>> response = agent.run("What are the benefits of RAG?")
"""

import logging
from typing import Any

from agentic_ai.core.base_agent import BaseAgent
from agentic_ai.core.base_llm import BaseLLM
from agentic_ai.core.base_tool import BaseTool
from agentic_ai.core.base_vectorstore import BaseVectorStore
from agentic_ai.core.memory import MemoryStore
from agentic_ai.core.models import AgentState, AgentStatus, Message, Role

# Configure module-level logger.
logger = logging.getLogger(__name__)

# Default system prompt for the RAG agent, instructing it to use retrieved context.
RAG_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on \
the provided context documents. Always ground your answers in the given context. If the \
context doesn't contain enough information to answer the question, say so clearly rather \
than making up an answer.

When citing information, reference the source document when possible. Be concise but \
thorough in your responses."""


class RAGAgent(BaseAgent):
    """
    Retrieval-Augmented Generation agent.

    Enhances LLM responses by retrieving relevant documents from a vector
    store and including them as context in the prompt. This grounds the
    LLM's response in actual data, reducing hallucinations and enabling
    answers about domain-specific information.

    Attributes:
        vector_store: The vector store containing indexed documents.
        top_k:        Number of documents to retrieve per query.
        include_sources: Whether to append source citations to the response.

    Example:
        >>> agent = RAGAgent(llm=llm, vector_store=store, top_k=5)
        >>> result = agent.run("Explain the architecture of transformers.")
    """

    def __init__(
        self,
        llm: BaseLLM,
        vector_store: BaseVectorStore,
        tools: list[BaseTool] | None = None,
        memory: MemoryStore | None = None,
        system_prompt: str = RAG_SYSTEM_PROMPT,
        top_k: int = 5,
        include_sources: bool = True,
        max_steps: int = 3,
    ) -> None:
        """
        Initialize the RAG agent with an LLM and vector store.

        Args:
            llm:             The language model for generating responses.
            vector_store:    The vector store for document retrieval.
            tools:           Optional additional tools the agent can use.
            memory:          Optional memory store for conversation persistence.
            system_prompt:   System prompt for the agent. Default is optimized for RAG.
            top_k:           Number of documents to retrieve per query. Default is 5.
            include_sources: Whether to include source document references
                             in the response. Default is True.
            max_steps:       Maximum reasoning steps. Default is 3 (retrieve,
                             generate, optionally refine).
        """
        super().__init__(
            llm=llm,
            tools=tools,
            memory=memory,
            system_prompt=system_prompt,
            max_steps=max_steps,
        )

        # Store the vector store for document retrieval.
        self.vector_store = vector_store
        # Number of documents to retrieve per query.
        self.top_k = top_k
        # Whether to include source citations in the response.
        self.include_sources = include_sources

    def _retrieve_context(self, query: str) -> list[dict[str, Any]]:
        """
        Retrieve relevant documents from the vector store.

        Performs a similarity search against the vector store and returns
        the matched documents with their content and metadata.

        Args:
            query: The search query to find relevant documents.

        Returns:
            A list of dictionaries, each containing 'content', 'metadata',
            and optionally 'score' for each retrieved document.
        """
        logger.debug("Retrieving top-%d documents for query: %s", self.top_k, query[:100])

        # Perform similarity search with scores if supported.
        results_with_scores = self.vector_store.similarity_search_with_scores(
            query=query,
            k=self.top_k,
        )

        # Format results as a list of dictionaries.
        context_docs: list[dict[str, Any]] = []
        for doc, score in results_with_scores:
            context_docs.append({
                "content": doc.content,
                "metadata": doc.metadata,
                "score": score,
            })

        logger.info("Retrieved %d relevant documents", len(context_docs))
        return context_docs

    def _format_context(self, documents: list[dict[str, Any]]) -> str:
        """
        Format retrieved documents into a context string for the LLM prompt.

        Creates a numbered list of document contents with source information,
        making it easy for the LLM to reference specific documents.

        Args:
            documents: List of document dictionaries from _retrieve_context().

        Returns:
            A formatted string containing all document contents with headers.
        """
        if not documents:
            return "No relevant documents were found for this query."

        # Build a formatted context string with numbered documents.
        context_parts: list[str] = []
        for i, doc in enumerate(documents, 1):
            # Extract source information from metadata if available.
            source = doc["metadata"].get("source", "Unknown source")
            score = doc.get("score", 0.0)

            # Format each document with a header showing its source and relevance.
            context_parts.append(
                f"[Document {i}] (Source: {source}, Relevance: {score:.2f})\n"
                f"{doc['content']}"
            )

        return "\n\n---\n\n".join(context_parts)

    def step(self, state: AgentState) -> AgentState:
        """
        Execute one step of the RAG pipeline.

        Step 0: Retrieve relevant documents and add context to the conversation.
        Step 1: Generate a response using the LLM with the retrieved context.
        Step 2+: Handle any follow-up refinement if needed.

        Args:
            state: The current agent state with conversation history.

        Returns:
            The updated agent state after this step.
        """
        if state.current_step == 0:
            # Step 0: Retrieve relevant documents based on the user's query.
            # Find the user's query from the message history.
            user_query = ""
            for msg in reversed(state.messages):
                if msg.role == Role.USER:
                    user_query = msg.content
                    break

            if not user_query:
                state.status = AgentStatus.FAILED
                return state

            # Retrieve relevant documents from the vector store.
            retrieved_docs = self._retrieve_context(user_query)
            state.intermediate_results = retrieved_docs

            # Format the context and inject it into the conversation.
            context_str = self._format_context(retrieved_docs)
            context_message = Message(
                role=Role.SYSTEM,
                content=(
                    f"Here are the relevant documents retrieved for the user's query:\n\n"
                    f"{context_str}\n\n"
                    f"Use these documents to answer the user's question. "
                    f"Cite specific documents when possible."
                ),
            )

            # Insert the context message before the user's query.
            state.messages.insert(-1, context_message)

            logger.info("Injected context from %d documents", len(retrieved_docs))

        elif state.current_step == 1:
            # Step 1: Generate the response using the LLM with context.
            response = self.llm.chat(state.messages)

            # Optionally append source citations to the response.
            if self.include_sources and state.intermediate_results:
                sources_text = "\n\n**Sources:**\n"
                for i, doc in enumerate(state.intermediate_results, 1):
                    source = doc["metadata"].get("source", "Unknown")
                    sources_text += f"- [{i}] {source}\n"
                response.content += sources_text

            # Add the response to the conversation history.
            state.messages.append(response)

            # Mark the agent as completed after generating the response.
            state.status = AgentStatus.COMPLETED

        return state
