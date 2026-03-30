"""
RAG pipeline evaluation module.

Provides metrics and evaluation tools for assessing the quality of
RAG pipeline outputs. Implements common evaluation criteria:

    - **Faithfulness**: Does the answer only contain information
      from the retrieved context? (No hallucination.)

    - **AnswerRelevance**: Is the answer relevant to the question?

    - **ContextRelevance**: Are the retrieved documents relevant
      to the question?

    - **ContextPrecision**: What fraction of retrieved documents
      are actually relevant? (Precision metric.)

    - **AnswerCorrectness**: Does the answer match a reference
      answer? (Requires ground truth.)

All evaluators use an LLM as a judge to score quality on a 0-1 scale,
following the LLM-as-judge paradigm used by RAGAS and similar frameworks.

Example:
    >>> from agentic_ai.rag.evaluation import RAGEvaluator
    >>> evaluator = RAGEvaluator(llm=judge_llm)
    >>> scores = evaluator.evaluate(
    ...     question="What is RAG?",
    ...     answer="RAG combines retrieval with generation...",
    ...     contexts=["RAG is a technique that..."],
    ...     reference="RAG retrieves documents and uses them...",
    ... )
    >>> print(scores)
    # {"faithfulness": 0.9, "answer_relevance": 0.85, ...}
"""

import logging
import re
from dataclasses import dataclass, field

from agentic_ai.core.base_llm import BaseLLM
from agentic_ai.core.models import Message, Role

# Configure module-level logger.
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """
    Container for RAG evaluation results.

    Attributes:
        faithfulness:      Score (0-1) for how well the answer is grounded
                           in the retrieved context.
        answer_relevance:  Score (0-1) for how relevant the answer is
                           to the question.
        context_relevance: Score (0-1) for how relevant the retrieved
                           contexts are to the question.
        context_precision: Score (0-1) for the fraction of retrieved
                           contexts that are relevant.
        answer_correctness: Score (0-1) for how correct the answer is
                            compared to a reference answer.
        details:           Additional details or explanations from the evaluator.
    """

    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_relevance: float = 0.0
    context_precision: float = 0.0
    answer_correctness: float = 0.0
    details: dict[str, str] = field(default_factory=dict)

    def overall_score(self) -> float:
        """
        Compute a weighted average of all evaluation metrics.

        Returns:
            The overall quality score (0.0 to 1.0).
        """
        scores = [
            self.faithfulness,
            self.answer_relevance,
            self.context_relevance,
            self.context_precision,
            self.answer_correctness,
        ]
        # Filter out zero scores (metrics not evaluated).
        non_zero = [s for s in scores if s > 0]
        if not non_zero:
            return 0.0
        return sum(non_zero) / len(non_zero)

    def to_dict(self) -> dict[str, float | str]:
        """
        Convert evaluation results to a dictionary.

        Returns:
            A dictionary with metric names and scores.
        """
        return {
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "context_relevance": self.context_relevance,
            "context_precision": self.context_precision,
            "answer_correctness": self.answer_correctness,
            "overall_score": self.overall_score(),
        }


class RAGEvaluator:
    """
    Comprehensive evaluator for RAG pipeline outputs.

    Uses an LLM as a judge to evaluate multiple quality dimensions
    of a RAG system's output. Each metric is scored on a 0-1 scale.

    Attributes:
        llm: The LLM used as a judge for evaluation.

    Example:
        >>> evaluator = RAGEvaluator(llm=judge_llm)
        >>> result = evaluator.evaluate(
        ...     question="What is Python?",
        ...     answer="Python is a programming language...",
        ...     contexts=["Python is a high-level language..."],
        ... )
        >>> print(result.faithfulness, result.answer_relevance)
    """

    def __init__(self, llm: BaseLLM) -> None:
        """
        Initialize the RAG evaluator.

        Args:
            llm: The LLM to use as a judge. Should be a capable model
                 (e.g., GPT-4, Claude 3 Opus) for reliable evaluation.
        """
        self.llm = llm

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        reference: str | None = None,
    ) -> EvaluationResult:
        """
        Evaluate a RAG pipeline output across all metrics.

        Args:
            question:  The user's input question.
            answer:    The RAG pipeline's generated answer.
            contexts:  The retrieved context documents used to generate the answer.
            reference: Optional ground-truth reference answer for correctness scoring.

        Returns:
            An EvaluationResult with scores for each metric.
        """
        result = EvaluationResult()

        # Evaluate faithfulness (is the answer grounded in context?).
        result.faithfulness = self._evaluate_faithfulness(
            question, answer, contexts
        )

        # Evaluate answer relevance (does the answer address the question?).
        result.answer_relevance = self._evaluate_answer_relevance(
            question, answer
        )

        # Evaluate context relevance (are the contexts relevant?).
        result.context_relevance = self._evaluate_context_relevance(
            question, contexts
        )

        # Evaluate context precision (what fraction of contexts are useful?).
        result.context_precision = self._evaluate_context_precision(
            question, contexts
        )

        # Evaluate answer correctness (if reference is provided).
        if reference:
            result.answer_correctness = self._evaluate_answer_correctness(
                question, answer, reference
            )

        logger.info(
            "RAG evaluation complete — overall score: %.2f",
            result.overall_score(),
        )

        return result

    def _evaluate_faithfulness(
        self, question: str, answer: str, contexts: list[str]
    ) -> float:
        """
        Evaluate if the answer is grounded in the retrieved contexts.

        Checks whether every claim in the answer can be traced back
        to information in the provided contexts. A high faithfulness
        score means low hallucination.

        Args:
            question: The user's question.
            answer:   The generated answer.
            contexts: The retrieved context documents.

        Returns:
            Faithfulness score (0.0 to 1.0).
        """
        from agentic_ai.prompts.templates import FAITHFULNESS_EVAL_PROMPT

        context_str = "\n---\n".join(contexts)
        prompt = FAITHFULNESS_EVAL_PROMPT.format(
            question=question,
            answer=answer,
            context=context_str,
        )

        return self._get_score(prompt, "faithfulness")

    def _evaluate_answer_relevance(
        self, question: str, answer: str
    ) -> float:
        """
        Evaluate if the answer is relevant to the question.

        Checks whether the answer actually addresses what was asked,
        regardless of factual accuracy.

        Args:
            question: The user's question.
            answer:   The generated answer.

        Returns:
            Relevance score (0.0 to 1.0).
        """
        from agentic_ai.prompts.templates import ANSWER_RELEVANCE_EVAL_PROMPT

        prompt = ANSWER_RELEVANCE_EVAL_PROMPT.format(
            question=question,
            answer=answer,
        )

        return self._get_score(prompt, "answer_relevance")

    def _evaluate_context_relevance(
        self, question: str, contexts: list[str]
    ) -> float:
        """
        Evaluate if the retrieved contexts are relevant to the question.

        Scores how well the retrieved documents relate to the question.

        Args:
            question: The user's question.
            contexts: The retrieved context documents.

        Returns:
            Context relevance score (0.0 to 1.0).
        """
        from agentic_ai.prompts.templates import CONTEXT_RELEVANCE_EVAL_PROMPT

        context_str = "\n---\n".join(contexts)
        prompt = CONTEXT_RELEVANCE_EVAL_PROMPT.format(
            question=question,
            context=context_str,
        )

        return self._get_score(prompt, "context_relevance")

    def _evaluate_context_precision(
        self, question: str, contexts: list[str]
    ) -> float:
        """
        Evaluate what fraction of retrieved contexts are relevant.

        Asks the LLM to judge each context individually, then computes
        the precision as the fraction of relevant contexts.

        Args:
            question: The user's question.
            contexts: The retrieved context documents.

        Returns:
            Precision score (0.0 to 1.0).
        """
        if not contexts:
            return 0.0

        relevant_count = 0
        for ctx in contexts:
            prompt = (
                f"Is the following context relevant to answering the question?\n\n"
                f"Question: {question}\n\n"
                f"Context: {ctx[:500]}\n\n"
                f"Respond with only 'YES' or 'NO'."
            )
            messages = [Message(role=Role.USER, content=prompt)]

            try:
                response = self.llm.chat(messages)
                if "YES" in response.content.upper():
                    relevant_count += 1
            except Exception as e:
                logger.warning("Context precision eval failed: %s", e)

        return relevant_count / len(contexts)

    def _evaluate_answer_correctness(
        self, question: str, answer: str, reference: str
    ) -> float:
        """
        Evaluate how correct the answer is compared to a reference.

        Compares the generated answer to a ground-truth reference answer,
        scoring based on factual overlap and accuracy.

        Args:
            question:  The user's question.
            answer:    The generated answer.
            reference: The ground-truth reference answer.

        Returns:
            Correctness score (0.0 to 1.0).
        """
        from agentic_ai.prompts.templates import ANSWER_CORRECTNESS_EVAL_PROMPT

        prompt = ANSWER_CORRECTNESS_EVAL_PROMPT.format(
            question=question,
            answer=answer,
            reference=reference,
        )

        return self._get_score(prompt, "answer_correctness")

    def _get_score(self, prompt: str, metric_name: str) -> float:
        """
        Send an evaluation prompt to the LLM and extract a numeric score.

        Args:
            prompt:      The evaluation prompt.
            metric_name: The name of the metric being evaluated (for logging).

        Returns:
            A score between 0.0 and 1.0.
        """
        messages = [Message(role=Role.USER, content=prompt)]

        try:
            response = self.llm.chat(messages)
            score_text = response.content.strip()

            # Extract numeric score from the response.
            numbers = re.findall(r'(\d+(?:\.\d+)?)', score_text)
            if numbers:
                score = float(numbers[0])
                # Normalize to 0-1 range if the score is on a 0-10 scale.
                if score > 1.0:
                    score = score / 10.0
                score = min(max(score, 0.0), 1.0)
                logger.debug("%s score: %.2f", metric_name, score)
                return score

        except Exception as e:
            logger.warning("Failed to evaluate %s: %s", metric_name, e)

        return 0.0
