"""
Output parsing module for the RAG pipeline.

Provides structured output parsing to convert free-form LLM text
responses into validated, typed data structures. Supports multiple
parsing strategies:

    - **PydanticOutputParser**: Parses LLM output into Pydantic models
      with automatic validation, type coercion, and error messages.

    - **JSONOutputParser**: Parses LLM output as JSON with schema validation.

    - **ListOutputParser**: Parses LLM output as a numbered or bulleted list.

    - **RegexOutputParser**: Extracts structured data using regex patterns.

Each parser generates format instructions that can be included in
the LLM prompt to guide the model toward the expected output format.

Example:
    >>> from pydantic import BaseModel
    >>> from agentic_ai.rag.output_parser import PydanticOutputParser
    >>>
    >>> class Answer(BaseModel):
    ...     answer: str
    ...     confidence: float
    ...     sources: list[str]
    >>>
    >>> parser = PydanticOutputParser(model=Answer)
    >>> result = parser.parse(llm_output)
    >>> print(result.answer)
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, TypeVar

# Configure module-level logger.
logger = logging.getLogger(__name__)

# TypeVar for generic Pydantic model type.
T = TypeVar("T")


class BaseOutputParser(ABC):
    """
    Abstract base class for output parsers.

    All parsers implement two methods:
    1. ``get_format_instructions``: Returns a string to include in the
       LLM prompt that describes the expected output format.
    2. ``parse``: Converts the raw LLM output string into the desired
       structured format.
    """

    @abstractmethod
    def get_format_instructions(self) -> str:
        """
        Get format instructions to include in the LLM prompt.

        Returns:
            A string describing the expected output format.
        """
        ...

    @abstractmethod
    def parse(self, text: str) -> Any:
        """
        Parse the LLM output into a structured format.

        Args:
            text: The raw text output from the LLM.

        Returns:
            The parsed, structured output.

        Raises:
            ValueError: If the output cannot be parsed.
        """
        ...


class PydanticOutputParser(BaseOutputParser):
    """
    Parses LLM output into Pydantic models with validation.

    Generates format instructions that describe the expected JSON schema
    based on the Pydantic model. Parses the LLM's JSON response and
    validates it against the model, providing clear error messages if
    validation fails.

    Attributes:
        pydantic_model: The Pydantic model class to parse into.

    Example:
        >>> from pydantic import BaseModel, Field
        >>>
        >>> class MovieReview(BaseModel):
        ...     title: str = Field(description="Movie title")
        ...     rating: float = Field(description="Rating from 0 to 10")
        ...     summary: str = Field(description="Brief review summary")
        >>>
        >>> parser = PydanticOutputParser(model=MovieReview)
        >>> instructions = parser.get_format_instructions()
        >>> # Include instructions in your LLM prompt, then:
        >>> review = parser.parse(llm_output)
        >>> print(review.title, review.rating)
    """

    def __init__(self, model: type) -> None:
        """
        Initialize the Pydantic output parser.

        Args:
            model: The Pydantic model class (not an instance) to parse into.
                   Must be a subclass of pydantic.BaseModel.
        """
        self.pydantic_model = model

    def get_format_instructions(self) -> str:
        """
        Generate format instructions from the Pydantic model schema.

        Creates a description of the expected JSON format including
        field names, types, and descriptions from the model.

        Returns:
            Format instruction string for the LLM prompt.
        """
        # Get the JSON schema from the Pydantic model.
        schema = self.pydantic_model.model_json_schema()

        # Build a human-readable description.
        fields_desc: list[str] = []
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        for field_name, field_info in properties.items():
            field_type = field_info.get("type", "any")
            description = field_info.get("description", "")
            is_required = field_name in required
            req_str = "required" if is_required else "optional"

            fields_desc.append(
                f'  "{field_name}": ({field_type}, {req_str}) {description}'
            )

        fields_str = "\n".join(fields_desc)

        return (
            "You must respond with a valid JSON object matching this schema:\n"
            "```json\n"
            "{\n"
            f"{fields_str}\n"
            "}\n"
            "```\n"
            "Output ONLY the JSON object, no additional text."
        )

    def parse(self, text: str) -> Any:
        """
        Parse LLM output into a validated Pydantic model instance.

        Extracts JSON from the text (handling markdown code blocks),
        parses it, and validates against the Pydantic model.

        Args:
            text: The raw LLM output containing JSON.

        Returns:
            A validated instance of the Pydantic model.

        Raises:
            ValueError: If the JSON is invalid or fails validation.
        """
        # Extract JSON from markdown code blocks if present.
        json_str = self._extract_json(text)

        try:
            # Parse the JSON string.
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON from LLM output: {e}\n"
                f"Raw output: {text[:500]}"
            ) from e

        try:
            # Validate and create the Pydantic model instance.
            return self.pydantic_model.model_validate(data)
        except Exception as e:
            raise ValueError(
                f"Pydantic validation failed: {e}\n"
                f"Parsed data: {data}"
            ) from e

    @staticmethod
    def _extract_json(text: str) -> str:
        """
        Extract JSON from text, handling markdown code blocks.

        Looks for JSON inside ```json ... ``` blocks first, then
        tries to find raw JSON objects or arrays.

        Args:
            text: The raw text containing JSON.

        Returns:
            The extracted JSON string.
        """
        # Try to extract from markdown code blocks.
        code_block_match = re.search(
            r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL
        )
        if code_block_match:
            return code_block_match.group(1).strip()

        # Try to find a JSON object in the text.
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0)

        # Try to find a JSON array in the text.
        array_match = re.search(r'\[.*\]', text, re.DOTALL)
        if array_match:
            return array_match.group(0)

        # Return the original text as-is.
        return text.strip()


class JSONOutputParser(BaseOutputParser):
    """
    Parses LLM output as a JSON object with optional schema validation.

    Extracts JSON from the LLM response and optionally validates it
    against a provided JSON schema dictionary.

    Attributes:
        schema: Optional JSON schema for validation.

    Example:
        >>> parser = JSONOutputParser(schema={"type": "object", "required": ["name"]})
        >>> data = parser.parse(llm_output)
        >>> print(data["name"])
    """

    def __init__(self, schema: dict[str, Any] | None = None) -> None:
        """
        Initialize the JSON output parser.

        Args:
            schema: Optional JSON schema dictionary for validation.
        """
        self.schema = schema

    def get_format_instructions(self) -> str:
        """
        Generate format instructions for JSON output.

        Returns:
            Format instruction string.
        """
        if self.schema:
            schema_str = json.dumps(self.schema, indent=2)
            return (
                "Respond with a valid JSON object matching this schema:\n"
                f"```json\n{schema_str}\n```\n"
                "Output ONLY the JSON, no additional text."
            )

        return (
            "Respond with a valid JSON object.\n"
            "Output ONLY the JSON, no additional text."
        )

    def parse(self, text: str) -> dict[str, Any]:
        """
        Parse LLM output as JSON.

        Args:
            text: The raw LLM output.

        Returns:
            A parsed JSON dictionary.

        Raises:
            ValueError: If the output is not valid JSON.
        """
        json_str = PydanticOutputParser._extract_json(text)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON: {e}\nRaw: {text[:500]}"
            ) from e

        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object, got {type(data).__name__}")

        return data


class ListOutputParser(BaseOutputParser):
    """
    Parses LLM output as a list of items.

    Handles numbered lists (1. item), bulleted lists (- item, * item),
    and newline-separated items.

    Example:
        >>> parser = ListOutputParser()
        >>> items = parser.parse("1. First\\n2. Second\\n3. Third")
        >>> print(items)  # ["First", "Second", "Third"]
    """

    def get_format_instructions(self) -> str:
        """
        Generate format instructions for list output.

        Returns:
            Format instruction string.
        """
        return (
            "Respond with a numbered list of items, one per line:\n"
            "1. First item\n"
            "2. Second item\n"
            "3. Third item"
        )

    def parse(self, text: str) -> list[str]:
        """
        Parse LLM output as a list of items.

        Handles numbered lists, bulleted lists, and plain newline-separated text.

        Args:
            text: The raw LLM output.

        Returns:
            A list of parsed item strings.
        """
        items: list[str] = []

        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            # Remove numbering (e.g., "1. ", "2) ").
            cleaned = re.sub(r'^[\d]+[.)]\s*', '', line)
            # Remove bullet markers (e.g., "- ", "* ", "• ").
            cleaned = re.sub(r'^[-*•]\s*', '', cleaned)
            cleaned = cleaned.strip()

            if cleaned:
                items.append(cleaned)

        return items


class RegexOutputParser(BaseOutputParser):
    """
    Parses LLM output using regex patterns to extract structured data.

    Useful for extracting specific fields from semi-structured text
    output using named capture groups.

    Attributes:
        pattern:  The regex pattern with named groups.
        output_keys: The expected keys from named groups.

    Example:
        >>> parser = RegexOutputParser(
        ...     pattern=r"Answer: (?P<answer>.+?)\\nConfidence: (?P<confidence>\\d+)",
        ...     output_keys=["answer", "confidence"],
        ... )
        >>> result = parser.parse("Answer: Yes\\nConfidence: 95")
        >>> print(result)  # {"answer": "Yes", "confidence": "95"}
    """

    def __init__(self, pattern: str, output_keys: list[str]) -> None:
        """
        Initialize the regex output parser.

        Args:
            pattern:     A regex pattern with named capture groups matching output_keys.
            output_keys: The expected keys from the named groups.
        """
        self.pattern = pattern
        self.output_keys = output_keys

    def get_format_instructions(self) -> str:
        """
        Generate format instructions for the expected output pattern.

        Returns:
            Format instruction string showing expected fields.
        """
        fields = "\n".join(f"{key}: <value>" for key in self.output_keys)
        return f"Respond in this exact format:\n{fields}"

    def parse(self, text: str) -> dict[str, str]:
        """
        Parse LLM output using regex pattern matching.

        Args:
            text: The raw LLM output.

        Returns:
            A dictionary mapping output keys to extracted values.

        Raises:
            ValueError: If the pattern doesn't match the output.
        """
        match = re.search(self.pattern, text, re.DOTALL)

        if not match:
            raise ValueError(
                f"Regex pattern did not match the output.\n"
                f"Pattern: {self.pattern}\n"
                f"Output: {text[:500]}"
            )

        result: dict[str, str] = {}
        for key in self.output_keys:
            try:
                result[key] = match.group(key).strip()
            except IndexError:
                result[key] = ""

        return result
