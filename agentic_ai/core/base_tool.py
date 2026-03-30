"""
Abstract base class for all tools that agents can invoke.

Tools are the mechanism by which agents interact with the external world.
Each tool represents a specific capability (e.g., web search, calculator,
file reader, API call) that the agent can choose to invoke during its
reasoning process. Tools are described to the LLM via JSON schemas, and
the LLM emits ToolCall objects to request their execution.

The framework automatically converts BaseTool subclasses into the JSON
schema format expected by LLM providers, so tool authors only need to
define the tool's name, description, parameters, and execution logic.

Example:
    >>> class Calculator(BaseTool):
    ...     name = "calculator"
    ...     description = "Evaluate a mathematical expression."
    ...     parameters = {"expression": {"type": "string", "description": "Math expression"}}
    ...     def execute(self, expression: str) -> str:
    ...         return str(eval(expression))
    >>> tool = Calculator()
    >>> tool.execute(expression="2 + 2")
    '4'
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """
    Abstract base class that all tools must inherit from.

    Subclasses must define the class-level attributes `name`, `description`,
    and `parameters`, and implement the `execute()` method. The framework
    uses these attributes to generate tool schemas for LLM function calling
    and to route ToolCall requests to the correct tool.

    Class Attributes:
        name:        A unique identifier for the tool (e.g., "web_search").
                     Must be a valid Python identifier (letters, digits, underscores).
        description: A human-readable description of what the tool does. This
                     description is sent to the LLM to help it decide when to
                     use the tool, so it should be clear and concise.
        parameters:  A dictionary describing the tool's input parameters in
                     JSON Schema format. Each key is a parameter name, and the
                     value describes its type and purpose.

    Methods:
        execute(**kwargs):    Run the tool with the given arguments.
        to_schema():          Convert the tool to a JSON schema for LLM APIs.
        validate_args(**kwargs): Validate arguments before execution.
    """

    # Subclasses must override these class attributes.
    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}

    @abstractmethod
    def execute(self, **kwargs: Any) -> str:
        """
        Execute the tool with the provided keyword arguments.

        This is the core method that performs the tool's action. Implementations
        should handle errors gracefully and return informative error messages
        rather than raising exceptions, since the output is fed back to the LLM.

        Args:
            **kwargs: Keyword arguments matching the tool's parameter schema.
                      The exact arguments depend on the specific tool.

        Returns:
            A string representation of the tool's output. This string is
            included in the conversation history as a TOOL message.

        Raises:
            ValueError: If required arguments are missing or invalid.
        """
        ...

    def to_schema(self) -> dict[str, Any]:
        """
        Convert the tool definition to a JSON schema dictionary.

        This schema follows the OpenAI function-calling format and is used
        by LLM providers to understand what tools are available and how to
        invoke them. Most providers accept this format directly or with
        minor adaptations.

        Returns:
            A dictionary with 'type', 'function' keys containing the tool's
            name, description, and parameter schema.

        Example:
            >>> tool.to_schema()
            {
                'type': 'function',
                'function': {
                    'name': 'calculator',
                    'description': 'Evaluate a mathematical expression.',
                    'parameters': {
                        'type': 'object',
                        'properties': {'expression': {'type': 'string'}},
                        'required': ['expression']
                    }
                }
            }
        """
        # Build the properties dictionary from the tool's parameters definition.
        properties = {}
        required_params = []
        for param_name, param_info in self.parameters.items():
            # Each parameter must have at least a 'type' field.
            properties[param_name] = {
                "type": param_info.get("type", "string"),
                "description": param_info.get("description", ""),
            }
            # Parameters are required by default unless explicitly marked optional.
            if param_info.get("required", True):
                required_params.append(param_name)

        # Assemble the complete schema in OpenAI function-calling format.
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required_params,
                },
            },
        }

    def validate_args(self, **kwargs: Any) -> bool:
        """
        Validate that the provided arguments match the tool's parameter schema.

        This method checks that all required parameters are present and that
        no unexpected parameters are provided. It does not perform type
        checking beyond presence validation.

        Args:
            **kwargs: The arguments to validate against the parameter schema.

        Returns:
            True if all required parameters are present and no unknown
            parameters are provided.

        Raises:
            ValueError: If required parameters are missing or unknown
                        parameters are provided.
        """
        # Extract the set of required parameter names from the schema.
        required = {
            name
            for name, info in self.parameters.items()
            if info.get("required", True)
        }
        # Check that all required parameters are provided.
        missing = required - set(kwargs.keys())
        if missing:
            raise ValueError(
                f"Missing required parameters for tool '{self.name}': {missing}"
            )

        # Check that no unknown parameters are provided.
        known = set(self.parameters.keys())
        unknown = set(kwargs.keys()) - known
        if unknown:
            raise ValueError(
                f"Unknown parameters for tool '{self.name}': {unknown}"
            )

        return True
