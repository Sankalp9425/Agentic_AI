"""
HTTP request MCP connector for making API calls.

This tool allows agents to make HTTP requests to arbitrary APIs, enabling
integration with REST APIs, webhooks, and other web services. It supports
all standard HTTP methods and custom headers/body content.

This is a general-purpose tool for API integration. For specific services
(like Google Search or email), prefer the dedicated connectors as they
provide better error handling and result formatting.

Requirements:
    pip install requests

Example:
    >>> from agentic_ai.mcp.http_tool import HTTPRequestTool
    >>> http = HTTPRequestTool()
    >>> result = http.execute(
    ...     url="https://api.example.com/data",
    ...     method="GET",
    ...     headers='{"Authorization": "Bearer token123"}',
    ... )
"""

import json
import logging
from typing import Any

import requests

from agentic_ai.core.base_tool import BaseTool

# Configure module-level logger.
logger = logging.getLogger(__name__)


class HTTPRequestTool(BaseTool):
    """
    MCP-compatible tool for making HTTP requests.

    Supports GET, POST, PUT, PATCH, and DELETE methods with custom headers
    and request bodies. Parses JSON responses automatically and includes
    status code and headers in the output.

    Attributes:
        name:        "http_request" - the tool identifier.
        description: Describes the HTTP capability to the LLM.
        parameters:  URL, method, headers, and body parameters.
        _timeout:    Request timeout in seconds.
        _max_response_chars: Maximum characters to return from response body.

    Example:
        >>> tool = HTTPRequestTool(timeout=30)
        >>> tool.execute(url="https://api.github.com/repos/python/cpython", method="GET")
    """

    name = "http_request"
    description = (
        "Make an HTTP request to any URL. Supports GET, POST, PUT, PATCH, and DELETE "
        "methods. Use this to interact with REST APIs, fetch data from endpoints, "
        "or send data to webhooks."
    )
    parameters = {
        "url": {
            "type": "string",
            "description": "The full URL to send the request to.",
            "required": True,
        },
        "method": {
            "type": "string",
            "description": "HTTP method: GET, POST, PUT, PATCH, or DELETE. Default is 'GET'.",
            "required": False,
        },
        "headers": {
            "type": "string",
            "description": "JSON string of request headers (e.g., '{\"Authorization\": \"Bearer token\"}').",
            "required": False,
        },
        "body": {
            "type": "string",
            "description": "Request body content. For JSON APIs, provide a JSON string.",
            "required": False,
        },
    }

    def __init__(
        self,
        timeout: int = 30,
        max_response_chars: int = 5000,
    ) -> None:
        """
        Initialize the HTTP request tool.

        Args:
            timeout:            Request timeout in seconds. Default is 30.
            max_response_chars: Maximum characters from the response body
                                to return. Default is 5000.
        """
        self._timeout = timeout
        self._max_response_chars = max_response_chars

    def execute(self, **kwargs: Any) -> str:
        """
        Make an HTTP request and return the response.

        Sends the request with the specified method, headers, and body.
        Returns the status code, response headers, and body content.

        Args:
            **kwargs: Must include 'url'. Optionally includes 'method',
                      'headers' (JSON string), and 'body'.

        Returns:
            A formatted string with status code, headers, and response body.
        """
        url = kwargs.get("url", "")
        method = kwargs.get("method", "GET").upper()
        headers_str = kwargs.get("headers", "")
        body = kwargs.get("body", "")

        if not url:
            return "Error: 'url' parameter is required."

        # Validate the HTTP method.
        valid_methods = {"GET", "POST", "PUT", "PATCH", "DELETE"}
        if method not in valid_methods:
            return f"Error: Invalid method '{method}'. Use one of: {valid_methods}"

        # Parse headers from JSON string.
        request_headers: dict[str, str] = {}
        if headers_str:
            try:
                request_headers = json.loads(headers_str)
            except json.JSONDecodeError:
                return "Error: 'headers' must be a valid JSON string."

        try:
            logger.info("Making %s request to: %s", method, url)

            # Build the request parameters.
            request_kwargs: dict[str, Any] = {
                "method": method,
                "url": url,
                "headers": request_headers,
                "timeout": self._timeout,
            }

            # Add request body for methods that support it.
            if body and method in {"POST", "PUT", "PATCH"}:
                # Try to parse as JSON; if that fails, send as plain text.
                try:
                    json_body = json.loads(body)
                    request_kwargs["json"] = json_body
                except json.JSONDecodeError:
                    request_kwargs["data"] = body

            # Make the request.
            response = requests.request(**request_kwargs)

            # Format the response.
            status_line = f"Status: {response.status_code} {response.reason}"

            # Include relevant response headers.
            response_headers = dict(response.headers)
            content_type = response_headers.get("Content-Type", "unknown")

            # Get the response body.
            response_body = response.text
            if len(response_body) > self._max_response_chars:
                response_body = (
                    response_body[: self._max_response_chars]
                    + "\n\n[Response truncated...]"
                )

            # Try to pretty-print JSON responses.
            if "json" in content_type.lower():
                try:
                    parsed = response.json()
                    response_body = json.dumps(parsed, indent=2)
                    if len(response_body) > self._max_response_chars:
                        response_body = (
                            response_body[: self._max_response_chars]
                            + "\n\n[Response truncated...]"
                        )
                except (json.JSONDecodeError, ValueError):
                    pass  # Keep the raw text if JSON parsing fails.

            return (
                f"{status_line}\n"
                f"Content-Type: {content_type}\n\n"
                f"{response_body}"
            )

        except requests.exceptions.Timeout:
            return f"Error: Request to {url} timed out after {self._timeout} seconds."

        except requests.exceptions.ConnectionError:
            return f"Error: Could not connect to {url}. Check the URL and your network connection."

        except Exception as e:
            logger.error("HTTP request error: %s", e)
            return f"Error making HTTP request: {e!s}"
