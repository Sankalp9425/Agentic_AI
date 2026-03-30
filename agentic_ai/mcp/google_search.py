"""
Google Search MCP connector for web search capabilities.

This tool allows agents to perform web searches using Google's Custom Search
JSON API. It returns structured search results including titles, URLs, and
snippets that the agent can use for information gathering and research tasks.

To use this tool, you need:
    1. A Google Cloud project with the Custom Search API enabled.
    2. An API key from the Google Cloud Console.
    3. A Custom Search Engine (CSE) ID configured to search the web.

Setup guide: https://developers.google.com/custom-search/v1/overview

Requirements:
    pip install requests

Example:
    >>> from agentic_ai.mcp.google_search import GoogleSearchTool
    >>> search = GoogleSearchTool(
    ...     api_key="AIza...",
    ...     search_engine_id="your-cse-id",
    ... )
    >>> results = search.execute(query="Python best practices 2024", num_results="5")
"""

import logging
from typing import Any

import requests

from agentic_ai.core.base_tool import BaseTool

# Configure module-level logger.
logger = logging.getLogger(__name__)

# Google Custom Search API endpoint.
GOOGLE_SEARCH_API_URL = "https://www.googleapis.com/customsearch/v1"


class GoogleSearchTool(BaseTool):
    """
    MCP-compatible tool for performing Google web searches.

    Uses Google's Custom Search JSON API to perform web searches and return
    structured results. Each result includes the page title, URL, and a
    text snippet for the agent to evaluate.

    Attributes:
        name:        "google_search" - the tool identifier.
        description: Describes the search capability to the LLM.
        parameters:  Defines search query and result count parameters.
        _api_key:    Google Cloud API key for authentication.
        _cse_id:     Custom Search Engine ID for web search scope.

    Example:
        >>> tool = GoogleSearchTool(api_key="AIza...", search_engine_id="abc123")
        >>> tool.execute(query="latest AI research papers")
    """

    name = "google_search"
    description = (
        "Search the web using Google. Returns titles, URLs, and snippets for each "
        "result. Use this to find current information, research topics, or verify facts."
    )
    parameters = {
        "query": {
            "type": "string",
            "description": "The search query string (e.g., 'Python web frameworks comparison').",
            "required": True,
        },
        "num_results": {
            "type": "string",
            "description": "Number of results to return (1-10). Default is '5'.",
            "required": False,
        },
    }

    def __init__(
        self,
        api_key: str,
        search_engine_id: str,
        timeout: int = 10,
    ) -> None:
        """
        Initialize the Google Search tool.

        Args:
            api_key:          Google Cloud API key with Custom Search API enabled.
            search_engine_id: The Custom Search Engine (CSE) ID. Create one at
                              https://programmablesearchengine.google.com/
            timeout:          HTTP request timeout in seconds. Default is 10.
        """
        self._api_key = api_key
        self._cse_id = search_engine_id
        self._timeout = timeout

    def execute(self, **kwargs: Any) -> str:
        """
        Perform a Google search and return formatted results.

        Calls the Google Custom Search API, parses the response, and
        formats the results as a readable string for the agent.

        Args:
            **kwargs: Must include 'query'. Optionally includes 'num_results'.

        Returns:
            A formatted string with search results (title, URL, snippet per result),
            or an error message if the search fails.
        """
        query = kwargs.get("query", "")
        num_results_str = kwargs.get("num_results", "5")

        if not query:
            return "Error: 'query' parameter is required."

        # Parse and validate the number of results.
        try:
            num_results = int(num_results_str)
            num_results = max(1, min(10, num_results))  # Clamp to 1-10.
        except (ValueError, TypeError):
            num_results = 5

        try:
            # Build the API request parameters.
            params = {
                "key": self._api_key,
                "cx": self._cse_id,
                "q": query,
                "num": num_results,
            }

            logger.info("Performing Google search: '%s'", query)

            # Make the API request.
            response = requests.get(
                GOOGLE_SEARCH_API_URL,
                params=params,
                timeout=self._timeout,
            )
            response.raise_for_status()

            # Parse the JSON response.
            data = response.json()
            items = data.get("items", [])

            if not items:
                return f"No results found for query: '{query}'"

            # Format results as a readable string.
            formatted_results: list[str] = []
            for i, item in enumerate(items, 1):
                title = item.get("title", "No title")
                link = item.get("link", "No URL")
                snippet = item.get("snippet", "No description available.")

                formatted_results.append(
                    f"{i}. **{title}**\n"
                    f"   URL: {link}\n"
                    f"   {snippet}"
                )

            # Include total results count for context.
            total_results = data.get("searchInformation", {}).get(
                "formattedTotalResults", "unknown"
            )

            header = f"Search results for '{query}' ({total_results} total results):\n\n"
            return header + "\n\n".join(formatted_results)

        except requests.exceptions.Timeout:
            logger.error("Google search timed out for query: '%s'", query)
            return "Error: Search request timed out. Try again or simplify the query."

        except requests.exceptions.HTTPError as e:
            logger.error("Google search API error: %s", e)
            if e.response is not None and e.response.status_code == 403:
                return "Error: API key is invalid or quota exceeded. Check your Google Cloud API key."
            return f"Error: Search API returned an error: {e!s}"

        except Exception as e:
            logger.error("Unexpected error during search: %s", e)
            return f"Error performing search: {e!s}"
