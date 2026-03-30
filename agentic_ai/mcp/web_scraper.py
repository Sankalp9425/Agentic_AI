"""
Web scraper MCP connector for extracting content from web pages.

This tool allows agents to fetch and extract readable text content from
web pages. It uses the requests library for HTTP fetching and BeautifulSoup
for HTML parsing, extracting the main text content while stripping scripts,
styles, and navigation elements.

This is useful for:
    - Following up on search results to read full articles.
    - Extracting data from specific web pages.
    - Gathering information from documentation sites.

Requirements:
    pip install requests beautifulsoup4

Example:
    >>> from agentic_ai.mcp.web_scraper import WebScraperTool
    >>> scraper = WebScraperTool()
    >>> content = scraper.execute(url="https://example.com/article")
"""

import logging
from typing import Any

import requests

from agentic_ai.core.base_tool import BaseTool

# Configure module-level logger.
logger = logging.getLogger(__name__)


class WebScraperTool(BaseTool):
    """
    MCP-compatible tool for scraping text content from web pages.

    Fetches a URL and extracts the main text content using BeautifulSoup.
    Strips scripts, styles, and navigation elements to return clean,
    readable text. Includes a character limit to prevent overwhelming
    the agent with extremely long pages.

    Attributes:
        name:        "web_scraper" - the tool identifier.
        description: Describes the scraping capability to the LLM.
        parameters:  URL and optional max character limit.
        _timeout:    HTTP request timeout in seconds.
        _max_chars:  Maximum characters to return from scraped content.
        _headers:    HTTP headers sent with requests (includes User-Agent).

    Example:
        >>> scraper = WebScraperTool(max_chars=5000)
        >>> content = scraper.execute(url="https://docs.python.org/3/")
    """

    name = "web_scraper"
    description = (
        "Fetch and extract readable text content from a web page URL. "
        "Returns the main text content, stripped of HTML tags, scripts, and styles. "
        "Use this to read articles, documentation, or any web page content."
    )
    parameters = {
        "url": {
            "type": "string",
            "description": "The full URL to scrape (must start with http:// or https://).",
            "required": True,
        },
        "max_chars": {
            "type": "string",
            "description": "Maximum characters to return. Default is '5000'.",
            "required": False,
        },
    }

    def __init__(
        self,
        timeout: int = 15,
        max_chars: int = 5000,
    ) -> None:
        """
        Initialize the web scraper tool.

        Args:
            timeout:   HTTP request timeout in seconds. Default is 15.
            max_chars: Maximum characters to return. Default is 5000.
        """
        self._timeout = timeout
        self._max_chars = max_chars
        # Set a realistic User-Agent to avoid being blocked by websites.
        self._headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; AgenticAI/1.0; "
                "+https://github.com/agentic-ai)"
            ),
        }

    def execute(self, **kwargs: Any) -> str:
        """
        Fetch a web page and extract its readable text content.

        Downloads the HTML, parses it with BeautifulSoup, removes non-content
        elements, and returns the cleaned text. Truncates to max_chars if
        the content is too long.

        Args:
            **kwargs: Must include 'url'. Optionally includes 'max_chars'.

        Returns:
            The extracted text content, or an error message.
        """
        url = kwargs.get("url", "")
        max_chars_str = kwargs.get("max_chars", str(self._max_chars))

        if not url:
            return "Error: 'url' parameter is required."

        # Validate URL format.
        if not url.startswith(("http://", "https://")):
            return "Error: URL must start with 'http://' or 'https://'."

        # Parse the max_chars parameter.
        try:
            max_chars = int(max_chars_str)
        except (ValueError, TypeError):
            max_chars = self._max_chars

        try:
            # Import BeautifulSoup for HTML parsing.
            try:
                from bs4 import BeautifulSoup
            except ImportError as e:
                raise ImportError(
                    "The 'beautifulsoup4' package is required for WebScraperTool. "
                    "Install it with: pip install beautifulsoup4"
                ) from e

            logger.info("Scraping URL: %s", url)

            # Fetch the web page.
            response = requests.get(
                url,
                headers=self._headers,
                timeout=self._timeout,
            )
            response.raise_for_status()

            # Parse the HTML content.
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove non-content elements that add noise.
            for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
                element.decompose()

            # Extract the text content.
            text = soup.get_text(separator="\n", strip=True)

            # Clean up excessive whitespace while preserving paragraph breaks.
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            cleaned_text = "\n".join(lines)

            # Truncate if the content exceeds the character limit.
            if len(cleaned_text) > max_chars:
                cleaned_text = cleaned_text[:max_chars] + "\n\n[Content truncated...]"

            logger.info(
                "Scraped %d characters from %s", len(cleaned_text), url
            )
            return f"Content from {url}:\n\n{cleaned_text}"

        except requests.exceptions.Timeout:
            return f"Error: Request to {url} timed out after {self._timeout} seconds."

        except requests.exceptions.HTTPError as e:
            return f"Error: HTTP {e.response.status_code if e.response is not None else 'unknown'} when fetching {url}."

        except Exception as e:
            logger.error("Error scraping %s: %s", url, e)
            return f"Error scraping URL: {e!s}"
