"""
File operation MCP connectors for reading and writing local files.

These tools allow agents to interact with the local file system,
enabling tasks like reading configuration files, writing reports,
processing data files, and managing local resources.

Security note: File operations are restricted to a configurable base
directory to prevent unauthorized access to the system. By default,
only the current working directory is accessible.

Example:
    >>> from agentic_ai.mcp.file_tools import FileReaderTool, FileWriterTool
    >>> reader = FileReaderTool(base_directory="/home/user/data")
    >>> writer = FileWriterTool(base_directory="/home/user/output")
    >>> agent = ReActAgent(llm=llm, tools=[reader, writer])
"""

import logging
from pathlib import Path
from typing import Any

from agentic_ai.core.base_tool import BaseTool

# Configure module-level logger.
logger = logging.getLogger(__name__)


class FileReaderTool(BaseTool):
    """
    MCP-compatible tool for reading file contents.

    Reads text files from the local file system within a sandboxed base
    directory. Supports reading entire files or specific line ranges.

    Attributes:
        name:        "read_file" - the tool identifier.
        description: Describes the reading capability to the LLM.
        parameters:  File path and optional line range parameters.
        _base_dir:   The base directory for sandboxed file access.
        _max_chars:  Maximum characters to read from a file.

    Example:
        >>> reader = FileReaderTool(base_directory="/home/user/project")
        >>> content = reader.execute(file_path="config.yaml")
    """

    name = "read_file"
    description = (
        "Read the contents of a file. Provide a file path relative to the "
        "working directory. Returns the file's text content."
    )
    parameters = {
        "file_path": {
            "type": "string",
            "description": "Path to the file to read (relative to working directory).",
            "required": True,
        },
        "max_chars": {
            "type": "string",
            "description": "Maximum characters to read. Default is '10000'.",
            "required": False,
        },
    }

    def __init__(
        self,
        base_directory: str = ".",
        max_chars: int = 10000,
    ) -> None:
        """
        Initialize the file reader with a sandboxed base directory.

        Args:
            base_directory: The root directory for file access. All file paths
                            are resolved relative to this directory, and access
                            outside it is denied. Default is current directory.
            max_chars:      Maximum characters to read. Default is 10000.
        """
        # Resolve the base directory to an absolute path.
        self._base_dir = Path(base_directory).resolve()
        self._max_chars = max_chars

    def _resolve_safe_path(self, file_path: str) -> Path | None:
        """
        Resolve a file path and verify it's within the sandbox.

        Prevents directory traversal attacks by ensuring the resolved path
        is within the configured base directory.

        Args:
            file_path: The requested file path (may be relative or absolute).

        Returns:
            The resolved Path if safe, or None if the path escapes the sandbox.
        """
        # Resolve the full path relative to the base directory.
        resolved = (self._base_dir / file_path).resolve()

        # Verify the resolved path is within the base directory.
        try:
            resolved.relative_to(self._base_dir)
            return resolved
        except ValueError:
            # Path escapes the sandbox.
            return None

    def execute(self, **kwargs: Any) -> str:
        """
        Read the contents of a file.

        Resolves the file path, checks it's within the sandbox, reads the
        content, and returns it. Handles encoding errors gracefully.

        Args:
            **kwargs: Must include 'file_path'. Optionally includes 'max_chars'.

        Returns:
            The file's text content, or an error message.
        """
        file_path = kwargs.get("file_path", "")
        max_chars_str = kwargs.get("max_chars", str(self._max_chars))

        if not file_path:
            return "Error: 'file_path' parameter is required."

        try:
            max_chars = int(max_chars_str)
        except (ValueError, TypeError):
            max_chars = self._max_chars

        # Resolve and validate the file path.
        safe_path = self._resolve_safe_path(file_path)
        if safe_path is None:
            return f"Error: Access denied. Path '{file_path}' is outside the allowed directory."

        # Check if the file exists.
        if not safe_path.exists():
            return f"Error: File not found: '{file_path}'"

        if not safe_path.is_file():
            return f"Error: '{file_path}' is not a regular file."

        try:
            # Read the file content with UTF-8 encoding.
            content = safe_path.read_text(encoding="utf-8")

            # Truncate if content exceeds the limit.
            if len(content) > max_chars:
                content = content[:max_chars] + "\n\n[File content truncated...]"

            logger.info("Read %d characters from: %s", len(content), file_path)
            return content

        except UnicodeDecodeError:
            return f"Error: File '{file_path}' is not a valid text file (binary or unsupported encoding)."

        except PermissionError:
            return f"Error: Permission denied reading file: '{file_path}'"

        except Exception as e:
            return f"Error reading file: {e!s}"


class FileWriterTool(BaseTool):
    """
    MCP-compatible tool for writing content to files.

    Writes text content to files within a sandboxed base directory.
    Supports creating new files and overwriting or appending to existing ones.

    Attributes:
        name:        "write_file" - the tool identifier.
        description: Describes the writing capability to the LLM.
        parameters:  File path, content, and write mode parameters.
        _base_dir:   The base directory for sandboxed file access.

    Example:
        >>> writer = FileWriterTool(base_directory="/home/user/output")
        >>> result = writer.execute(
        ...     file_path="report.md",
        ...     content="# Analysis Report\n\nFindings...",
        ... )
    """

    name = "write_file"
    description = (
        "Write content to a file. Creates the file if it doesn't exist. "
        "Can overwrite existing content or append to the file."
    )
    parameters = {
        "file_path": {
            "type": "string",
            "description": "Path to the file to write (relative to working directory).",
            "required": True,
        },
        "content": {
            "type": "string",
            "description": "The text content to write to the file.",
            "required": True,
        },
        "mode": {
            "type": "string",
            "description": "Write mode: 'overwrite' (default) or 'append'.",
            "required": False,
        },
    }

    def __init__(
        self,
        base_directory: str = ".",
    ) -> None:
        """
        Initialize the file writer with a sandboxed base directory.

        Args:
            base_directory: The root directory for file access. Default is
                            the current directory.
        """
        self._base_dir = Path(base_directory).resolve()

    def _resolve_safe_path(self, file_path: str) -> Path | None:
        """
        Resolve a file path and verify it's within the sandbox.

        Args:
            file_path: The requested file path.

        Returns:
            The resolved Path if safe, or None if it escapes the sandbox.
        """
        resolved = (self._base_dir / file_path).resolve()
        try:
            resolved.relative_to(self._base_dir)
            return resolved
        except ValueError:
            return None

    def execute(self, **kwargs: Any) -> str:
        """
        Write content to a file.

        Creates parent directories if they don't exist. Supports both
        overwrite and append modes.

        Args:
            **kwargs: Must include 'file_path' and 'content'. Optionally
                      includes 'mode' ("overwrite" or "append").

        Returns:
            A success confirmation or error message.
        """
        file_path = kwargs.get("file_path", "")
        content = kwargs.get("content", "")
        mode = kwargs.get("mode", "overwrite")

        if not file_path:
            return "Error: 'file_path' parameter is required."

        # Resolve and validate the path.
        safe_path = self._resolve_safe_path(file_path)
        if safe_path is None:
            return f"Error: Access denied. Path '{file_path}' is outside the allowed directory."

        try:
            # Create parent directories if they don't exist.
            safe_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to the file based on the specified mode.
            if mode == "append":
                with open(safe_path, "a", encoding="utf-8") as f:
                    f.write(content)
                action = "Appended"
            else:
                safe_path.write_text(content, encoding="utf-8")
                action = "Wrote"

            logger.info(
                "%s %d characters to: %s", action, len(content), file_path
            )
            return f"{action} {len(content)} characters to '{file_path}' successfully."

        except PermissionError:
            return f"Error: Permission denied writing to: '{file_path}'"

        except Exception as e:
            return f"Error writing file: {e!s}"
