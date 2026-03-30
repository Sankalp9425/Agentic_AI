"""
Centralized logging configuration for the Agentic AI framework.

Provides a consistent logging setup across all modules with structured
output formatting, configurable log levels, and optional file output.
All framework modules use Python's standard logging library with loggers
named after their module path (e.g., 'agentic_ai.llms.openai_llm').

Usage:
    >>> from agentic_ai.utils.logger import setup_logging
    >>> setup_logging(level="DEBUG", log_file="agent.log")
    >>> # All framework loggers will now output at DEBUG level to both
    >>> # the console and the specified log file.
"""

import logging
import sys


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    format_string: str | None = None,
) -> logging.Logger:
    """
    Configure logging for the entire Agentic AI framework.

    Sets up the root 'agentic_ai' logger with console output and optional
    file output. All sub-module loggers automatically inherit this
    configuration through Python's logging hierarchy.

    Args:
        level:         The minimum log level to display. Valid values are
                       "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
                       Default is "INFO".
        log_file:      Optional file path to write logs to. If provided,
                       logs are written to both console and the file.
                       The file handler uses append mode.
        format_string: Optional custom format string for log messages.
                       If not provided, uses a sensible default format.

    Returns:
        The configured root 'agentic_ai' logger instance.

    Example:
        >>> logger = setup_logging(level="DEBUG", log_file="debug.log")
        >>> logger.info("Logging is configured!")
    """
    # Get the root logger for the agentic_ai package.
    # All sub-module loggers (e.g., agentic_ai.llms.openai_llm) will
    # inherit handlers and level from this logger.
    root_logger = logging.getLogger("agentic_ai")

    # Parse the log level string to a logging constant.
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)

    # Use the default format if none provided.
    if not format_string:
        format_string = (
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )

    # Create the formatter.
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Remove any existing handlers to prevent duplicate output when
    # setup_logging is called multiple times.
    root_logger.handlers.clear()

    # Add a console handler that writes to stderr.
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add an optional file handler.
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        root_logger.info("Logging to file: %s", log_file)

    root_logger.info("Logging configured at %s level", level.upper())

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module within the agentic_ai namespace.

    This is a convenience function that ensures all loggers follow the
    'agentic_ai.module_name' naming convention.

    Args:
        name: The module or component name (e.g., "llms.openai", "agents.react").

    Returns:
        A Logger instance with the properly namespaced name.

    Example:
        >>> logger = get_logger("agents.react")
        >>> logger.info("ReAct agent initialized")
    """
    # Prefix with 'agentic_ai.' if not already present.
    if not name.startswith("agentic_ai."):
        name = f"agentic_ai.{name}"

    return logging.getLogger(name)
