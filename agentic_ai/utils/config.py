"""
Configuration management utilities for the Agentic AI framework.

Provides helpers for loading configuration from environment variables,
.env files, and YAML/JSON configuration files. This centralizes
configuration handling so that API keys, model names, and other
settings can be managed consistently across the framework.

Features:
    - Load API keys from environment variables.
    - Parse .env files for local development.
    - Validate required configuration parameters.
    - Provide sensible defaults for optional parameters.

Example:
    >>> from agentic_ai.utils.config import load_config, get_api_key
    >>> config = load_config("config.yaml")
    >>> openai_key = get_api_key("OPENAI_API_KEY")
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

# Configure module-level logger.
logger = logging.getLogger(__name__)


def get_api_key(env_var_name: str, required: bool = True) -> str:
    """
    Retrieve an API key from environment variables.

    Checks for the specified environment variable and returns its value.
    If the variable is not set and required=True, raises a ValueError
    with helpful instructions.

    Args:
        env_var_name: The name of the environment variable containing the
                      API key (e.g., "OPENAI_API_KEY").
        required:     Whether the key is required. If True (default) and
                      the variable is not set, raises ValueError. If False,
                      returns an empty string.

    Returns:
        The API key value as a string.

    Raises:
        ValueError: If the key is required but the environment variable
                    is not set or is empty.

    Example:
        >>> key = get_api_key("OPENAI_API_KEY")
        >>> # Returns the value of $OPENAI_API_KEY
    """
    # Retrieve the environment variable value.
    value = os.environ.get(env_var_name, "").strip()

    if not value and required:
        raise ValueError(
            f"Required environment variable '{env_var_name}' is not set. "
            f"Set it with: export {env_var_name}='your-api-key'"
        )

    if not value:
        logger.warning("Optional API key '%s' not found", env_var_name)

    return value


def load_env_file(file_path: str = ".env") -> dict[str, str]:
    """
    Load environment variables from a .env file.

    Parses a .env file and sets the variables in the process environment.
    Existing environment variables are NOT overwritten (environment takes
    precedence over .env file values).

    File format:
        KEY=value
        KEY="quoted value"
        # Comments are ignored
        EMPTY_KEY=

    Args:
        file_path: Path to the .env file. Default is ".env" in the
                   current directory.

    Returns:
        A dictionary of all key-value pairs loaded from the file.

    Example:
        >>> loaded = load_env_file(".env")
        >>> print(loaded)  # {"OPENAI_API_KEY": "sk-...", "MODEL": "gpt-4"}
    """
    env_path = Path(file_path)
    loaded_vars: dict[str, str] = {}

    # Return empty if the file doesn't exist.
    if not env_path.exists():
        logger.debug("No .env file found at: %s", file_path)
        return loaded_vars

    logger.info("Loading environment from: %s", file_path)

    with open(env_path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            # Strip whitespace and skip empty lines and comments.
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Split on the first '=' to get key and value.
            if "=" not in line:
                logger.warning(
                    "Invalid line %d in %s: %s", line_number, file_path, line
                )
                continue

            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()

            # Remove surrounding quotes from the value.
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]

            loaded_vars[key] = value

            # Set in environment only if not already set.
            if key not in os.environ:
                os.environ[key] = value

    logger.info("Loaded %d variables from %s", len(loaded_vars), file_path)
    return loaded_vars


def load_config(file_path: str) -> dict[str, Any]:
    """
    Load configuration from a JSON or YAML file.

    Automatically detects the file format based on the extension and
    parses the contents into a Python dictionary.

    Supported formats:
        - .json: Standard JSON format.
        - .yaml / .yml: YAML format (requires PyYAML package).

    Args:
        file_path: Path to the configuration file.

    Returns:
        A dictionary containing the parsed configuration.

    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        ValueError: If the file format is unsupported.

    Example:
        >>> config = load_config("config.yaml")
        >>> print(config["llm"]["model"])
        'gpt-4'
    """
    config_path = Path(file_path)

    # Verify the file exists.
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    # Determine the file format from the extension.
    suffix = config_path.suffix.lower()

    if suffix == ".json":
        # Parse JSON configuration.
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        logger.info("Loaded JSON config from: %s", file_path)
        return config

    elif suffix in (".yaml", ".yml"):
        # Parse YAML configuration (requires PyYAML).
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "The 'PyYAML' package is required for YAML config files. "
                "Install it with: pip install pyyaml"
            ) from e

        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info("Loaded YAML config from: %s", file_path)
        return config or {}

    else:
        raise ValueError(
            f"Unsupported config file format: '{suffix}'. "
            f"Use .json, .yaml, or .yml."
        )


def validate_config(
    config: dict[str, Any],
    required_keys: list[str],
) -> list[str]:
    """
    Validate that a configuration dictionary contains all required keys.

    Checks for the presence of each required key and returns a list of
    any missing keys. Does not check the values of the keys.

    Args:
        config:        The configuration dictionary to validate.
        required_keys: A list of key names that must be present.

    Returns:
        A list of missing key names. Empty list if all keys are present.

    Example:
        >>> config = {"model": "gpt-4", "api_key": "sk-..."}
        >>> missing = validate_config(config, ["model", "api_key", "temperature"])
        >>> print(missing)  # ["temperature"]
    """
    missing = [key for key in required_keys if key not in config]

    if missing:
        logger.warning("Missing configuration keys: %s", missing)

    return missing
