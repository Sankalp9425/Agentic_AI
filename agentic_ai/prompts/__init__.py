"""
Configurable prompt templates for the agentic AI framework.

Centralizes all prompt templates used across the framework into a single
module for easy customization and maintenance. Templates use Python's
str.format() syntax for variable substitution.

Modules:
    - templates: All prompt templates organized by category (RAG, agents,
      evaluation, retrieval).

Usage:
    >>> from agentic_ai.prompts.templates import RAG_SYSTEM_PROMPT
    >>> prompt = RAG_SYSTEM_PROMPT.format(context="...", question="...")
"""
