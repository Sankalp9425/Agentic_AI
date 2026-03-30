"""
MCP (Model Context Protocol) connector implementations.

MCP connectors provide agents with access to external services and APIs
through a standardized tool interface. Each connector wraps a specific
service (email, web search, file operations, etc.) as a BaseTool that
agents can invoke during their reasoning process.

The Model Context Protocol (MCP) defines a standard way for AI models
to interact with external tools and data sources. These connectors
implement the MCP tool interface, making them compatible with any
MCP-compliant agent framework.

Supported Connectors:
    - EmailTool:        Send emails via SMTP.
    - GoogleSearchTool: Perform web searches via Google's Custom Search API.
    - WebScraperTool:   Fetch and extract content from web pages.
    - FileReaderTool:   Read content from local files.
    - FileWriterTool:   Write content to local files.
    - ShellTool:        Execute shell commands (sandboxed).
    - HTTPRequestTool:  Make HTTP requests to arbitrary APIs.

Example:
    >>> from agentic_ai.mcp.email import EmailTool
    >>> email_tool = EmailTool(smtp_host="smtp.gmail.com", ...)
    >>> agent = ReActAgent(llm=llm, tools=[email_tool])
    >>> agent.run("Send a summary email to john@example.com")
"""
