"""
Email MCP connector for sending emails via SMTP.

This tool allows agents to send emails as part of their task execution.
It connects to an SMTP server (Gmail, Outlook, custom SMTP, etc.) and
sends formatted emails with optional HTML content.

Security note: This tool requires SMTP credentials. For Gmail, use an
App Password rather than your main account password. Never hardcode
credentials - use environment variables or a secrets manager.

Requirements:
    No additional packages required (uses Python's built-in smtplib and email).

Example:
    >>> from agentic_ai.mcp.email_tool import EmailTool
    >>> email = EmailTool(
    ...     smtp_host="smtp.gmail.com",
    ...     smtp_port=587,
    ...     username="your-email@gmail.com",
    ...     password="your-app-password",
    ... )
    >>> result = email.execute(
    ...     to="recipient@example.com",
    ...     subject="Hello from Agentic AI",
    ...     body="This email was sent by an AI agent!",
    ... )
"""

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from agentic_ai.core.base_tool import BaseTool

# Configure module-level logger.
logger = logging.getLogger(__name__)


class EmailTool(BaseTool):
    """
    MCP-compatible tool for sending emails via SMTP.

    Connects to a configured SMTP server and sends emails with plain text
    or HTML content. Supports TLS encryption for secure transmission.

    Attributes:
        name:        "send_email" - the tool identifier for LLM function calling.
        description: Describes the tool's capability to the LLM.
        parameters:  Defines the expected input parameters (to, subject, body, etc.).
        _smtp_host:  The SMTP server hostname.
        _smtp_port:  The SMTP server port (typically 587 for TLS, 465 for SSL).
        _username:   The SMTP authentication username (usually the sender's email).
        _password:   The SMTP authentication password or app password.
        _from_email: The sender's email address shown in the "From" field.
        _use_tls:    Whether to use TLS encryption. Default is True.

    Example:
        >>> tool = EmailTool(
        ...     smtp_host="smtp.gmail.com",
        ...     smtp_port=587,
        ...     username="agent@gmail.com",
        ...     password="app-password",
        ... )
        >>> tool.execute(
        ...     to="user@example.com",
        ...     subject="Report Ready",
        ...     body="Your weekly report is attached.",
        ... )
    """

    name = "send_email"
    description = (
        "Send an email to a specified recipient. Supports plain text and HTML content. "
        "Use this when you need to communicate results, send reports, or notify users."
    )
    parameters = {
        "to": {
            "type": "string",
            "description": "Recipient email address (e.g., 'user@example.com').",
            "required": True,
        },
        "subject": {
            "type": "string",
            "description": "Email subject line.",
            "required": True,
        },
        "body": {
            "type": "string",
            "description": "Email body content (plain text or HTML).",
            "required": True,
        },
        "html": {
            "type": "string",
            "description": "Optional HTML version of the email body for rich formatting.",
            "required": False,
        },
        "cc": {
            "type": "string",
            "description": "Optional CC recipients, comma-separated.",
            "required": False,
        },
    }

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int = 587,
        username: str = "",
        password: str = "",
        from_email: str | None = None,
        use_tls: bool = True,
    ) -> None:
        """
        Initialize the email tool with SMTP server configuration.

        Args:
            smtp_host:  SMTP server hostname (e.g., "smtp.gmail.com").
            smtp_port:  SMTP server port. Default is 587 (STARTTLS).
                        Use 465 for implicit SSL/TLS.
            username:   SMTP authentication username (usually sender's email).
            password:   SMTP authentication password or app-specific password.
            from_email: The "From" email address. Defaults to username if not specified.
            use_tls:    Whether to use STARTTLS encryption. Default is True.
        """
        self._smtp_host = smtp_host
        self._smtp_port = smtp_port
        self._username = username
        self._password = password
        # Use the username as the sender address if not explicitly provided.
        self._from_email = from_email or username
        self._use_tls = use_tls

    def execute(self, **kwargs: Any) -> str:
        """
        Send an email to the specified recipient.

        Constructs a MIME email message, connects to the SMTP server,
        authenticates, and sends the email. Returns a success or error message.

        Args:
            **kwargs: Must include 'to', 'subject', and 'body'. Optionally
                      includes 'html' for rich formatting and 'cc' for copies.

        Returns:
            A string confirming successful delivery or describing the error.
        """
        # Extract required parameters.
        to_email = kwargs.get("to", "")
        subject = kwargs.get("subject", "")
        body = kwargs.get("body", "")
        html_body = kwargs.get("html", "")
        cc = kwargs.get("cc", "")

        if not to_email or not subject:
            return "Error: 'to' and 'subject' are required parameters."

        try:
            # Create a multipart email message.
            message = MIMEMultipart("alternative")
            message["From"] = self._from_email
            message["To"] = to_email
            message["Subject"] = subject

            # Add CC recipients if specified.
            if cc:
                message["Cc"] = cc

            # Attach the plain text version of the body.
            message.attach(MIMEText(body, "plain"))

            # Attach the HTML version if provided.
            if html_body:
                message.attach(MIMEText(html_body, "html"))

            # Determine all recipients (To + CC).
            all_recipients = [to_email]
            if cc:
                all_recipients.extend(
                    addr.strip() for addr in cc.split(",") if addr.strip()
                )

            # Connect to the SMTP server and send the email.
            logger.info(
                "Connecting to SMTP server %s:%d", self._smtp_host, self._smtp_port
            )

            with smtplib.SMTP(self._smtp_host, self._smtp_port) as server:
                # Upgrade the connection to TLS if configured.
                if self._use_tls:
                    server.starttls()

                # Authenticate with the SMTP server.
                if self._username and self._password:
                    server.login(self._username, self._password)

                # Send the email.
                server.sendmail(self._from_email, all_recipients, message.as_string())

            logger.info("Email sent successfully to %s", to_email)
            return f"Email sent successfully to {to_email} with subject '{subject}'."

        except smtplib.SMTPAuthenticationError:
            error_msg = (
                "Authentication failed. Check your username and password. "
                "For Gmail, use an App Password instead of your account password."
            )
            logger.error("SMTP authentication failed: %s", error_msg)
            return f"Error: {error_msg}"

        except smtplib.SMTPException as e:
            logger.error("SMTP error: %s", e)
            return f"Error sending email: {e!s}"

        except Exception as e:
            logger.error("Unexpected error sending email: %s", e)
            return f"Error: {e!s}"
