# =============================================================================
# Dockerfile for Agentic AI Framework
# =============================================================================
# Multi-stage build for a production-ready container.
#
# Build:
#   docker build -t agentic-ai .
#
# Run:
#   docker run -p 8000:8000 --env-file .env agentic-ai
#
# Run with specific extras:
#   docker build --build-arg EXTRAS="openai,chroma" -t agentic-ai .

# ---------------------------------------------------------------------------
# Stage 1: Build dependencies
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# Set build arguments for optional extras.
ARG EXTRAS="all"

# Set working directory.
WORKDIR /app

# Install system dependencies required for building Python packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency files first for better layer caching.
COPY pyproject.toml README.md ./

# Create a virtual environment and install dependencies.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install the package with specified extras.
COPY . .
RUN pip install --no-cache-dir -e ".[$EXTRAS]"

# ---------------------------------------------------------------------------
# Stage 2: Production image
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS production

# Install runtime system dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder stage.
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory.
WORKDIR /app

# Copy the application code.
COPY . .

# Create a non-root user for security.
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# Expose the API port.
EXPOSE 8000

# Set environment variables.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8000

# Health check.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start the API server using uvicorn.
CMD ["uvicorn", "agentic_ai.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
