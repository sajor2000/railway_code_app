# Incremental Deployment - Step 1: Medical APIs only
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8080
ENV PYTHONPATH=/app

# Create app directory
WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install core dependencies for medical APIs
RUN pip install --no-cache-dir \
    fastapi==0.109.0 \
    uvicorn[standard]==0.27.0 \
    python-dotenv==1.0.0 \
    pydantic>=2.5.0 \
    pydantic-settings>=2.1.0 \
    httpx>=0.26.0 \
    requests>=2.31.0 \
    aiohttp>=3.9.0

# Copy backend source
COPY backend/ ./backend/

# Create required directories and init files
RUN mkdir -p backend/uploads backend/static && \
    touch backend/__init__.py backend/utils/__init__.py backend/services/__init__.py

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:$PORT/api/health || exit 1

# Expose port
EXPOSE $PORT

# Use incremental server for now
CMD ["uvicorn", "backend.server_incremental:app", "--host", "0.0.0.0", "--port", "8080"]