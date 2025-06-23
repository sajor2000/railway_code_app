# Production Dockerfile with progressive initialization
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8080
ENV PYTHONPATH=/app

# Create app directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies in stages for better error handling
# Stage 1: Core dependencies (must succeed)
RUN pip install --no-cache-dir \
    fastapi==0.109.0 \
    uvicorn[standard]==0.27.0 \
    pydantic>=2.5.0 \
    pydantic-settings>=2.1.0 \
    python-dotenv>=1.0.0 \
    requests>=2.31.0 \
    httpx>=0.26.0 \
    aiohttp>=3.9.0 \
    pandas>=2.1.0 \
    numpy>=1.24.0 \
    redis>=5.0.0 \
    motor>=3.3.0

# Stage 2: ML dependencies (allow failures for now)
RUN pip install --no-cache-dir \
    openai>=1.12.0 \
    pinecone-client==5.0.1 \
    biobert-embedding==0.1.2 \
    sentence-transformers>=2.2.0 \
    || echo "Some ML dependencies failed, continuing..."

# Stage 3: Remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt || echo "Some optional dependencies failed"

# Copy backend source
COPY backend/ ./backend/

# Create required directories
RUN mkdir -p backend/uploads backend/static

# Create __init__.py files for proper module imports
RUN touch backend/__init__.py backend/utils/__init__.py backend/services/__init__.py

# Health check with reasonable timeout
HEALTHCHECK --interval=30s --timeout=20s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:$PORT/api/health || exit 1

# Expose port
EXPOSE $PORT

# Start server with proper module path
CMD ["sh", "-c", "cd /app && uvicorn backend.server:app --host 0.0.0.0 --port $PORT --workers 1"]