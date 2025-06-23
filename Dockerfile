# TEMPORARY: Minimal Dockerfile to debug GCP deployment
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8080

# Create app directory
WORKDIR /app

# Install only essential dependencies first
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install minimal Python dependencies
RUN pip install --no-cache-dir \
    fastapi==0.109.0 \
    uvicorn[standard]==0.27.0 \
    python-dotenv==1.0.0 \
    pydantic==2.5.0 \
    pydantic-settings==2.1.0

# Copy backend source
COPY backend/ ./backend/

# Create required directories
RUN mkdir -p backend/uploads backend/static

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:$PORT/api/health || exit 1

# Expose port
EXPOSE $PORT

# Use minimal server temporarily
CMD uvicorn backend.server_minimal:app --host 0.0.0.0 --port $PORT