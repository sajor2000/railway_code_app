# GCP Cloud Run Optimized Dockerfile for Medical Coding Intelligence Platform
# Multi-stage build for React frontend + FastAPI backend

# Stage 1: Build React Frontend
FROM node:18-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy package files
COPY frontend/package*.json ./

# Install dependencies (use npm install as fallback for missing package-lock.json)
RUN npm ci --omit=dev || npm install --omit=dev

# Copy frontend source
COPY frontend/ ./

# Build React app for production
RUN npm run build

# Stage 2: Python Backend with Static Files
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8080

# Create app directory
WORKDIR /app

# Install system dependencies for medical libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python requirements
COPY requirements.txt .

# Install Python dependencies with cache
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download embedding model to avoid startup delays
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" || echo "Model download failed, will download on first use"

# Copy backend source code
COPY backend/ ./backend/

# Copy built React app from frontend stage
COPY --from=frontend-builder /app/frontend/build ./backend/static/

# Create uploads directory
RUN mkdir -p backend/uploads

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Health check with longer start period for ML model loading
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:$PORT/api/health || exit 1

# Expose port
EXPOSE $PORT

# Start command for Cloud Run
CMD exec uvicorn backend.server:app --host 0.0.0.0 --port $PORT --workers 1