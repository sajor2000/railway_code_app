"""Minimal FastAPI server for GCP Cloud Run deployment testing."""

import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Medical Coding Intelligence Platform - Minimal",
    version="2.1.0-minimal",
    description="Minimal server for deployment testing"
)

# Basic health check that responds immediately
@app.get("/api/health")
async def health_check():
    """Minimal health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Medical Coding Intelligence Platform",
        "version": "2.1.0-minimal",
        "port": os.getenv("PORT", "8080"),
        "message": "Minimal server running successfully"
    }

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Medical Coding Intelligence Platform - Minimal Server",
        "status": "running",
        "endpoints": ["/", "/api/health", "/api/test"]
    }

@app.get("/api/test")
async def test_endpoint():
    """Test endpoint to verify API routing."""
    return {
        "message": "API is working",
        "timestamp": datetime.now().isoformat(),
        "environment": {
            "port": os.getenv("PORT", "8080"),
            "python_version": os.sys.version,
            "working_dir": str(Path.cwd())
        }
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Minimal startup event."""
    logger.info(f"🚀 Starting minimal server on PORT: {os.getenv('PORT', '8080')}")
    logger.info("✅ Minimal server startup complete")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Minimal shutdown event."""
    logger.info("👋 Shutting down minimal server")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)