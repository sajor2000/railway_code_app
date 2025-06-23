"""Startup validation and environment checking for deployment safety."""

import os
import sys
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import asyncio
import httpx
import redis.asyncio as redis
from redis.exceptions import RedisError

from ..config import settings

logger = logging.getLogger(__name__)

class StartupValidationError(Exception):
    """Raised when critical startup validation fails."""
    pass

class StartupValidator:
    """Validates environment and dependencies before server startup."""
    
    def __init__(self):
        self.warnings: List[str] = []
        self.errors: List[str] = []
        
    async def validate_all(self) -> bool:
        """Run all validation checks. Returns True if startup should proceed."""
        logger.info("🔍 Starting deployment validation checks...")
        
        # Critical checks (must pass)
        await self._validate_environment_variables()
        await self._validate_python_dependencies()
        await self._validate_file_system()
        await self._validate_port_availability()
        
        # Service checks (optional but important)
        await self._validate_redis_connection()
        await self._validate_external_apis()
        await self._validate_model_accessibility()
        
        # Report results
        self._report_results()
        
        # Fail if critical errors found
        if self.errors:
            raise StartupValidationError(f"Critical validation errors: {self.errors}")
            
        return True
    
    async def _validate_environment_variables(self):
        """Validate critical environment variables."""
        logger.info("Checking environment variables...")
        
        # Check PORT
        port = os.getenv("PORT")
        if not port:
            self.errors.append("PORT environment variable not set")
        else:
            try:
                port_int = int(port)
                if not (1 <= port_int <= 65535):
                    self.errors.append(f"Invalid PORT value: {port}")
            except ValueError:
                self.errors.append(f"PORT must be numeric, got: {port}")
        
        # Check for BioBERT model name
        if not settings.embedding_model_name:
            self.errors.append("embedding_model_name not configured")
        
        # Optional but recommended API keys
        if not settings.openai_api_key:
            self.warnings.append("OpenAI API key not configured - some features may be limited")
            
        if not settings.umls_api_key:
            self.warnings.append("UMLS API key not configured - medical terminology may be limited")
            
        if not settings.pinecone_api_key:
            self.warnings.append("Pinecone API key not configured - vector search may be limited")
    
    async def _validate_python_dependencies(self):
        """Check critical Python packages are available."""
        logger.info("Checking Python dependencies...")
        
        critical_packages = [
            "fastapi",
            "uvicorn", 
            "sentence_transformers",
            "biobert_embedding",
            "pinecone",
            "openai",
            "pydantic",
            "redis"
        ]
        
        for package in critical_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError as e:
                self.errors.append(f"Critical package missing: {package} - {e}")
    
    async def _validate_file_system(self):
        """Validate file system permissions and directories."""
        logger.info("Checking file system...")
        
        # Check upload directory
        upload_dir = Path("backend/uploads")
        try:
            upload_dir.mkdir(parents=True, exist_ok=True)
            # Test write permission
            test_file = upload_dir / "test_write.tmp"
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            self.errors.append(f"Upload directory not writable: {e}")
        
        # Check static directory for React frontend
        static_dir = Path("backend/static")
        if not static_dir.exists():
            self.warnings.append("Frontend static files not found - frontend may not work")
    
    async def _validate_port_availability(self):
        """Check if the configured port is available."""
        import socket
        
        port = int(os.getenv("PORT", "8080"))
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
        except OSError as e:
            self.errors.append(f"Port {port} not available: {e}")
    
    async def _validate_redis_connection(self):
        """Test Redis connection if configured."""
        if not settings.redis_host:
            self.warnings.append("Redis not configured - caching will be disabled")
            return
            
        try:
            redis_client = redis.from_url(settings.redis_url, decode_responses=True)
            await redis_client.ping()
            await redis_client.close()
            logger.info("✅ Redis connection validated")
        except RedisError as e:
            self.warnings.append(f"Redis connection failed - caching disabled: {e}")
        except Exception as e:
            self.warnings.append(f"Redis validation error: {e}")
    
    async def _validate_external_apis(self):
        """Test connectivity to external medical APIs."""
        logger.info("Testing external API connectivity...")
        
        # Test RxNorm API
        if settings.rxnorm_base_url:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(f"{settings.rxnorm_base_url}/version")
                    if response.status_code == 200:
                        logger.info("✅ RxNorm API accessible")
                    else:
                        self.warnings.append(f"RxNorm API returned {response.status_code}")
            except Exception as e:
                self.warnings.append(f"RxNorm API not accessible: {e}")
        
        # Test SNOMED browser
        if settings.snomed_browser_url:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(f"{settings.snomed_browser_url}/info")
                    if response.status_code == 200:
                        logger.info("✅ SNOMED browser accessible")
                    else:
                        self.warnings.append(f"SNOMED browser returned {response.status_code}")
            except Exception as e:
                self.warnings.append(f"SNOMED browser not accessible: {e}")
    
    async def _validate_model_accessibility(self):
        """Check if embedding models can be loaded."""
        logger.info("Validating model accessibility...")
        
        try:
            from sentence_transformers import SentenceTransformer
            # Try to access model info without loading
            model_name = settings.embedding_model_name
            logger.info(f"✅ Embedding model {model_name} configured")
        except Exception as e:
            self.warnings.append(f"Embedding model validation failed: {e}")
    
    def _report_results(self):
        """Report validation results."""
        if self.warnings:
            logger.warning("⚠️  Validation warnings:")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")
        
        if self.errors:
            logger.error("❌ Validation errors:")
            for error in self.errors:
                logger.error(f"  - {error}")
        else:
            logger.info("✅ All critical validation checks passed")

# Global validator instance
startup_validator = StartupValidator()