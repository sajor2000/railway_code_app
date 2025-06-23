"""Staged server with core medical coding features but optimized startup."""

import os
import sys
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Medical Coding Intelligence Platform",
    version="2.1.0-staged",
    description="Medical coding with staged initialization"
)

api_router = APIRouter(prefix="/api")

# Global flags for feature availability
FEATURES_AVAILABLE = {
    "embeddings": False,
    "biobert": False,
    "openai": False,
    "medical_apis": False,
    "csv_processing": False
}

# =================== MODELS ===================
class TerminologyQuery(BaseModel):
    query: str
    ontologies: List[str] = ["umls", "rxnorm", "snomed", "icd10", "loinc"]
    max_results: int = 20
    semantic_search: bool = False
    expand_abbreviations: bool = True
    confidence_threshold: float = 0.0

class MedicalConcept(BaseModel):
    concept_name: str
    ontology: str
    concept_id: str
    score: float
    definition: Optional[str] = None
    semantic_types: Optional[List[str]] = None
    confidence_score: float = 0.0

# =================== HEALTH CHECK ===================
@api_router.get("/health")
async def health_check():
    """Health check with feature status."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Medical Coding Intelligence Platform",
        "version": "2.1.0-staged",
        "port": os.getenv("PORT", "8080"),
        "features_available": FEATURES_AVAILABLE,
        "message": "Core server running - features loading progressively"
    }

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Medical Coding Intelligence Platform - Staged Server",
        "status": "running",
        "api_docs": "/docs",
        "health": "/api/health"
    }

# =================== CORE MEDICAL SEARCH ===================
@api_router.post("/search/unified")
async def unified_search(query: TerminologyQuery):
    """Medical terminology search with progressive feature loading."""
    try:
        results = []
        
        # Step 1: Try medical APIs if available
        if FEATURES_AVAILABLE["medical_apis"]:
            try:
                from .services.api_clients import MedicalAPIClient
                client = MedicalAPIClient()
                
                # Search each requested ontology
                if "umls" in query.ontologies:
                    umls_results = await client.search_umls(query.query)
                    results.extend(umls_results)
                    
                if "rxnorm" in query.ontologies:
                    rxnorm_results = await client.search_rxnorm(query.query)
                    results.extend(rxnorm_results)
                    
            except Exception as e:
                logger.warning(f"Medical API search failed: {e}")
        
        # Step 2: If no results and embeddings available, try semantic search
        if not results and FEATURES_AVAILABLE["embeddings"] and query.semantic_search:
            logger.info("Would perform semantic search here")
            # Semantic search would go here
        
        # Convert to response format
        concepts = []
        for result in results[:query.max_results]:
            concept = MedicalConcept(
                concept_name=result.get('name', ''),
                ontology=result.get('source', 'unknown'),
                concept_id=result.get('code', ''),
                score=result.get('score', 0.0),
                definition=result.get('definition'),
                confidence_score=result.get('score', 0.0)
            )
            concepts.append(concept)
        
        return {
            "query": query.query,
            "concepts": concepts,
            "total_results": len(concepts),
            "features_used": [k for k, v in FEATURES_AVAILABLE.items() if v]
        }
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =================== PROGRESSIVE INITIALIZATION ===================
async def initialize_medical_apis():
    """Initialize medical API clients."""
    try:
        from .services.api_clients import MedicalAPIClient
        from .config import settings
        
        # Test if we can create the client
        client = MedicalAPIClient()
        FEATURES_AVAILABLE["medical_apis"] = True
        logger.info("✅ Medical APIs initialized")
    except Exception as e:
        logger.warning(f"⚠️ Medical APIs not available: {e}")

async def initialize_embeddings():
    """Initialize embedding models progressively."""
    try:
        # Only import if needed
        from sentence_transformers import SentenceTransformer
        
        # Don't load model yet, just check availability
        FEATURES_AVAILABLE["embeddings"] = True
        logger.info("✅ Embeddings available (will load on first use)")
    except ImportError as e:
        logger.warning(f"⚠️ Embeddings not available: {e}")

async def initialize_openai():
    """Initialize OpenAI client."""
    try:
        from openai import OpenAI
        from .config import settings
        
        if settings.openai_api_key:
            FEATURES_AVAILABLE["openai"] = True
            logger.info("✅ OpenAI available")
    except Exception as e:
        logger.warning(f"⚠️ OpenAI not available: {e}")

# =================== STARTUP ===================
@app.on_event("startup")
async def startup_event():
    """Progressive startup - don't block on failures."""
    logger.info("🚀 Starting Medical Coding Intelligence Platform (Staged)")
    
    # Start server immediately, then initialize features
    asyncio.create_task(progressive_initialization())
    
    logger.info("✅ Server started - features loading progressively")

async def progressive_initialization():
    """Initialize features progressively without blocking startup."""
    await asyncio.sleep(1)  # Let server start first
    
    # Initialize in order of importance
    await initialize_medical_apis()
    await initialize_embeddings()
    await initialize_openai()
    
    logger.info(f"📊 Features available: {FEATURES_AVAILABLE}")

# Include router
app.include_router(api_router)

# Add CORS if needed
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)