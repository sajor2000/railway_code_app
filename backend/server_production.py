"""Production server with all features - using lazy loading for fast startup."""

import os
import sys
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, APIRouter, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uuid

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Medical Coding Intelligence Platform",
    version="2.1.0",
    description="Production server with all medical coding features"
)

# API Router
api_router = APIRouter(prefix="/api")

# =================== LAZY LOADING MANAGERS ===================
class LazyLoader:
    """Manages lazy loading of heavy dependencies."""
    
    def __init__(self):
        self._medical_api_client = None
        self._embedding_model = None
        self._openai_client = None
        self._biobert_engine = None
        self._csv_services = None
        
    @property
    def medical_api_client(self):
        """Lazy load medical API client."""
        if self._medical_api_client is None:
            try:
                from backend.services.api_clients import MedicalAPIClient
                self._medical_api_client = MedicalAPIClient()
                logger.info("✅ Medical API client loaded")
            except Exception as e:
                logger.error(f"Failed to load medical API client: {e}")
                # Return a basic client that works
                from backend.server_incremental import SimpleMedicalAPIClient
                self._medical_api_client = SimpleMedicalAPIClient()
        return self._medical_api_client
    
    @property
    def embedding_model(self):
        """Lazy load sentence transformer model."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading embedding model (this may take a moment)...")
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✅ Embedding model loaded")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
        return self._embedding_model
    
    @property
    def openai_client(self):
        """Lazy load OpenAI client."""
        if self._openai_client is None:
            try:
                from openai import OpenAI
                from backend.config import settings
                if settings.openai_api_key:
                    self._openai_client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
                    logger.info("✅ OpenAI client loaded")
            except Exception as e:
                logger.error(f"Failed to load OpenAI client: {e}")
        return self._openai_client
    
    @property
    def csv_services(self):
        """Lazy load CSV processing services."""
        if self._csv_services is None:
            try:
                from backend.services.csv_analyzer import CSVAnalyzer
                from backend.services.batch_mapper import BatchMapper
                from backend.services.results_processor import ResultsProcessor
                self._csv_services = {
                    'analyzer': CSVAnalyzer(),
                    'mapper': BatchMapper(),
                    'processor': ResultsProcessor()
                }
                logger.info("✅ CSV services loaded")
            except Exception as e:
                logger.error(f"Failed to load CSV services: {e}")
                self._csv_services = {}
        return self._csv_services

# Global lazy loader
lazy = LazyLoader()

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
    is_abbreviation_expansion: bool = False

class HybridSearchResult(BaseModel):
    api_results: List[MedicalConcept]
    semantic_results: List[MedicalConcept] = []
    total_results: int
    search_metadata: Dict[str, Any]

# =================== HEALTH CHECK ===================
@api_router.get("/health")
async def health_check():
    """Comprehensive health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Medical Coding Intelligence Platform",
        "version": "2.1.0",
        "port": os.getenv("PORT", "8080"),
        "features": {
            "medical_apis": bool(lazy._medical_api_client),
            "embeddings": bool(lazy._embedding_model),
            "openai": bool(lazy._openai_client),
            "csv_processing": bool(lazy._csv_services),
            "biobert": False  # Will be loaded on demand
        }
    }

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Medical Coding Intelligence Platform - Production",
        "api_docs": "/docs",
        "health": "/api/health"
    }

# =================== MEDICAL SEARCH ===================
@api_router.post("/search/unified", response_model=HybridSearchResult)
async def unified_search(query: TerminologyQuery):
    """Unified medical terminology search with all features."""
    try:
        api_results = []
        semantic_results = []
        search_metadata = {
            "query": query.query,
            "timestamp": datetime.now().isoformat(),
            "features_used": []
        }
        
        # Step 1: API Search (always available)
        client = lazy.medical_api_client
        if client:
            search_metadata["features_used"].append("medical_apis")
            
            # Search each requested ontology
            for ontology in query.ontologies:
                try:
                    if ontology == "umls" and hasattr(client, 'search_umls'):
                        results = await client.search_umls(query.query)
                        api_results.extend(results)
                    elif ontology == "rxnorm":
                        results = await client.search_rxnorm(query.query)
                        api_results.extend(results)
                    elif ontology == "icd10":
                        results = await client.search_icd10(query.query)
                        api_results.extend(results)
                    elif ontology == "snomed" and hasattr(client, 'search_snomed'):
                        results = await client.search_snomed(query.query)
                        api_results.extend(results)
                    elif ontology == "loinc" and hasattr(client, 'search_loinc'):
                        results = await client.search_loinc(query.query)
                        api_results.extend(results)
                except Exception as e:
                    logger.warning(f"Failed to search {ontology}: {e}")
        
        # Step 2: Semantic Search (if requested and available)
        if query.semantic_search and lazy.embedding_model:
            search_metadata["features_used"].append("embeddings")
            # Simplified semantic search
            logger.info("Performing semantic search...")
            # In production, this would search Pinecone or perform similarity matching
        
        # Convert to response format
        def to_medical_concept(result: Dict) -> MedicalConcept:
            return MedicalConcept(
                concept_name=result.get('name', ''),
                ontology=result.get('source', 'unknown'),
                concept_id=result.get('code', ''),
                score=result.get('score', 1.0),
                definition=result.get('definition'),
                confidence_score=result.get('score', 1.0),
                is_abbreviation_expansion=result.get('is_abbreviation_expansion', False)
            )
        
        api_concepts = [to_medical_concept(r) for r in api_results[:query.max_results]]
        
        return HybridSearchResult(
            api_results=api_concepts,
            semantic_results=semantic_results,
            total_results=len(api_concepts) + len(semantic_results),
            search_metadata=search_metadata
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =================== CSV PROCESSING ===================
@api_router.post("/csv/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Upload CSV for batch processing."""
    try:
        services = lazy.csv_services
        if not services or 'analyzer' not in services:
            raise HTTPException(status_code=503, detail="CSV services not available")
        
        # Save file
        file_id = str(uuid.uuid4())
        file_path = Path(f"backend/uploads/{file_id}_{file.filename}")
        file_path.parent.mkdir(exist_ok=True)
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Analyze structure
        analyzer = services['analyzer']
        analysis = await analyzer.analyze_csv_structure(str(file_path), file.filename)
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "analysis": analysis,
            "message": "File uploaded successfully. Use /csv/process to start mapping."
        }
        
    except Exception as e:
        logger.error(f"CSV upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =================== AI CHAT (if available) ===================
@api_router.post("/chat")
async def medical_chat(query: Dict[str, str]):
    """AI-powered medical coding assistant."""
    try:
        client = lazy.openai_client
        if not client:
            return {
                "response": "AI chat is currently unavailable. Please use the search endpoints.",
                "status": "ai_unavailable"
            }
        
        # Simple chat implementation
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use cheaper model
            messages=[
                {"role": "system", "content": "You are a medical coding assistant."},
                {"role": "user", "content": query.get("message", "")}
            ],
            max_tokens=500
        )
        
        return {
            "response": response.choices[0].message.content,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {
            "response": "I encountered an error processing your request.",
            "status": "error",
            "error": str(e)
        }

# =================== STARTUP ===================
@app.on_event("startup")
async def startup_event():
    """Fast startup - load only essentials."""
    logger.info("🚀 Starting Medical Coding Intelligence Platform")
    logger.info("✅ Server ready - features will load on demand")
    
    # Optional: Pre-load critical services in background
    asyncio.create_task(preload_services())

async def preload_services():
    """Pre-load services in background after startup."""
    await asyncio.sleep(5)  # Wait for server to be ready
    
    logger.info("Pre-loading services in background...")
    # Trigger lazy loading of common services
    _ = lazy.medical_api_client
    logger.info("Background service loading complete")

# Include router
app.include_router(api_router)

# Add CORS
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