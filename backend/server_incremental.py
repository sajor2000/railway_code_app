"""Incremental server - Step 1: Add medical APIs without ML models."""

import os
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Medical Coding Intelligence Platform",
    version="2.1.0-incremental",
    description="Medical coding with incremental features"
)

# API Router
api_router = APIRouter(prefix="/api")

# =================== MODELS ===================
class TerminologyQuery(BaseModel):
    query: str
    ontologies: List[str] = ["umls", "rxnorm", "snomed", "icd10", "loinc"]
    max_results: int = 20
    expand_abbreviations: bool = True
    confidence_threshold: float = 0.0

class MedicalConcept(BaseModel):
    concept_name: str
    ontology: str
    concept_id: str
    score: float
    definition: Optional[str] = None
    confidence_score: float = 0.0

# =================== MEDICAL API CLIENT ===================
class SimpleMedicalAPIClient:
    """Simplified medical API client without heavy dependencies."""
    
    def __init__(self):
        # Try to load config, but don't fail if not available
        try:
            from ..config import settings
            self.umls_api_key = settings.umls_api_key.get_secret_value() if settings.umls_api_key else None
        except:
            logger.warning("Config not available, using environment variables")
            self.umls_api_key = os.getenv("UMLS_API_KEY")
    
    async def search_rxnorm(self, query: str) -> List[Dict]:
        """Search RxNorm API."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                url = "https://rxnav.nlm.nih.gov/REST/drugs.json"
                params = {"name": query}
                response = await client.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    results = []
                    
                    # Parse RxNorm response
                    drug_group = data.get("drugGroup", {})
                    concept_group = drug_group.get("conceptGroup", [])
                    
                    for group in concept_group:
                        concepts = group.get("conceptProperties", [])
                        for concept in concepts:
                            results.append({
                                "name": concept.get("name", ""),
                                "code": concept.get("rxcui", ""),
                                "source": "RxNorm",
                                "score": 1.0,
                                "definition": f"RxNorm concept: {concept.get('name', '')}"
                            })
                    
                    return results[:10]  # Limit results
                    
        except Exception as e:
            logger.error(f"RxNorm search error: {e}")
        
        return []
    
    async def search_icd10(self, query: str) -> List[Dict]:
        """Search ICD-10 using NIH Clinical Tables API."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                url = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"
                params = {
                    "sf": "code,name",
                    "terms": query,
                    "maxList": 10
                }
                response = await client.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    results = []
                    
                    # Parse response [total, codes_array]
                    if len(data) >= 4:
                        codes = data[3]  # The actual results
                        for code_info in codes:
                            if len(code_info) >= 2:
                                results.append({
                                    "name": code_info[1],
                                    "code": code_info[0],
                                    "source": "ICD-10-CM",
                                    "score": 1.0,
                                    "definition": f"ICD-10 diagnosis code: {code_info[1]}"
                                })
                    
                    return results
                    
        except Exception as e:
            logger.error(f"ICD-10 search error: {e}")
        
        return []

# Global client
medical_api_client = SimpleMedicalAPIClient()

# =================== ENDPOINTS ===================
@api_router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Medical Coding Intelligence Platform",
        "version": "2.1.0-incremental",
        "port": os.getenv("PORT", "8080"),
        "features": {
            "medical_apis": "enabled",
            "ml_models": "disabled",
            "embeddings": "disabled"
        }
    }

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Medical Coding Intelligence Platform - Incremental",
        "api_docs": "/docs",
        "health": "/api/health"
    }

@api_router.post("/search/unified")
async def unified_search(query: TerminologyQuery):
    """Medical terminology search using only APIs (no ML)."""
    try:
        all_results = []
        
        # Search requested ontologies
        if "rxnorm" in query.ontologies:
            rxnorm_results = await medical_api_client.search_rxnorm(query.query)
            all_results.extend(rxnorm_results)
        
        if "icd10" in query.ontologies:
            icd10_results = await medical_api_client.search_icd10(query.query)
            all_results.extend(icd10_results)
        
        # Convert to response format
        concepts = []
        for result in all_results[:query.max_results]:
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
            "message": "Results from medical APIs only (ML models disabled)"
        }
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =================== STARTUP ===================
@app.on_event("startup")
async def startup_event():
    """Minimal startup."""
    logger.info(f"🚀 Starting incremental server on PORT: {os.getenv('PORT', '8080')}")
    logger.info("✅ Medical APIs enabled, ML models disabled")

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