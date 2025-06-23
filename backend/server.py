from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, Request
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import asyncio
import json
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import requests
import httpx
import base64
from urllib.parse import quote
import pandas as pd
import io
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np
from difflib import SequenceMatcher
import re
import csv

# Import configuration
from .config import get_settings, settings

# Import error handling
from .exceptions import (
    APIError, UMLSAPIError, RxNormAPIError, WHOICDAPIError,
    LOINCAPIError, SNOMEDAPIError, ValidationError
)
from .utils.retry import api_retry

# Import services
from .services.cache import cache_service, api_cache
from .services.html_export import html_export_service
from .services.api_clients import MedicalAPIClient
from .services.csv_analyzer import CSVAnalyzer
from .services.batch_mapper import BatchMapper
from .services.results_processor import ResultsProcessor

# Pinecone imports
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logging.warning("Pinecone not installed. RAG functionality will be limited.")

# Import Pydantic AI research workflow
from .workflows.research_workflow import (
    execute_medical_concept_search,
    ResearchTerminologyDeps,
    create_streamlined_medical_workflow
)

# Import hybrid search components
from .integrations.biobert_rag import get_biobert_rag_engine, BioBERTRAGEngine
from .integrations.hybrid_search import hybrid_search_engine, HybridSearchResult

ROOT_DIR = Path(__file__).parent.parent  # Go up to project root  
load_dotenv(ROOT_DIR / '.env')
print(f"Loading .env from: {ROOT_DIR / '.env'}")  # Debug logging

# Debug: Print API key status (first/last 4 chars only for security)
print("🔧 API Configuration Status:")
if settings.openai_api_key:
    key = settings.openai_api_key.get_secret_value()
    print(f"  ✅ OpenAI: {key[:4]}...{key[-4:]}")
else:
    print("  ❌ OpenAI: Not configured")

if settings.umls_api_key:
    key = settings.umls_api_key.get_secret_value()
    print(f"  ✅ UMLS: {key[:4]}...{key[-4:]}")
else:
    print("  ❌ UMLS: Not configured")

if settings.pinecone_api_key:
    key = settings.pinecone_api_key.get_secret_value()
    print(f"  ✅ Pinecone: {key[:4]}...{key[-4:]}")
else:
    print("  ❌ Pinecone: Not configured")

print(f"  📂 Working Directory: {Path.cwd()}")

# MongoDB connection
client = AsyncIOMotorClient(settings.mongo_url)
db = client[settings.db_name]

# OpenAI client
openai_client = OpenAI(api_key=settings.openai_api_key.get_secret_value() if settings.openai_api_key else None)

# Sentence transformer for embeddings (existing)
embedding_model = SentenceTransformer(settings.embedding_model_name)

# Create the main app
app = FastAPI(title=settings.app_name, version=settings.app_version)
api_router = APIRouter(prefix=settings.api_prefix)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize medical API client
medical_api_client = MedicalAPIClient()

# Initialize CSV processing services
csv_analyzer = CSVAnalyzer()
batch_mapper = BatchMapper()
results_processor = ResultsProcessor()

# Create uploads directory
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

# Store uploaded files metadata
uploaded_files = {}

# Global BioBERT RAG engine (will be initialized lazily)
biobert_rag_engine = None

# Abbreviation engine will be initialized after class definition

# =================== MEDICAL ABBREVIATIONS DATABASE ===================
MEDICAL_ABBREVIATIONS = {
    # Cardiovascular
    "HTN": ["hypertension", "high blood pressure", "elevated blood pressure"],
    "MI": ["myocardial infarction", "heart attack", "acute myocardial infarction"],
    "CHF": ["congestive heart failure", "heart failure", "cardiac failure"],
    "CAD": ["coronary artery disease", "coronary heart disease"],
    "PAD": ["peripheral artery disease", "peripheral arterial disease"],
    "AF": ["atrial fibrillation", "atrial fib"],
    "SVT": ["supraventricular tachycardia"],
    "VT": ["ventricular tachycardia"],
    "VF": ["ventricular fibrillation"],
    
    # Endocrine
    "DM": ["diabetes mellitus", "diabetes"],
    "T1DM": ["type 1 diabetes mellitus", "type 1 diabetes"],
    "T2DM": ["type 2 diabetes mellitus", "type 2 diabetes"],
    "DKA": ["diabetic ketoacidosis"],
    "HHS": ["hyperglycemic hyperosmolar syndrome"],
    "TSH": ["thyroid stimulating hormone"],
    
    # Respiratory
    "COPD": ["chronic obstructive pulmonary disease", "emphysema", "chronic bronchitis"],
    "ARDS": ["acute respiratory distress syndrome"],
    "PE": ["pulmonary embolism", "lung clot"],
    "PNA": ["pneumonia"],
    "URI": ["upper respiratory infection"],
    "SOB": ["shortness of breath", "dyspnea"],
    
    # Hematology
    "DVT": ["deep vein thrombosis", "blood clot"],
    "VTE": ["venous thromboembolism"],
    "DIC": ["disseminated intravascular coagulation"],
    "ITP": ["idiopathic thrombocytopenic purpura"],
    
    # Neurology
    "CVA": ["cerebrovascular accident", "stroke"],
    "TIA": ["transient ischemic attack", "mini stroke"],
    "SAH": ["subarachnoid hemorrhage"],
    "ICH": ["intracerebral hemorrhage"],
    "MS": ["multiple sclerosis"],
    "PD": ["parkinson disease", "parkinson's disease"],
    
    # Gastroenterology
    "IBD": ["inflammatory bowel disease"],
    "UC": ["ulcerative colitis"],
    "CD": ["crohn disease", "crohn's disease"],
    "GERD": ["gastroesophageal reflux disease"],
    "PUD": ["peptic ulcer disease"],
    "GIB": ["gastrointestinal bleeding"],
    
    # Infectious Disease
    "UTI": ["urinary tract infection"],
    "HAI": ["healthcare associated infection"],
    "MRSA": ["methicillin resistant staphylococcus aureus"],
    "VRE": ["vancomycin resistant enterococcus"],
    "C diff": ["clostridioides difficile", "clostridium difficile"],
    
    # Critical Care
    "SIRS": ["systemic inflammatory response syndrome"],
    "MODS": ["multiple organ dysfunction syndrome"],
    "AKI": ["acute kidney injury"],
    "CKD": ["chronic kidney disease"],
    "ARF": ["acute renal failure"],
    "ESRD": ["end stage renal disease"],
    
    # Cancer
    "CA": ["cancer", "carcinoma"],
    "NHL": ["non hodgkin lymphoma"],
    "ALL": ["acute lymphoblastic leukemia"],
    "AML": ["acute myeloid leukemia"],
    "CLL": ["chronic lymphocytic leukemia"],
    "CML": ["chronic myeloid leukemia"]
}

# =================== USAGE GUIDES ===================
USAGE_GUIDE = {
    "overview": {
        "title": "Medical Research Intelligence Platform v2.1 - Hybrid API + BioBERT RAG",
        "description": "Advanced AI-powered platform combining authoritative medical APIs with BioBERT semantic search for comprehensive medical concept discovery.",
        "key_features": [
            "🔍 Hybrid search: API + BioBERT RAG",
            "🏥 Authoritative medical codes from live APIs",
            "🧬 Semantic discovery using BioBERT embeddings",
            "🎯 Confidence scoring and validation",
            "🗺️ Cross-ontology mapping across 5 major systems",
            "🤖 AI-powered medical assistant with hybrid context",
            "📊 Enhanced batch processing with semantic enrichment",
            "🔤 70+ medical abbreviation database"
        ]
    },
    "hybrid_search": {
        "title": "🧬 Hybrid Search (API + BioBERT)",
        "purpose": "Combines authoritative API results with BioBERT semantic discovery for comprehensive medical concept mapping",
        "when_to_use": "When you need both official codes AND semantically related concepts for complete research coverage",
        "benefits": [
            "✅ Authoritative results from official medical APIs",
            "✅ Semantic discovery of related concepts via BioBERT",
            "✅ Validation of RAG results against API data",
            "✅ Discovery of concepts not found in API searches",
            "✅ Enhanced confidence scoring and quality indicators"
        ],
        "result_types": [
            "🔗 API Results: Official, authoritative medical codes",
            "✅ Validated RAG: BioBERT results confirmed by APIs",
            "🔍 Discovery: Related concepts found only via semantic search",
            "🎯 Hybrid Confidence: Overall quality score for all results"
        ]
    }
}

# =================== MODELS ===================
class MedicalQuery(BaseModel):
    query: str
    ontologies: Optional[List[str]] = ["umls", "rxnorm", "icd10", "snomed", "loinc"]
    expand_abbreviations: Optional[bool] = True
    semantic_search: Optional[bool] = True
    confidence_threshold: Optional[float] = 0.5
    search_mode: Optional[str] = "hybrid"  # "api_only", "rag_only", "hybrid"
    focus_type: Optional[str] = "comprehensive"  # "authoritative", "discovery", "comprehensive"
    
    # New Pydantic AI workflow fields
    research_intent: Optional[str] = None  # "cohort_definition", "literature_mapping", etc.
    concept_type: Optional[str] = None     # "condition", "procedure", "medication", etc.

class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None

class MedicalConcept(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    concept_name: str
    source_ontology: str
    concept_id: str
    definition: Optional[str] = None
    synonyms: Optional[List[str]] = []
    semantic_types: Optional[List[str]] = []
    confidence_score: Optional[float] = 1.0
    is_abbreviation_expansion: Optional[bool] = False
    search_method: Optional[str] = "api"  # "api", "biobert_rag", "hybrid"
    is_rag_result: Optional[bool] = False
    api_validated: Optional[Any] = None  # None, True, False, "partial"
    validation_confidence: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class CrossOntologyMapping(BaseModel):
    original_term: str
    expanded_terms: List[str]
    mappings: Dict[str, List[MedicalConcept]]
    confidence_score: float

class MappingResult(BaseModel):
    original_term: str
    mapped_concepts: List[MedicalConcept]
    confidence_score: float

class HybridSearchResponse(BaseModel):
    api_results: List[MedicalConcept]
    rag_results: List[MedicalConcept]
    validated_results: List[MedicalConcept]
    discovery_results: List[MedicalConcept]
    hybrid_confidence: float
    search_metadata: Dict[str, Any]

class BioBERTDataUpload(BaseModel):
    file_type: str  # "json", "csv", "pickle"
    file_path: str

# CSV Processing Models
class ColumnMapping(BaseModel):
    column: str
    medical_type: str
    terminology_systems: List[str]
    search_mode: str = "api_only"
    confidence_threshold: float = 0.5

class CSVMappingConfig(BaseModel):
    columns: List[ColumnMapping]
    batch_size: Optional[int] = 50
    max_concurrent: Optional[int] = 5

class CSVUploadResponse(BaseModel):
    file_id: str
    filename: str
    analysis: Dict[str, Any]
    suggested_mappings: Dict[str, Any]

class CSVProcessingResponse(BaseModel):
    job_id: str
    status: str
    message: str

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    current_operation: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None

# Pinecone Configuration Model
class PineconeConfig(BaseModel):
    api_key: str
    index_name: str = "medical-biobert"
    environment: Optional[str] = None

# BioBERT Data Management Endpoints
@api_router.post("/biobert/upload-data")
async def upload_biobert_data(upload_request: BioBERTDataUpload):
    """Upload BioBERT embeddings data"""
    try:
        if not biobert_rag_engine.is_initialized:
            await biobert_rag_engine.initialize()
        
        file_path = upload_request.file_path
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        if upload_request.file_type == "json":
            await biobert_rag_engine.load_biobert_data_from_json(file_path)
        elif upload_request.file_type == "csv":
            await biobert_rag_engine.load_biobert_data_from_csv(file_path)
        elif upload_request.file_type == "pickle":
            await biobert_rag_engine.load_biobert_data_from_pickle(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use 'json', 'csv', or 'pickle'")
        
        # Get updated stats
        stats = await biobert_rag_engine.get_stats()
        
        return {
            "message": "BioBERT data uploaded successfully",
            "file_path": file_path,
            "file_type": upload_request.file_type,
            "collection_stats": stats
        }
        
    except Exception as e:
        logger.error(f"BioBERT data upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/pinecone/create-index")
async def create_pinecone_index(config: PineconeConfig):
    """Create a new Pinecone index"""
    try:
        if not PINECONE_AVAILABLE:
            raise HTTPException(status_code=400, detail="Pinecone not available")
        
        # Initialize Pinecone client
        pc = Pinecone(api_key=config.api_key)
        
        # Check if index already exists
        indexes = pc.list_indexes()
        index_names = [idx.name for idx in indexes]
        
        if config.index_name in index_names:
            return {
                "message": f"Index '{config.index_name}' already exists",
                "index_name": config.index_name,
                "status": "exists"
            }
        
        # Create index with appropriate dimensions for BioBERT embeddings
        pc.create_index(
            name=config.index_name,
            dimension=768,  # BioBERT embedding dimension
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'  # Free tier region
            )
        )
        
        return {
            "message": f"Index '{config.index_name}' created successfully",
            "index_name": config.index_name,
            "dimension": 768,
            "metric": "cosine",
            "status": "created"
        }
        
    except Exception as e:
        logger.error(f"Pinecone index creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pinecone index creation failed: {str(e)}")

@api_router.post("/pinecone/configure")
async def configure_pinecone(config: PineconeConfig):
    """Configure Pinecone integration"""
    try:
        # Update environment variables
        os.environ['PINECONE_API_KEY'] = config.api_key
        os.environ['PINECONE_INDEX_NAME'] = config.index_name
        if config.environment:
            os.environ['PINECONE_ENVIRONMENT'] = config.environment
        
        # Reinitialize BioBERT engine with Pinecone
        global biobert_rag_engine
        biobert_rag_engine = BioBERTRAGEngine(
            vector_db_type="pinecone",
            pinecone_api_key=config.api_key,
            pinecone_index_name=config.index_name,
            pinecone_environment=config.environment
        )
        
        await biobert_rag_engine.initialize()
        
        # Get stats to verify connection
        stats = await biobert_rag_engine.get_stats()
        
        return {
            "message": "Pinecone configured successfully",
            "index_name": config.index_name,
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Pinecone configuration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pinecone configuration failed: {str(e)}")

@api_router.get("/pinecone/status")
async def get_pinecone_status():
    """Get Pinecone connection status"""
    try:
        global biobert_rag_engine
        
        # Initialize if not yet done
        if biobert_rag_engine is None:
            try:
                biobert_rag_engine = await get_biobert_rag_engine()
            except Exception as e:
                logger.error(f"Failed to initialize BioBERT engine: {e}")
                return {"status": "error", "error": str(e), "database_type": "pinecone"}
        
        if not biobert_rag_engine.is_initialized:
            return {"status": "not_initialized", "database_type": "unknown"}
        
        stats = await biobert_rag_engine.get_stats()
        
        return {
            "status": "connected" if stats else "error",
            "database_type": stats.get("database_type", "unknown"),
            "stats": stats
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@api_router.post("/vector-db/switch")
async def switch_vector_database(db_type: str = "pinecone"):
    """Initialize Pinecone vector database"""
    try:
        pinecone_api_key = settings.pinecone_api_key.get_secret_value() if settings.pinecone_api_key else None
        if not pinecone_api_key:
            raise HTTPException(status_code=400, detail="Pinecone API key not configured")
        
        # Get or create the BioBERT RAG engine
        biobert_rag_engine = await get_biobert_rag_engine()
        stats = await biobert_rag_engine.get_stats()
        
        return {
            "message": "Using Pinecone vector database",
            "database_type": "pinecone",
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Vector database initialization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =================== MEDICAL ABBREVIATION ENGINE ===================
class MedicalAbbreviationEngine:
    def __init__(self):
        self.abbreviations = MEDICAL_ABBREVIATIONS
        
    def expand_abbreviations(self, query: str) -> List[str]:
        """Expand medical abbreviations in query"""
        expanded_terms = [query]  # Always include original
        
        # Check for exact abbreviation matches
        query_upper = query.upper()
        if query_upper in self.abbreviations:
            expanded_terms.extend(self.abbreviations[query_upper])
        
        # Check for abbreviations within text
        words = re.split(r'[,;\s]+', query)
        for word in words:
            word_upper = word.upper().strip()
            if word_upper in self.abbreviations:
                for expansion in self.abbreviations[word_upper]:
                    expanded_terms.append(expansion)
                    # Also create version with expansion substituted
                    expanded_query = query.replace(word, expansion)
                    expanded_terms.append(expanded_query)
        
        return list(set(expanded_terms))  # Remove duplicates
    
    def get_abbreviation_confidence(self, original: str, expanded: str) -> float:
        """Calculate confidence score for abbreviation expansion"""
        if original.upper() in self.abbreviations:
            return 0.95
        return 0.8

# Create abbreviation engine instance
abbreviation_engine = MedicalAbbreviationEngine()

# Setup sample BioBERT data function
async def setup_sample_biobert_data():
    """Setup sample BioBERT data for demonstration"""
    try:
        sample_concepts = [
            {
                "concept": "Type 2 Diabetes Mellitus",
                "concept_id": "E11.9",
                "definition": "A metabolic disorder characterized by high blood sugar",
                "ontology": "ICD-10"
            },
            {
                "concept": "Diabetes mellitus",
                "concept_id": "73211009",
                "definition": "A group of metabolic disorders characterized by hyperglycemia",
                "ontology": "SNOMED CT"
            },
            {
                "concept": "Metformin",
                "concept_id": "6809",
                "definition": "An oral antidiabetic drug in the biguanide class",
                "ontology": "RxNorm"
            },
            # Sepsis-related concepts for testing
            {
                "concept": "Sepsis, unspecified organism",
                "concept_id": "A41.9",
                "definition": "Life-threatening organ dysfunction caused by dysregulated host response to infection",
                "ontology": "ICD-10"
            },
            {
                "concept": "Severe sepsis",
                "concept_id": "R65.20",
                "definition": "Sepsis with acute organ dysfunction",
                "ontology": "ICD-10"
            },
            {
                "concept": "Septic shock",
                "concept_id": "R65.21",
                "definition": "Sepsis with hypotension despite adequate fluid resuscitation",
                "ontology": "ICD-10"
            },
            {
                "concept": "Sepsis",
                "concept_id": "91302008",
                "definition": "Systemic inflammatory response syndrome due to infection",
                "ontology": "SNOMED CT"
            },
            {
                "concept": "Septicemia",
                "concept_id": "76571007",
                "definition": "Presence of pathogenic microorganisms or their toxins in the blood",
                "ontology": "SNOMED CT"
            },
            {
                "concept": "Neonatal sepsis",
                "concept_id": "P36.9",
                "definition": "Bacterial sepsis of newborn, unspecified",
                "ontology": "ICD-10"
            },
            {
                "concept": "Streptococcal sepsis",
                "concept_id": "A40.9",
                "definition": "Streptococcal sepsis, unspecified",
                "ontology": "ICD-10"
            },
            {
                "concept": "Sepsis due to Escherichia coli",
                "concept_id": "A41.51",
                "definition": "Sepsis caused by E. coli infection",
                "ontology": "ICD-10"
            }
        ]
        
        # Add to BioBERT engine
        result = await biobert_rag_engine.add_concepts(sample_concepts)
        logger.info(f"Added sample BioBERT data: {result}")
        return result
    except Exception as e:
        logger.error(f"Failed to setup sample data: {e}")
        return {"error": str(e)}

# =================== SEMANTIC SEARCH ENGINE ===================
class SemanticSearchEngine:
    def __init__(self, model):
        self.model = model
        
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text"""
        return self.model.encode(text)
    
    def semantic_similarity(self, query: str, concepts: List[MedicalConcept]) -> List[MedicalConcept]:
        """Add semantic similarity scores to concepts"""
        if not concepts:
            return concepts
            
        query_embedding = self.get_embedding(query)
        
        for concept in concepts:
            concept_text = f"{concept.concept_name} {concept.definition or ''}"
            concept_embedding = self.get_embedding(concept_text)
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, concept_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(concept_embedding)
            )
            
            # Adjust confidence based on semantic similarity
            concept.confidence_score = float(similarity)
        
        return sorted(concepts, key=lambda x: x.confidence_score, reverse=True)

# Create semantic search engine instance  
semantic_engine = SemanticSearchEngine(embedding_model)

# =================== CSV EXPORT UTILITIES ===================
def create_csv_from_concepts(concepts: List[MedicalConcept], original_query: str = "") -> str:
    """Create CSV content from medical concepts"""
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    header = [
        'Original_Query',
        'Concept_Name', 
        'Source_Ontology', 
        'Concept_ID', 
        'Definition',
        'Confidence_Score',
        'Search_Method',
        'Is_RAG_Result',
        'API_Validated',
        'Validation_Confidence',
        'Is_Abbreviation_Expansion',
        'Semantic_Types',
        'Timestamp'
    ]
    writer.writerow(header)
    
    # Write data
    for concept in concepts:
        row = [
            original_query,
            concept.concept_name,
            concept.source_ontology,
            concept.concept_id,
            concept.definition or '',
            f"{concept.confidence_score:.3f}" if concept.confidence_score else '',
            concept.search_method or 'api',
            'Yes' if concept.is_rag_result else 'No',
            str(concept.api_validated) if concept.api_validated is not None else '',
            f"{concept.validation_confidence:.3f}" if concept.validation_confidence else '',
            'Yes' if concept.is_abbreviation_expansion else 'No',
            '; '.join(concept.semantic_types) if concept.semantic_types else '',
            concept.timestamp.isoformat() if concept.timestamp else ''
        ]
        writer.writerow(row)
    
    return output.getvalue()

def create_research_csv_with_attribution(research_results: Dict[str, Any], original_query: str) -> str:
    """Create research-grade CSV from Pydantic AI workflow results with full attribution"""
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write enhanced research header
    header = [
        'Original_Query',
        'Research_Intent',
        'Concept_Type',
        'Mapping_Strategy',
        'System',
        'Code',
        'Character_Pattern',
        'Preferred_Term',
        'Hierarchy_Path',
        'Semantic_Type',
        'Confidence_Score',
        'Source_Method',
        'Validation_Status',
        'Research_Notes',
        'Cross_Mappings',
        'Code_Set_Inclusion',
        'Agent_Used',
        'Workflow_Type',
        'Timestamp'
    ]
    writer.writerow(header)
    
    # Extract research analysis
    analysis = research_results.get('research_analysis', {})
    research_intent = analysis.get('research_intent', 'Not specified')
    concept_type = analysis.get('concept_type', 'Not specified')
    mapping_strategy = analysis.get('mapping_strategy', 'Not specified')
    
    # Write terminology mappings
    terminology_mappings = research_results.get('terminology_mappings', {})
    for system, mappings in terminology_mappings.items():
        for mapping in mappings:
            row = [
                original_query,
                research_intent,
                concept_type,
                mapping_strategy,
                system,
                mapping.get('primary_code', ''),
                mapping.get('character_pattern', ''),
                mapping.get('preferred_term', ''),
                ' > '.join(mapping.get('hierarchy_path', [])),
                mapping.get('semantic_type', ''),
                f"{mapping.get('confidence_score', 0):.3f}",
                mapping.get('source_method', 'api'),
                mapping.get('validation_status', 'pending'),
                mapping.get('research_notes', ''),
                'Available' if research_results.get('cross_mappings') else 'None',
                'Available' if research_results.get('code_set') else 'None',
                'TerminologyMappingAgent',
                'Pydantic AI + LangGraph',
                datetime.utcnow().isoformat()
            ]
            writer.writerow(row)
    
    # Write cross-mappings if available
    cross_mappings = research_results.get('cross_mappings', [])
    if cross_mappings:
        for cross_mapping in cross_mappings:
            row = [
                original_query,
                research_intent,
                concept_type,
                'cross_terminology_harmonization',
                f"{cross_mapping.get('source_system', '')} → Multiple",
                cross_mapping.get('source_code', ''),
                'Cross-mapping',
                cross_mapping.get('source_term', ''),
                '',
                'Cross-reference',
                f"{cross_mapping.get('mapping_confidence', 0):.3f}",
                'cross_harmonization',
                'validated',
                cross_mapping.get('research_implications', ''),
                'This record',
                'N/A',
                'CrossTerminologyHarmonizer',
                'Pydantic AI + LangGraph',
                datetime.utcnow().isoformat()
            ]
            writer.writerow(row)
    
    return output.getvalue()

def create_enhanced_csv_with_attribution(hybrid_results: HybridSearchResponse, original_query: str) -> str:
    """Create comprehensive CSV from enhanced hybrid search results with full source attribution"""
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write enhanced header with source attribution
    header = [
        'Original_Query',
        'Result_Type',
        'Data_Source',
        'Concept_Name',
        'Source_Ontology',
        'Concept_ID',
        'Definition',
        'Confidence_Score',
        'Search_Method',
        'API_Validated',
        'Validation_Confidence',
        'Hybrid_Confidence',
        'Is_RAG_Result',
        'Pinecone_Enabled',
        'Data_Provenance',
        'Timestamp'
    ]
    writer.writerow(header)
    
    # Helper function to write concept rows with enhanced attribution
    def write_enhanced_concept_rows(concepts, result_type, data_source):
        for concept in concepts:
            # Determine data provenance
            if concept.search_method == "api":
                provenance = f"API_Direct_{concept.source_ontology}"
            elif concept.search_method == "pinecone_rag":
                provenance = f"Pinecone_Vector_Database"
            else:
                provenance = f"Hybrid_{concept.search_method}"
            
            # Enhanced attribution information
            api_validated_status = "Yes" if concept.api_validated is True else ("No" if concept.api_validated is False else "N/A")
            
            row = [
                original_query,
                result_type,
                data_source,
                concept.concept_name,
                concept.source_ontology,
                concept.concept_id,
                concept.definition or '',
                f"{concept.confidence_score:.3f}" if concept.confidence_score else '',
                concept.search_method or 'api',
                api_validated_status,
                f"{concept.validation_confidence:.3f}" if concept.validation_confidence else '',
                f"{hybrid_results.hybrid_confidence:.3f}",
                'Yes' if concept.is_rag_result else 'No',
                'Yes',  # Pinecone enabled for enhanced search
                provenance,
                concept.timestamp.isoformat() if concept.timestamp else datetime.utcnow().isoformat()
            ]
            writer.writerow(row)
    
    # Write different result types with clear attribution
    write_enhanced_concept_rows(hybrid_results.api_results, "API_Authoritative", "Medical_APIs")
    write_enhanced_concept_rows(hybrid_results.validated_results, "RAG_Validated", "Pinecone_Validated")
    write_enhanced_concept_rows(hybrid_results.discovery_results, "RAG_Discovery", "Pinecone_Semantic")
    
    return output.getvalue()



# =================== ENHANCED API ENDPOINTS ===================

@api_router.get("/")
async def root():
    return {"message": "Medical Research Intelligence Platform v2.1 - Hybrid API + BioBERT RAG Search!"}

@api_router.get("/usage-guide")
async def get_usage_guide():
    """Get comprehensive usage guide for the platform"""
    return USAGE_GUIDE

# Create research dependencies instance
research_deps = ResearchTerminologyDeps()

# NEW: Pydantic AI + LangGraph Research Workflow Endpoint
@api_router.post("/research-query")
async def research_terminology_query(query: MedicalQuery):
    """
    Advanced medical terminology research using Pydantic AI + LangGraph workflow
    
    This endpoint implements the full research workflow:
    1. Research Query Analysis (Pydantic AI agent)
    2. Parallel Terminology Mapping (across multiple systems)
    3. Cross-Terminology Harmonization 
    4. Research Code Set Building (for cohort definition/outcome measurement)
    5. Research-Grade Output Formatting
    """
    try:
        # Execute research workflow using Pydantic AI + LangGraph
        result = await execute_medical_concept_search(
            query=query.query,
            research_intent=getattr(query, 'research_intent', None),
            concept_type=getattr(query, 'concept_type', None),
            target_systems=[ont.upper() for ont in query.ontologies],
            search_mode=getattr(query, 'search_mode', 'api_only'),
            deps=research_deps,
            medical_api_client=medical_api_client
        )
        
        # Store research query in database
        research_record = {
            "id": str(uuid.uuid4()),
            "query": query.query,
            "research_intent": getattr(query, 'research_intent', None),
            "concept_type": getattr(query, 'concept_type', None),
            "search_mode": getattr(query, 'search_mode', 'api_only'),
            "workflow_type": "pydantic_ai_langgraph",
            "success": result.get('success', False),
            "terminology_count": sum(len(mappings) for mappings in result.get('terminology_mappings', {}).values()),
            "cross_mappings_count": len(result.get('cross_mappings', [])),
            "has_code_set": result.get('code_set') is not None,
            "timestamp": datetime.utcnow()
        }
        
        # Store in database (with error handling)
        try:
            await db.research_queries.insert_one(research_record)
            logger.info("Research query record saved to database")
        except Exception as e:
            logger.warning(f"Database insertion failed (continuing without): {e}")
            # Continue without database storage
        
        return result
        
    except Exception as e:
        logger.error(f"Research workflow failed: {e}")
        raise HTTPException(status_code=500, detail=f"Research workflow failed: {str(e)}")

@api_router.post("/research-query/download-csv")
async def download_research_query_csv(query: MedicalQuery):
    """Download research query results as comprehensive CSV with research attribution"""
    # Execute research workflow
    research_results = await research_terminology_query(query)
    
    # Create research-grade CSV content
    csv_content = create_research_csv_with_attribution(research_results, query.query)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"research_terminology_{timestamp}.csv"
    
    # Return as streaming response
    return StreamingResponse(
        io.BytesIO(csv_content.encode('utf-8')),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

# Traditional search endpoint (preserved for compatibility)
@api_router.post("/search", response_model=List[MedicalConcept])
async def search_medical_concepts(query: MedicalQuery):
    """Enhanced search across multiple medical ontologies with abbreviation expansion and semantic search"""
    try:
        all_results = []
        
        # Phase 1: Abbreviation Expansion
        search_terms = [query.query]
        if query.expand_abbreviations:
            expanded_terms = abbreviation_engine.expand_abbreviations(query.query)
            search_terms = expanded_terms
        
        # Search each term across ontologies
        for term in search_terms:
            is_expansion = term != query.query
            
            # Search each requested ontology
            if "umls" in query.ontologies:
                umls_results = await medical_api_client.search_umls(term)
                if umls_results:
                    for result in umls_results:
                        result['is_abbreviation_expansion'] = is_expansion
                    all_results.extend(umls_results)
            
            if "rxnorm" in query.ontologies:
                rxnorm_results = await medical_api_client.search_rxnorm(term)
                if rxnorm_results:
                    for result in rxnorm_results:
                        result['is_abbreviation_expansion'] = is_expansion
                    all_results.extend(rxnorm_results)
            
            if "icd10" in query.ontologies:
                icd10_results = await medical_api_client.search_icd10(term)
                if icd10_results:
                    for result in icd10_results:
                        result['is_abbreviation_expansion'] = is_expansion
                    all_results.extend(icd10_results)
            
            if "snomed" in query.ontologies:
                snomed_results = await medical_api_client.search_snomed(term)
                if snomed_results:
                    for result in snomed_results:
                        result['is_abbreviation_expansion'] = is_expansion
                    all_results.extend(snomed_results)
            
            if "loinc" in query.ontologies:
                loinc_results = await medical_api_client.search_loinc(term)
                if loinc_results:
                    for result in loinc_results:
                        result['is_abbreviation_expansion'] = is_expansion
                    all_results.extend(loinc_results)
        
        # Convert to MedicalConcept objects
        concepts = []
        for result in all_results:
            if result and result.get('concept_id') and result.get('concept_name'):
                concept = MedicalConcept(
                    concept_name=result['concept_name'],
                    source_ontology=result['source_ontology'],
                    concept_id=result['concept_id'],
                    definition=result.get('definition'),
                    synonyms=result.get('synonyms', []),
                    semantic_types=result.get('semantic_types', []),
                    is_abbreviation_expansion=result.get('is_abbreviation_expansion', False),
                    confidence_score=1.0,
                    search_method="api"
                )
                concepts.append(concept)
        
        # Phase 2: Semantic Enhancement
        if query.semantic_search and concepts:
            concepts = semantic_engine.semantic_similarity(query.query, concepts)
        
        # Filter by confidence threshold
        concepts = [c for c in concepts if c.confidence_score >= query.confidence_threshold]
        
        # Store in database (with error handling)
        try:
            # Temporarily comment out database insertion to fix timeout
            # for concept in concepts:
            #     await db.medical_concepts.insert_one(concept.dict())
            logger.info(f"Search completed with {len(concepts)} results")
        except Exception as e:
            logger.error(f"Database insertion failed: {e}")
            # Continue without database storage
            pass
        
        return concepts
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        # Raise the actual error so we can debug the real issue
        raise HTTPException(status_code=500, detail=f"Medical search failed: {str(e)}")

# NEW: Hybrid search endpoint
@api_router.post("/hybrid-search", response_model=HybridSearchResponse)
async def hybrid_search_medical_concepts(query: MedicalQuery):
    """
    Advanced hybrid search combining API calls with BioBERT RAG for comprehensive medical concept discovery
    """
    try:
        # Initialize hybrid search engine if needed
        if not hybrid_search_engine.biobert_engine.is_initialized:
            await hybrid_search_engine.initialize()
            # Set up sample data if BioBERT collection is empty
            stats = await biobert_rag_engine.get_stats()
            if stats.get('total_concepts', 0) == 0:
                logger.info("Setting up sample BioBERT data for demonstration")
                await setup_sample_biobert_data()
        
        # Create a robust API search function wrapper with detailed logging
        async def robust_api_search(query_dict):
            logger.info(f"🔗 API Search called with: {query_dict}")
            try:
                # Create MedicalQuery object with defaults
                api_query = MedicalQuery(
                    query=query_dict.get('query', ''),
                    ontologies=query_dict.get('ontologies', ['umls']),
                    expand_abbreviations=query_dict.get('expand_abbreviations', True),
                    semantic_search=query_dict.get('semantic_search', True),
                    confidence_threshold=query_dict.get('confidence_threshold', 0.5)
                )
                logger.info(f"🔗 Created MedicalQuery: {api_query.query} with {len(api_query.ontologies)} ontologies")
                
                # Call the search function
                results = await search_medical_concepts(api_query)
                logger.info(f"🔗 API search returned {len(results)} results")
                
                # Convert results to dicts consistently
                dict_results = []
                for result in results:
                    if hasattr(result, 'dict'):
                        dict_results.append(result.dict())
                    elif isinstance(result, dict):
                        dict_results.append(result)
                    else:
                        logger.warning(f"🔗 Unexpected result type: {type(result)}")
                        
                logger.info(f"🔗 Converted to {len(dict_results)} dict results")
                return dict_results
                
            except Exception as e:
                logger.error(f"🔗 API search failed in hybrid search: {e}")
                import traceback
                logger.error(f"🔗 Full traceback: {traceback.format_exc()}")
                return []
        
        # Perform hybrid search with robust API function
        logger.info(f"🔄 Starting hybrid search for: {query.query}")
        hybrid_result = await hybrid_search_engine.hybrid_search(
            query=query.query,
            ontologies=query.ontologies,
            api_search_func=robust_api_search,
            expand_abbreviations=query.expand_abbreviations,
            semantic_search=query.semantic_search,
            confidence_threshold=query.confidence_threshold
        )
        logger.info(f"🔄 Hybrid search completed with {len(hybrid_result.api_results)} API + {len(hybrid_result.rag_results)} RAG results")
        
        # Helper function to convert RAG results to MedicalConcept format
        def normalize_rag_result(rag_result):
            """Convert RAG result to MedicalConcept-compatible format"""
            if isinstance(rag_result, dict):
                return {
                    'concept_name': rag_result.get('concept_name', rag_result.get('concept', '')),
                    'concept_id': rag_result.get('concept_id', rag_result.get('id', '')),
                    'source_ontology': rag_result.get('source_ontology', rag_result.get('ontology', 'PINECONE')),
                    'definition': rag_result.get('definition', rag_result.get('text', '')),
                    'confidence_score': rag_result.get('confidence_score', rag_result.get('score', 0.5)),
                    'search_method': rag_result.get('search_method', 'biobert_rag'),
                    'is_rag_result': True,
                    'synonyms': rag_result.get('synonyms', []),
                    'semantic_types': rag_result.get('semantic_types', [])
                }
            return rag_result
        
        # Normalize all RAG-based results
        normalized_rag_results = [normalize_rag_result(r) for r in hybrid_result.rag_results]
        normalized_validated_results = [normalize_rag_result(r) for r in hybrid_result.validated_results]
        normalized_discovery_results = [normalize_rag_result(r) for r in hybrid_result.discovery_results]
        
        # Convert results to response format
        response = HybridSearchResponse(
            api_results=hybrid_result.api_results,
            rag_results=normalized_rag_results,
            validated_results=normalized_validated_results,
            discovery_results=normalized_discovery_results,
            hybrid_confidence=hybrid_result.hybrid_confidence,
            search_metadata=hybrid_result.search_metadata
        )
        
        # Store hybrid search results
        search_record = {
            "id": str(uuid.uuid4()),
            "query": query.query,
            "search_type": "hybrid",
            "api_results_count": len(hybrid_result.api_results),
            "rag_results_count": len(hybrid_result.rag_results),
            "validated_count": len(hybrid_result.validated_results),
            "discovery_count": len(hybrid_result.discovery_results),
            "hybrid_confidence": hybrid_result.hybrid_confidence,
            "timestamp": datetime.utcnow()
        }
        
        # Store in database (with error handling)
        try:
            await db.hybrid_searches.insert_one(search_record)
            logger.info("Hybrid search record saved to database")
        except Exception as e:
            logger.warning(f"Database insertion failed (continuing without): {e}")
            # Continue without database storage
        
        return response
        
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {str(e)}")

@api_router.post("/hybrid-search/download-csv")
async def download_hybrid_search_csv(query: MedicalQuery):
    """Download hybrid search results as comprehensive CSV"""
    # Perform hybrid search
    hybrid_results = await hybrid_search_medical_concepts(query)
    
    # Create CSV content
    csv_content = create_enhanced_csv_with_attribution(hybrid_results, query.query)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"hybrid_search_{timestamp}.csv"
    
    # Return as streaming response
    return StreamingResponse(
        io.BytesIO(csv_content.encode('utf-8')),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@api_router.get("/related-concepts/{concept_id}")
async def get_related_concepts(concept_id: str, ontology: str, limit: int = 10):
    """Get semantically related concepts using BioBERT"""
    try:
        if not hybrid_search_engine.biobert_engine.is_initialized:
            await hybrid_search_engine.initialize()
        
        related_concepts = await hybrid_search_engine.get_related_concepts(
            concept_id=concept_id,
            ontology=ontology,
            limit=limit
        )
        
        return {
            "concept_id": concept_id,
            "ontology": ontology,
            "related_concepts": related_concepts,
            "count": len(related_concepts)
        }
        
    except Exception as e:
        logger.error(f"Failed to get related concepts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# BioBERT Data Management Endpoints
@api_router.post("/biobert/upload-data")
async def upload_biobert_data(upload_request: BioBERTDataUpload):
    """Upload BioBERT embeddings data"""
    try:
        if not biobert_rag_engine.is_initialized:
            await biobert_rag_engine.initialize()
        
        file_path = upload_request.file_path
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        if upload_request.file_type == "json":
            await biobert_rag_engine.load_biobert_data_from_json(file_path)
        elif upload_request.file_type == "csv":
            await biobert_rag_engine.load_biobert_data_from_csv(file_path)
        elif upload_request.file_type == "pickle":
            await biobert_rag_engine.load_biobert_data_from_pickle(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use 'json', 'csv', or 'pickle'")
        
        # Get updated stats
        stats = await biobert_rag_engine.get_stats()
        
        return {
            "message": "BioBERT data uploaded successfully",
            "file_path": file_path,
            "file_type": upload_request.file_type,
            "collection_stats": stats
        }
        
    except Exception as e:
        logger.error(f"BioBERT data upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/biobert/stats")
async def get_biobert_stats():
    """Get BioBERT collection statistics"""
    try:
        if not biobert_rag_engine.is_initialized:
            await biobert_rag_engine.initialize()
        
        stats = await biobert_rag_engine.get_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get BioBERT stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/biobert/setup-sample")
async def setup_sample_data():
    """Set up sample BioBERT data for testing"""
    try:
        if not biobert_rag_engine.is_initialized:
            await biobert_rag_engine.initialize()
        
        await setup_sample_biobert_data()
        stats = await biobert_rag_engine.get_stats()
        
        return {
            "message": "Sample BioBERT data created successfully",
            "collection_stats": stats
        }
        
    except Exception as e:
        logger.error(f"Failed to setup sample data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# NEW: Debug endpoint for direct BioBERT search
@api_router.get("/debug/biobert-search")
async def debug_biobert_search(query: str, limit: int = 5):
    """Debug endpoint to test BioBERT search directly"""
    try:
        logger.info(f"🔍 Debug BioBERT search for: {query}")
        
        # Initialize if needed
        if not biobert_rag_engine.is_initialized:
            await biobert_rag_engine.initialize()
        
        # Direct search using BioBERT engine
        results = await biobert_rag_engine.search(query, limit=limit)
        
        # Get stats
        stats = await biobert_rag_engine.get_stats()
        
        return {
            "query": query,
            "results_count": len(results),
            "results": results,
            "index_stats": stats,
            "embedding_model": biobert_rag_engine.embedding_model.__class__.__name__ if biobert_rag_engine.embedding_model else "fallback"
        }
        
    except Exception as e:
        logger.error(f"Debug BioBERT search failed: {e}")
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "query": query
        }

# NEW: Debug endpoint to test field mapping
@api_router.get("/debug/field-mapping-test")
async def debug_field_mapping():
    """Test field mapping with sample Pinecone data"""
    try:
        # Sample data structures from Pinecone
        sample_records = [
            {"concept": "Diabetes mellitus", "code": "E11.9", "ontology": "ICD-10"},
            {"concept_name": "Type 2 diabetes", "concept_id": "73211009", "source": "SNOMED"},
            {"description": "Hypertension", "id": "I10", "code_system": "ICD-10"},
            {"text": "Essential hypertension", "code": "1201005", "source_ontology": "SNOMED"}
        ]
        
        # Process through BioBERT field standardization
        standardized = []
        for record in sample_records:
            metadata = record
            
            concept_name = (metadata.get('concept', '') or 
                           metadata.get('concept_name', '') or
                           metadata.get('description', '') or 
                           metadata.get('text', '') or
                           metadata.get('name', ''))
            
            concept_id = (metadata.get('concept_id', '') or
                         metadata.get('code', '') or
                         metadata.get('id', ''))
            
            ontology = (metadata.get('ontology', '') or 
                       metadata.get('source', '') or
                       metadata.get('code_system', '') or
                       metadata.get('source_ontology', '') or
                       'UNKNOWN')
            
            standardized.append({
                "original": record,
                "standardized": {
                    "concept_name": concept_name,
                    "concept_id": concept_id,
                    "ontology": ontology
                }
            })
        
        return {
            "message": "Field mapping test results",
            "samples_tested": len(sample_records),
            "results": standardized
        }
        
    except Exception as e:
        logger.error(f"Field mapping test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# NEW: Debug endpoint to diagnose embedding mismatch
@api_router.get("/debug/embedding-diagnostic")
async def debug_embedding_diagnostic(query: str = "diabetes"):
    """Diagnose embedding generation and search issues"""
    try:
        logger.info(f"🔬 Running embedding diagnostic for: {query}")
        
        # Initialize if needed
        if not biobert_rag_engine.is_initialized:
            await biobert_rag_engine.initialize()
        
        # Process the query
        from backend.integrations.medical_query_processor import medical_query_processor
        processed = medical_query_processor.process_query(query)
        
        # Generate embedding
        embedding = biobert_rag_engine.get_embedding(query)
        
        # Get a few direct results
        results = await biobert_rag_engine._search_pinecone(query, limit=5)
        
        # Analyze results
        result_analysis = []
        for result in results:
            result_analysis.append({
                "concept_name": result.get('concept_name', 'N/A'),
                "concept_id": result.get('concept_id', 'N/A'),
                "score": result.get('confidence_score', 0),
                "ontology": result.get('ontology', 'N/A'),
                "metadata_keys": list(result.get('metadata', {}).keys()) if 'metadata' in result else []
            })
        
        return {
            "query": query,
            "processed_query": {
                "normalized": processed.normalized_query,
                "expanded_terms": processed.expanded_terms[:5],
                "concept_type": processed.concept_type
            },
            "embedding_info": {
                "model": biobert_rag_engine.embedding_model.__class__.__name__ if biobert_rag_engine.embedding_model else "BioBERT",
                "method": biobert_rag_engine.embedding_method,
                "dimension": len(embedding),
                "norm": float(np.linalg.norm(embedding)),
                "sample_values": embedding[:5].tolist()
            },
            "search_results": {
                "count": len(results),
                "top_results": result_analysis
            },
            "recommendations": [
                "The low relevance scores suggest embedding mismatch",
                "Consider re-indexing Pinecone with current embedding model",
                "Or use the exact same model that created the Pinecone vectors"
            ]
        }
        
    except Exception as e:
        logger.error(f"Embedding diagnostic failed: {e}")
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# Enhanced Chat with Hybrid Context
@api_router.post("/chat")
async def chat_with_medical_ai(message: ChatMessage):
    """Enhanced AI chat with hybrid medical context (API + BioBERT)"""
    try:
        # Enhanced search with hybrid results
        search_query = MedicalQuery(
            query=message.message,
            expand_abbreviations=True,
            semantic_search=True,
            confidence_threshold=0.3,
            search_mode="hybrid"
        )
        
        # Try hybrid search first, fall back to regular search if needed
        try:
            hybrid_results = await hybrid_search_medical_concepts(search_query)
            all_concepts = (hybrid_results.api_results + 
                          hybrid_results.validated_results + 
                          hybrid_results.discovery_results)
        except Exception as e:
            logger.warning(f"Hybrid search failed, falling back to regular search: {e}")
            all_concepts = await search_medical_concepts(search_query)
            hybrid_results = None
        
        # Prepare enhanced context for AI
        context = "You are an advanced medical informatics AI assistant with access to both authoritative medical APIs and BioBERT semantic search. "
        
        if hybrid_results:
            context += f"Based on hybrid search results for '{message.message}':\n\n"
            
            if hybrid_results.api_results:
                context += f"**AUTHORITATIVE API RESULTS ({len(hybrid_results.api_results)}):**\n"
                for concept in hybrid_results.api_results[:5]:
                    context += f"- {concept.concept_name} ({concept.source_ontology}: {concept.concept_id})\n"
                context += "\n"
            
            if hybrid_results.validated_results:
                context += f"**VALIDATED SEMANTIC RESULTS ({len(hybrid_results.validated_results)}):**\n"
                for concept in hybrid_results.validated_results[:5]:
                    validation_indicator = "✅" if concept.api_validated == True else "⚠️"
                    context += f"- {concept.concept_name} ({concept.source_ontology}: {concept.concept_id}) {validation_indicator}\n"
                context += "\n"
            
            if hybrid_results.discovery_results:
                context += f"**SEMANTIC DISCOVERIES ({len(hybrid_results.discovery_results)}):**\n"
                for concept in hybrid_results.discovery_results[:3]:
                    context += f"- {concept.concept_name} (BioBERT discovery: {concept.confidence_score:.2f})\n"
                context += "\n"
            
            context += f"**Hybrid Confidence Score:** {hybrid_results.hybrid_confidence:.2f}\n\n"
        
        else:
            context += f"Based on medical concepts found for '{message.message}':\n\n"
            for concept in all_concepts[:10]:
                confidence_indicator = "🔥" if concept.confidence_score > 0.8 else "✓" if concept.confidence_score > 0.6 else "~"
                context += f"- {concept.concept_name} ({concept.source_ontology}: {concept.concept_id}) {confidence_indicator}\n"
            context += "\n"
        
        context += f"\nUser question: {message.message}\n\n"
        context += "Provide a comprehensive response that:\n"
        context += "1. Explains the medical concept clearly\n"
        context += "2. Lists specific codes from relevant ontologies\n"
        context += "3. Distinguishes between authoritative API results and semantic discoveries\n"
        context += "4. Explains any abbreviation expansions made\n"
        context += "5. Provides clinical context when appropriate\n"
        context += "6. Mentions confidence levels and validation status\n"
        
        # Get AI response
        response = openai_client.chat.completions.create(
            model=settings.primary_model,
            messages=[
                {"role": "system", "content": "You are an expert medical informatics AI assistant with access to both authoritative medical APIs and BioBERT semantic search. You help researchers understand the difference between official codes and semantically related concepts."},
                {"role": "user", "content": context}
            ],
            max_tokens=1500
        )
        
        ai_response = response.choices[0].message.content
        
        # Store enhanced chat history
        chat_record = {
            "id": str(uuid.uuid4()),
            "session_id": message.session_id or str(uuid.uuid4()),
            "user_message": message.message,
            "ai_response": ai_response,
            "concepts_found": len(all_concepts),
            "search_type": "hybrid" if hybrid_results else "api_only",
            "hybrid_confidence": hybrid_results.hybrid_confidence if hybrid_results else None,
            "timestamp": datetime.utcnow()
        }
        
        # Store in database (with error handling)
        try:
            await db.chat_history.insert_one(chat_record)
            logger.info("Chat history record saved to database")
        except Exception as e:
            logger.warning(f"Database insertion failed (continuing without): {e}")
            # Continue without database storage
        
        return {
            "response": ai_response,
            "concepts": all_concepts[:15],  # Changed from concepts_found to match frontend
            "search_type": "hybrid" if hybrid_results else "api_only",
            "hybrid_metadata": hybrid_results.search_metadata if hybrid_results else None,
            "conversation_id": chat_record["session_id"]  # Changed from session_id to match frontend
        }
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Preserve existing endpoints (search/download-csv, cross-ontology-mapping, etc.)
@api_router.post("/search/download-csv")
async def download_search_csv(query: MedicalQuery):
    """Download search results as CSV"""
    # Perform search
    concepts = await search_medical_concepts(query)
    
    # Create CSV content
    csv_content = create_csv_from_concepts(concepts, query.query)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"medical_search_{timestamp}.csv"
    
    # Return as streaming response
    return StreamingResponse(
        io.BytesIO(csv_content.encode('utf-8')),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@api_router.post("/search/export-html")
async def export_search_html(query: MedicalQuery, research_context: Optional[str] = None):
    """Export search results as beautiful HTML for researchers"""
    # Perform search
    concepts = await search_medical_concepts(query)
    
    # Convert concepts to dict format for HTML export
    results = []
    for concept in concepts:
        result_dict = concept.dict() if hasattr(concept, 'dict') else concept
        # Ensure display_name is set
        if 'display_name' not in result_dict:
            result_dict['display_name'] = result_dict.get('concept_name', 'Unknown')
        results.append(result_dict)
    
    # Generate HTML
    html_content = html_export_service.export_results(
        results=results,
        query=query.query,
        research_context=research_context,
        target_systems=query.ontologies if query.ontologies else None,
        search_mode="api"
    )
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"medical_terminology_results_{timestamp}.html"
    
    # Return as HTML response
    return StreamingResponse(
        io.BytesIO(html_content.encode('utf-8')),
        media_type="text/html",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@api_router.post("/hybrid-search/export-html")
async def export_hybrid_search_html(
    query: MedicalQuery, 
    research_context: Optional[str] = None,
    research_intent: Optional[str] = None
):
    """Export hybrid search results as beautiful HTML for researchers"""
    try:
        # Initialize hybrid search engine if needed
        if not hybrid_search_engine.biobert_engine.is_initialized:
            await hybrid_search_engine.initialize()
        
        # Perform hybrid search
        hybrid_result = await hybrid_search_engine.hybrid_search(
            query=query.query,
            ontologies=query.ontologies,
            api_search_func=lambda q: search_medical_concepts(MedicalQuery(**q)),
            expand_abbreviations=query.expand_abbreviations,
            semantic_search=query.semantic_search,
            confidence_threshold=query.confidence_threshold
        )
        
        # Combine all results
        all_results = []
        
        # Add validated results first (highest priority)
        for result in hybrid_result.validated_results:
            result_dict = result.dict() if hasattr(result, 'dict') else result
            result_dict['api_validated'] = True
            result_dict['display_name'] = result_dict.get('concept_name', 'Unknown')
            all_results.append(result_dict)
        
        # Add discovery results
        for result in hybrid_result.discovery_results:
            result_dict = result.dict() if hasattr(result, 'dict') else result
            result_dict['display_name'] = result_dict.get('concept_name', 'Unknown')
            all_results.append(result_dict)
        
        # Generate HTML
        html_content = html_export_service.export_results(
            results=all_results,
            query=query.query,
            research_context=research_context,
            research_intent=research_intent,
            target_systems=query.ontologies if query.ontologies else None,
            search_mode="hybrid"
        )
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"medical_terminology_hybrid_results_{timestamp}.html"
        
        # Return as HTML response
        return StreamingResponse(
            io.BytesIO(html_content.encode('utf-8')),
            media_type="text/html",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        logger.error(f"HTML export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/research-query/export-html")
async def export_research_query_html(
    query: str,
    research_intent: str = "cohort_definition",
    concept_type: str = "diagnosis",
    target_systems: List[str] = ["ICD-10-CM", "SNOMED CT"],
    search_mode: str = "hybrid"
):
    """Export research query results as beautiful HTML for researchers"""
    try:
        # Create dependencies
        deps = ResearchTerminologyDeps()
        
        # Execute research query
        result = await execute_medical_concept_search(
            query=query,
            research_intent=research_intent,
            concept_type=concept_type,
            target_systems=target_systems,
            search_mode=search_mode,
            deps=deps,
            medical_api_client=medical_api_client
        )
        
        # Parse the research output to extract results
        results = []
        
        # Extract code set results if available
        if "code_set" in result and result["code_set"]:
            code_set = result["code_set"]
            for system, codes in code_set.get("code_sets", {}).items():
                for code_info in codes:
                    results.append({
                        "code": code_info.get("code", ""),
                        "display_name": code_info.get("display_name", ""),
                        "system": system,
                        "confidence_score": code_info.get("confidence_score", 0.95),
                        "definition": code_info.get("rationale", ""),
                        "research_notes": code_info.get("usage_notes", ""),
                        "search_method": search_mode,
                        "character_pattern": {
                            "description": code_info.get("pattern_description", "")
                        }
                    })
        
        # Generate HTML
        html_content = html_export_service.export_results(
            results=results,
            query=query,
            research_context=f"{research_intent} - {concept_type}",
            research_intent=research_intent,
            target_systems=target_systems,
            search_mode=search_mode
        )
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_query_results_{timestamp}.html"
        
        # Return as HTML response
        return StreamingResponse(
            io.BytesIO(html_content.encode('utf-8')),
            media_type="text/html",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        logger.error(f"Research query HTML export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/debug/api-status")
async def debug_api_status():
    """Debug endpoint to test individual API connections"""
    status = {
        "timestamp": datetime.now().isoformat(),
        "working_directory": str(Path.cwd()),
        "env_file_path": str(ROOT_DIR / '.env'),
        "env_file_exists": (ROOT_DIR / '.env').exists(),
        "apis": {}
    }
    
    # Test OpenAI
    try:
        if settings.openai_api_key:
            test_client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
            # Simple test call
            response = test_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            status["apis"]["openai"] = {
                "status": "✅ Connected",
                "model": "gpt-4o",
                "test_response": response.choices[0].message.content[:20]
            }
        else:
            status["apis"]["openai"] = {"status": "❌ No API key"}
    except Exception as e:
        status["apis"]["openai"] = {"status": f"❌ Error: {str(e)}"}
    
    # Test UMLS
    try:
        if settings.umls_api_key:
            umls_results = await medical_api_client.search_umls("diabetes")
            status["apis"]["umls"] = {
                "status": "✅ Connected", 
                "test_results": len(umls_results),
                "sample": umls_results[0]["concept_name"] if umls_results else "No results"
            }
        else:
            status["apis"]["umls"] = {"status": "❌ No API key"}
    except Exception as e:
        status["apis"]["umls"] = {"status": f"❌ Error: {str(e)}"}
    
    # Test RxNorm (public API)
    try:
        rxnorm_results = await medical_api_client.search_rxnorm("metformin")
        status["apis"]["rxnorm"] = {
            "status": "✅ Connected",
            "test_results": len(rxnorm_results),
            "sample": rxnorm_results[0]["concept_name"] if rxnorm_results else "No results"
        }
    except Exception as e:
        status["apis"]["rxnorm"] = {"status": f"❌ Error: {str(e)}"}
    
    return status

@api_router.get("/stats")
async def get_system_stats():
    """Get enhanced system statistics including BioBERT"""
    total_concepts = await db.medical_concepts.count_documents({})
    total_chats = await db.chat_history.count_documents({})
    total_batch_jobs = await db.batch_jobs.count_documents({})
    total_hybrid_searches = await db.hybrid_searches.count_documents({})
    
    # Get ontology breakdown
    ontology_stats = {}
    for ontology in ["UMLS", "RxNorm", "ICD-10", "SNOMED CT", "LOINC"]:
        count = await db.medical_concepts.count_documents({"source_ontology": ontology})
        ontology_stats[ontology] = count
    
    # Get BioBERT stats
    biobert_stats = {}
    try:
        if biobert_rag_engine.is_initialized:
            biobert_stats = await biobert_rag_engine.get_collection_stats()
    except Exception as e:
        logger.warning(f"Could not get BioBERT stats: {e}")
    
    return {
        "total_concepts_searched": total_concepts,
        "total_chats": total_chats,
        "total_batch_jobs": total_batch_jobs,
        "total_hybrid_searches": total_hybrid_searches,
        "ontology_breakdown": ontology_stats,
        "biobert_stats": biobert_stats,
        "available_ontologies": ["UMLS", "RxNorm", "ICD-10", "SNOMED CT", "LOINC"],
        "supported_abbreviations": len(MEDICAL_ABBREVIATIONS),
        "features": [
            "abbreviation_expansion", 
            "semantic_search", 
            "cross_ontology_mapping", 
            "confidence_scoring", 
            "csv_export",
            "hybrid_api_biobert_search",
            "rag_validation",
            "concept_discovery"
        ],
        "system_status": "operational",
        "version": "2.1.0"
    }

@api_router.get("/debug-pipeline/{query}")
async def debug_complete_pipeline(query: str):
    """Comprehensive debug of the entire hybrid search pipeline"""
    try:
        debug_info = {
            "query": query,
            "pipeline_steps": []
        }
        
        # Step 1: Test API search directly
        try:
            api_query = MedicalQuery(query=query, ontologies=["umls"], confidence_threshold=0.5)
            api_results = await search_medical_concepts(api_query)
            debug_info["step_1_direct_api"] = {
                "success": True,
                "count": len(api_results),
                "sample": api_results[0].dict() if api_results else None
            }
        except Exception as e:
            debug_info["step_1_direct_api"] = {"success": False, "error": str(e)}
        
        # Step 2: Test RAG search directly
        try:
            rag_engine = await get_biobert_rag_engine()
            rag_results = await rag_engine.search(query, limit=5)
            debug_info["step_2_direct_rag"] = {
                "success": True,
                "count": len(rag_results),
                "sample": rag_results[0] if rag_results else None
            }
        except Exception as e:
            debug_info["step_2_direct_rag"] = {"success": False, "error": str(e)}
        
        # Step 3: Test API function wrapper
        try:
            async def test_api_wrapper(query_dict):
                api_query = MedicalQuery(
                    query=query_dict.get('query', ''),
                    ontologies=query_dict.get('ontologies', ['umls']),
                    confidence_threshold=query_dict.get('confidence_threshold', 0.5)
                )
                results = await search_medical_concepts(api_query)
                return [result.dict() if hasattr(result, 'dict') else result for result in results]
            
            wrapped_results = await test_api_wrapper({"query": query, "ontologies": ["umls"]})
            debug_info["step_3_api_wrapper"] = {
                "success": True,
                "count": len(wrapped_results),
                "sample": wrapped_results[0] if wrapped_results else None
            }
        except Exception as e:
            debug_info["step_3_api_wrapper"] = {"success": False, "error": str(e)}
        
        # Step 4: Test hybrid search engine directly
        try:
            result = await hybrid_search_engine.hybrid_search(
                query=query,
                ontologies=["umls"],
                api_search_func=test_api_wrapper,
                confidence_threshold=0.5,
                rag_limit=5
            )
            debug_info["step_4_hybrid_engine"] = {
                "success": True,
                "api_count": len(result.api_results),
                "rag_count": len(result.rag_results),
                "validated_count": len(result.validated_results),
                "discovery_count": len(result.discovery_results),
                "confidence": result.hybrid_confidence
            }
        except Exception as e:
            debug_info["step_4_hybrid_engine"] = {"success": False, "error": str(e)}
        
        return debug_info
        
    except Exception as e:
        import traceback
        return {
            "query": query,
            "pipeline_error": str(e),
            "traceback": traceback.format_exc()
        }

@api_router.get("/test-rag/{query}")
async def test_rag_direct(query: str):
    """Direct test of RAG functionality for debugging"""
    try:
        # Test both direct RAG and hybrid search engine
        rag_engine = await get_biobert_rag_engine()
        rag_results = await rag_engine.search(query, limit=5)
        
        # Test hybrid search engine's RAG access
        hybrid_rag_results = await hybrid_search_engine.biobert_engine.search(query, limit=5)
        
        return {
            "query": query,
            "direct_rag": {
                "initialized": rag_engine.is_initialized,
                "biobert_loaded": rag_engine.biobert_model is not None,
                "pinecone_connected": rag_engine.pinecone_index is not None,
                "results_count": len(rag_results),
                "results": rag_results[:2]
            },
            "hybrid_rag": {
                "initialized": hybrid_search_engine.biobert_engine.is_initialized if hybrid_search_engine.biobert_engine else False,
                "same_instance": hybrid_search_engine.biobert_engine is rag_engine,
                "results_count": len(hybrid_rag_results),
                "results": hybrid_rag_results[:2]
            }
        }
        
    except Exception as e:
        import traceback
        return {
            "query": query,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@api_router.get("/abbreviations")
async def get_medical_abbreviations():
    """Get list of supported medical abbreviations"""
    return {
        "total_abbreviations": len(MEDICAL_ABBREVIATIONS),
        "abbreviations": MEDICAL_ABBREVIATIONS,
        "categories": {
            "cardiovascular": ["HTN", "MI", "CHF", "CAD", "PAD", "AF"],
            "endocrine": ["DM", "T1DM", "T2DM", "DKA", "HHS"],
            "respiratory": ["COPD", "ARDS", "PE", "PNA", "SOB"],
            "neurology": ["CVA", "TIA", "SAH", "ICH", "MS", "PD"],
            "infectious": ["UTI", "HAI", "MRSA", "VRE"],
            "critical_care": ["SIRS", "MODS", "AKI", "CKD", "ARF"]
        }
    }

@api_router.get("/cache/stats")
async def get_cache_stats():
    """Get Redis cache statistics"""
    stats = await cache_service.get_stats()
    return {
        "cache_stats": stats,
        "cache_enabled": stats.get("status") == "connected"
    }

@api_router.post("/cache/clear")
async def clear_cache(pattern: str = "*"):
    """Clear cache entries matching pattern"""
    if pattern == "*":
        # Clear all cache
        cleared = await cache_service.clear_pattern("*")
    else:
        cleared = await cache_service.clear_pattern(pattern)
    
    return {
        "message": f"Cleared {cleared} cache entries",
        "pattern": pattern
    }

@api_router.get("/history")
async def get_chat_history(session_id: Optional[str] = None):
    """Get enhanced chat history"""
    query = {"session_id": session_id} if session_id else {}
    history = await db.chat_history.find(query).sort("timestamp", -1).limit(50).to_list(50)
    return history

# =================== CSV PROCESSING ENDPOINTS ===================

@api_router.post("/csv/upload", response_model=CSVUploadResponse)
async def upload_csv_file(file: UploadFile = File(...)):
    """Upload CSV file for medical concept mapping analysis"""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        
        # Save uploaded file
        file_path = UPLOADS_DIR / f"{file_id}_{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Analyze CSV structure
        analysis = await csv_analyzer.analyze_csv_structure(str(file_path), file.filename)
        
        # Store file metadata
        uploaded_files[file_id] = {
            'filename': file.filename,
            'file_path': str(file_path),
            'upload_time': datetime.now(),
            'analysis': analysis
        }
        
        return CSVUploadResponse(
            file_id=file_id,
            filename=file.filename,
            analysis=analysis,
            suggested_mappings=analysis.get('mapping_suggestions', {})
        )
        
    except Exception as e:
        logger.error(f"CSV upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@api_router.post("/csv/process", response_model=CSVProcessingResponse)
async def process_csv_mapping(file_id: str, mapping_config: CSVMappingConfig):
    """Process CSV file with medical concept mapping"""
    try:
        # Verify file exists
        if file_id not in uploaded_files:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_info = uploaded_files[file_id]
        file_path = file_info['file_path']
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Start background processing
        asyncio.create_task(
            batch_mapper.process_csv_batch(
                file_path=file_path,
                mapping_config=mapping_config.dict(),
                job_id=job_id
            )
        )
        
        return CSVProcessingResponse(
            job_id=job_id,
            status="processing",
            message="CSV processing started. Use the job ID to check progress."
        )
        
    except Exception as e:
        logger.error(f"CSV processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@api_router.get("/csv/status/{job_id}", response_model=JobStatus)
async def get_processing_status(job_id: str):
    """Get the status of a CSV processing job"""
    try:
        status = batch_mapper.get_job_status(job_id)
        
        if status.get('status') == 'not_found':
            raise HTTPException(status_code=404, detail="Job not found")
        
        return JobStatus(
            job_id=job_id,
            status=status.get('status', 'unknown'),
            progress=status.get('progress', 0),
            current_operation=status.get('current_operation', ''),
            start_time=status.get('start_time'),
            end_time=status.get('end_time'),
            error=status.get('error')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@api_router.get("/csv/download/{job_id}")
async def download_processed_csv(job_id: str):
    """Download the processed CSV file with mappings"""
    try:
        # Get job status
        status = batch_mapper.get_job_status(job_id)
        
        if status.get('status') != 'completed':
            raise HTTPException(status_code=400, detail="Job not completed yet")
        
        output_file = status.get('output_file')
        if not output_file or not Path(output_file).exists():
            raise HTTPException(status_code=404, detail="Output file not found")
        
        # Create file response
        def file_generator():
            with open(output_file, 'rb') as f:
                while chunk := f.read(8192):
                    yield chunk
        
        filename = Path(output_file).name
        
        return StreamingResponse(
            file_generator(),
            media_type='text/csv',
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@api_router.get("/csv/report/{job_id}")
async def get_mapping_report(job_id: str):
    """Get comprehensive mapping report for a completed job"""
    try:
        # Get job status
        status = batch_mapper.get_job_status(job_id)
        
        if status.get('status') != 'completed':
            raise HTTPException(status_code=400, detail="Job not completed yet")
        
        summary = status.get('summary', {})
        
        return {
            'job_id': job_id,
            'status': status,
            'mapping_summary': summary,
            'recommendations': summary.get('recommendations', []),
            'quality_assessment': {
                'success_rate': summary.get('success_rate', 0),
                'average_confidence': summary.get('average_confidence', 0),
                'total_concepts': summary.get('total_concepts_processed', 0)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@api_router.delete("/csv/cleanup")
async def cleanup_old_files(max_age_hours: int = 24):
    """Clean up old uploaded files and processing jobs"""
    try:
        current_time = datetime.now()
        files_removed = 0
        
        # Clean up old uploaded files
        files_to_remove = []
        for file_id, file_info in uploaded_files.items():
            upload_time = file_info.get('upload_time', current_time)
            if (current_time - upload_time).total_seconds() > max_age_hours * 3600:
                files_to_remove.append(file_id)
                
                # Remove physical file
                file_path = Path(file_info['file_path'])
                if file_path.exists():
                    file_path.unlink()
                    files_removed += 1
        
        # Remove from memory
        for file_id in files_to_remove:
            del uploaded_files[file_id]
        
        # Clean up old jobs
        batch_mapper.cleanup_old_jobs(max_age_hours)
        
        return {
            'message': f'Cleanup completed. Removed {files_removed} files.',
            'files_removed': files_removed,
            'max_age_hours': max_age_hours
        }
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

# NEW: Advanced Research Workflow Endpoint
@api_router.post("/research-workflow")
async def execute_research_workflow(
    query: str,
    user_intent: str = "comprehensive",
    concept_type: str = None
):
    """
    Execute the streamlined Pydantic AI + LangGraph research workflow for advanced medical concept analysis
    """
    try:
        logger.info(f"🧠 Research Workflow: Starting analysis for '{query}'")
        
        # Create dependencies
        deps = ResearchTerminologyDeps()
        deps.medical_client = medical_api_client
        
        # Execute the streamlined research workflow
        result = await execute_medical_concept_search(
            query=query,
            user_intent=user_intent,
            concept_type=concept_type,
            deps=deps,
            medical_client=medical_api_client,
            hybrid_engine=hybrid_search_engine
        )
        
        logger.info(f"🧠 Research Workflow: Completed with {result.get('total_concepts_found', 0)} concepts found")
        
        return {
            "success": result.get("success", False),
            "query": result.get("query"),
            "search_strategy": result.get("search_strategy"),
            "results": {
                "api_results": result.get("api_results", []),
                "rag_results": result.get("rag_results", []),
                "total_found": result.get("total_concepts_found", 0)
            },
            "analysis": {
                "synthesis_notes": result.get("synthesis_notes", ""),
                "coverage_analysis": result.get("coverage_analysis", ""),
                "quality_score": result.get("quality_score", 0.0)
            },
            "recommendations": result.get("recommended_actions", []),
            "formatted_output": result.get("formatted_output", ""),
            "metadata": result.get("metadata", {}),
            "errors": result.get("errors", [])
        }
        
    except Exception as e:
        logger.error(f"🧠 Research Workflow failed: {e}")
        import traceback
        logger.error(f"🧠 Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Research workflow failed: {str(e)}")

# Include the router in the main app
app.include_router(api_router)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for React frontend
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logger.info(f"Mounted static files from {static_dir}")

# Serve React app
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the React frontend index.html"""
    static_dir = Path(__file__).parent / "static"
    index_file = static_dir / "index.html"
    
    if index_file.exists():
        return FileResponse(index_file)
    else:
        return HTMLResponse("""
        <html>
            <head><title>Medical Coding Intelligence Platform</title></head>
            <body>
                <h1>Medical Coding Intelligence Platform</h1>
                <p>Backend is running. Frontend build not found.</p>
                <p>API endpoints available at <a href="/docs">/docs</a></p>
            </body>
        </html>
        """)

# Catch-all route to serve React app for client-side routing
@app.get("/{path:path}")
async def serve_frontend_routes(path: str):
    """Serve React app for all non-API routes (client-side routing)"""
    # Skip API routes
    if path.startswith("api/") or path.startswith("docs") or path.startswith("openapi.json"):
        raise HTTPException(status_code=404, detail="Not found")
    
    static_dir = Path(__file__).parent / "static"
    
    # Try to serve the specific file first
    requested_file = static_dir / path
    if requested_file.exists() and requested_file.is_file():
        return FileResponse(requested_file)
    
    # Otherwise serve index.html for React routing
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    else:
        raise HTTPException(status_code=404, detail="Frontend not found")

# Initialize hybrid search engine on startup
@app.on_event("startup")
async def startup_event():
    try:
        logger.info("Initializing Medical Research Intelligence Platform v2.1")
        
        # Initialize cache service
        await cache_service.initialize()
        logger.info("Redis cache service initialized successfully")
        
        # Initialize hybrid search engine
        await hybrid_search_engine.initialize()
        logger.info("Hybrid search engine initialized successfully")
    except Exception as e:
        logger.warning(f"Could not initialize services: {e}")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
    await cache_service.close()