# BioBERT RAG Integration Module with Pinecone Support
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import logging
import os
from pathlib import Path
import pickle
import asyncio

# Set up logger first
logger = logging.getLogger(__name__)

from .medical_query_processor import medical_query_processor, ProcessedQuery

# Import embedding models
try:
    from biobert_embedding.embedding import BiobertEmbedding
    BIOBERT_EMBEDDING_AVAILABLE = True
except ImportError:
    BIOBERT_EMBEDDING_AVAILABLE = False
    # Try our custom implementation
    try:
        from .biobert_original import BiobertEmbedding
        BIOBERT_EMBEDDING_AVAILABLE = True
        logger.info("Using custom BioBERT implementation")
    except ImportError:
        logger.warning("BioBERT embedding not available")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
import asyncio

# Pinecone imports
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logging.warning("Pinecone not installed. RAG functionality will be limited.")

logger = logging.getLogger(__name__)

class BioBERTRAGEngine:
    """
    BioBERT-powered RAG engine with Pinecone vector database
    """
    
    def __init__(self, 
                 pinecone_api_key: Optional[str] = None,
                 pinecone_index_name: Optional[str] = None,
                 pinecone_environment: Optional[str] = None):
        
        self.pinecone_api_key = pinecone_api_key or os.environ.get('PINECONE_API_KEY')
        self.pinecone_index_name = pinecone_index_name or os.environ.get('PINECONE_INDEX_NAME', 'medical-biobert')
        self.pinecone_environment = pinecone_environment or os.environ.get('PINECONE_ENVIRONMENT')
        
        # Pinecone components
        self.pinecone_client = None
        self.pinecone_index = None
        
        # Embedding models
        self.embedding_model = None  # Sentence transformer
        self.biobert_model = None    # Original BioBERT
        self.biobert_tokenizer = None
        self.embedding_cache = {}     # Cache embeddings
        self.embedding_method = 'auto'  # 'biobert', 'sentence_transformer', or 'auto'
        
        # State
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize Pinecone vector database"""
        try:
            await self._initialize_pinecone()
            
            # Try to detect which embedding model to use based on Pinecone metadata
            await self._detect_embedding_model()
            
            # Priority 1: Use original BioBERT embeddings (most likely to match Pinecone)
            if BIOBERT_EMBEDDING_AVAILABLE:
                try:
                    logger.info("🧬 Loading original BioBERT embeddings (best match for Pinecone)...")
                    self.biobert_original = BiobertEmbedding()
                    self.embedding_method = 'biobert_original'
                    logger.info("✅ Loaded original BioBERT embeddings - this should match Pinecone vectors!")
                    
                    # Test embedding generation
                    test_embedding = self.biobert_original.sentence_vector("test")
                    logger.info(f"✅ BioBERT embedding test successful, dimension: {len(test_embedding)}")
                    
                except Exception as e:
                    logger.warning(f"Could not load original BioBERT embeddings: {e}")
                    self.embedding_method = 'auto'
            
            # Fallback options if original BioBERT not available
            if self.embedding_method != 'biobert_original':
                # Try transformers BioBERT
                if self.embedding_method == 'biobert' and TRANSFORMERS_AVAILABLE:
                    try:
                        logger.info("Loading BioBERT model via transformers...")
                        model_name = 'dmis-lab/biobert-v1.1'
                        self.biobert_tokenizer = AutoTokenizer.from_pretrained(model_name)
                        self.biobert_model = AutoModel.from_pretrained(model_name)
                        self.biobert_model.eval()
                        logger.info("✅ Loaded BioBERT via transformers")
                    except Exception as e:
                        logger.warning(f"Could not load BioBERT model: {e}")
                        self.embedding_method = 'sentence_transformer'
                
                # Final fallback to sentence transformers
                if self.embedding_method != 'biobert' and SENTENCE_TRANSFORMERS_AVAILABLE:
                    try:
                        self.embedding_model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
                        logger.info("✅ Loaded PubMedBERT sentence transformer (fallback)")
                    except Exception as e:
                        logger.warning(f"Could not load any embedding model: {e}")
            
            self.is_initialized = True
            logger.info("BioBERT RAG engine initialized successfully using Pinecone")
            
        except Exception as e:
            logger.error(f"Failed to initialize BioBERT RAG engine: {e}")
            raise
    
    async def _initialize_pinecone(self):
        """Initialize Pinecone vector database"""
        if not PINECONE_AVAILABLE:
            raise Exception("Pinecone not available")
        
        if not self.pinecone_api_key:
            raise Exception("Pinecone API key not provided")
        
        try:
            # Initialize Pinecone client with new API
            self.pinecone_client = Pinecone(api_key=self.pinecone_api_key)
            
            # Connect to existing index
            self.pinecone_index = self.pinecone_client.Index(self.pinecone_index_name)
            
            logger.info(f"Connected to Pinecone index: {self.pinecone_index_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise
    
    async def _detect_embedding_model(self):
        """Detect which embedding model was used for Pinecone vectors"""
        try:
            # Sample a few vectors to check dimensionality and metadata
            stats = await self._get_pinecone_stats()
            dimension = stats.get('dimension', 768)
            
            if dimension == 768:
                # Both BioBERT and many sentence transformers use 768
                # Try to detect from index name or metadata
                if 'biobert' in self.pinecone_index_name.lower():
                    self.embedding_method = 'biobert'
                    logger.info("Detected BioBERT embeddings in Pinecone")
                else:
                    self.embedding_method = 'sentence_transformer'
                    logger.info("Detected sentence transformer embeddings")
            else:
                self.embedding_method = 'sentence_transformer'
                logger.info(f"Non-standard dimension {dimension}, using sentence transformers")
                
        except Exception as e:
            logger.warning(f"Could not detect embedding model: {e}")
            self.embedding_method = 'auto'
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using appropriate model"""
        # Check cache first
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        embedding = None
        
        # Priority 1: Use original BioBERT embeddings
        if self.embedding_method == 'biobert_original' and self.biobert_original:
            try:
                # Get sentence embedding from original BioBERT
                embedding = np.array(self.biobert_original.sentence_vector(text))
                logger.debug(f"Generated embedding with original BioBERT for: {text[:50]}...")
                    
            except Exception as e:
                logger.error(f"Original BioBERT embedding failed: {e}")
                embedding = None
        
        # Fallback 1: Use transformers BioBERT
        elif self.embedding_method == 'biobert' and self.biobert_model and self.biobert_tokenizer:
            try:
                # Tokenize and encode with BioBERT
                inputs = self.biobert_tokenizer(text, return_tensors='pt', 
                                               truncation=True, max_length=512, 
                                               padding=True)
                
                with torch.no_grad():
                    outputs = self.biobert_model(**inputs)
                    # Use CLS token embedding
                    embedding = outputs.last_hidden_state[:, 0, :].numpy()[0]
                    
            except Exception as e:
                logger.error(f"BioBERT embedding failed: {e}")
                embedding = None
        
        # Fallback 2: Use sentence transformer
        if embedding is None and self.embedding_model:
            try:
                embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            except Exception as e:
                logger.error(f"Sentence transformer failed: {e}")
                embedding = None
        
        # Ultimate fallback
        if embedding is None:
            logger.warning("Using fallback embedding generation")
            text_hash = hash(text.lower())
            np.random.seed(abs(text_hash) % (2**32))
            embedding = np.random.randn(768)
        
        # Ensure numpy array and normalize
        embedding = np.array(embedding)
        embedding = embedding / np.linalg.norm(embedding)
        
        # Cache it
        self.embedding_cache[text] = embedding
        return embedding
    
    def get_biobert_embedding(self, text: str) -> np.ndarray:
        """Alias for backward compatibility"""
        return self.get_embedding(text)
    
    async def semantic_search(self, query: str, top_k: int = 10, score_threshold: float = 0.5) -> List[Dict]:
        """Perform semantic search using Pinecone's inference API"""
        if not self.is_initialized:
            return []
        
        try:
            # Use Pinecone's new semantic search API
            results = self.pinecone_index.search(
                query={
                    "top_k": top_k,
                    "inputs": {
                        'text': query
                    }
                },
                search_params={
                    "alpha": 0.5,
                    "include_metadata": True
                }
            )
            
            # Process results
            search_results = []
            for match in results.get('matches', []):
                if match.get('score', 0) >= score_threshold:
                    search_results.append({
                        'concept': match.get('metadata', {}).get('concept', ''),
                        'definition': match.get('metadata', {}).get('definition', ''),
                        'ontology': match.get('metadata', {}).get('ontology', ''),
                        'score': match.get('score', 0),
                        'search_method': 'biobert_rag_pinecone'
                    })
            
            return search_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    async def search(self, query: str, limit: int = 10, ontology_filter: Optional[str] = None) -> List[Dict]:
        """Enhanced search using BioBERT embeddings with medical query processing"""
        if not self.is_initialized:
            logger.warning("BioBERT RAG engine not initialized")
            return []
        
        try:
            # Process the medical query
            processed_query = medical_query_processor.process_query(query)
            logger.info(f"🔍 Processed query: {processed_query.normalized_query}")
            if processed_query.abbreviations_expanded:
                logger.info(f"📝 Expanded abbreviations: {processed_query.abbreviations_expanded}")
            if len(processed_query.expanded_terms) > 1:
                logger.info(f"🔗 Related terms: {processed_query.expanded_terms[:3]}")
            
            # Generate multiple search queries
            search_queries = medical_query_processor.generate_search_queries(processed_query)
            
            # Perform multi-query search
            all_results = []
            seen_ids = set()
            
            for i, search_query in enumerate(search_queries[:3]):  # Limit to top 3 queries
                logger.info(f"🔎 Searching with query {i+1}: {search_query}")
                
                # Search with each query
                results = await self._search_pinecone(
                    search_query, 
                    limit=limit * 2,  # Get more results per query
                    ontology_filter=ontology_filter
                )
                
                # Deduplicate and add results
                for result in results:
                    result_id = result.get('concept_id', '')
                    if result_id and result_id not in seen_ids:
                        seen_ids.add(result_id)
                        # Boost score for primary query matches
                        if i == 0:
                            result['confidence_score'] = result.get('confidence_score', 0) * 1.2
                        all_results.append(result)
            
            # Re-rank results based on medical relevance
            ranked_results = self._rerank_medical_results(
                all_results, 
                processed_query, 
                limit
            )
            
            return ranked_results
            
        except Exception as e:
            logger.error(f"Enhanced search failed: {e}")
            # Fallback to simple search
            return await self._search_pinecone(query, limit, ontology_filter)
    
    async def _search_pinecone(self, query: str, limit: int, ontology_filter: Optional[str] = None) -> List[Dict]:
        """Search Pinecone"""
        if not self.pinecone_index:
            logger.error("Pinecone index not initialized")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.get_embedding(query)
            
            # Prepare filter if ontology is specified
            filter_dict = {}
            if ontology_filter:
                filter_dict["ontology"] = {"$eq": ontology_filter}
            
            # Query Pinecone
            results = self.pinecone_index.query(
                vector=query_embedding.tolist(),
                top_k=limit,
                include_metadata=True,
                filter=filter_dict if filter_dict else None
            )
            
            # Process results with standardized field names
            search_results = []
            for match in results.get('matches', []):
                metadata = match.get('metadata', {})
                
                # Standardize field names from various sources
                concept_name = (metadata.get('concept', '') or 
                               metadata.get('concept_name', '') or
                               metadata.get('description', '') or 
                               metadata.get('text', '') or
                               metadata.get('name', ''))
                
                concept_id = (metadata.get('concept_id', '') or
                             metadata.get('code', '') or
                             metadata.get('id', '') or
                             match.get('id', ''))
                
                definition = (metadata.get('definition', '') or 
                             metadata.get('description', '') or
                             metadata.get('text', '') or
                             metadata.get('summary', ''))
                
                ontology = (metadata.get('ontology', '') or 
                           metadata.get('source', '') or
                           metadata.get('code_system', '') or
                           metadata.get('source_ontology', '') or
                           'PINECONE')
                
                # Create standardized result
                search_results.append({
                    'concept_name': concept_name,  # Standardized field name
                    'concept_id': concept_id,
                    'definition': definition,
                    'source_ontology': ontology,  # Changed from 'ontology' to match MedicalConcept model
                    'confidence_score': match.get('score', 0),  # Standardized score field
                    'search_method': 'biobert_rag_pinecone',
                    'is_rag_result': True,
                    'embedding_model': 'sentence_transformer',
                    'metadata': metadata,
                    # Keep original fields for backward compatibility
                    'concept': concept_name,
                    'ontology': ontology,  # Keep for backward compatibility
                    'score': match.get('score', 0)
                })
            
            return search_results
            
        except Exception as e:
            logger.error(f"Pinecone search failed: {e}")
            return []
    
    def _rerank_medical_results(self, results: List[Dict], processed_query: ProcessedQuery, limit: int) -> List[Dict]:
        """Re-rank results based on medical relevance"""
        # Score adjustments based on medical relevance
        for result in results:
            score = result.get('confidence_score', 0.5)
            concept_name = result.get('concept_name', '').lower()
            ontology = result.get('ontology', '').lower()
            
            # Boost exact matches
            if processed_query.normalized_query.lower() in concept_name:
                score *= 1.5
            
            # Boost matches with expanded terms
            for term in processed_query.expanded_terms:
                if term.lower() in concept_name:
                    score *= 1.2
                    break
            
            # Boost based on concept type match
            if processed_query.concept_type:
                if processed_query.concept_type == 'disease' and any(term in concept_name for term in ['disease', 'disorder', 'syndrome']):
                    score *= 1.1
                elif processed_query.concept_type == 'medication' and any(term in ontology for term in ['rxnorm', 'drug']):
                    score *= 1.1
                elif processed_query.concept_type == 'procedure' and any(term in concept_name for term in ['procedure', 'surgery', 'therapy']):
                    score *= 1.1
            
            # Penalize unlikely matches
            if processed_query.concept_type == 'disease' and 'procedure' in concept_name:
                score *= 0.7
            elif processed_query.concept_type == 'medication' and 'disease' in concept_name:
                score *= 0.7
            
            # Update score
            result['confidence_score'] = min(score, 1.0)  # Cap at 1.0
            result['reranking_applied'] = True
        
        # Sort by new scores
        ranked_results = sorted(results, key=lambda x: x.get('confidence_score', 0), reverse=True)
        
        return ranked_results[:limit]
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get vector database statistics"""
        if not self.is_initialized:
            return {"error": "Not initialized"}
        
        try:
            return await self._get_pinecone_stats()
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}
    
    async def _get_pinecone_stats(self) -> Dict[str, Any]:
        """Get Pinecone statistics"""
        try:
            stats = self.pinecone_index.describe_index_stats()
            return {
                'total_vectors': stats.get('total_vector_count', 0),
                'dimension': stats.get('dimension', 0),
                'index_name': self.pinecone_index_name,
                'database_type': 'pinecone'
            }
        except Exception as e:
            logger.error(f"Failed to get Pinecone stats: {e}")
            return {"error": str(e)}
    
    async def add_concepts(self, concepts: List[Dict[str, str]]) -> Dict[str, Any]:
        """Add medical concepts to Pinecone"""
        if not self.is_initialized:
            return {"error": "Not initialized"}
        
        try:
            vectors_to_upsert = []
            
            for i, concept in enumerate(concepts):
                # Generate embedding
                text_to_embed = f"{concept.get('concept', '')} {concept.get('definition', '')}"
                embedding = self.get_biobert_embedding(text_to_embed)
                
                # Prepare vector
                vector_id = f"concept_{i}_{concept.get('concept', '').replace(' ', '_')}"
                vectors_to_upsert.append({
                    "id": vector_id,
                    "values": embedding.tolist(),
                    "metadata": {
                        "concept": concept.get('concept', ''),
                        "definition": concept.get('definition', ''),
                        "ontology": concept.get('ontology', 'unknown')
                    }
                })
            
            # Upsert to Pinecone
            self.pinecone_index.upsert(vectors=vectors_to_upsert)
            
            return {
                "message": f"Successfully added {len(concepts)} concepts",
                "count": len(concepts)
            }
            
        except Exception as e:
            logger.error(f"Failed to add concepts: {e}")
            return {"error": str(e)}

# Factory function to create BioBERT RAG Engine
async def create_biobert_rag_engine(vector_db_type: str = "pinecone") -> BioBERTRAGEngine:
    """Factory function to create and initialize BioBERT RAG engine"""
    if not PINECONE_AVAILABLE:
        raise Exception("Pinecone not available. Please install pinecone-client.")
    
    engine = BioBERTRAGEngine()
    await engine.initialize()
    logger.info("BioBERT engine configured for Pinecone")
    return engine

# Global instance
_biobert_rag_engine = None

async def get_biobert_rag_engine() -> BioBERTRAGEngine:
    """Get or create global BioBERT RAG engine instance"""
    global _biobert_rag_engine
    
    if _biobert_rag_engine is None:
        _biobert_rag_engine = await create_biobert_rag_engine()
    
    return _biobert_rag_engine

def reset_biobert_rag_engine():
    """Reset the global BioBERT RAG engine instance"""
    global _biobert_rag_engine
    _biobert_rag_engine = None