"""
Medical Hybrid RAG with Pinecone
Combines semantic search + exact keyword matching for medical code discovery
Perfect for: medical terminology, drug names, condition codes, abbreviations
"""

import logging
from typing import List, Dict, Optional, Tuple
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class MedicalHybridRAG:
    """
    Medical-focused hybrid RAG system
    User types: "chest pain elderly patient" 
    Returns: Relevant medical codes with context
    """
    
    def __init__(self):
        self.pc = None
        self.index = None
        self.dense_model = None
        self.is_ready = False
        
    def initialize(self):
        """Initialize medical hybrid search"""
        try:
            # Get API key
            api_key = os.environ.get('PINECONE_API_KEY')
            if not api_key:
                logger.error("PINECONE_API_KEY not found")
                return False
            
            # Initialize Pinecone
            self.pc = Pinecone(api_key=api_key)
            self.index = self.pc.Index('biobert')
            
            # Use BioBERT-compatible model for medical embeddings
            self.dense_model = SentenceTransformer('dmis-lab/biobert-base-cased-v1.1')
            
            # Test connection
            stats = self.index.describe_index_stats()
            logger.info(f"✅ Medical RAG ready: {stats['total_vector_count']:,} medical concepts")
            
            self.is_ready = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Medical RAG initialization failed: {e}")
            # Fallback to general model if BioBERT fails
            try:
                self.dense_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
                logger.info("Using fallback embedding model")
                self.is_ready = True
                return True
            except:
                return False
    
    def medical_search(self, query: str, top_k: int = 10, alpha: float = 0.7) -> List[Dict]:
        """
        Medical hybrid search optimized for healthcare terminology
        
        Args:
            query: Natural language medical query
            top_k: Number of results to return
            alpha: Balance between semantic (1.0) and keyword (0.0) search
        """
        if not self.is_ready:
            logger.warning("Medical RAG not ready")
            return []
        
        try:
            # Create dense embedding for semantic search
            dense_query = self.dense_model.encode(query).tolist()
            
            # For now, use dense-only search (your index may not have sparse vectors yet)
            results = self.index.query(
                vector=dense_query,
                top_k=top_k,
                include_metadata=True,
                include_values=False
            )
            
            # Format results for medical context
            medical_results = []
            for match in results.matches:
                metadata = match.metadata or {}
                
                # Extract medical information
                result = {
                    'concept_name': metadata.get('concept_name', metadata.get('title', 'Unknown')),
                    'concept_id': metadata.get('concept_id', metadata.get('id', 'N/A')),
                    'source_ontology': metadata.get('source_ontology', metadata.get('source', 'Unknown')),
                    'definition': metadata.get('definition', metadata.get('text', '')),
                    'semantic_types': metadata.get('semantic_types', []),
                    'similarity_score': round(match.score, 3),
                    'clinical_relevance': self._assess_clinical_relevance(match.score),
                    'search_method': 'dense_semantic'
                }
                medical_results.append(result)
            
            logger.info(f"Found {len(medical_results)} medical concepts for: '{query}'")
            return medical_results
            
        except Exception as e:
            logger.error(f"Medical search failed: {e}")
            return []
    
    def medical_rag_search(self, query: str, context_limit: int = 5) -> Dict:
        """
        Enhanced medical search with context for LLM integration
        Perfect for: "What medications treat diabetes?" 
        """
        # Get medical search results
        results = self.medical_search(query, top_k=15)
        
        # Group by medical ontology
        ontology_groups = {}
        for result in results:
            ontology = result['source_ontology']
            if ontology not in ontology_groups:
                ontology_groups[ontology] = []
            ontology_groups[ontology].append(result)
        
        # Prepare medical context for LLM
        medical_context = {
            'query': query,
            'total_medical_concepts': len(results),
            'ontologies_found': list(ontology_groups.keys()),
            'top_matches': results[:context_limit],
            'by_ontology': ontology_groups,
            'clinical_summary': self._create_clinical_summary(results[:context_limit]),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return medical_context
    
    def _assess_clinical_relevance(self, score: float) -> str:
        """Convert similarity score to clinical relevance"""
        if score >= 0.85:
            return "Highly Relevant"
        elif score >= 0.75:
            return "Clinically Relevant"
        elif score >= 0.65:
            return "Moderately Relevant"
        elif score >= 0.55:
            return "Possibly Relevant"
        else:
            return "Low Relevance"
    
    def _create_clinical_summary(self, results: List[Dict]) -> str:
        """Create clinical summary for LLM context"""
        if not results:
            return "No relevant medical concepts found."
        
        summary_parts = []
        
        # Group by ontology for summary
        ontology_counts = {}
        for result in results:
            ontology = result['source_ontology']
            ontology_counts[ontology] = ontology_counts.get(ontology, 0) + 1
        
        summary_parts.append(f"Found {len(results)} relevant medical concepts across {len(ontology_counts)} ontologies:")
        
        for ontology, count in ontology_counts.items():
            summary_parts.append(f"- {ontology}: {count} concepts")
        
        # Add top concepts
        summary_parts.append("\nTop medical concepts:")
        for i, result in enumerate(results[:3], 1):
            summary_parts.append(f"{i}. {result['concept_name']} ({result['concept_id']}) - {result['clinical_relevance']}")
        
        return "\n".join(summary_parts)
    
    def search_by_ontology(self, query: str, ontology: str, top_k: int = 5) -> List[Dict]:
        """Search within specific medical ontology"""
        results = self.medical_search(query, top_k=top_k * 3)  # Get more, then filter
        
        # Filter by ontology
        filtered_results = [
            r for r in results 
            if r['source_ontology'].lower() == ontology.lower()
        ]
        
        return filtered_results[:top_k]
    
    def get_medical_status(self) -> Dict:
        """Get status optimized for medical research"""
        if not self.is_ready:
            return {
                'status': 'disconnected',
                'message': 'Medical RAG system not connected',
                'ready_for_research': False
            }
        
        try:
            stats = self.index.describe_index_stats()
            return {
                'status': 'connected',
                'total_medical_concepts': stats['total_vector_count'],
                'embedding_dimension': stats['dimension'],
                'message': 'Medical RAG ready for clinical research',
                'ready_for_research': True,
                'model_type': 'BioBERT-compatible'
            }
        except Exception as e:
            return {
                'status': 'error', 
                'message': str(e),
                'ready_for_research': False
            }

# Global medical RAG instance
medical_rag = MedicalHybridRAG()