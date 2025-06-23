"""
Simple Pinecone Integration for Medical Code Discovery
Focus: User types natural language, gets relevant medical codes
"""

import logging
from typing import List, Dict, Optional
from pinecone import Pinecone
import numpy as np
from sentence_transformers import SentenceTransformer
import os

logger = logging.getLogger(__name__)

class SimplePineconeSearch:
    """Simple, user-focused Pinecone search for medical codes"""
    
    def __init__(self):
        self.pc = None
        self.index = None
        self.embedding_model = None
        self.is_ready = False
        
    def initialize(self):
        """Initialize Pinecone connection"""
        try:
            # Get API key
            api_key = os.environ.get('PINECONE_API_KEY')
            if not api_key:
                logger.error("PINECONE_API_KEY not found")
                return False
            
            # Initialize Pinecone
            self.pc = Pinecone(api_key=api_key)
            self.index = self.pc.Index('biobert')
            
            # Initialize embedding model for queries (768D to match your index)
            self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            
            # Test connection
            stats = self.index.describe_index_stats()
            logger.info(f"✅ Pinecone connected: {stats['total_vector_count']} vectors")
            
            self.is_ready = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Pinecone initialization failed: {e}")
            return False
    
    def semantic_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        User-friendly semantic search
        Input: Natural language query like "chest pain in elderly"
        Output: Relevant medical codes with explanations
        """
        if not self.is_ready:
            logger.warning("Pinecone not ready")
            return []
        
        try:
            # Convert text to embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Search Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                include_values=False
            )
            
            # Format for users
            formatted_results = []
            for match in results.matches:
                metadata = match.metadata or {}
                
                # Extract useful information
                result = {
                    'concept_name': metadata.get('concept_name', 'Unknown'),
                    'concept_id': metadata.get('concept_id', 'N/A'),
                    'source_ontology': metadata.get('source_ontology', 'Unknown'),
                    'definition': metadata.get('definition', ''),
                    'semantic_types': metadata.get('semantic_types', []),
                    'similarity_score': round(match.score, 3),
                    'match_strength': self._get_match_strength(match.score)
                }
                formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} results for: '{query}'")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def _get_match_strength(self, score: float) -> str:
        """Convert score to user-friendly strength indicator"""
        if score >= 0.9:
            return "Excellent Match"
        elif score >= 0.8:
            return "Very Good Match"
        elif score >= 0.7:
            return "Good Match"
        elif score >= 0.6:
            return "Fair Match"
        else:
            return "Weak Match"
    
    def search_with_context(self, query: str, ontology_filter: Optional[str] = None) -> Dict:
        """
        Enhanced search with context for LLM integration
        """
        # Get semantic results
        results = self.semantic_search(query, top_k=15)
        
        # Filter by ontology if specified
        if ontology_filter:
            results = [r for r in results if r['source_ontology'].lower() == ontology_filter.lower()]
        
        # Group by ontology
        grouped_results = {}
        for result in results:
            ontology = result['source_ontology']
            if ontology not in grouped_results:
                grouped_results[ontology] = []
            grouped_results[ontology].append(result)
        
        # Prepare context for LLM
        context_summary = {
            'query': query,
            'total_matches': len(results),
            'ontologies_found': list(grouped_results.keys()),
            'best_matches': results[:5],  # Top 5 for LLM context
            'grouped_by_ontology': grouped_results
        }
        
        return context_summary
    
    def get_status(self) -> Dict:
        """Get simple status for frontend"""
        if not self.is_ready:
            return {'status': 'disconnected', 'message': 'Pinecone not connected'}
        
        try:
            stats = self.index.describe_index_stats()
            return {
                'status': 'connected',
                'total_vectors': stats['total_vector_count'],
                'dimension': stats['dimension'],
                'message': 'Pinecone ready for semantic search'
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

# Global instance
pinecone_search = SimplePineconeSearch()