# Hybrid Search Engine: API + BioBERT RAG
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from .biobert_rag import get_biobert_rag_engine
import json

logger = logging.getLogger(__name__)

@dataclass
class HybridSearchResult:
    """Structured result from hybrid search"""
    api_results: List[Dict[str, Any]]
    rag_results: List[Dict[str, Any]]
    validated_results: List[Dict[str, Any]]
    discovery_results: List[Dict[str, Any]]
    hybrid_confidence: float
    search_metadata: Dict[str, Any]

class HybridMedicalSearchEngine:
    """
    Advanced hybrid search combining API calls with BioBERT RAG for comprehensive medical concept discovery
    """
    
    def __init__(self):
        self.biobert_engine = None  # Will be initialized async
        self.similarity_threshold = 0.15   # Very low threshold to include more results
        self.validation_threshold = 0.1    # Minimal threshold for validation
        
    async def initialize(self):
        """Initialize the hybrid search engine"""
        try:
            self.biobert_engine = await get_biobert_rag_engine()
            logger.info("Hybrid Medical Search Engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize hybrid search engine: {e}")
            self.biobert_engine = None
    
    async def hybrid_search(
        self,
        query: str,
        ontologies: List[str],
        api_search_func,  # Function to call existing API search
        expand_abbreviations: bool = True,
        semantic_search: bool = True,
        confidence_threshold: float = 0.5,
        rag_limit: int = 20
    ) -> HybridSearchResult:
        """
        Perform hybrid search combining API and BioBERT RAG
        """
        try:
            # Phase 1: API Search (Authoritative)
            logger.info(f"Phase 1: API search for '{query}'")
            api_results = await api_search_func({
                'query': query,
                'ontologies': ontologies,
                'expand_abbreviations': expand_abbreviations,
                'semantic_search': semantic_search,
                'confidence_threshold': confidence_threshold
            })
            
            # Phase 2: BioBERT RAG Search (Semantic Discovery)
            logger.info(f"🧬 Phase 2: BioBERT RAG search for '{query}'")
            try:
                raw_rag_results = await self.biobert_engine.search(
                    query=query,
                    limit=rag_limit
                )
                logger.info(f"🧬 RAG search returned {len(raw_rag_results)} raw results")
                
                # RAG results are already normalized in BioBERT engine
                rag_results = raw_rag_results
                
                logger.info(f"🧬 Normalized {len(rag_results)} RAG results")
                if rag_results:
                    logger.info(f"🧬 Sample RAG result: {rag_results[0].get('concept_name', rag_results[0].get('concept', 'N/A'))[:50]}...")
                    
            except Exception as e:
                logger.error(f"🧬 RAG search failed: {e}")
                import traceback
                logger.error(f"🧬 RAG traceback: {traceback.format_exc()}")
                rag_results = []
            
            # Phase 3: Cross-Validation and Enhancement
            logger.info("Phase 3: Cross-validation and result enhancement")
            validated_results, discovery_results = await self._validate_and_enhance_results(
                api_results, rag_results
            )
            logger.info(f"Validation returned {len(validated_results)} validated, {len(discovery_results)} discovery")
            
            # Phase 4: Calculate hybrid confidence
            hybrid_confidence = self._calculate_hybrid_confidence(
                api_results, validated_results, discovery_results
            )
            
            # Compile search metadata
            search_metadata = {
                'api_results_count': len(api_results),
                'rag_results_count': len(rag_results),
                'validated_count': len(validated_results),
                'discovery_count': len(discovery_results),
                'search_strategy': 'hybrid_api_biobert',
                'confidence_threshold': confidence_threshold,
                'rag_limit': rag_limit
            }
            
            return HybridSearchResult(
                api_results=api_results,
                rag_results=rag_results,  # Keep original RAG results
                validated_results=validated_results,
                discovery_results=discovery_results,
                hybrid_confidence=hybrid_confidence,
                search_metadata=search_metadata
            )
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            # Return partial results if possible
            return HybridSearchResult(
                api_results=api_results if 'api_results' in locals() else [],
                rag_results=[],
                validated_results=[],
                discovery_results=[],
                hybrid_confidence=0.0,
                search_metadata={'error': str(e)}
            )
    
    async def _validate_and_enhance_results(
        self, 
        api_results: List[Dict], 
        rag_results: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Validate RAG results against API results and enhance with additional metadata
        """
        logger.info(f"🔍 Validation input: {len(api_results)} API, {len(rag_results)} RAG results")
        validated_results = []
        discovery_results = []
        
        # For now, bypass complex validation and put all RAG results in discovery
        # This ensures we don't lose results due to validation issues
        for i, rag_result in enumerate(rag_results):
            try:
                logger.info(f"🔍 Processing RAG result {i}: {rag_result.get('concept', 'N/A')[:30]}")
                
                # Find potential matches in API results
                best_match, similarity_score = self._find_best_api_match(rag_result, api_results)
                logger.info(f"🔍 Best match similarity: {similarity_score:.3f}")
                
                # Create enhanced result with consistent fields
                enhanced_rag_result = rag_result.copy() if isinstance(rag_result, dict) else {}
                
                # Ensure all required fields exist
                enhanced_rag_result.setdefault('confidence_score', enhanced_rag_result.get('score', 0.5))
                enhanced_rag_result.setdefault('concept_name', enhanced_rag_result.get('concept', ''))
                enhanced_rag_result.setdefault('concept_id', enhanced_rag_result.get('id', f'RAG_{i}'))
                enhanced_rag_result.setdefault('definition', enhanced_rag_result.get('text', ''))
                enhanced_rag_result.setdefault('source_ontology', enhanced_rag_result.get('ontology', enhanced_rag_result.get('source', 'PINECONE')))
                
                enhanced_rag_result['validation_metadata'] = {
                    'best_api_match': best_match.get('concept_id') if best_match else None,
                    'api_similarity_score': similarity_score,
                    'validation_method': 'simplified_semantic_similarity'
                }
                
                # Simplified validation - include all results but categorize them
                if similarity_score >= self.validation_threshold:
                    # Some match found
                    enhanced_rag_result.update({
                        'api_validated': True if similarity_score >= 0.5 else 'partial',
                        'validation_confidence': similarity_score,
                        'api_match': best_match
                    })
                    validated_results.append(enhanced_rag_result)
                    logger.info(f"🔍 Result {i} -> VALIDATED (similarity: {similarity_score:.3f})")
                else:
                    # Pure discovery result
                    enhanced_rag_result.update({
                        'api_validated': False,
                        'validation_confidence': similarity_score,
                        'discovery_type': 'semantic_discovery'
                    })
                    discovery_results.append(enhanced_rag_result)
                    logger.info(f"🔍 Result {i} -> DISCOVERY (similarity: {similarity_score:.3f})")
                    
            except Exception as e:
                logger.error(f"🔍 Error processing RAG result {i}: {e}")
                # Don't lose the result - add to discovery even if processing failed
                discovery_results.append(rag_result)
                continue
        
        return validated_results, discovery_results
    
    def _find_best_api_match(self, rag_result: Dict, api_results: List[Dict]) -> Tuple[Optional[Dict], float]:
        """
        Find the best matching API result for a RAG result
        """
        if not api_results:
            return None, 0.0
        
        # Use standardized field names
        rag_name = (rag_result.get('concept_name', '') or rag_result.get('concept', '')).lower()
        rag_id = rag_result.get('concept_id', '')
        
        best_match = None
        best_score = 0.0
        
        for api_result in api_results:
            # Check for exact ID match
            if hasattr(api_result, 'concept_id') and api_result.concept_id == rag_id:
                return api_result.__dict__, 1.0
            elif isinstance(api_result, dict) and api_result.get('concept_id') == rag_id:
                return api_result, 1.0
            
            # Check for name similarity
            api_name = ""
            if hasattr(api_result, 'concept_name'):
                api_name = api_result.concept_name.lower()
            elif isinstance(api_result, dict):
                api_name = api_result.get('concept_name', '').lower()
            
            if api_name:
                # Simple similarity based on common words
                rag_words = set(rag_name.split())
                api_words = set(api_name.split())
                
                if rag_words and api_words:
                    intersection = rag_words.intersection(api_words)
                    union = rag_words.union(api_words)
                    jaccard_similarity = len(intersection) / len(union) if union else 0.0
                    
                    if jaccard_similarity > best_score:
                        best_score = jaccard_similarity
                        best_match = api_result.__dict__ if hasattr(api_result, '__dict__') else api_result
        
        return best_match, best_score
    
    def _calculate_hybrid_confidence(
        self, 
        api_results: List[Dict], 
        validated_results: List[Dict], 
        discovery_results: List[Dict]
    ) -> float:
        """
        Calculate overall confidence score for hybrid search results
        """
        scores = []
        
        # API results get highest weight
        for result in api_results:
            if hasattr(result, 'confidence_score'):
                scores.append(result.confidence_score * 1.0)  # Full weight
            elif isinstance(result, dict) and 'confidence_score' in result:
                scores.append(result['confidence_score'] * 1.0)
        
        # Validated RAG results get medium weight
        for result in validated_results:
            score = result.get('confidence_score', result.get('score', 0.5))
            scores.append(score * 0.8)  # 80% weight
        
        # Discovery results get lower weight
        for result in discovery_results:
            score = result.get('confidence_score', result.get('score', 0.5))
            scores.append(score * 0.6)  # 60% weight
        
        return sum(scores) / len(scores) if scores else 0.0
    
    async def search_with_focus(
        self,
        query: str,
        focus_type: str,  # 'authoritative', 'discovery', 'comprehensive'
        ontologies: List[str],
        api_search_func,
        **kwargs
    ) -> HybridSearchResult:
        """
        Focused search based on research needs
        """
        if focus_type == 'authoritative':
            # Emphasize API results
            kwargs['rag_limit'] = 10
            kwargs['confidence_threshold'] = 0.7
            
        elif focus_type == 'discovery':
            # Emphasize RAG discovery
            kwargs['rag_limit'] = 50
            kwargs['confidence_threshold'] = 0.3
            
        elif focus_type == 'comprehensive':
            # Balanced approach
            kwargs['rag_limit'] = 30
            kwargs['confidence_threshold'] = 0.5
        
        return await self.hybrid_search(
            query=query,
            ontologies=ontologies,
            api_search_func=api_search_func,
            **kwargs
        )
    
    async def get_related_concepts(
        self,
        concept_id: str,
        ontology: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        Find related concepts using BioBERT semantic similarity
        """
        try:
            # Search for the original concept in BioBERT data
            base_results = await self.biobert_engine.semantic_search(
                query=concept_id,
                top_k=1
            )
            
            if not base_results:
                return []
            
            base_concept = base_results[0]
            
            # Use the concept name to find semantically similar concepts
            related_results = await self.biobert_engine.semantic_search(
                query=base_concept['concept_name'],
                top_k=limit + 1  # +1 to exclude the original concept
            )
            
            # Filter out the original concept
            filtered_results = [
                result for result in related_results 
                if result['concept_id'] != concept_id
            ]
            
            return filtered_results[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get related concepts: {e}")
            return []

# Global hybrid search engine instance
hybrid_search_engine = HybridMedicalSearchEngine()