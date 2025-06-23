"""
Batch Mapping Service for Medical Concepts
Processes medical concepts in batches and maps them to standardized terminologies
"""

import asyncio
import pandas as pd
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import uuid
from concurrent.futures import ThreadPoolExecutor
import time

from .api_clients import MedicalAPIClient
from ..integrations.hybrid_search import hybrid_search_engine
from ..config import settings

logger = logging.getLogger(__name__)

class BatchMapper:
    """Intelligent batch mapper for medical concepts"""
    
    def __init__(self):
        self.medical_client = MedicalAPIClient()
        self.processing_jobs = {}  # Store job status
        self.max_concurrent = 5  # Limit concurrent API calls
        self.rate_limit_delay = 0.1  # Delay between API calls
        
    async def process_csv_batch(
        self, 
        file_path: str, 
        mapping_config: Dict[str, Any],
        job_id: str = None
    ) -> Dict[str, Any]:
        """
        Process CSV file with batch mapping of medical concepts
        """
        if not job_id:
            job_id = str(uuid.uuid4())
        
        try:
            # Initialize job tracking
            self.processing_jobs[job_id] = {
                'status': 'starting',
                'progress': 0,
                'total_concepts': 0,
                'processed_concepts': 0,
                'start_time': datetime.now(),
                'current_operation': 'Loading data...'
            }
            
            # Load CSV data
            df = pd.read_csv(file_path)
            
            # Extract medical concepts for mapping
            concepts_to_map = self._extract_concepts_for_mapping(df, mapping_config)
            
            self.processing_jobs[job_id].update({
                'total_concepts': len(concepts_to_map),
                'status': 'processing',
                'current_operation': 'Starting concept mapping...'
            })
            
            # Process concepts in batches
            mapped_results = await self._process_concepts_batch(concepts_to_map, mapping_config, job_id)
            
            # Generate enhanced CSV
            enhanced_df = self._generate_enhanced_csv(df, mapped_results, mapping_config)
            
            # Save enhanced CSV
            output_path = file_path.replace('.csv', f'_enhanced_{job_id[:8]}.csv')
            enhanced_df.to_csv(output_path, index=False)
            
            # Generate summary report
            summary = self._generate_mapping_summary(mapped_results, mapping_config)
            
            # Update job status
            self.processing_jobs[job_id].update({
                'status': 'completed',
                'progress': 100,
                'end_time': datetime.now(),
                'output_file': output_path,
                'summary': summary,
                'current_operation': 'Completed'
            })
            
            return {
                'job_id': job_id,
                'status': 'completed',
                'output_file': output_path,
                'summary': summary,
                'enhanced_data_preview': enhanced_df.head(5).to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"Batch processing failed for job {job_id}: {e}")
            self.processing_jobs[job_id].update({
                'status': 'failed',
                'error': str(e),
                'end_time': datetime.now()
            })
            raise
    
    def _extract_concepts_for_mapping(self, df: pd.DataFrame, mapping_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract unique medical concepts from specified columns"""
        
        concepts_to_map = []
        
        for column_config in mapping_config.get('columns', []):
            column_name = column_config['column']
            if column_name not in df.columns:
                continue
            
            # Get unique non-null values from column
            unique_values = df[column_name].dropna().astype(str).unique()
            
            for value in unique_values:
                value = str(value).strip()
                if value and len(value) > 1:  # Skip empty or single character values
                    concepts_to_map.append({
                        'original_value': value,
                        'column': column_name,
                        'medical_type': column_config.get('medical_type', 'condition'),
                        'terminology_systems': column_config.get('terminology_systems', ['UMLS']),
                        'search_mode': column_config.get('search_mode', 'api_only'),
                        'confidence_threshold': column_config.get('confidence_threshold', 0.5)
                    })
        
        return concepts_to_map
    
    async def _process_concepts_batch(
        self, 
        concepts: List[Dict[str, Any]], 
        mapping_config: Dict[str, Any],
        job_id: str
    ) -> Dict[str, Any]:
        """Process concepts in batches with rate limiting"""
        
        mapped_results = {}
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_single_concept(concept: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
            async with semaphore:
                try:
                    # Add rate limiting delay
                    await asyncio.sleep(self.rate_limit_delay)
                    
                    # Update progress
                    self.processing_jobs[job_id]['current_operation'] = f"Mapping: {concept['original_value'][:50]}..."
                    
                    # Process the concept
                    result = await self._map_single_concept(concept)
                    
                    # Update progress
                    self.processing_jobs[job_id]['processed_concepts'] += 1
                    progress = (self.processing_jobs[job_id]['processed_concepts'] / 
                              self.processing_jobs[job_id]['total_concepts']) * 100
                    self.processing_jobs[job_id]['progress'] = round(progress, 1)
                    
                    return concept['original_value'], result
                    
                except Exception as e:
                    logger.error(f"Failed to map concept '{concept['original_value']}': {e}")
                    return concept['original_value'], {
                        'status': 'error',
                        'error': str(e),
                        'mappings': {}
                    }
        
        # Process all concepts concurrently
        tasks = [process_single_concept(concept) for concept in concepts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Compile results
        for result in results:
            if isinstance(result, tuple):
                original_value, mapping_result = result
                mapped_results[original_value] = mapping_result
            else:
                logger.error(f"Unexpected result type: {type(result)}")
        
        return mapped_results
    
    async def _map_single_concept(self, concept: Dict[str, Any]) -> Dict[str, Any]:
        """Map a single medical concept to terminology systems"""
        
        try:
            original_value = concept['original_value']
            search_mode = concept['search_mode']
            terminology_systems = concept['terminology_systems']
            
            # Prepare search parameters
            search_params = {
                'query': original_value,
                'ontologies': [system.lower().replace('-', '').replace(' ', '') for system in terminology_systems],
                'expand_abbreviations': True,
                'semantic_search': True,
                'confidence_threshold': concept['confidence_threshold']
            }
            
            # Perform search based on mode
            if search_mode == 'hybrid':
                # Use hybrid search
                result = await hybrid_search_engine.hybrid_search(
                    query=original_value,
                    ontologies=search_params['ontologies'],
                    api_search_func=self._api_search_wrapper,
                    expand_abbreviations=True,
                    semantic_search=True,
                    confidence_threshold=concept['confidence_threshold']
                )
                
                # Process hybrid results
                mappings = self._process_hybrid_results(result, terminology_systems)
                
            else:
                # Use API-only search
                api_results = await self.medical_client.search_across_apis(
                    query=original_value,
                    ontologies=search_params['ontologies']
                )
                
                # Process API results
                mappings = self._process_api_results(api_results, terminology_systems)
            
            return {
                'status': 'success',
                'original_value': original_value,
                'mappings': mappings,
                'search_mode': search_mode,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error mapping concept '{concept['original_value']}': {e}")
            return {
                'status': 'error',
                'original_value': concept['original_value'],
                'error': str(e),
                'mappings': {}
            }
    
    async def _api_search_wrapper(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Wrapper for API search to work with hybrid search"""
        try:
            results = await self.medical_client.search_across_apis(
                query=params['query'],
                ontologies=params['ontologies']
            )
            
            # Flatten results from different ontologies
            flattened = []
            for ontology, ontology_results in results.items():
                flattened.extend(ontology_results)
            
            return flattened
            
        except Exception as e:
            logger.error(f"API search wrapper error: {e}")
            return []
    
    def _process_hybrid_results(self, hybrid_result, terminology_systems: List[str]) -> Dict[str, Any]:
        """Process hybrid search results into standardized format"""
        
        mappings = {}
        
        # Process API results
        for result in hybrid_result.api_results:
            ontology = result.get('source_ontology', '').upper()
            mappings.setdefault(ontology, []).append({
                'code': result.get('concept_id', ''),
                'name': result.get('concept_name', ''),
                'confidence': result.get('confidence_score', 0.0),
                'source': 'api',
                'definition': result.get('definition', '')
            })
        
        # Process validated results
        for result in hybrid_result.validated_results:
            ontology = result.get('source_ontology', '').upper()
            mappings.setdefault(ontology, []).append({
                'code': result.get('concept_id', ''),
                'name': result.get('concept_name', ''),
                'confidence': result.get('validation_confidence', 0.0),
                'source': 'validated_rag',
                'definition': result.get('definition', '')
            })
        
        # Process discovery results
        for result in hybrid_result.discovery_results:
            ontology = result.get('source_ontology', '').upper()
            mappings.setdefault(ontology, []).append({
                'code': result.get('concept_id', ''),
                'name': result.get('concept_name', ''),
                'confidence': result.get('confidence_score', 0.0),
                'source': 'discovery',
                'definition': result.get('definition', '')
            })
        
        # Sort by confidence and take best matches
        for ontology in mappings:
            mappings[ontology] = sorted(mappings[ontology], key=lambda x: x['confidence'], reverse=True)[:3]
        
        return mappings
    
    def _process_api_results(self, api_results: Dict[str, List], terminology_systems: List[str]) -> Dict[str, Any]:
        """Process API-only results into standardized format"""
        
        mappings = {}
        
        for ontology, results in api_results.items():
            if results:
                ontology_upper = ontology.upper()
                mappings[ontology_upper] = []
                
                for result in results[:3]:  # Take top 3 results
                    mappings[ontology_upper].append({
                        'code': result.get('concept_id', ''),
                        'name': result.get('concept_name', ''),
                        'confidence': result.get('confidence_score', 0.0),
                        'source': 'api',
                        'definition': result.get('definition', '')
                    })
        
        return mappings
    
    def _generate_enhanced_csv(
        self, 
        original_df: pd.DataFrame, 
        mapped_results: Dict[str, Any],
        mapping_config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Generate enhanced CSV with mapping results"""
        
        enhanced_df = original_df.copy()
        
        # Add mapping columns for each configured column
        for column_config in mapping_config.get('columns', []):
            column_name = column_config['column']
            terminology_systems = column_config.get('terminology_systems', ['UMLS'])
            
            if column_name not in enhanced_df.columns:
                continue
            
            # Add columns for each terminology system
            for system in terminology_systems:
                system_upper = system.upper().replace('-', '_').replace(' ', '_')
                
                # Add code, name, and confidence columns
                enhanced_df[f"{column_name}_{system_upper}_CODE"] = ""
                enhanced_df[f"{column_name}_{system_upper}_NAME"] = ""
                enhanced_df[f"{column_name}_{system_upper}_CONFIDENCE"] = 0.0
            
            # Add general mapping status column
            enhanced_df[f"{column_name}_MAPPING_STATUS"] = ""
            enhanced_df[f"{column_name}_ALTERNATIVES"] = ""
            
            # Fill in the mapping data
            for idx, row in enhanced_df.iterrows():
                original_value = str(row[column_name])
                
                if original_value in mapped_results:
                    mapping_data = mapped_results[original_value]
                    
                    if mapping_data.get('status') == 'success':
                        mappings = mapping_data.get('mappings', {})
                        alternatives = []
                        
                        # Fill in best matches for each system
                        for system in terminology_systems:
                            system_upper = system.upper().replace('-', '_').replace(' ', '_')
                            
                            if system_upper in mappings and mappings[system_upper]:
                                best_match = mappings[system_upper][0]  # Take best match
                                
                                enhanced_df.at[idx, f"{column_name}_{system_upper}_CODE"] = best_match['code']
                                enhanced_df.at[idx, f"{column_name}_{system_upper}_NAME"] = best_match['name']
                                enhanced_df.at[idx, f"{column_name}_{system_upper}_CONFIDENCE"] = best_match['confidence']
                                
                                # Collect alternatives
                                if len(mappings[system_upper]) > 1:
                                    alternatives.extend([
                                        f"{alt['code']}:{alt['name']} ({alt['confidence']:.2f})"
                                        for alt in mappings[system_upper][1:]
                                    ])
                        
                        # Set mapping status
                        if any(mappings.values()):
                            enhanced_df.at[idx, f"{column_name}_MAPPING_STATUS"] = "success"
                        else:
                            enhanced_df.at[idx, f"{column_name}_MAPPING_STATUS"] = "no_matches"
                        
                        # Set alternatives
                        enhanced_df.at[idx, f"{column_name}_ALTERNATIVES"] = "; ".join(alternatives[:5])
                        
                    else:
                        enhanced_df.at[idx, f"{column_name}_MAPPING_STATUS"] = "failed"
                else:
                    enhanced_df.at[idx, f"{column_name}_MAPPING_STATUS"] = "not_processed"
        
        return enhanced_df
    
    def _generate_mapping_summary(self, mapped_results: Dict[str, Any], mapping_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary report of mapping results"""
        
        total_concepts = len(mapped_results)
        successful_mappings = sum(1 for result in mapped_results.values() if result.get('status') == 'success')
        failed_mappings = total_concepts - successful_mappings
        
        # Count mappings by terminology system
        system_counts = {}
        for result in mapped_results.values():
            if result.get('status') == 'success':
                mappings = result.get('mappings', {})
                for system in mappings:
                    system_counts[system] = system_counts.get(system, 0) + len(mappings[system])
        
        # Calculate confidence distribution
        confidence_scores = []
        for result in mapped_results.values():
            if result.get('status') == 'success':
                mappings = result.get('mappings', {})
                for system_mappings in mappings.values():
                    for mapping in system_mappings:
                        confidence_scores.append(mapping.get('confidence', 0.0))
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        return {
            'total_concepts_processed': total_concepts,
            'successful_mappings': successful_mappings,
            'failed_mappings': failed_mappings,
            'success_rate': (successful_mappings / total_concepts * 100) if total_concepts > 0 else 0,
            'terminology_system_counts': system_counts,
            'average_confidence': round(avg_confidence, 3),
            'high_confidence_mappings': sum(1 for score in confidence_scores if score > 0.8),
            'low_confidence_mappings': sum(1 for score in confidence_scores if score < 0.5),
            'processing_timestamp': datetime.now().isoformat()
        }
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get the status of a processing job"""
        return self.processing_jobs.get(job_id, {'status': 'not_found'})
    
    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Clean up old job records"""
        current_time = datetime.now()
        jobs_to_remove = []
        
        for job_id, job_data in self.processing_jobs.items():
            start_time = job_data.get('start_time')
            if start_time and (current_time - start_time).total_seconds() > max_age_hours * 3600:
                jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del self.processing_jobs[job_id]