"""Medical API client implementations."""

import httpx
import logging
from typing import List, Dict, Optional
from difflib import SequenceMatcher

from ..config import settings
from ..exceptions import (
    UMLSAPIError, RxNormAPIError, WHOICDAPIError,
    LOINCAPIError, SNOMEDAPIError, RateLimitError, TimeoutError
)
from ..utils.retry import api_retry
from .cache import api_cache

logger = logging.getLogger(__name__)


class MedicalAPIClient:
    """Client for interacting with various medical terminology APIs."""
    
    def __init__(self):
        self.umls_api_key = settings.umls_api_key.get_secret_value() if settings.umls_api_key else None
        self.umls_username = settings.umls_username
        self.rxnorm_base_url = settings.rxnorm_base_url
        self.who_icd_client_id = settings.who_icd_client_id.get_secret_value() if settings.who_icd_client_id else None
        self.who_icd_client_secret = settings.who_icd_client_secret.get_secret_value() if settings.who_icd_client_secret else None
        self.loinc_username = settings.loinc_username
        self.loinc_password = settings.loinc_password.get_secret_value() if settings.loinc_password else None
        self.snomed_browser_url = settings.snomed_browser_url
        
    @api_cache
    @api_retry
    async def search_umls(self, query: str) -> List[Dict]:
        """Search UMLS Metathesaurus"""
        if not self.umls_api_key:
            return []
            
        try:
            url = f"https://uts-ws.nlm.nih.gov/rest/search/current"
            params = {
                'string': query,
                'apiKey': self.umls_api_key,
                'returnIdType': 'concept',
                'pageSize': 20
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    results = []
                    for result in data.get('result', {}).get('results', []):
                        results.append({
                            'concept_id': result.get('ui'),
                            'concept_name': result.get('name'),
                            'source_ontology': 'UMLS',
                            'definition': result.get('rootSource'),
                            'semantic_types': [result.get('rootSource')]
                        })
                    return results
                elif response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    raise RateLimitError(f"UMLS API rate limit exceeded", retry_after=retry_after)
                else:
                    raise UMLSAPIError(f"UMLS API error: {response.status_code}")
        except httpx.TimeoutException:
            raise TimeoutError("UMLS API request timed out", timeout=30.0)
        except Exception as e:
            if isinstance(e, (UMLSAPIError, RateLimitError, TimeoutError)):
                raise
            logger.error(f"UMLS search error: {e}")
            return []

    @api_cache
    @api_retry
    async def search_rxnorm(self, query: str) -> List[Dict]:
        """Search RxNorm for medication codes"""
        try:
            url = f"{self.rxnorm_base_url}/drugs.json"
            params = {'name': query}
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    results = []
                    drugs = data.get('drugGroup', {}).get('conceptGroup', [])
                    for group in drugs:
                        for concept in group.get('conceptProperties', []):
                            results.append({
                                'concept_id': concept.get('rxcui'),
                                'concept_name': concept.get('name'),
                                'source_ontology': 'RxNorm',
                                'definition': concept.get('synonym'),
                                'semantic_types': ['Drug']
                            })
                    return results
                elif response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    raise RateLimitError(f"RxNorm API rate limit exceeded", retry_after=retry_after)
                else:
                    raise RxNormAPIError(f"RxNorm API error: {response.status_code}")
        except httpx.TimeoutException:
            raise TimeoutError("RxNorm API request timed out", timeout=30.0)
        except Exception as e:
            if isinstance(e, (RxNormAPIError, RateLimitError, TimeoutError)):
                raise
            logger.error(f"RxNorm search error: {e}")
            return []

    @api_cache
    @api_retry
    async def search_icd10(self, query: str) -> List[Dict]:
        """Search ICD-10-CM codes using ClinicalTables.nlm.nih.gov API"""
        try:
            # Use the public ICD-10-CM API from ClinicalTables
            url = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"
            params = {
                'sf': 'code,name',  # Search both code and name fields
                'terms': query,
                'maxList': 20,
                'df': 'code,name'   # Display both code and name
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    results = []
                    
                    # ClinicalTables API returns [total, codes, null, display_data]
                    if data and len(data) >= 4:
                        total_count = data[0] if len(data) > 0 else 0
                        codes = data[1] if len(data) > 1 else []
                        # data[2] is null, skip it
                        display_data = data[3] if len(data) > 3 else []
                        
                        # Process display_data which contains [code, name] pairs
                        for item in display_data:
                            if isinstance(item, list) and len(item) >= 2:
                                icd_code = item[0]
                                icd_name = item[1]
                                
                                if icd_code and icd_name:
                                    results.append({
                                        'concept_id': icd_code,
                                        'concept_name': icd_name,
                                        'source_ontology': 'ICD-10-CM',
                                        'definition': icd_name,  # Use the full name as definition
                                        'semantic_types': ['Disease Classification']
                                    })
                    return results
                elif response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    raise RateLimitError(f"ICD-10-CM API rate limit exceeded", retry_after=retry_after)
                else:
                    raise WHOICDAPIError(f"ICD-10-CM API error: {response.status_code}")
        except httpx.TimeoutException:
            raise TimeoutError("ICD-10-CM API request timed out", timeout=30.0)
        except Exception as e:
            if isinstance(e, (WHOICDAPIError, RateLimitError, TimeoutError)):
                raise
            logger.error(f"ICD-10-CM search error: {e}")
            return []

    async def search_snomed(self, query: str) -> List[Dict]:
        """Search SNOMED CT using fallback approach with common medical concepts"""
        try:
            # Since SNOMED CT API access is limited, we'll use a fallback approach
            # with common medical concepts for demonstration purposes
            common_snomed_concepts = {
                'pneumonia': [
                    {'id': '233604007', 'term': 'Pneumonia', 'definition': 'Infection of the lung'},
                    {'id': '385093006', 'term': 'Community acquired pneumonia', 'definition': 'Pneumonia acquired outside hospital'},
                    {'id': '444814009', 'term': 'Viral pneumonia', 'definition': 'Pneumonia caused by virus'}
                ],
                'diabetes': [
                    {'id': '73211009', 'term': 'Diabetes mellitus', 'definition': 'Disorder of glucose metabolism'},
                    {'id': '46635009', 'term': 'Type 1 diabetes mellitus', 'definition': 'Insulin dependent diabetes'},
                    {'id': '44054006', 'term': 'Type 2 diabetes mellitus', 'definition': 'Non-insulin dependent diabetes'}
                ],
                'hypertension': [
                    {'id': '38341003', 'term': 'Hypertensive disorder', 'definition': 'High blood pressure'},
                    {'id': '59621000', 'term': 'Essential hypertension', 'definition': 'Primary high blood pressure'}
                ],
                'heart': [
                    {'id': '22298006', 'term': 'Myocardial infarction', 'definition': 'Heart attack'},
                    {'id': '84114007', 'term': 'Heart failure', 'definition': 'Inability of heart to pump effectively'}
                ],
                'asthma': [
                    {'id': '195967001', 'term': 'Asthma', 'definition': 'Chronic respiratory disorder'},
                    {'id': '370218001', 'term': 'Acute severe asthma', 'definition': 'Severe asthma attack'}
                ],
                'sepsis': [
                    {'id': '91302008', 'term': 'Sepsis (disorder)', 'definition': 'Life-threatening organ dysfunction caused by dysregulated host response to infection'},
                    {'id': '17294005', 'term': 'Septic shock', 'definition': 'Sepsis with hypotension despite adequate fluid resuscitation'}
                ]
            }
            
            results = []
            query_lower = query.lower()
            
            # Search for matching concepts
            for term, concepts in common_snomed_concepts.items():
                if term in query_lower:
                    for concept in concepts:
                        results.append({
                            'concept_id': concept['id'],
                            'concept_name': concept['term'],
                            'source_ontology': 'SNOMED CT',
                            'definition': concept['definition'],
                            'semantic_types': ['Clinical Finding']
                        })
                    break
            
            # If no direct match, use fuzzy matching
            if not results:
                best_match = None
                best_score = 0
                for term, concepts in common_snomed_concepts.items():
                    score = SequenceMatcher(None, query_lower, term).ratio()
                    if score > best_score and score > 0.6:
                        best_score = score
                        best_match = concepts
                
                if best_match:
                    for concept in best_match:
                        results.append({
                            'concept_id': concept['id'],
                            'concept_name': concept['term'],
                            'source_ontology': 'SNOMED CT',
                            'definition': concept['definition'],
                            'semantic_types': ['Clinical Finding'],
                            'confidence_score': best_score
                        })
            
            return results
        except Exception as e:
            logger.error(f"SNOMED search error: {e}")
            return []

    @api_cache
    @api_retry
    async def search_loinc(self, query: str) -> List[Dict]:
        """Search LOINC codes using public API"""
        try:
            # Use public LOINC search API
            url = "https://clinicaltables.nlm.nih.gov/api/loinc_items/v3/search"
            params = {
                'terms': query,
                'maxList': 20,
                'df': 'LOINC_NUM,LONG_COMMON_NAME,SHORTNAME'
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    results = []
                    
                    # ClinicalTables API returns [total, loinc_codes, null, detailed_data]
                    if data and len(data) >= 4:
                        total_count = data[0] if len(data) > 0 else 0
                        loinc_codes = data[1] if len(data) > 1 else []
                        # data[2] is null, skip it
                        detailed_data = data[3] if len(data) > 3 else []
                        
                        # Process detailed_data which contains [LOINC_NUM, LONG_COMMON_NAME, SHORTNAME]
                        for item in detailed_data:
                            if isinstance(item, list) and len(item) >= 3:
                                loinc_code = item[0]
                                long_name = item[1]
                                short_name = item[2]
                                
                                if loinc_code and long_name:
                                    results.append({
                                        'concept_id': loinc_code,
                                        'concept_name': long_name,
                                        'source_ontology': 'LOINC',
                                        'definition': short_name,
                                        'semantic_types': ['Laboratory Test']
                                    })
                    return results
                elif response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    raise RateLimitError(f"LOINC API rate limit exceeded", retry_after=retry_after)
                else:
                    raise LOINCAPIError(f"LOINC API error: {response.status_code}")
        except httpx.TimeoutException:
            raise TimeoutError("LOINC API request timed out", timeout=30.0)
        except Exception as e:
            if isinstance(e, (LOINCAPIError, RateLimitError, TimeoutError)):
                raise
            logger.error(f"LOINC search error: {e}")
            return []

    async def search_across_apis(self, query: str, ontologies: Optional[List[str]] = None) -> Dict[str, List[Dict]]:
        """Search across multiple medical APIs based on specified ontologies"""
        results = {}
        
        # Default to all available ontologies if none specified
        if not ontologies:
            ontologies = ['UMLS', 'RxNorm', 'ICD-10-CM', 'SNOMED CT', 'LOINC']
        
        # Map ontology names to search methods
        search_methods = {
            'UMLS': self.search_umls,
            'RxNorm': self.search_rxnorm,
            'ICD-10': self.search_icd10,
            'ICD-10-CM': self.search_icd10,  # Using ICD-10-CM ClinicalTables API
            'SNOMED CT': self.search_snomed,
            'SNOMED': self.search_snomed,
            'LOINC': self.search_loinc
        }
        
        # Execute searches for requested ontologies
        for ontology in ontologies:
            if ontology in search_methods:
                try:
                    results[ontology] = await search_methods[ontology](query)
                except Exception as e:
                    logger.error(f"Error searching {ontology}: {e}")
                    results[ontology] = []
        
        return results