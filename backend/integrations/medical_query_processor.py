"""
Medical Query Processor for Enhanced RAG Search
Handles query expansion, abbreviation resolution, and medical term normalization
"""

import re
import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ProcessedQuery:
    """Processed medical query with expansions and metadata"""
    original_query: str
    normalized_query: str
    expanded_terms: List[str]
    abbreviations_expanded: Dict[str, str]
    concept_type: Optional[str]
    medical_context: str

class MedicalQueryProcessor:
    """Processes medical queries for improved RAG search"""
    
    def __init__(self):
        # Common medical abbreviations
        self.medical_abbreviations = {
            # Diseases/Conditions
            "dm": ["diabetes mellitus", "diabetes"],
            "t2dm": ["type 2 diabetes mellitus", "type 2 diabetes", "diabetes type 2"],
            "t1dm": ["type 1 diabetes mellitus", "type 1 diabetes", "diabetes type 1"],
            "htn": ["hypertension", "high blood pressure"],
            "copd": ["chronic obstructive pulmonary disease"],
            "cad": ["coronary artery disease"],
            "chf": ["congestive heart failure", "heart failure"],
            "mi": ["myocardial infarction", "heart attack"],
            "cva": ["cerebrovascular accident", "stroke"],
            "pe": ["pulmonary embolism"],
            "dvt": ["deep vein thrombosis"],
            "uti": ["urinary tract infection"],
            "ckd": ["chronic kidney disease"],
            "esrd": ["end stage renal disease"],
            "gerd": ["gastroesophageal reflux disease"],
            "ra": ["rheumatoid arthritis"],
            "oa": ["osteoarthritis"],
            "ms": ["multiple sclerosis"],
            "sle": ["systemic lupus erythematosus", "lupus"],
            
            # Procedures
            "cabg": ["coronary artery bypass graft", "bypass surgery"],
            "pci": ["percutaneous coronary intervention"],
            "tka": ["total knee arthroplasty", "knee replacement"],
            "tha": ["total hip arthroplasty", "hip replacement"],
            
            # Lab tests
            "cbc": ["complete blood count"],
            "bmp": ["basic metabolic panel"],
            "cmp": ["comprehensive metabolic panel"],
            "lfts": ["liver function tests"],
            "tsh": ["thyroid stimulating hormone"],
            "hba1c": ["hemoglobin a1c", "glycated hemoglobin"],
            "psa": ["prostate specific antigen"],
            
            # Medications
            "asa": ["aspirin", "acetylsalicylic acid"],
            "apap": ["acetaminophen", "paracetamol"],
            "nsaid": ["non-steroidal anti-inflammatory drug"],
            "ppi": ["proton pump inhibitor"],
            "ssri": ["selective serotonin reuptake inhibitor"],
            "ace": ["angiotensin converting enzyme"],
            "arb": ["angiotensin receptor blocker"],
        }
        
        # Medical synonyms and related terms
        self.medical_synonyms = {
            # Diabetes related
            "diabetes": ["diabetes mellitus", "diabetic", "hyperglycemia"],
            "diabetes mellitus": ["diabetes", "DM", "diabetic condition"],
            "type 2 diabetes": ["T2DM", "type II diabetes", "adult onset diabetes", "NIDDM"],
            "type 1 diabetes": ["T1DM", "type I diabetes", "juvenile diabetes", "IDDM"],
            
            # Cardiovascular
            "hypertension": ["high blood pressure", "HTN", "elevated blood pressure"],
            "heart attack": ["myocardial infarction", "MI", "cardiac infarction"],
            "heart failure": ["CHF", "congestive heart failure", "cardiac failure"],
            "stroke": ["CVA", "cerebrovascular accident", "brain attack"],
            
            # Respiratory
            "asthma": ["bronchial asthma", "reactive airway disease"],
            "pneumonia": ["lung infection", "chest infection"],
            
            # General symptoms
            "chest pain": ["angina", "chest discomfort", "thoracic pain"],
            "shortness of breath": ["dyspnea", "SOB", "breathing difficulty"],
            "headache": ["cephalgia", "head pain"],
            "fever": ["pyrexia", "elevated temperature", "febrile"],
        }
        
        # Concept type patterns
        self.concept_patterns = {
            "disease": ["disease", "disorder", "syndrome", "condition", "illness"],
            "symptom": ["pain", "ache", "fever", "cough", "nausea", "fatigue"],
            "procedure": ["surgery", "procedure", "operation", "test", "exam"],
            "medication": ["drug", "medication", "medicine", "pill", "tablet"],
            "anatomy": ["organ", "tissue", "bone", "muscle", "nerve"],
        }
    
    def process_query(self, query: str) -> ProcessedQuery:
        """Process a medical query with all enhancements"""
        # Normalize the query
        normalized = self._normalize_text(query)
        
        # Expand abbreviations
        expanded_query, abbreviations = self._expand_abbreviations(normalized)
        
        # Get expanded terms (synonyms)
        expanded_terms = self._get_expanded_terms(expanded_query)
        
        # Detect concept type
        concept_type = self._detect_concept_type(expanded_query)
        
        # Generate medical context
        medical_context = self._generate_medical_context(expanded_query, concept_type)
        
        return ProcessedQuery(
            original_query=query,
            normalized_query=normalized,
            expanded_terms=expanded_terms,
            abbreviations_expanded=abbreviations,
            concept_type=concept_type,
            medical_context=medical_context
        )
    
    def _normalize_text(self, text: str) -> str:
        """Normalize medical text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Handle common variations
        text = re.sub(r'\bdiabetes\s+mellitus\s+type\s+2\b', 'type 2 diabetes mellitus', text)
        text = re.sub(r'\bdiabetes\s+mellitus\s+type\s+1\b', 'type 1 diabetes mellitus', text)
        text = re.sub(r'\btype\s+ii\s+diabetes\b', 'type 2 diabetes', text)
        text = re.sub(r'\btype\s+i\s+diabetes\b', 'type 1 diabetes', text)
        
        return text
    
    def _expand_abbreviations(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Expand medical abbreviations in text"""
        expanded = text
        expansions = {}
        
        # Sort by length to handle longer abbreviations first
        sorted_abbrevs = sorted(self.medical_abbreviations.keys(), key=len, reverse=True)
        
        for abbrev in sorted_abbrevs:
            # Use word boundaries to match whole words only
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            if re.search(pattern, expanded, re.IGNORECASE):
                # Use the first expansion as the primary one
                expansion = self.medical_abbreviations[abbrev][0]
                expanded = re.sub(pattern, expansion, expanded, flags=re.IGNORECASE)
                expansions[abbrev] = expansion
        
        return expanded, expansions
    
    def _get_expanded_terms(self, text: str) -> List[str]:
        """Get expanded terms including synonyms"""
        expanded_terms = set()
        
        # Add the original text
        expanded_terms.add(text)
        
        # Check each word/phrase for synonyms
        for term, synonyms in self.medical_synonyms.items():
            if term in text.lower():
                expanded_terms.update(synonyms)
        
        # Add partial matches for multi-word terms
        words = text.split()
        for word in words:
            for term, synonyms in self.medical_synonyms.items():
                if word in term or term in word:
                    expanded_terms.update(synonyms)
        
        return list(expanded_terms)
    
    def _detect_concept_type(self, text: str) -> Optional[str]:
        """Detect the medical concept type from text"""
        text_lower = text.lower()
        
        # Check for concept type indicators
        for concept_type, patterns in self.concept_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return concept_type
        
        # Check specific terms
        if any(term in text_lower for term in ["diabetes", "hypertension", "cancer", "infection"]):
            return "disease"
        elif any(term in text_lower for term in ["surgery", "biopsy", "x-ray", "mri", "ct scan"]):
            return "procedure"
        elif any(term in text_lower for term in ["metformin", "insulin", "statin", "antibiotic"]):
            return "medication"
        
        return None
    
    def _generate_medical_context(self, query: str, concept_type: Optional[str]) -> str:
        """Generate medical context for the query"""
        contexts = []
        
        # Add base medical context
        contexts.append("medical terminology")
        
        # Add concept type context
        if concept_type:
            contexts.append(f"{concept_type} concept")
        
        # Add specific domain context based on terms
        query_lower = query.lower()
        
        if any(term in query_lower for term in ["diabetes", "glucose", "insulin", "hba1c"]):
            contexts.append("endocrinology")
            contexts.append("metabolic disorders")
        elif any(term in query_lower for term in ["heart", "cardiac", "coronary", "hypertension"]):
            contexts.append("cardiology")
            contexts.append("cardiovascular system")
        elif any(term in query_lower for term in ["lung", "respiratory", "asthma", "copd"]):
            contexts.append("pulmonology")
            contexts.append("respiratory system")
        elif any(term in query_lower for term in ["kidney", "renal", "nephro"]):
            contexts.append("nephrology")
            contexts.append("renal system")
        elif any(term in query_lower for term in ["liver", "hepatic", "cirrhosis"]):
            contexts.append("hepatology")
            contexts.append("hepatic system")
        elif any(term in query_lower for term in ["cancer", "tumor", "oncology"]):
            contexts.append("oncology")
            contexts.append("neoplastic diseases")
        
        return " ".join(contexts)
    
    def generate_search_queries(self, processed_query: ProcessedQuery) -> List[str]:
        """Generate multiple search queries for better coverage"""
        queries = []
        
        # Original query
        queries.append(processed_query.original_query)
        
        # Normalized query
        if processed_query.normalized_query != processed_query.original_query:
            queries.append(processed_query.normalized_query)
        
        # Top expanded terms (limit to avoid too many queries)
        queries.extend(processed_query.expanded_terms[:3])
        
        # Add contextualized queries
        if processed_query.concept_type:
            queries.append(f"{processed_query.normalized_query} {processed_query.concept_type}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)
        
        return unique_queries

# Global instance for easy access
medical_query_processor = MedicalQueryProcessor()