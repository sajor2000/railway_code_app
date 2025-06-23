# Medical Terminology Research Agents using Pydantic AI + LangGraph
import os
import json
import asyncio
from typing import Dict, List, Literal, Optional, TypedDict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
ROOT_DIR = Path(__file__).parent.parent.parent  # Go up to project root
load_dotenv(ROOT_DIR / '.env')

from openai import OpenAI

# Import for Pydantic AI (we'll use OpenAI client as base)
import logging
logger = logging.getLogger(__name__)

# Lazy OpenAI client initialization
_openai_client = None

def get_openai_client():
    """Get or create OpenAI client"""
    global _openai_client
    if _openai_client is None:
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client

class ResearchTerminologyDeps:
    """Dependencies for medical research agents"""
    def __init__(self):
        self.pinecone_client = None  # Will be injected
        self.api_clients = {}  # Medical API clients
        self.medical_client = None  # Will be injected

# =================== PYDANTIC MODELS ===================

class SearchStrategyAnalysis(BaseModel):
    search_mode: Literal["api_only", "hybrid", "discovery_focused"]
    concept_type: Literal["condition", "procedure", "medication", "laboratory", "observation", "device", "general"]
    target_ontologies: List[Literal["umls", "snomed", "icd10", "loinc", "rxnorm"]]
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    rag_limit: int = Field(ge=5, le=100)
    expand_abbreviations: bool = True
    semantic_search: bool = True
    strategy_rationale: str

class EnhancedResult(BaseModel):
    concept_name: str
    concept_id: str
    definition: str
    ontology: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    source_method: Literal["api", "rag", "hybrid"] = "api"
    clinical_context: str
    usage_notes: str
    related_concepts: List[str] = []
    metadata: Dict[str, Any] = {}

class ComprehensiveSearchResult(BaseModel):
    query: str
    search_strategy: SearchStrategyAnalysis
    api_results: List[EnhancedResult]
    rag_results: List[EnhancedResult]
    synthesis_notes: str
    coverage_analysis: str
    recommended_actions: List[str]
    result_quality_score: float = Field(ge=0.0, le=1.0)
    total_concepts_found: int
    processing_metadata: Dict[str, Any] = {}

# =================== PYDANTIC AI AGENTS ===================

class QueryStrategyAgent:
    """Agent for determining optimal search strategy for medical concepts"""
    
    def __init__(self, deps: ResearchTerminologyDeps):
        self.deps = deps
        self.system_prompt = """# MEDICAL CONCEPT SEARCH STRATEGY OPTIMIZER

## ROLE
You are a medical concept search optimization specialist who determines the best search strategy for finding comprehensive medical terminology information. Your goal is to route queries to the most effective combination of API searches and semantic discovery methods.

## INPUT
You receive medical concept queries containing:
- Clinical terms, conditions, procedures, medications, or observations
- User intent (comprehensive search, quick lookup, discovery research)
- Available data sources (APIs: UMLS, SNOMED, ICD-10, LOINC, RxNorm; RAG: BioBERT semantic database)
- Quality requirements and time constraints

## STEPS
1. **Analyze Query Type**: Identify concept type and complexity (single term vs multi-concept)
2. **Determine Search Mode**: Choose api_only (fast, authoritative), hybrid (comprehensive), or discovery_focused (exploratory)
3. **Select Target Ontologies**: Prioritize relevant APIs based on concept type
4. **Set Search Parameters**: Configure confidence thresholds, result limits, and search options
5. **Plan Enhancement Strategy**: Determine if semantic RAG search will add value
6. **Generate Strategy Rationale**: Document reasoning for approach selection

## EXPECTATION
Output a structured SearchStrategyAnalysis containing:
- Optimal search_mode selection (api_only/hybrid/discovery_focused)
- Accurate concept_type identification
- Prioritized target_ontologies list
- Appropriate confidence_threshold (0.3-0.8 range)
- Suitable rag_limit (5-50 for efficiency)
- Boolean flags for expand_abbreviations and semantic_search
- Clear strategy_rationale explaining the approach

## NARROWING
**Search Mode Guidelines:**
- api_only: Single concept, need authoritative codes quickly
- hybrid: Complex concepts, want comprehensive coverage with semantic discovery
- discovery_focused: Exploratory research, unknown terminology, broad concept families

**Ontology Selection:**
- umls: General medical concepts, broad coverage
- snomed: Clinical conditions and procedures
- icd10: Billing/administrative codes, standardized diagnoses
- loinc: Laboratory tests and observations
- rxnorm: Medications and pharmaceutical concepts

**Parameter Guidelines:**
- confidence_threshold: 0.3-0.5 for discovery, 0.6-0.8 for precision
- rag_limit: 5-15 for speed, 20-50 for comprehensiveness
- Always enable expand_abbreviations for medical queries
- Enable semantic_search for hybrid and discovery modes

**Optimization Focus:**
- Balance speed vs comprehensiveness based on user intent
- Minimize API calls while maximizing result quality
- Leverage semantic RAG for concept discovery when beneficial
- Ensure scalable approach for batch processing"""

    async def determine_strategy(self, query: str, user_intent: str = "comprehensive", concept_type: str = None) -> SearchStrategyAnalysis:
        """Determine optimal search strategy for medical concept query"""
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""
            Determine the optimal search strategy for: "{query}"
            
            Context:
            - User Intent: {user_intent} (quick lookup, comprehensive search, discovery research)
            - Concept Type Hint: {concept_type or 'Auto-detect'}
            
            Analyze and determine:
            1. Best search_mode: api_only (fast), hybrid (comprehensive), discovery_focused (exploratory)
            2. Concept type: condition, procedure, medication, laboratory, observation, device, general
            3. Target ontologies: umls, snomed, icd10, loinc, rxnorm (prioritized list)
            4. Confidence threshold: 0.3-0.8 based on precision needs
            5. RAG limit: 5-50 based on comprehensiveness needs
            6. Enable abbreviation expansion and semantic search
            7. Strategy rationale explaining the approach
            
            Respond in JSON format matching the SearchStrategyAnalysis schema.
            """}
        ]
        
        try:
            client = get_openai_client()
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            result_data = json.loads(response.choices[0].message.content)
            return SearchStrategyAnalysis(**result_data)
            
        except Exception as e:
            logger.error(f"Strategy analysis failed: {e}")
            # Return sensible default strategy
            return SearchStrategyAnalysis(
                search_mode="hybrid",
                concept_type=concept_type or "general",
                target_ontologies=["umls", "snomed", "icd10"],
                confidence_threshold=0.5,
                rag_limit=20,
                expand_abbreviations=True,
                semantic_search=True,
                strategy_rationale=f"Default hybrid strategy for query: {query}"
            )

class ResultsEnhancementAgent:
    """Agent for enhancing search results with clinical context and metadata"""
    
    def __init__(self, deps: ResearchTerminologyDeps):
        self.deps = deps
        self.system_prompt = """# MEDICAL CONCEPT RESULTS ENHANCEMENT SPECIALIST

## ROLE
You are a medical concept enhancement specialist who adds clinical context, usage guidance, and professional metadata to search results. Your goal is to make medical terminology results more useful and actionable for healthcare professionals and researchers.

## INPUT
You receive:
- Raw search results from medical APIs (UMLS, SNOMED, ICD-10, LOINC, RxNorm)
- RAG semantic search results from BioBERT database
- Original query context and search strategy
- User requirements for clinical context and usage guidance

## STEPS
1. **Normalize Result Formats**: Standardize field names and structure across different APIs
2. **Add Clinical Context**: Provide clinical usage notes, contraindications, and relevant scenarios
3. **Generate Usage Guidance**: Explain when and how to use each concept in practice
4. **Identify Related Concepts**: Suggest hierarchically related or semantically similar terms
5. **Calculate Enhancement Scores**: Assess result quality and clinical relevance
6. **Add Professional Metadata**: Include source attribution, confidence levels, and validation status
7. **Format for Export**: Prepare results for professional HTML/PDF documentation

## EXPECTATION
Output an array of EnhancedResult objects containing:
- concept_name: Clear, professional concept name
- concept_id: Primary identifier/code
- definition: Comprehensive clinical definition
- ontology: Source terminology system
- confidence_score: Result quality score (0.0-1.0)
- source_method: Origin tracking (api/rag/hybrid)
- clinical_context: Practical usage scenarios and clinical relevance
- usage_notes: Professional guidance for implementation
- related_concepts: List of semantically related terms
- metadata: Additional technical and source information

## NARROWING
**Enhancement Quality Standards:**
- All results must have meaningful clinical context
- Usage notes should be practical and actionable
- Confidence scores must reflect both technical accuracy and clinical relevance
- Related concepts should be genuinely useful, not just semantically similar

**Clinical Context Requirements:**
- Include typical use cases and scenarios
- Note any important contraindications or limitations
- Explain relationship to common clinical workflows
- Provide guidance for documentation and coding

**Professional Standards:**
- Use clear, non-technical language where possible
- Maintain clinical accuracy and precision
- Include appropriate disclaimers for clinical use
- Ensure results are suitable for professional documentation"""

    async def enhance_results(self, query: str, api_results: List[Dict], rag_results: List[Dict], strategy: SearchStrategyAnalysis) -> List[EnhancedResult]:
        """Enhance search results with clinical context and professional metadata"""
        
        # Combine and limit results for processing
        combined_results = []
        combined_results.extend(api_results[:15])  # Top 15 API results
        combined_results.extend(rag_results[:10])  # Top 10 RAG results
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""
            Enhance these medical concept search results for query: "{query}"
            
            Search Results: {json.dumps(combined_results, indent=2)}
            
            Search Strategy Context:
            - Search Mode: {strategy.search_mode}
            - Concept Type: {strategy.concept_type}
            - Target Ontologies: {strategy.target_ontologies}
            
            For each relevant result, provide enhanced information:
            1. Clear concept name and primary identifier
            2. Comprehensive clinical definition
            3. Source ontology/terminology
            4. Quality confidence score
            5. Source method (api/rag/hybrid)
            6. Clinical context and usage scenarios
            7. Professional usage notes and guidance
            8. Related concepts that might be relevant
            9. Technical metadata and source information
            
            Focus on the most clinically relevant and high-quality results.
            Return a JSON array of EnhancedResult objects.
            """}
        ]
        
        try:
            client = get_openai_client()
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result_data = json.loads(response.choices[0].message.content)
            enhanced_results = []
            
            for result_data_item in result_data.get('enhanced_results', []):
                enhanced_results.append(EnhancedResult(**result_data_item))
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Results enhancement failed: {e}")
            # Return basic enhanced results as fallback
            fallback_results = []
            for i, result in enumerate(combined_results[:10]):
                try:
                    enhanced = EnhancedResult(
                        concept_name=result.get('concept_name', result.get('concept', f'Concept {i+1}')),
                        concept_id=result.get('concept_id', result.get('code', f'ID_{i+1}')),
                        definition=result.get('definition', result.get('description', 'No definition available')),
                        ontology=result.get('ontology', result.get('source', result.get('code_system', 'Unknown'))),
                        confidence_score=result.get('confidence_score', result.get('score', 0.5)),
                        source_method="api" if 'is_rag_result' not in result else "rag",
                        clinical_context="Professional medical context - consult current clinical guidelines",
                        usage_notes="Verify current clinical standards and institutional protocols",
                        related_concepts=[],
                        metadata={"source": "fallback_enhancement", "original_result": result}
                    )
                    fallback_results.append(enhanced)
                except Exception as inner_e:
                    logger.error(f"Failed to create fallback result {i}: {inner_e}")
                    continue
            
            return fallback_results

class ComprehensiveSynthesisAgent:
    """Agent for synthesizing comprehensive search results and generating final output"""
    
    def __init__(self, deps: ResearchTerminologyDeps):
        self.deps = deps
        self.system_prompt = """# MEDICAL CONCEPT SYNTHESIS AND DOCUMENTATION SPECIALIST

## ROLE
You are a medical concept synthesis specialist who creates comprehensive, professional documentation from medical terminology search results. Your expertise focuses on combining API and semantic search results into actionable, well-organized information suitable for healthcare professionals and researchers.

## INPUT
You receive:
- Enhanced results from API searches across multiple medical ontologies
- Enhanced results from semantic RAG discovery searches
- Original search query and strategy analysis
- User requirements for output format and depth

## STEPS
1. **Analyze Result Coverage**: Review API and RAG results for comprehensive concept coverage
2. **Identify Knowledge Gaps**: Determine if important related concepts are missing
3. **Organize by Relevance**: Prioritize results by clinical importance and search relevance
4. **Synthesize Key Findings**: Create coherent narrative connecting related concepts
5. **Assess Overall Quality**: Evaluate completeness and clinical utility of result set
6. **Generate Recommendations**: Suggest additional searches or related concepts to explore
7. **Format Professional Output**: Create well-structured documentation suitable for clinical use
8. **Add Quality Metadata**: Include search statistics and confidence assessments

## EXPECTATION
Output a ComprehensiveSearchResult object containing:
- query: Original search query
- search_strategy: Strategy analysis used
- api_results: Top enhanced API results
- rag_results: Top enhanced RAG results  
- synthesis_notes: Narrative summary of findings and key insights
- coverage_analysis: Assessment of concept coverage and any gaps
- recommended_actions: Suggestions for follow-up searches or related concepts
- result_quality_score: Overall assessment of search success (0.0-1.0)
- total_concepts_found: Count of unique concepts discovered
- processing_metadata: Technical details about search execution

## NARROWING
**Synthesis Quality Standards:**
- Focus on clinically actionable and relevant concepts
- Prioritize high-confidence results but include promising discoveries
- Ensure narrative coherence in synthesis notes
- Provide practical recommendations for further research

**Coverage Assessment Requirements:**
- Identify major concept families represented
- Note any obvious gaps in terminology coverage
- Assess balance between authoritative API results and discovery RAG results
- Consider clinical workflow relevance

**Professional Documentation Standards:**
- Use clear, professional language suitable for healthcare settings
- Structure information logically (most relevant first)
- Include appropriate caveats about clinical use
- Provide actionable next steps and recommendations

**Quality Metrics:**
- Result quality score should reflect both quantity and clinical relevance
- Consider diversity of sources and concept types
- Account for user intent (quick lookup vs comprehensive research)
- Factor in completeness of concept coverage for the query domain"""

    async def synthesize_comprehensive_results(
        self, 
        query: str, 
        strategy: SearchStrategyAnalysis,
        api_enhanced_results: List[EnhancedResult], 
        rag_enhanced_results: List[EnhancedResult]
    ) -> ComprehensiveSearchResult:
        """Synthesize comprehensive search results with professional analysis"""
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""
            Synthesize comprehensive results for medical concept search: "{query}"
            
            Search Strategy Used:
            - Mode: {strategy.search_mode}
            - Concept Type: {strategy.concept_type}
            - Target Ontologies: {strategy.target_ontologies}
            
            API Enhanced Results: {json.dumps([r.dict() for r in api_enhanced_results[:10]], indent=2)}
            
            RAG Enhanced Results: {json.dumps([r.dict() for r in rag_enhanced_results[:10]], indent=2)}
            
            Create a comprehensive synthesis including:
            1. Overall assessment of concept coverage and quality
            2. Key insights and clinical relevance of findings
            3. Analysis of coverage gaps or areas for further exploration
            4. Recommended follow-up actions or related searches
            5. Quality score reflecting search success and clinical utility
            6. Professional summary suitable for healthcare documentation
            
            Return a ComprehensiveSearchResult JSON object.
            """}
        ]
        
        try:
            client = get_openai_client()
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result_data = json.loads(response.choices[0].message.content)
            
            # Ensure required fields are present
            result_data.setdefault('query', query)
            result_data.setdefault('search_strategy', strategy.dict())
            result_data.setdefault('api_results', [r.dict() for r in api_enhanced_results[:10]])
            result_data.setdefault('rag_results', [r.dict() for r in rag_enhanced_results[:10]])
            result_data.setdefault('total_concepts_found', len(api_enhanced_results) + len(rag_enhanced_results))
            
            return ComprehensiveSearchResult(**result_data)
            
        except Exception as e:
            logger.error(f"Comprehensive synthesis failed: {e}")
            # Return basic synthesis as fallback
            total_results = len(api_enhanced_results) + len(rag_enhanced_results)
            
            return ComprehensiveSearchResult(
                query=query,
                search_strategy=strategy,
                api_results=[r.dict() for r in api_enhanced_results[:10]],
                rag_results=[r.dict() for r in rag_enhanced_results[:10]],
                synthesis_notes=f"Search completed for '{query}'. Found {len(api_enhanced_results)} API results and {len(rag_enhanced_results)} semantic discovery results. Results cover key medical concepts related to the query.",
                coverage_analysis=f"Coverage includes {len(set(r.ontology for r in api_enhanced_results + rag_enhanced_results))} different ontologies. Represents comprehensive medical terminology coverage.",
                recommended_actions=["Review top results for clinical relevance", "Consider related concept searches if needed", "Verify current clinical guidelines"],
                result_quality_score=min(0.8, total_results / 20.0),  # Basic quality heuristic
                total_concepts_found=total_results,
                processing_metadata={
                    "synthesis_method": "fallback",
                    "api_results_count": len(api_enhanced_results),
                    "rag_results_count": len(rag_enhanced_results),
                    "search_mode": strategy.search_mode
                }
            )

# Note: ResearchCodeSetBuilder removed - functionality integrated into ComprehensiveSynthesisAgent

# =================== AGENT FACTORY ===================

def create_research_agents(deps: ResearchTerminologyDeps):
    """Create streamlined research agents focused on practical medical concept search"""
    return {
        'strategy_agent': QueryStrategyAgent(deps),
        'enhancement_agent': ResultsEnhancementAgent(deps),
        'synthesis_agent': ComprehensiveSynthesisAgent(deps)
    }