# LangGraph Workflow for Medical Terminology Research
import asyncio
import json
from typing import TypedDict, Dict, List, Any, Optional
from datetime import datetime
import logging

# LangGraph imports (we'll simulate the structure since LangGraph may not be available)
try:
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    # Fallback implementation
    LANGGRAPH_AVAILABLE = False
    logging.warning("LangGraph not available, using fallback implementation")

from ..agents.research_agents import (
    SearchStrategyAnalysis, 
    EnhancedResult, 
    ComprehensiveSearchResult,
    ResearchTerminologyDeps,
    create_research_agents
)

logger = logging.getLogger(__name__)

# =================== STATE DEFINITION ===================

class MedicalSearchState(TypedDict):
    query: str
    user_intent: Optional[str]
    concept_type: Optional[str]
    
    # Strategy analysis
    strategy: Optional[SearchStrategyAnalysis]
    
    # Raw search results
    api_search_results: List[Dict]
    rag_search_results: List[Dict]
    
    # Enhanced results
    api_enhanced_results: List[EnhancedResult]
    rag_enhanced_results: List[EnhancedResult]
    
    # Final synthesis
    comprehensive_result: Optional[ComprehensiveSearchResult]
    
    # Processing metadata
    metadata: Dict[str, Any]
    errors: List[str]

# =================== WORKFLOW NODES ===================

async def determine_search_strategy(state: MedicalSearchState, agents: Dict) -> Dict:
    """Determine optimal search strategy for the medical query"""
    
    try:
        strategy_agent = agents['strategy_agent']
        strategy = await strategy_agent.determine_strategy(
            query=state['query'],
            user_intent=state.get('user_intent', 'comprehensive'),
            concept_type=state.get('concept_type')
        )
        
        return {
            "strategy": strategy,
            "metadata": {
                **state.get("metadata", {}),
                "strategy_determined": datetime.utcnow().isoformat(),
                "search_mode": strategy.search_mode,
                "target_ontologies": strategy.target_ontologies
            }
        }
        
    except Exception as e:
        logger.error(f"Strategy determination failed: {e}")
        return {
            "errors": state.get("errors", []) + [f"Strategy determination failed: {str(e)}"]
        }

async def execute_medical_searches(state: MedicalSearchState, agents: Dict, medical_client, hybrid_engine) -> Dict:
    """Execute medical concept searches using API and RAG methods"""
    
    try:
        query = state['query']
        strategy = state['strategy']
        
        api_results = []
        rag_results = []
        
        # Execute searches based on strategy
        if strategy.search_mode in ['api_only', 'hybrid']:
            # API searches across target ontologies
            for ontology in strategy.target_ontologies:
                try:
                    if ontology == 'umls':
                        results = await medical_client.search_umls(query)
                    elif ontology == 'snomed':
                        results = await medical_client.search_snomed(query)
                    elif ontology == 'icd10':
                        results = await medical_client.search_icd10(query)
                    elif ontology == 'loinc':
                        results = await medical_client.search_loinc(query)
                    elif ontology == 'rxnorm':
                        results = await medical_client.search_rxnorm(query)
                    else:
                        results = []
                    
                    # Add ontology source to each result
                    for result in results:
                        if hasattr(result, '__dict__'):
                            result.source_ontology = ontology
                        elif isinstance(result, dict):
                            result['source_ontology'] = ontology
                    
                    api_results.extend(results)
                    
                except Exception as e:
                    logger.error(f"API search failed for {ontology}: {e}")
                    continue
        
        # RAG search for hybrid and discovery modes
        if strategy.search_mode in ['hybrid', 'discovery_focused']:
            try:
                # Use BioBERT RAG engine directly
                if hybrid_engine and hybrid_engine.biobert_engine:
                    raw_rag_results = await hybrid_engine.biobert_engine.search(
                        query=query,
                        limit=strategy.rag_limit
                    )
                    
                    # Normalize RAG results
                    for result in raw_rag_results:
                        normalized = {
                            'concept_name': result.get('concept', result.get('concept_name', '')),
                            'concept_id': result.get('concept_id', result.get('code', '')),
                            'definition': result.get('definition', result.get('description', result.get('text', ''))),
                            'ontology': result.get('ontology', result.get('source', 'RAG')),
                            'confidence_score': result.get('score', result.get('confidence_score', 0.5)),
                            'source_method': 'rag',
                            'is_rag_result': True
                        }
                        rag_results.append(normalized)
                        
            except Exception as e:
                logger.error(f"RAG search failed: {e}")
        
        return {
            "api_search_results": api_results,
            "rag_search_results": rag_results,
            "metadata": {
                **state.get("metadata", {}),
                "searches_completed": datetime.utcnow().isoformat(),
                "api_results_count": len(api_results),
                "rag_results_count": len(rag_results)
            }
        }
        
    except Exception as e:
        logger.error(f"Medical searches failed: {e}")
        return {
            "errors": state.get("errors", []) + [f"Medical searches failed: {str(e)}"]
        }

async def enhance_search_results(state: MedicalSearchState, agents: Dict) -> Dict:
    """Enhance search results with clinical context and metadata"""
    
    try:
        enhancement_agent = agents['enhancement_agent']
        
        api_results = state.get('api_search_results', [])
        rag_results = state.get('rag_search_results', [])
        strategy = state['strategy']
        
        # Convert objects to dicts for API processing
        api_dicts = []
        for result in api_results:
            if hasattr(result, '__dict__'):
                api_dicts.append(result.__dict__)
            else:
                api_dicts.append(result)
        
        enhanced_results = await enhancement_agent.enhance_results(
            query=state['query'],
            api_results=api_dicts,
            rag_results=rag_results,
            strategy=strategy
        )
        
        # Separate enhanced results by source
        api_enhanced = [r for r in enhanced_results if r.source_method in ['api', 'hybrid']]
        rag_enhanced = [r for r in enhanced_results if r.source_method == 'rag']
        
        return {
            "api_enhanced_results": api_enhanced,
            "rag_enhanced_results": rag_enhanced,
            "metadata": {
                **state.get("metadata", {}),
                "enhancement_completed": datetime.utcnow().isoformat(),
                "api_enhanced_count": len(api_enhanced),
                "rag_enhanced_count": len(rag_enhanced)
            }
        }
        
    except Exception as e:
        logger.error(f"Results enhancement failed: {e}")
        return {
            "errors": state.get("errors", []) + [f"Enhancement failed: {str(e)}"]
        }

async def synthesize_comprehensive_results(state: MedicalSearchState, agents: Dict) -> Dict:
    """Synthesize enhanced results into comprehensive final output"""
    
    try:
        synthesis_agent = agents['synthesis_agent']
        
        api_enhanced = state.get('api_enhanced_results', [])
        rag_enhanced = state.get('rag_enhanced_results', [])
        strategy = state['strategy']
        
        comprehensive_result = await synthesis_agent.synthesize_comprehensive_results(
            query=state['query'],
            strategy=strategy,
            api_enhanced_results=api_enhanced,
            rag_enhanced_results=rag_enhanced
        )
        
        return {
            "comprehensive_result": comprehensive_result,
            "metadata": {
                **state.get("metadata", {}),
                "synthesis_completed": datetime.utcnow().isoformat(),
                "final_quality_score": comprehensive_result.result_quality_score
            }
        }
        
    except Exception as e:
        logger.error(f"Comprehensive synthesis failed: {e}")
        return {
            "errors": state.get("errors", []) + [f"Synthesis failed: {str(e)}"]
        }

async def format_professional_output(state: MedicalSearchState) -> Dict:
    """Format comprehensive results for professional documentation"""
    
    try:
        comprehensive_result = state.get('comprehensive_result')
        strategy = state.get('strategy')
        
        if not comprehensive_result:
            return {
                "formatted_output": "No comprehensive results available",
                "errors": state.get("errors", []) + ["No comprehensive results to format"]
            }
        
        # Generate professional markdown output
        output = f"""# Medical Concept Search Results
## Query: {state['query']}
## Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}

### Search Strategy
- **Search Mode**: {strategy.search_mode if strategy else 'Not specified'}
- **Concept Type**: {strategy.concept_type if strategy else 'Not specified'}
- **Target Ontologies**: {', '.join(strategy.target_ontologies) if strategy else 'Not specified'}
- **Quality Score**: {comprehensive_result.result_quality_score:.2f}/1.0

### Summary
{comprehensive_result.synthesis_notes}

### Coverage Analysis
{comprehensive_result.coverage_analysis}

### API Results ({len(comprehensive_result.api_results)} found)
"""
        
        # Add top API results
        for i, result in enumerate(comprehensive_result.api_results[:10], 1):
            if isinstance(result, dict):
                concept_name = result.get('concept_name', 'Unknown')
                concept_id = result.get('concept_id', 'N/A')
                ontology = result.get('ontology', 'Unknown')
                confidence = result.get('confidence_score', 0.0)
                
                output += f"\n**{i}. {concept_name}**\n"
                output += f"- Code: `{concept_id}`\n"
                output += f"- Source: {ontology}\n"
                output += f"- Confidence: {confidence:.2f}\n"
        
        # Add RAG results if available
        if comprehensive_result.rag_results:
            output += f"\n### Semantic Discovery Results ({len(comprehensive_result.rag_results)} found)\n"
            
            for i, result in enumerate(comprehensive_result.rag_results[:5], 1):
                if isinstance(result, dict):
                    concept_name = result.get('concept_name', 'Unknown')
                    definition = result.get('definition', 'No definition available')
                    confidence = result.get('confidence_score', 0.0)
                    
                    output += f"\n**{i}. {concept_name}**\n"
                    output += f"- Definition: {definition[:200]}...\n"
                    output += f"- Confidence: {confidence:.2f}\n"
        
        # Add recommendations
        if comprehensive_result.recommended_actions:
            output += "\n### Recommended Actions\n"
            for action in comprehensive_result.recommended_actions:
                output += f"- {action}\n"
        
        # Add metadata
        metadata = state.get('metadata', {})
        if metadata:
            output += "\n### Processing Details\n"
            for key, value in metadata.items():
                if not key.endswith('_completed'):  # Skip timestamp details
                    output += f"- {key}: {value}\n"
        
        return {
            "formatted_output": output,
            "comprehensive_result": comprehensive_result,
            "metadata": {
                **metadata,
                "output_formatted": datetime.utcnow().isoformat(),
                "total_concepts": comprehensive_result.total_concepts_found
            }
        }
        
    except Exception as e:
        logger.error(f"Output formatting failed: {e}")
        return {
            "formatted_output": f"Error formatting output: {str(e)}",
            "errors": state.get("errors", []) + [f"Output formatting failed: {str(e)}"]
        }

# =================== WORKFLOW ORCHESTRATOR ===================

class StreamlinedMedicalWorkflow:
    """Streamlined workflow orchestrator for medical concept search and analysis"""
    
    def __init__(self, deps: ResearchTerminologyDeps, medical_client, hybrid_engine=None):
        self.deps = deps
        self.medical_client = medical_client
        self.hybrid_engine = hybrid_engine
        self.agents = create_research_agents(deps)
    
    async def execute_workflow(self, initial_state: MedicalSearchState) -> MedicalSearchState:
        """Execute the streamlined medical search workflow"""
        
        state = initial_state.copy()
        state['errors'] = []
        state['metadata'] = {'workflow_started': datetime.utcnow().isoformat()}
        
        try:
            # Step 1: Determine search strategy
            logger.info("Step 1: Determining search strategy")
            strategy_result = await determine_search_strategy(state, self.agents)
            state.update(strategy_result)
            
            if state.get('errors'):
                return state
            
            # Step 2: Execute medical searches
            logger.info("Step 2: Executing medical searches")
            search_result = await execute_medical_searches(state, self.agents, self.medical_client, self.hybrid_engine)
            state.update(search_result)
            
            if state.get('errors'):
                return state
            
            # Step 3: Enhance search results
            logger.info("Step 3: Enhancing search results")
            enhancement_result = await enhance_search_results(state, self.agents)
            state.update(enhancement_result)
            
            if state.get('errors'):
                return state
            
            # Step 4: Synthesize comprehensive results
            logger.info("Step 4: Synthesizing comprehensive results")
            synthesis_result = await synthesize_comprehensive_results(state, self.agents)
            state.update(synthesis_result)
            
            if state.get('errors'):
                return state
            
            # Step 5: Format professional output
            logger.info("Step 5: Formatting professional output")
            output_result = await format_professional_output(state)
            state.update(output_result)
            
            state['metadata']['workflow_completed'] = datetime.utcnow().isoformat()
            
            return state
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            state['errors'] = state.get('errors', []) + [f"Workflow failed: {str(e)}"]
            state['formatted_output'] = f"Workflow execution failed: {str(e)}"
            return state

# =================== WORKFLOW FACTORY ===================

def create_streamlined_medical_workflow(deps: ResearchTerminologyDeps, medical_client, hybrid_engine=None):
    """Create the streamlined medical concept search workflow"""
    return StreamlinedMedicalWorkflow(deps, medical_client, hybrid_engine)

# =================== SIMPLIFIED API INTERFACE ===================

async def execute_medical_concept_search(
    query: str,
    user_intent: str = "comprehensive",
    concept_type: str = None,
    deps: ResearchTerminologyDeps = None,
    medical_client = None,
    hybrid_engine = None
) -> Dict[str, Any]:
    """Execute a streamlined medical concept search query"""
    
    # Create initial state
    initial_state = MedicalSearchState(
        query=query,
        user_intent=user_intent,
        concept_type=concept_type,
        strategy=None,
        api_search_results=[],
        rag_search_results=[],
        api_enhanced_results=[],
        rag_enhanced_results=[],
        comprehensive_result=None,
        metadata={},
        errors=[]
    )
    
    # Create and execute workflow
    workflow = create_streamlined_medical_workflow(deps, medical_client, hybrid_engine)
    result_state = await workflow.execute_workflow(initial_state)
    
    # Convert to API response format
    comprehensive_result = result_state.get('comprehensive_result')
    
    return {
        "query": result_state['query'],
        "search_strategy": result_state.get('strategy').dict() if result_state.get('strategy') else None,
        "api_results": comprehensive_result.api_results if comprehensive_result else [],
        "rag_results": comprehensive_result.rag_results if comprehensive_result else [],
        "synthesis_notes": comprehensive_result.synthesis_notes if comprehensive_result else "",
        "coverage_analysis": comprehensive_result.coverage_analysis if comprehensive_result else "",
        "recommended_actions": comprehensive_result.recommended_actions if comprehensive_result else [],
        "quality_score": comprehensive_result.result_quality_score if comprehensive_result else 0.0,
        "total_concepts_found": comprehensive_result.total_concepts_found if comprehensive_result else 0,
        "formatted_output": result_state.get('formatted_output', ''),
        "metadata": result_state.get('metadata', {}),
        "errors": result_state.get('errors', []),
        "success": len(result_state.get('errors', [])) == 0
    }