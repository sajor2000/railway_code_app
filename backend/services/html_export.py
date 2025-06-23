"""HTML export service for medical terminology results."""

from typing import List, Dict, Any, Optional
from datetime import datetime
from jinja2 import Environment, Template
import html

# Remove unused import for now


class HTMLExportService:
    """Service for exporting medical terminology results to beautiful HTML."""
    
    def __init__(self):
        self.template = self._create_template()
    
    def _create_template(self) -> Template:
        """Create Jinja2 template for HTML export."""
        template_str = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Terminology Results - {{ query }}</title>
    <style>
        :root {
            --primary-color: #2c3e50;
            --secondary-color: #3498db;
            --success-color: #27ae60;
            --warning-color: #f39c12;
            --danger-color: #e74c3c;
            --light-bg: #f8f9fa;
            --white: #ffffff;
            --border-color: #dee2e6;
            --text-muted: #6c757d;
            --shadow: 0 2px 4px rgba(0,0,0,0.1);
            --shadow-lg: 0 4px 15px rgba(0,0,0,0.15);
        }
        
        * {
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
            line-height: 1.6;
            color: var(--primary-color);
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
        }
        .header {
            background: var(--white);
            padding: 40px;
            border-radius: 15px;
            box-shadow: var(--shadow-lg);
            margin-bottom: 30px;
            position: relative;
            overflow: hidden;
        }
        
        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--secondary-color), var(--success-color));
        }
        
        h1 {
            color: var(--primary-color);
            margin: 0 0 15px 0;
            font-size: 32px;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        h1::before {
            content: '🏥';
            font-size: 28px;
        }
        .query-info {
            color: var(--text-muted);
            font-size: 15px;
            line-height: 1.8;
            background: var(--light-bg);
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }
        
        .query-info-item {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        
        .query-info strong {
            color: var(--primary-color);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 12px;
            letter-spacing: 0.5px;
        }
        
        .query-value {
            color: var(--primary-color);
            font-weight: 500;
            font-size: 14px;
        }
        .research-context {
            background: #ecf0f1;
            padding: 15px;
            border-radius: 6px;
            margin-top: 15px;
            font-size: 14px;
            color: #2c3e50;
        }
        .results-summary {
            background: var(--white);
            padding: 30px;
            border-radius: 15px;
            box-shadow: var(--shadow-lg);
            margin-bottom: 30px;
            position: relative;
        }
        
        .results-summary h2 {
            color: var(--primary-color);
            margin: 0 0 25px 0;
            font-size: 24px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .results-summary h2::before {
            content: '📊';
            font-size: 20px;
        }
        
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px;
        }
        
        .summary-item {
            text-align: center;
            padding: 25px 20px;
            background: linear-gradient(135deg, var(--light-bg) 0%, #e9ecef 100%);
            border-radius: 12px;
            border: 1px solid var(--border-color);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            position: relative;
            overflow: hidden;
        }
        
        .summary-item::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: var(--secondary-color);
        }
        
        .summary-item:hover {
            transform: translateY(-3px);
            box-shadow: var(--shadow-lg);
        }
        
        .summary-number {
            font-size: 32px;
            font-weight: 800;
            color: var(--secondary-color);
            margin-bottom: 8px;
            line-height: 1;
        }
        
        .summary-label {
            font-size: 13px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
        }
        .results-container {
            display: grid;
            gap: 20px;
        }
        .result-card {
            background: var(--white);
            padding: 30px;
            border-radius: 15px;
            box-shadow: var(--shadow);
            transition: all 0.3s ease;
            border: 1px solid var(--border-color);
            position: relative;
            overflow: hidden;
        }
        
        .result-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--secondary-color), var(--success-color));
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        
        .result-card:hover {
            transform: translateY(-5px);
            box-shadow: var(--shadow-lg);
            border-color: var(--secondary-color);
        }
        
        .result-card:hover::before {
            opacity: 1;
        }
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 15px;
        }
        .result-title {
            font-size: 20px;
            font-weight: 600;
            color: #2c3e50;
            margin: 0;
            flex: 1;
        }
        .badges {
            display: flex;
            gap: 8px;
            align-items: center;
        }
        .confidence-badge {
            background: linear-gradient(135deg, #d4edda, #c3e6cb);
            color: var(--success-color);
            padding: 8px 16px;
            border-radius: 25px;
            font-size: 13px;
            font-weight: 600;
            border: 1px solid #c3e6cb;
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        .confidence-badge::before {
            content: '✓';
            font-size: 11px;
        }
        
        .confidence-low {
            background: linear-gradient(135deg, #fff3cd, #ffeaa7);
            color: var(--warning-color);
            border-color: #ffeaa7;
        }
        
        .confidence-low::before {
            content: '⚠';
        }
        
        .confidence-medium {
            background: linear-gradient(135deg, #cfe2ff, #b3d7ff);
            color: var(--secondary-color);
            border-color: #b3d7ff;
        }
        
        .confidence-medium::before {
            content: '~';
        }
        .validation-badge {
            background: linear-gradient(135deg, #e1f5fe, #b3e5fc);
            color: #01579b;
            padding: 8px 16px;
            border-radius: 25px;
            font-size: 13px;
            font-weight: 600;
            border: 1px solid #b3e5fc;
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        .validation-badge::before {
            content: '🛡';
            font-size: 11px;
        }
        .code-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }
        .info-item {
            background: #f8f9fa;
            padding: 10px 15px;
            border-radius: 6px;
            border-left: 3px solid #3498db;
        }
        .info-label {
            font-size: 12px;
            color: #7f8c8d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .info-value {
            font-weight: 600;
            color: #2c3e50;
            margin-top: 2px;
            font-family: 'Monaco', 'Consolas', monospace;
        }
        .research-notes {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 15px;
            border-radius: 6px;
            margin: 15px 0;
            font-size: 14px;
            color: #856404;
        }
        .research-notes strong {
            color: #704700;
        }
        .hierarchy {
            background: #e3f2fd;
            padding: 10px 15px;
            border-radius: 6px;
            margin: 10px 0;
            font-size: 14px;
            color: #1565c0;
        }
        .hierarchy-path {
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 13px;
        }
        .definition {
            color: #555;
            font-style: italic;
            margin: 10px 0;
            line-height: 1.5;
            padding: 10px;
            background: #fafafa;
            border-radius: 6px;
        }
        .synonyms {
            margin: 10px 0;
        }
        .synonym-tag {
            display: inline-block;
            background: #e0e0e0;
            padding: 3px 10px;
            margin: 2px;
            border-radius: 15px;
            font-size: 12px;
        }
        .relationships {
            margin: 15px 0;
            padding: 15px;
            background: #f0f8ff;
            border-radius: 6px;
        }
        .relationship-type {
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 5px;
        }
        .relationship-codes {
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 13px;
            color: #555;
        }
        .source-indicator {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            text-transform: uppercase;
            font-weight: 600;
        }
        .source-api {
            background: linear-gradient(135deg, #e8f5e9, #c3e6cb);
            color: var(--success-color);
            border: 1px solid #c3e6cb;
        }
        
        .source-api::before {
            content: '🔗 ';
            font-size: 10px;
        }
        
        .source-pinecone {
            background: linear-gradient(135deg, #f3e5f5, #e1bee7);
            color: #6a1b9a;
            border: 1px solid #e1bee7;
        }
        
        .source-pinecone::before {
            content: '🧬 ';
            font-size: 10px;
        }
        
        .source-hybrid {
            background: linear-gradient(135deg, #e3f2fd, #bbdefb);
            color: #1565c0;
            border: 1px solid #bbdefb;
        }
        
        .source-hybrid::before {
            content: '⚡ ';
            font-size: 10px;
        }
        .temporal-info {
            background: #f5f5f5;
            padding: 10px;
            border-radius: 6px;
            margin: 10px 0;
            font-size: 13px;
            color: #666;
        }
        .export-footer {
            margin-top: 40px;
            padding: 30px;
            background: var(--white);
            border-radius: 15px;
            box-shadow: var(--shadow-lg);
            text-align: center;
            font-size: 13px;
            color: var(--text-muted);
            border-top: 3px solid var(--secondary-color);
        }
        
        .medical-disclaimer {
            background: linear-gradient(135deg, #fff3cd, #ffeaa7);
            border: 1px solid #ffc107;
            border-radius: 12px;
            padding: 20px;
            margin: 20px 0;
            color: #856404;
            font-weight: 500;
            position: relative;
        }
        
        .medical-disclaimer::before {
            content: '⚠️';
            font-size: 20px;
            position: absolute;
            top: 20px;
            left: 20px;
        }
        
        .medical-disclaimer h3 {
            margin: 0 0 10px 30px;
            color: #704700;
            font-size: 16px;
        }
        
        .medical-disclaimer p {
            margin: 5px 0 0 30px;
            line-height: 1.6;
        }
        .citation {
            background: #ecf0f1;
            padding: 15px;
            border-radius: 6px;
            margin: 20px 0;
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 12px;
            color: #34495e;
        }
        @media print {
            body {
                background: white;
            }
            .result-card:hover {
                transform: none;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Medical Terminology Search Results</h1>
        <div class="query-info">
            <div class="query-info-item">
                <strong>Query</strong>
                <span class="query-value">{{ query }}</span>
            </div>
            <div class="query-info-item">
                <strong>Generated</strong>
                <span class="query-value">{{ timestamp }}</span>
            </div>
            <div class="query-info-item">
                <strong>Search Mode</strong>
                <span class="query-value">
                    {% if search_mode == 'hybrid' %}
                        🧬 API + RAG Hybrid
                    {% elif search_mode == 'api' %}
                        🔗 API Only
                    {% else %}
                        {{ search_mode | title }}
                    {% endif %}
                </span>
            </div>
            {% if target_systems %}
            <div class="query-info-item">
                <strong>Target Systems</strong>
                <span class="query-value">{{ target_systems | join(', ') }}</span>
            </div>
            {% endif %}
            {% if research_intent %}
            <div class="query-info-item">
                <strong>Research Intent</strong>
                <span class="query-value">{{ research_intent | title }}</span>
            </div>
            {% endif %}
        </div>
        {% if research_context %}
        <div class="research-context">
            <strong>Research Context:</strong> {{ research_context }}
        </div>
        {% endif %}
    </div>
    
    {% if summary %}
    <div class="results-summary">
        <h2>Results Summary</h2>
        <div class="summary-grid">
            <div class="summary-item">
                <div class="summary-number">{{ summary.total_results }}</div>
                <div class="summary-label">Total Results</div>
            </div>
            <div class="summary-item">
                <div class="summary-number">{{ summary.unique_systems }}</div>
                <div class="summary-label">Terminology Systems</div>
            </div>
            {% if search_mode == 'hybrid' %}
            <div class="summary-item">
                <div class="summary-number">{{ summary.api_results }}</div>
                <div class="summary-label">🔗 API Results</div>
            </div>
            <div class="summary-item">
                <div class="summary-number">{{ summary.semantic_results }}</div>
                <div class="summary-label">🧬 RAG Discoveries</div>
            </div>
            {% else %}
            <div class="summary-item">
                <div class="summary-number">{{ summary.api_results }}</div>
                <div class="summary-label">Authoritative Results</div>
            </div>
            <div class="summary-item">
                <div class="summary-number">{{ (summary.total_results / summary.unique_systems) | round | int }}</div>
                <div class="summary-label">Avg per System</div>
            </div>
            {% endif %}
        </div>
    </div>
    {% endif %}
    
    <div class="results-container">
        {% for result in results %}
        <div class="result-card">
            <div class="result-header">
                <h2 class="result-title">{{ result.display_name }}</h2>
                <div class="badges">
                    {% if result.confidence_score %}
                    <span class="confidence-badge {% if result.confidence_score < 0.7 %}confidence-low{% elif result.confidence_score < 0.85 %}confidence-medium{% endif %}">
                        {{ (result.confidence_score * 100) | round }}% match
                    </span>
                    {% endif %}
                    {% if result.api_validated %}
                    <span class="validation-badge">✓ Validated</span>
                    {% endif %}
                </div>
            </div>
            
            <div class="code-info">
                <div class="info-item">
                    <div class="info-label">Code</div>
                    <div class="info-value">{{ result.code }}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">System</div>
                    <div class="info-value">{{ result.system }}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Pattern</div>
                    <div class="info-value">{{ result.character_pattern.description }}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Source</div>
                    <div class="info-value">
                        <span class="source-indicator source-{{ result.search_method }}">{{ result.search_method }}</span>
                    </div>
                </div>
                {% if result.version %}
                <div class="info-item">
                    <div class="info-label">Version</div>
                    <div class="info-value">{{ result.version }}</div>
                </div>
                {% endif %}
            </div>
            
            {% if result.definition %}
            <div class="definition">
                {{ result.definition }}
            </div>
            {% endif %}
            
            {% if result.hierarchy %}
            <div class="hierarchy">
                <strong>Hierarchy:</strong> 
                <span class="hierarchy-path">{{ result.hierarchy | join(' → ') }}</span>
            </div>
            {% endif %}
            
            {% if result.research_notes %}
            <div class="research-notes">
                <strong>Research Notes:</strong> {{ result.research_notes }}
            </div>
            {% endif %}
            
            {% if result.temporal_validity %}
            <div class="temporal-info">
                <strong>Temporal Validity:</strong>
                {% if result.temporal_validity.valid_from %}
                Valid from: {{ result.temporal_validity.valid_from }}
                {% endif %}
                {% if result.temporal_validity.valid_to %}
                | Valid to: {{ result.temporal_validity.valid_to }}
                {% endif %}
                {% if result.temporal_validity.replaced_by %}
                | Replaced by: {{ result.temporal_validity.replaced_by }}
                {% endif %}
            </div>
            {% endif %}
            
            {% if result.synonyms %}
            <div class="synonyms">
                <strong>Synonyms:</strong>
                {% for synonym in result.synonyms %}
                <span class="synonym-tag">{{ synonym }}</span>
                {% endfor %}
            </div>
            {% endif %}
            
            {% if result.relationships %}
            <div class="relationships">
                {% for rel_type, codes in result.relationships.items() %}
                <div style="margin-bottom: 10px;">
                    <div class="relationship-type">{{ rel_type }}:</div>
                    <div class="relationship-codes">{{ codes | join(', ') }}</div>
                </div>
                {% endfor %}
            </div>
            {% endif %}
        </div>
        {% endfor %}
    </div>
    
    <div class="medical-disclaimer">
        <h3>Medical Use Disclaimer</h3>
        <p><strong>Important:</strong> This report is generated for research and educational purposes only. All medical codes and terminology mappings should be verified against official, authoritative sources before use in clinical practice, billing, or patient care.</p>
        <p>The confidence scores and mappings are AI-generated and may not reflect the most current terminology standards. Always consult with medical coding professionals and reference the latest versions of official coding systems.</p>
    </div>

    {% if citation_text %}
    <div class="citation">
        <strong>📚 Suggested Citation:</strong><br>
        {{ citation_text }}
    </div>
    {% endif %}
    
    <div class="export-footer">
        <h3 style="margin: 0 0 15px 0; color: var(--primary-color);">📋 Export Information</h3>
        <p><strong>Generated by:</strong> Medical Terminology Intelligence Platform v{{ version }}</p>
        <p><strong>Export Date:</strong> {{ timestamp }}</p>
        <p><strong>Results Summary:</strong> {{ results | length }} concepts mapped across {{ unique_sources | length }} source{{ 's' if unique_sources | length != 1 else '' }}</p>
        {% if search_mode == 'hybrid' %}
        <p><strong>Search Technology:</strong> 🧬 Hybrid AI + API Discovery System</p>
        {% else %}
        <p><strong>Search Technology:</strong> 🔗 Authoritative API Search</p>
        {% endif %}
        <p style="margin-top: 15px; font-size: 12px; opacity: 0.8;">
            Powered by advanced medical AI and authoritative terminology APIs
        </p>
    </div>
</body>
</html>'''
        
        env = Environment()
        return env.from_string(template_str)
    
    def export_results(
        self,
        results: List[Dict[str, Any]],
        query: str,
        research_context: Optional[str] = None,
        research_intent: Optional[str] = None,
        target_systems: Optional[List[str]] = None,
        search_mode: str = "hybrid"
    ) -> str:
        """
        Export medical terminology results to beautiful HTML.
        
        Args:
            results: List of result dictionaries
            query: Original search query
            research_context: Optional research context
            research_intent: Type of research (cohort, mapping, etc.)
            target_systems: List of target terminology systems
            search_mode: Search mode used (api, semantic, hybrid)
            
        Returns:
            HTML string
        """
        # Process results for display
        processed_results = []
        unique_sources = set()
        api_count = 0
        semantic_count = 0
        
        for result in results:
            # Get search method
            search_method = result.get('search_method', 'api')
            if search_method == 'biobert_rag':
                search_method = 'pinecone'
            unique_sources.add(search_method)
            
            if search_method == 'api':
                api_count += 1
            else:
                semantic_count += 1
            
            # Add research notes based on system
            if not result.get('research_notes'):
                result['research_notes'] = self._generate_research_notes(
                    result.get('system', ''),
                    result.get('code', '')
                )
            
            # Add character pattern description if missing
            if 'character_pattern' not in result:
                result['character_pattern'] = {
                    'description': self._describe_code_pattern(
                        result.get('system', ''),
                        result.get('code', '')
                    )
                }
            
            processed_results.append(result)
        
        # Calculate summary statistics
        unique_systems = len(set(r.get('system', '') for r in results))
        
        summary = {
            'total_results': len(results),
            'unique_systems': unique_systems,
            'api_results': api_count,
            'semantic_results': semantic_count
        }
        
        # Generate citation
        citation_text = self._generate_citation(query, len(results))
        
        # Render template
        html_content = self.template.render(
            query=html.escape(query),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            results=processed_results,
            research_context=research_context,
            research_intent=research_intent,
            target_systems=target_systems,
            search_mode=search_mode,
            summary=summary,
            unique_sources=list(unique_sources),
            citation_text=citation_text,
            version="2.1.0"
        )
        
        return html_content
    
    def _generate_research_notes(self, system: str, code: str) -> str:
        """Generate research-specific notes based on terminology system."""
        notes = {
            "ICD-10-CM": "Commonly used in claims data and administrative databases",
            "SNOMED CT": "Most granular clinical terminology for EHR data; Includes relationships to other concepts",
            "LOINC": "Standard for laboratory test results; Essential for lab value harmonization across sites",
            "RxNorm": "Standard for medication data; Includes ingredient, strength, and dose form information",
            "CPT": "Procedure coding system; Important for identifying medical procedures and services",
            "UMLS": "Meta-thesaurus linking multiple vocabularies; Useful for cross-terminology mapping"
        }
        
        base_note = notes.get(system, "Standard medical terminology")
        
        # Add specific notes based on code patterns
        if system == "ICD-10-CM":
            if code.startswith(('F', 'G')):
                base_note += "; Mental/neurological condition codes"
            elif code.startswith('Z'):
                base_note += "; Factors influencing health status"
            elif '.' in code and len(code.split('.')[1]) > 2:
                base_note += "; Specific sub-category provides detailed classification"
        
        return base_note
    
    def _describe_code_pattern(self, system: str, code: str) -> str:
        """Describe the code pattern in human-readable format."""
        patterns = {
            "ICD-10-CM": self._describe_icd10_pattern,
            "SNOMED CT": lambda c: f"{len(c)} digits",
            "LOINC": lambda c: f"{len(c.split('-')[0])} digits + hyphen + {len(c.split('-')[1])} digit",
            "RxNorm": lambda c: f"{len(c)} digit identifier",
            "CPT": lambda c: f"{len(c)} digit procedure code"
        }
        
        pattern_func = patterns.get(system, lambda c: f"{len(c)} characters")
        return pattern_func(code)
    
    def _describe_icd10_pattern(self, code: str) -> str:
        """Describe ICD-10 code pattern."""
        if '.' in code:
            parts = code.split('.')
            return f"XXX.{'X' * len(parts[1])} ({len(parts[0])} alphanumeric + decimal + {len(parts[1])} more)"
        else:
            return f"{'X' * len(code)} ({len(code)} alphanumeric)"
    
    def _generate_citation(self, query: str, result_count: int) -> str:
        """Generate citation text for the export."""
        return (
            f"Medical Terminology System (2024). Search results for '{query}'. "
            f"Retrieved {datetime.now().strftime('%B %d, %Y')}. "
            f"Query returned {result_count} results from multiple terminology systems."
        )


# Export service instance
html_export_service = HTMLExportService()