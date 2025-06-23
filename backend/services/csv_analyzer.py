"""
CSV Analysis Service with LLM Integration
Analyzes uploaded CSV files to identify medical concepts and suggest mapping strategies
"""

import pandas as pd
import json
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from openai import OpenAI
import os
from datetime import datetime

from ..config import settings

logger = logging.getLogger(__name__)

class CSVAnalyzer:
    """Intelligent CSV analyzer for medical data using LLM"""
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=settings.openai_api_key.get_secret_value() if settings.openai_api_key else None)
        self.medical_keywords = {
            'condition': ['diagnosis', 'disease', 'condition', 'disorder', 'syndrome', 'pathology', 'illness'],
            'medication': ['drug', 'medication', 'medicine', 'pharmaceutical', 'prescription', 'treatment', 'therapy'],
            'procedure': ['procedure', 'surgery', 'operation', 'intervention', 'treatment', 'therapy'],
            'laboratory': ['lab', 'test', 'result', 'value', 'level', 'count', 'measure', 'assay'],
            'observation': ['observation', 'finding', 'assessment', 'evaluation', 'examination']
        }
        
    async def analyze_csv_structure(self, file_path: str, filename: str) -> Dict[str, Any]:
        """
        Analyze CSV structure and identify medical concepts using LLM
        """
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Basic file information
            file_info = {
                'filename': filename,
                'rows': len(df),
                'columns': len(df.columns),
                'size_mb': round(Path(file_path).stat().st_size / (1024 * 1024), 2),
                'timestamp': datetime.now().isoformat()
            }
            
            # Get sample data for LLM analysis
            headers = df.columns.tolist()
            sample_data = df.head(3).to_dict('records')
            
            # LLM analysis
            llm_analysis = await self._analyze_with_llm(headers, sample_data)
            
            # Rule-based analysis
            rule_analysis = self._analyze_with_rules(df)
            
            # Combine analyses
            combined_analysis = self._combine_analyses(llm_analysis, rule_analysis)
            
            # Generate mapping suggestions
            mapping_suggestions = self._generate_mapping_suggestions(combined_analysis)
            
            return {
                'file_info': file_info,
                'structure_analysis': {
                    'headers': headers,
                    'sample_data': sample_data[:3],  # Limit sample size
                    'data_types': df.dtypes.astype(str).to_dict(),
                    'null_counts': df.isnull().sum().to_dict()
                },
                'medical_analysis': combined_analysis,
                'mapping_suggestions': mapping_suggestions,
                'processing_recommendations': self._get_processing_recommendations(df, combined_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing CSV structure: {e}")
            raise ValueError(f"Failed to analyze CSV: {str(e)}")
    
    async def _analyze_with_llm(self, headers: List[str], sample_data: List[Dict]) -> Dict[str, Any]:
        """Use LLM to analyze CSV structure and identify medical concepts"""
        
        prompt = f"""You are a medical data analyst expert. Analyze this CSV structure and identify medical concepts.

CSV Headers: {headers}

Sample Data (first 3 rows):
{json.dumps(sample_data, indent=2)}

For each column, provide analysis in this JSON format:
{{
    "column_name": {{
        "is_medical_concept": true/false,
        "medical_type": "condition/medication/procedure/laboratory/observation/demographic/other",
        "confidence_score": 0.0-1.0,
        "reasoning": "explanation of classification",
        "data_quality": "excellent/good/fair/poor",
        "suggested_preprocessing": ["list of preprocessing steps"],
        "terminology_systems": ["UMLS", "ICD-10-CM", "SNOMED CT", "RxNorm", "LOINC"],
        "sample_concepts": ["list of medical concepts found in sample data"]
    }}
}}

Focus on:
1. Medical relevance of each column
2. Quality and consistency of medical data
3. Appropriate terminology systems for mapping
4. Data preprocessing needs
5. Potential mapping challenges

Respond with valid JSON only."""

        try:
            # Check if OpenAI client has valid API key
            if not self.openai_client or not self.openai_client.api_key:
                logger.warning("No OpenAI API key - using rule-based analysis only")
                return {}
                
            logger.info("Using GPT-4o for CSV analysis")
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a medical data analysis expert. Respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            # Fallback to rule-based analysis
            return {}
    
    def _analyze_with_rules(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Rule-based analysis for medical concept detection"""
        
        analysis = {}
        
        for column in df.columns:
            column_lower = column.lower()
            sample_values = df[column].dropna().astype(str).head(10).tolist()
            
            # Detect medical concept type
            medical_type = "other"
            confidence = 0.0
            terminology_systems = []
            
            # Check column name against medical keywords
            for concept_type, keywords in self.medical_keywords.items():
                if any(keyword in column_lower for keyword in keywords):
                    medical_type = concept_type
                    confidence = 0.7
                    break
            
            # Analyze sample values for medical patterns
            medical_patterns = self._detect_medical_patterns(sample_values)
            if medical_patterns['is_medical']:
                if confidence == 0.0:
                    medical_type = medical_patterns['type']
                    confidence = medical_patterns['confidence']
                
                # Suggest terminology systems
                if medical_type == "condition":
                    terminology_systems = ["UMLS", "ICD-10-CM", "SNOMED CT"]
                elif medical_type == "medication":
                    terminology_systems = ["UMLS", "RxNorm"]
                elif medical_type == "laboratory":
                    terminology_systems = ["UMLS", "LOINC"]
                elif medical_type == "procedure":
                    terminology_systems = ["UMLS", "SNOMED CT"]
                else:
                    terminology_systems = ["UMLS"]
            
            analysis[column] = {
                "is_medical_concept": confidence > 0.3,
                "medical_type": medical_type,
                "confidence_score": confidence,
                "reasoning": f"Rule-based analysis of column name and sample data",
                "data_quality": self._assess_data_quality(df[column]),
                "suggested_preprocessing": self._suggest_preprocessing(df[column]),
                "terminology_systems": terminology_systems,
                "sample_concepts": [str(val) for val in sample_values[:5] if self._looks_like_medical_concept(str(val))]
            }
        
        return analysis
    
    def _detect_medical_patterns(self, values: List[str]) -> Dict[str, Any]:
        """Detect medical patterns in data values"""
        
        medical_indicators = {
            'condition': ['diabetes', 'hypertension', 'pneumonia', 'sepsis', 'infection', 'syndrome', 'disease'],
            'medication': ['mg', 'ml', 'tablet', 'capsule', 'injection', 'dose'],
            'laboratory': ['mg/dl', 'mmol/l', 'positive', 'negative', 'normal', 'abnormal'],
            'procedure': ['surgery', 'biopsy', 'scan', 'x-ray', 'mri', 'ct']
        }
        
        scores = {concept_type: 0 for concept_type in medical_indicators.keys()}
        
        for value in values:
            value_lower = str(value).lower()
            for concept_type, indicators in medical_indicators.items():
                for indicator in indicators:
                    if indicator in value_lower:
                        scores[concept_type] += 1
        
        # Determine best match
        max_score = max(scores.values())
        if max_score > 0:
            best_type = max(scores, key=scores.get)
            confidence = min(max_score / len(values), 1.0)
            return {
                'is_medical': True,
                'type': best_type,
                'confidence': confidence
            }
        
        return {'is_medical': False, 'type': 'other', 'confidence': 0.0}
    
    def _looks_like_medical_concept(self, value: str) -> bool:
        """Check if a value looks like a medical concept"""
        value = str(value).strip().lower()
        
        # Skip empty, numeric, or very short values
        if not value or len(value) < 3 or value.isdigit():
            return False
        
        # Common medical terms
        medical_terms = [
            'diabetes', 'hypertension', 'pneumonia', 'sepsis', 'infection',
            'syndrome', 'disease', 'disorder', 'condition', 'mg', 'ml',
            'positive', 'negative', 'normal', 'abnormal', 'surgery', 'biopsy'
        ]
        
        return any(term in value for term in medical_terms)
    
    def _assess_data_quality(self, series: pd.Series) -> str:
        """Assess data quality of a pandas series"""
        null_percentage = series.isnull().sum() / len(series)
        unique_ratio = series.nunique() / len(series)
        
        if null_percentage < 0.05 and unique_ratio > 0.1:
            return "excellent"
        elif null_percentage < 0.15 and unique_ratio > 0.05:
            return "good"
        elif null_percentage < 0.30:
            return "fair"
        else:
            return "poor"
    
    def _suggest_preprocessing(self, series: pd.Series) -> List[str]:
        """Suggest preprocessing steps for a column"""
        suggestions = []
        
        # Check for missing values
        if series.isnull().any():
            suggestions.append("handle_missing_values")
        
        # Check for inconsistent formatting
        if series.dtype == 'object':
            sample_values = series.dropna().astype(str).head(20)
            if len(set(val.lower().strip() for val in sample_values)) != len(sample_values):
                suggestions.append("standardize_case_and_spacing")
        
        # Check for potential abbreviations
        if series.dtype == 'object':
            short_values = series.dropna().astype(str).str.len()
            if short_values.median() < 10:
                suggestions.append("expand_abbreviations")
        
        return suggestions
    
    def _combine_analyses(self, llm_analysis: Dict, rule_analysis: Dict) -> Dict[str, Any]:
        """Combine LLM and rule-based analyses"""
        
        combined = {}
        
        # Use all columns from rule analysis as baseline
        for column, rule_data in rule_analysis.items():
            combined[column] = rule_data.copy()
            
            # Override with LLM analysis if available and more confident
            if column in llm_analysis:
                llm_data = llm_analysis[column]
                
                # Use LLM data if it has higher confidence or provides more detail
                if (llm_data.get('confidence_score', 0) > rule_data.get('confidence_score', 0) or
                    len(llm_data.get('reasoning', '')) > len(rule_data.get('reasoning', ''))):
                    
                    combined[column].update({
                        'is_medical_concept': llm_data.get('is_medical_concept', rule_data['is_medical_concept']),
                        'medical_type': llm_data.get('medical_type', rule_data['medical_type']),
                        'confidence_score': max(llm_data.get('confidence_score', 0), rule_data['confidence_score']),
                        'reasoning': f"LLM: {llm_data.get('reasoning', '')} | Rule: {rule_data['reasoning']}",
                        'terminology_systems': llm_data.get('terminology_systems', rule_data['terminology_systems'])
                    })
        
        return combined
    
    def _generate_mapping_suggestions(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mapping suggestions based on analysis"""
        
        suggestions = {
            'recommended_columns': [],
            'mapping_strategies': {},
            'terminology_priorities': {},
            'processing_order': []
        }
        
        # Identify columns recommended for mapping
        for column, data in analysis.items():
            if data['is_medical_concept'] and data['confidence_score'] > 0.3:
                suggestions['recommended_columns'].append({
                    'column': column,
                    'type': data['medical_type'],
                    'confidence': data['confidence_score'],
                    'systems': data['terminology_systems']
                })
                
                # Suggest mapping strategy
                if data['confidence_score'] > 0.8:
                    strategy = "direct_mapping"
                elif data['confidence_score'] > 0.5:
                    strategy = "fuzzy_matching"
                else:
                    strategy = "manual_review"
                
                suggestions['mapping_strategies'][column] = strategy
                
                # Set terminology priorities
                suggestions['terminology_priorities'][column] = data['terminology_systems']
        
        # Sort by confidence for processing order
        suggestions['recommended_columns'].sort(key=lambda x: x['confidence'], reverse=True)
        suggestions['processing_order'] = [col['column'] for col in suggestions['recommended_columns']]
        
        return suggestions
    
    def _get_processing_recommendations(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get overall processing recommendations"""
        
        medical_columns = [col for col, data in analysis.items() if data['is_medical_concept']]
        total_medical_concepts = sum(len(analysis[col]['sample_concepts']) for col in medical_columns)
        
        return {
            'total_rows': len(df),
            'medical_columns_detected': len(medical_columns),
            'estimated_concepts_to_map': total_medical_concepts,
            'estimated_processing_time': f"{max(1, total_medical_concepts // 10)} minutes",
            'recommended_search_mode': "hybrid" if total_medical_concepts > 50 else "api_only",
            'batch_size_recommendation': min(100, max(10, len(df) // 10)),
            'quality_concerns': [
                col for col, data in analysis.items() 
                if data['is_medical_concept'] and data['data_quality'] in ['poor', 'fair']
            ]
        }