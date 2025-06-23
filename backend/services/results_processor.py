"""
Results Processing Service
Handles processing and formatting of CSV mapping results
"""

import pandas as pd
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import io
import zipfile
import tempfile

logger = logging.getLogger(__name__)

class ResultsProcessor:
    """Process and format CSV mapping results for download and analysis"""
    
    def __init__(self):
        pass
    
    def generate_comprehensive_report(
        self, 
        enhanced_df: pd.DataFrame, 
        mapping_summary: Dict[str, Any],
        original_analysis: Dict[str, Any],
        job_id: str
    ) -> Dict[str, Any]:
        """Generate comprehensive report package"""
        
        try:
            # Create temporary directory for files
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Generate main enhanced CSV
                enhanced_csv_path = temp_path / f"enhanced_data_{job_id[:8]}.csv"
                enhanced_df.to_csv(enhanced_csv_path, index=False)
                
                # Generate summary report
                summary_report = self._create_summary_report(mapping_summary, original_analysis)
                summary_path = temp_path / f"mapping_summary_{job_id[:8]}.json"
                with open(summary_path, 'w') as f:
                    json.dump(summary_report, f, indent=2)
                
                # Generate mapping statistics
                stats_df = self._create_mapping_statistics(enhanced_df, mapping_summary)
                stats_path = temp_path / f"mapping_statistics_{job_id[:8]}.csv"
                stats_df.to_csv(stats_path, index=False)
                
                # Generate quality report
                quality_report = self._create_quality_report(enhanced_df, mapping_summary)
                quality_path = temp_path / f"quality_report_{job_id[:8]}.json"
                with open(quality_path, 'w') as f:
                    json.dump(quality_report, f, indent=2)
                
                # Create ZIP package
                zip_path = temp_path / f"mapping_results_{job_id[:8]}.zip"
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    zipf.write(enhanced_csv_path, enhanced_csv_path.name)
                    zipf.write(summary_path, summary_path.name)
                    zipf.write(stats_path, stats_path.name)
                    zipf.write(quality_path, quality_path.name)
                
                # Read zip content for return
                with open(zip_path, 'rb') as f:
                    zip_content = f.read()
                
                return {
                    'status': 'success',
                    'zip_content': zip_content,
                    'zip_filename': f"mapping_results_{job_id[:8]}.zip",
                    'files_included': [
                        enhanced_csv_path.name,
                        summary_path.name,
                        stats_path.name,
                        quality_path.name
                    ],
                    'summary': summary_report
                }
                
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _create_summary_report(self, mapping_summary: Dict[str, Any], original_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed summary report"""
        
        return {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_type': 'Medical Concept Mapping Summary',
                'version': '1.0'
            },
            'original_file_analysis': {
                'file_info': original_analysis.get('file_info', {}),
                'medical_columns_detected': len([
                    col for col, data in original_analysis.get('medical_analysis', {}).items()
                    if data.get('is_medical_concept', False)
                ]),
                'data_quality_assessment': self._assess_overall_quality(original_analysis)
            },
            'mapping_results': mapping_summary,
            'recommendations': self._generate_recommendations(mapping_summary, original_analysis),
            'next_steps': self._suggest_next_steps(mapping_summary)
        }
    
    def _create_mapping_statistics(self, enhanced_df: pd.DataFrame, mapping_summary: Dict[str, Any]) -> pd.DataFrame:
        """Create detailed mapping statistics DataFrame"""
        
        stats_data = []
        
        # Identify mapping columns
        mapping_columns = [col for col in enhanced_df.columns if '_MAPPING_STATUS' in col]
        
        for status_col in mapping_columns:
            base_column = status_col.replace('_MAPPING_STATUS', '')
            
            # Count mapping statuses
            status_counts = enhanced_df[status_col].value_counts()
            
            # Find associated confidence columns
            confidence_cols = [col for col in enhanced_df.columns if base_column in col and 'CONFIDENCE' in col]
            
            for conf_col in confidence_cols:
                system = conf_col.replace(f"{base_column}_", "").replace("_CONFIDENCE", "")
                confidences = enhanced_df[conf_col][enhanced_df[conf_col] > 0]
                
                stats_data.append({
                    'column': base_column,
                    'terminology_system': system,
                    'total_concepts': len(enhanced_df),
                    'mapped_concepts': len(confidences),
                    'mapping_rate': len(confidences) / len(enhanced_df) * 100 if len(enhanced_df) > 0 else 0,
                    'avg_confidence': confidences.mean() if len(confidences) > 0 else 0,
                    'min_confidence': confidences.min() if len(confidences) > 0 else 0,
                    'max_confidence': confidences.max() if len(confidences) > 0 else 0,
                    'high_confidence_count': len(confidences[confidences > 0.8]) if len(confidences) > 0 else 0,
                    'medium_confidence_count': len(confidences[(confidences > 0.5) & (confidences <= 0.8)]) if len(confidences) > 0 else 0,
                    'low_confidence_count': len(confidences[confidences <= 0.5]) if len(confidences) > 0 else 0
                })
        
        return pd.DataFrame(stats_data)
    
    def _create_quality_report(self, enhanced_df: pd.DataFrame, mapping_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Create data quality assessment report"""
        
        quality_issues = []
        recommendations = []
        
        # Check for low confidence mappings
        confidence_cols = [col for col in enhanced_df.columns if 'CONFIDENCE' in col]
        for conf_col in confidence_cols:
            low_conf_count = len(enhanced_df[enhanced_df[conf_col] < 0.5])
            if low_conf_count > 0:
                quality_issues.append({
                    'type': 'low_confidence_mappings',
                    'column': conf_col,
                    'count': low_conf_count,
                    'percentage': low_conf_count / len(enhanced_df) * 100,
                    'severity': 'medium' if low_conf_count / len(enhanced_df) < 0.2 else 'high'
                })
                
                if low_conf_count / len(enhanced_df) > 0.1:
                    recommendations.append({
                        'type': 'manual_review',
                        'description': f"Manual review recommended for {low_conf_count} low-confidence mappings in {conf_col}",
                        'priority': 'high'
                    })
        
        # Check for failed mappings
        status_cols = [col for col in enhanced_df.columns if '_MAPPING_STATUS' in col]
        for status_col in status_cols:
            failed_count = len(enhanced_df[enhanced_df[status_col] == 'failed'])
            no_match_count = len(enhanced_df[enhanced_df[status_col] == 'no_matches'])
            
            if failed_count > 0:
                quality_issues.append({
                    'type': 'failed_mappings',
                    'column': status_col,
                    'count': failed_count,
                    'percentage': failed_count / len(enhanced_df) * 100,
                    'severity': 'high'
                })
            
            if no_match_count > 0:
                quality_issues.append({
                    'type': 'no_matches',
                    'column': status_col,
                    'count': no_match_count,
                    'percentage': no_match_count / len(enhanced_df) * 100,
                    'severity': 'medium'
                })
                
                recommendations.append({
                    'type': 'terminology_expansion',
                    'description': f"Consider additional terminology systems for {no_match_count} unmatched concepts in {status_col}",
                    'priority': 'medium'
                })
        
        # Check for inconsistent mappings
        inconsistencies = self._detect_mapping_inconsistencies(enhanced_df)
        if inconsistencies:
            quality_issues.extend(inconsistencies)
            recommendations.append({
                'type': 'consistency_review',
                'description': "Review inconsistent mappings for data standardization opportunities",
                'priority': 'medium'
            })
        
        return {
            'quality_score': self._calculate_quality_score(enhanced_df),
            'quality_issues': quality_issues,
            'recommendations': recommendations,
            'data_completeness': self._assess_data_completeness(enhanced_df),
            'mapping_consistency': self._assess_mapping_consistency(enhanced_df)
        }
    
    def _detect_mapping_inconsistencies(self, enhanced_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect inconsistent mappings in the data"""
        
        inconsistencies = []
        
        # Find original data columns and their corresponding mapping columns
        mapping_status_cols = [col for col in enhanced_df.columns if '_MAPPING_STATUS' in col]
        
        for status_col in mapping_status_cols:
            base_column = status_col.replace('_MAPPING_STATUS', '')
            
            if base_column not in enhanced_df.columns:
                continue
            
            # Group by original value and check for different mappings
            code_cols = [col for col in enhanced_df.columns if base_column in col and '_CODE' in col]
            
            for code_col in code_cols:
                # Group by original value and count unique codes
                grouped = enhanced_df.groupby(base_column)[code_col].nunique()
                inconsistent = grouped[grouped > 1]
                
                if len(inconsistent) > 0:
                    inconsistencies.append({
                        'type': 'inconsistent_mapping',
                        'column': base_column,
                        'mapping_column': code_col,
                        'inconsistent_values': inconsistent.index.tolist(),
                        'count': len(inconsistent),
                        'severity': 'medium'
                    })
        
        return inconsistencies
    
    def _calculate_quality_score(self, enhanced_df: pd.DataFrame) -> float:
        """Calculate overall mapping quality score (0-100)"""
        
        total_score = 0
        score_components = 0
        
        # Score based on mapping success rate
        status_cols = [col for col in enhanced_df.columns if '_MAPPING_STATUS' in col]
        if status_cols:
            successful_mappings = 0
            total_mappings = 0
            
            for status_col in status_cols:
                success_count = len(enhanced_df[enhanced_df[status_col] == 'success'])
                total_count = len(enhanced_df)
                successful_mappings += success_count
                total_mappings += total_count
            
            if total_mappings > 0:
                success_rate = successful_mappings / total_mappings
                total_score += success_rate * 40  # 40% weight for success rate
                score_components += 1
        
        # Score based on confidence levels
        confidence_cols = [col for col in enhanced_df.columns if 'CONFIDENCE' in col]
        if confidence_cols:
            avg_confidence = 0
            conf_count = 0
            
            for conf_col in confidence_cols:
                valid_confidences = enhanced_df[conf_col][enhanced_df[conf_col] > 0]
                if len(valid_confidences) > 0:
                    avg_confidence += valid_confidences.mean()
                    conf_count += 1
            
            if conf_count > 0:
                avg_confidence /= conf_count
                total_score += avg_confidence * 40  # 40% weight for confidence
                score_components += 1
        
        # Score based on data completeness
        completeness_score = self._assess_data_completeness(enhanced_df)
        total_score += completeness_score * 20  # 20% weight for completeness
        score_components += 1
        
        return total_score / score_components if score_components > 0 else 0
    
    def _assess_data_completeness(self, enhanced_df: pd.DataFrame) -> float:
        """Assess data completeness (0-100)"""
        
        mapping_cols = [col for col in enhanced_df.columns if any(suffix in col for suffix in ['_CODE', '_NAME', '_CONFIDENCE'])]
        
        if not mapping_cols:
            return 0
        
        total_cells = len(enhanced_df) * len(mapping_cols)
        filled_cells = 0
        
        for col in mapping_cols:
            if 'CONFIDENCE' in col:
                filled_cells += len(enhanced_df[enhanced_df[col] > 0])
            else:
                filled_cells += len(enhanced_df[enhanced_df[col].notna() & (enhanced_df[col] != '')])
        
        return (filled_cells / total_cells * 100) if total_cells > 0 else 0
    
    def _assess_mapping_consistency(self, enhanced_df: pd.DataFrame) -> float:
        """Assess mapping consistency (0-100)"""
        
        inconsistencies = self._detect_mapping_inconsistencies(enhanced_df)
        total_mappings = len(enhanced_df)
        
        if total_mappings == 0:
            return 100
        
        inconsistent_count = sum(inc.get('count', 0) for inc in inconsistencies)
        consistency_rate = 1 - (inconsistent_count / total_mappings)
        
        return max(0, consistency_rate * 100)
    
    def _assess_overall_quality(self, original_analysis: Dict[str, Any]) -> str:
        """Assess overall data quality from original analysis"""
        
        medical_analysis = original_analysis.get('medical_analysis', {})
        quality_scores = []
        
        for col_data in medical_analysis.values():
            quality = col_data.get('data_quality', 'poor')
            if quality == 'excellent':
                quality_scores.append(4)
            elif quality == 'good':
                quality_scores.append(3)
            elif quality == 'fair':
                quality_scores.append(2)
            else:
                quality_scores.append(1)
        
        if not quality_scores:
            return 'unknown'
        
        avg_score = sum(quality_scores) / len(quality_scores)
        
        if avg_score >= 3.5:
            return 'excellent'
        elif avg_score >= 2.5:
            return 'good'
        elif avg_score >= 1.5:
            return 'fair'
        else:
            return 'poor'
    
    def _generate_recommendations(self, mapping_summary: Dict[str, Any], original_analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate actionable recommendations"""
        
        recommendations = []
        success_rate = mapping_summary.get('success_rate', 0)
        avg_confidence = mapping_summary.get('average_confidence', 0)
        
        # Success rate recommendations
        if success_rate < 50:
            recommendations.append({
                'category': 'mapping_improvement',
                'priority': 'high',
                'description': 'Low mapping success rate detected. Consider using hybrid search mode or additional terminology systems.'
            })
        elif success_rate < 80:
            recommendations.append({
                'category': 'mapping_improvement',
                'priority': 'medium',
                'description': 'Moderate mapping success rate. Review failed mappings for data preprocessing opportunities.'
            })
        
        # Confidence recommendations
        if avg_confidence < 0.6:
            recommendations.append({
                'category': 'confidence_improvement',
                'priority': 'high',
                'description': 'Low average confidence scores. Manual review recommended for critical applications.'
            })
        
        # System-specific recommendations
        system_counts = mapping_summary.get('terminology_system_counts', {})
        if len(system_counts) < 2:
            recommendations.append({
                'category': 'coverage_expansion',
                'priority': 'medium',
                'description': 'Consider mapping to additional terminology systems for better coverage.'
            })
        
        return recommendations
    
    def _suggest_next_steps(self, mapping_summary: Dict[str, Any]) -> List[str]:
        """Suggest next steps based on mapping results"""
        
        steps = []
        success_rate = mapping_summary.get('success_rate', 0)
        low_conf_count = mapping_summary.get('low_confidence_mappings', 0)
        
        if success_rate > 80:
            steps.append("✅ High mapping success rate achieved. Proceed with data analysis.")
        else:
            steps.append("🔍 Review failed mappings and consider data preprocessing.")
        
        if low_conf_count > 0:
            steps.append(f"⚠️ Manual review recommended for {low_conf_count} low-confidence mappings.")
        
        steps.extend([
            "📊 Validate a sample of high-confidence mappings for accuracy.",
            "🔄 Consider iterative improvement based on domain expert feedback.",
            "📋 Document mapping decisions for reproducibility."
        ])
        
        return steps
    
    def create_validation_sample(self, enhanced_df: pd.DataFrame, sample_size: int = 20) -> pd.DataFrame:
        """Create a sample for manual validation"""
        
        # Sample high, medium, and low confidence mappings
        confidence_cols = [col for col in enhanced_df.columns if 'CONFIDENCE' in col]
        
        if not confidence_cols:
            return enhanced_df.head(sample_size)
        
        samples = []
        
        for conf_col in confidence_cols:
            valid_data = enhanced_df[enhanced_df[conf_col] > 0]
            
            if len(valid_data) > 0:
                # High confidence sample
                high_conf = valid_data[valid_data[conf_col] > 0.8]
                if len(high_conf) > 0:
                    samples.append(high_conf.sample(min(5, len(high_conf))))
                
                # Medium confidence sample
                med_conf = valid_data[(valid_data[conf_col] > 0.5) & (valid_data[conf_col] <= 0.8)]
                if len(med_conf) > 0:
                    samples.append(med_conf.sample(min(5, len(med_conf))))
                
                # Low confidence sample
                low_conf = valid_data[valid_data[conf_col] <= 0.5]
                if len(low_conf) > 0:
                    samples.append(low_conf.sample(min(5, len(low_conf))))
        
        if samples:
            combined_sample = pd.concat(samples).drop_duplicates()
            return combined_sample.head(sample_size)
        else:
            return enhanced_df.head(sample_size)