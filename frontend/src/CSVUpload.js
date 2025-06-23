import React, { useState, useRef } from 'react';
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

const CSVUpload = ({ onUploadSuccess, onUploadError }) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [mappingConfig, setMappingConfig] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingJob, setProcessingJob] = useState(null);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0 && files[0].type === 'text/csv') {
      handleFileUpload(files[0]);
    }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file && file.type === 'text/csv') {
      handleFileUpload(file);
    }
  };

  const handleFileUpload = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    setUploadProgress(0);

    try {
      const response = await axios.post(`${API_BASE_URL}/api/csv/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(percentCompleted);
        },
      });

      setAnalysisResult(response.data);
      setUploadProgress(100);
      
      // Generate initial mapping configuration from suggestions
      generateMappingConfig(response.data.suggested_mappings);
      
      if (onUploadSuccess) {
        onUploadSuccess(response.data);
      }
    } catch (error) {
      console.error('Upload failed:', error);
      setUploadProgress(null);
      if (onUploadError) {
        onUploadError(error.response?.data?.detail || 'Upload failed');
      }
    }
  };

  const generateMappingConfig = (suggestions) => {
    const config = {
      columns: suggestions.recommended_columns?.map(col => ({
        column: col.column,
        medical_type: col.type,
        terminology_systems: col.systems,
        search_mode: 'api_only',
        confidence_threshold: 0.5
      })) || [],
      batch_size: 50,
      max_concurrent: 5
    };
    setMappingConfig(config);
  };

  const updateColumnMapping = (index, field, value) => {
    const updatedConfig = { ...mappingConfig };
    updatedConfig.columns[index][field] = value;
    setMappingConfig(updatedConfig);
  };

  const startProcessing = async () => {
    if (!analysisResult || !mappingConfig) return;

    setIsProcessing(true);
    
    try {
      const response = await axios.post(`${API_BASE_URL}/api/csv/process`, {
        file_id: analysisResult.file_id,
        mapping_config: mappingConfig
      });

      setProcessingJob(response.data);
      
      // Start polling for status updates
      pollJobStatus(response.data.job_id);
      
    } catch (error) {
      console.error('Processing failed:', error);
      setIsProcessing(false);
      if (onUploadError) {
        onUploadError(error.response?.data?.detail || 'Processing failed');
      }
    }
  };

  const pollJobStatus = async (jobId) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/csv/status/${jobId}`);
      const status = response.data;

      setProcessingJob(status);

      if (status.status === 'completed') {
        setIsProcessing(false);
        // Auto-download results
        downloadResults(jobId);
      } else if (status.status === 'failed') {
        setIsProcessing(false);
        if (onUploadError) {
          onUploadError(status.error || 'Processing failed');
        }
      } else if (status.status === 'processing' || status.status === 'starting') {
        // Continue polling
        setTimeout(() => pollJobStatus(jobId), 2000);
      }
    } catch (error) {
      console.error('Status check failed:', error);
      setIsProcessing(false);
    }
  };

  const downloadResults = async (jobId) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/csv/download/${jobId}`, {
        responseType: 'blob'
      });

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `enhanced_medical_data_${jobId.slice(0, 8)}.zip`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Download failed:', error);
    }
  };

  const reset = () => {
    setUploadProgress(null);
    setAnalysisResult(null);
    setMappingConfig(null);
    setIsProcessing(false);
    setProcessingJob(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="csv-upload-container">
      {!analysisResult ? (
        <div 
          className={`upload-zone ${isDragOver ? 'drag-over' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <div className="upload-icon">📊</div>
          <div className="upload-title">Upload CSV for Medical Concept Mapping</div>
          <div className="upload-description">
            Drop your CSV file here or click to browse
          </div>
          <div className="upload-requirements">
            • CSV files only • AI will analyze structure • Batch process medical concepts
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv"
            onChange={handleFileSelect}
            style={{ display: 'none' }}
          />
        </div>
      ) : (
        <div className="analysis-results">
          <div className="file-info">
            <h3>📋 File Analysis Complete</h3>
            <div className="file-stats">
              <span>📄 {analysisResult.filename}</span>
              <span>📊 {analysisResult.analysis.file_info.rows} rows</span>
              <span>🏛️ {analysisResult.analysis.file_info.columns} columns</span>
              <span>⚕️ {analysisResult.analysis.medical_analysis ? 
                Object.values(analysisResult.analysis.medical_analysis).filter(col => col.is_medical_concept).length : 0} medical columns detected</span>
            </div>
          </div>

          {mappingConfig && (
            <div className="mapping-config">
              <h4>🎯 Mapping Configuration</h4>
              {mappingConfig.columns.map((column, index) => (
                <div key={index} className="column-config">
                  <div className="column-name">{column.column}</div>
                  <div className="column-controls">
                    <select
                      value={column.medical_type}
                      onChange={(e) => updateColumnMapping(index, 'medical_type', e.target.value)}
                    >
                      <option value="condition">Condition</option>
                      <option value="medication">Medication</option>
                      <option value="procedure">Procedure</option>
                      <option value="laboratory">Laboratory</option>
                      <option value="observation">Observation</option>
                    </select>
                    <select
                      value={column.search_mode}
                      onChange={(e) => updateColumnMapping(index, 'search_mode', e.target.value)}
                    >
                      <option value="api_only">API Only</option>
                      <option value="hybrid">Hybrid (API + AI)</option>
                    </select>
                    <input
                      type="number"
                      min="0.1"
                      max="1.0"
                      step="0.1"
                      value={column.confidence_threshold}
                      onChange={(e) => updateColumnMapping(index, 'confidence_threshold', parseFloat(e.target.value))}
                      placeholder="Confidence"
                    />
                  </div>
                </div>
              ))}
            </div>
          )}

          <div className="action-buttons">
            <button 
              className="process-btn"
              onClick={startProcessing}
              disabled={isProcessing || !mappingConfig?.columns.length}
            >
              {isProcessing ? 'Processing...' : '🚀 Start Medical Concept Mapping'}
            </button>
            <button className="reset-btn" onClick={reset}>
              🔄 Upload New File
            </button>
          </div>

          {processingJob && (
            <div className="processing-status">
              <div className="status-header">
                <span>📊 Processing Status: {processingJob.status}</span>
                {processingJob.progress !== undefined && (
                  <span>{processingJob.progress.toFixed(1)}%</span>
                )}
              </div>
              {processingJob.current_operation && (
                <div className="current-operation">
                  {processingJob.current_operation}
                </div>
              )}
              {processingJob.progress !== undefined && (
                <div className="progress-bar">
                  <div 
                    className="progress-fill"
                    style={{ width: `${processingJob.progress}%` }}
                  ></div>
                </div>
              )}
              {processingJob.processed_concepts !== undefined && (
                <div className="processing-stats">
                  Processed {processingJob.processed_concepts} of {processingJob.total_concepts} concepts
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {uploadProgress !== null && uploadProgress < 100 && (
        <div className="upload-progress">
          <div className="progress-bar">
            <div 
              className="progress-fill"
              style={{ width: `${uploadProgress}%` }}
            ></div>
          </div>
          <div className="progress-text">Uploading... {uploadProgress}%</div>
        </div>
      )}
    </div>
  );
};

export default CSVUpload;