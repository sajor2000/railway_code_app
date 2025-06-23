import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import CSVUpload from './CSVUpload';

// For single-service deployment, use relative URLs
const API_BASE_URL = process.env.REACT_APP_BACKEND_URL || (
  window.location.hostname === 'localhost' 
    ? 'http://localhost:8001'  // Development mode
    : ''  // Production: same origin
);

function App() {
  // State Management
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [searchMode, setSearchMode] = useState('api_only');
  const [vectorDbStatus, setVectorDbStatus] = useState(null);
  const [currentMode, setCurrentMode] = useState('chat'); // 'chat', 'research', or 'csv'
  const [conversationId, setConversationId] = useState(null);
  const [abbreviations, setAbbreviations] = useState(null);
  const [showAbbreviations, setShowAbbreviations] = useState(false);

  // Check vector database status on load
  useEffect(() => {
    checkVectorDbStatus();
    loadAbbreviations();
    // Add welcome message
    setMessages([{
      id: 'welcome',
      type: 'system',
      content: 'Welcome to the Medical Terminology Assistant! I\'m an AI-powered medical informatics assistant. Ask me about medical conditions, medications, lab tests, or diagnostic codes. I can search across 5 major medical terminology systems and provide comprehensive analysis.',
      timestamp: new Date()
    }]);
  }, []);

  const checkVectorDbStatus = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/pinecone/status`);
      setVectorDbStatus(response.data);
    } catch (error) {
      console.error('Error checking vector DB status:', error);
      setVectorDbStatus({ status: 'disconnected' });
    }
  };

  const loadAbbreviations = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/abbreviations`);
      setAbbreviations(response.data);
    } catch (error) {
      console.error('Error loading abbreviations:', error);
    }
  };

  const handleChat = async () => {
    if (!inputText.trim()) return;

    const userMessage = {
      id: Date.now() + '_user',
      type: 'user',
      content: inputText.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);

    try {
      // Use chat endpoint for conversational AI
      const chatPayload = {
        message: userMessage.content,
        conversation_id: conversationId,
        search_mode: searchMode
      };

      const response = await axios.post(`${API_BASE_URL}/api/chat`, chatPayload);
      
      // Update conversation ID if provided
      if (response.data.conversation_id) {
        setConversationId(response.data.conversation_id);
      }
      
      const assistantMessage = {
        id: Date.now() + '_assistant',
        type: 'assistant',
        content: response.data.response,
        concepts: response.data.concepts,
        searchMode: searchMode,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage = {
        id: Date.now() + '_error',
        type: 'error',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleResearch = async () => {
    if (!inputText.trim()) return;

    const userMessage = {
      id: Date.now() + '_user',
      type: 'user',
      content: inputText.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);

    try {
      const researchPayload = {
        query: userMessage.content,
        comprehensive: true,
        include_related: true
      };

      const response = await axios.post(`${API_BASE_URL}/api/research-workflow`, researchPayload);
      
      const assistantMessage = {
        id: Date.now() + '_research',
        type: 'research',
        content: response.data,
        query: userMessage.content,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Research error:', error);
      const errorMessage = {
        id: Date.now() + '_error',
        type: 'error',
        content: 'Sorry, I encountered an error during research. Please try again.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = () => {
    if (currentMode === 'chat') {
      handleChat();
    } else if (currentMode === 'research') {
      handleResearch();
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const downloadResults = async (query, mode, format = 'csv') => {
    try {
      let endpoint;
      if (format === 'html') {
        endpoint = mode === 'hybrid' ? '/api/hybrid-search/export-html' : '/api/search/export-html';
      } else {
        endpoint = mode === 'hybrid' ? '/api/hybrid-search/download-csv' : '/api/search';
      }
      
      const response = await axios.post(`${API_BASE_URL}${endpoint}`, {
        query: query,
        ontologies: ['umls', 'rxnorm', 'icd10', 'snomed', 'loinc'],
        expand_abbreviations: true,
        semantic_search: true,
        confidence_threshold: 0.5
      }, {
        responseType: 'blob'
      });

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      const extension = format === 'html' ? 'html' : 'csv';
      link.setAttribute('download', `medical_terminology_${query.replace(/\s+/g, '_')}_${Date.now()}.${extension}`);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (error) {
      console.error('Download error:', error);
      alert('Download failed. Please try again.');
    }
  };

  const handleCSVUploadSuccess = (uploadData) => {
    const successMessage = {
      id: Date.now() + '_csv_success',
      type: 'system',
      content: `✅ CSV file "${uploadData.filename}" uploaded and analyzed successfully! ${uploadData.analysis.file_info.rows} rows with ${Object.values(uploadData.analysis.medical_analysis).filter(col => col.is_medical_concept).length} medical columns detected.`,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, successMessage]);
  };

  const handleCSVUploadError = (error) => {
    const errorMessage = {
      id: Date.now() + '_csv_error',
      type: 'error',
      content: `❌ CSV upload failed: ${error}`,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, errorMessage]);
  };

  const renderMessage = (message) => {
    if (message.type === 'system') {
      return (
        <div key={message.id} className="message-container system-message">
          <div className="message-bubble system-bubble">
            <div className="message-icon">🏥</div>
            <div className="message-content">{message.content}</div>
          </div>
        </div>
      );
    }

    if (message.type === 'user') {
      return (
        <div key={message.id} className="message-container user-message">
          <div className="message-bubble user-bubble">
            <div className="message-content">{message.content}</div>
          </div>
        </div>
      );
    }

    if (message.type === 'error') {
      return (
        <div key={message.id} className="message-container assistant-message">
          <div className="message-bubble error-bubble">
            <div className="message-icon">⚠️</div>
            <div className="message-content">{message.content}</div>
          </div>
        </div>
      );
    }

    if (message.type === 'assistant') {
      return renderAssistantMessage(message);
    }

    if (message.type === 'research') {
      return renderResearchMessage(message);
    }

    return null;
  };

  const renderAssistantMessage = (message) => {
    // For chat responses, display the AI response directly
    if (typeof message.content === 'string') {
      return (
        <div key={message.id} className="message-container assistant-message">
          <div className="message-bubble assistant-bubble">
            <div className="message-icon">🤖</div>
            <div className="message-content">
              <div className="ai-response">{message.content}</div>
              {message.concepts && message.concepts.length > 0 && (
                <div className="concept-references">
                  <div className="reference-header">📚 Referenced Concepts:</div>
                  <div className="concept-list">
                    {message.concepts.slice(0, 5).map((concept, idx) => (
                      <div key={idx} className="concept-item">
                        <span className="concept-name">{concept.concept_name}</span>
                        <span className="concept-code">{concept.concept_id}</span>
                        <button 
                          onClick={() => exploreRelatedConcepts(concept.concept_id)}
                          className="explore-btn"
                          title="Explore related concepts"
                        >
                          🔗
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      );
    }
    
    // Legacy search results handling
    const results = message.content;
    const isHybrid = message.searchMode === 'hybrid';

    if (isHybrid && results) {
      return renderHybridMessage(message);
    } else if (results && Array.isArray(results)) {
      return renderAPIMessage(message);
    }

    return (
      <div key={message.id} className="message-container assistant-message">
        <div className="message-bubble assistant-bubble">
          <div className="message-icon">🔍</div>
          <div className="message-content">No results found for your query.</div>
        </div>
      </div>
    );
  };

  const renderAPIMessage = (message) => {
    const results = message.content;
    const totalResults = results.length;

    // Group results by ontology
    const groupedResults = results.reduce((acc, result) => {
      const ontology = result.source_ontology;
      if (!acc[ontology]) acc[ontology] = [];
      acc[ontology].push(result);
      return acc;
    }, {});

    return (
      <div key={message.id} className="message-container assistant-message">
        <div className="message-bubble assistant-bubble">
          <div className="message-header">
            <div className="message-icon">🔍</div>
            <div className="message-title">
              Medical Terminology Results
              <div className="message-subtitle">
                Found {totalResults} concepts using API search
              </div>
            </div>
            <div className="export-buttons">
              <button 
                onClick={() => downloadResults(message.query, message.searchMode, 'csv')}
                className="download-btn"
                title="Download CSV"
              >
                📊
              </button>
              <button 
                onClick={() => downloadResults(message.query, message.searchMode, 'html')}
                className="download-btn"
                title="Export as HTML Report"
              >
                📄
              </button>
            </div>
          </div>

          <div className="results-container">
            {Object.entries(groupedResults).map(([ontology, ontologyResults]) => (
              <div key={ontology} className="ontology-section">
                <div className="ontology-header">
                  <span className="ontology-name">{ontology}</span>
                  <span className="ontology-count">({ontologyResults.length})</span>
                </div>
                <div className="results-grid">
                  {ontologyResults.slice(0, 5).map((result, idx) => (
                    <div key={idx} className="result-card">
                      <div className="result-code">{result.concept_id}</div>
                      <div className="result-name">{result.concept_name}</div>
                      {result.definition && (
                        <div className="result-definition">{result.definition}</div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };

  const renderResearchMessage = (message) => {
    const research = message.content;
    
    return (
      <div key={message.id} className="message-container research-message">
        <div className="message-bubble research-bubble">
          <div className="message-header">
            <div className="message-icon">🔬</div>
            <div className="message-title">
              Comprehensive Research Analysis
              <div className="message-subtitle">
                Multi-agent analysis for "{message.query}"
              </div>
            </div>
            <button 
              onClick={() => exportResearchReport(message.query, research)}
              className="export-btn"
              title="Export as HTML Report"
            >
              📄 Export Report
            </button>
          </div>

          <div className="research-container">
            {research.summary && (
              <div className="research-section">
                <h3>📋 Executive Summary</h3>
                <p>{research.summary}</p>
              </div>
            )}

            {research.definitions && (
              <div className="research-section">
                <h3>📖 Definitions & Context</h3>
                <div className="definition-list">
                  {Object.entries(research.definitions).map(([term, def]) => (
                    <div key={term} className="definition-item">
                      <strong>{term}:</strong> {def}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {research.classifications && (
              <div className="research-section">
                <h3>🏷️ Classifications</h3>
                <div className="classification-grid">
                  {Object.entries(research.classifications).map(([system, codes]) => (
                    <div key={system} className="classification-item">
                      <h4>{system}</h4>
                      <ul>
                        {codes.map((code, idx) => (
                          <li key={idx}>{code.code}: {code.description}</li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {research.relationships && (
              <div className="research-section">
                <h3>🔗 Related Concepts</h3>
                <div className="relationship-list">
                  {research.relationships.map((rel, idx) => (
                    <div key={idx} className="relationship-item">
                      <span className="rel-type">{rel.type}:</span>
                      <span className="rel-concept">{rel.concept}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {research.clinical_context && (
              <div className="research-section">
                <h3>🏥 Clinical Context</h3>
                <p>{research.clinical_context}</p>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  const exploreRelatedConcepts = async (conceptId) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/related-concepts/${conceptId}`);
      const relatedMessage = {
        id: Date.now() + '_related',
        type: 'assistant',
        content: `Related concepts for ${conceptId}:\n\n${response.data.map(c => `• ${c.concept_name} (${c.concept_id})`).join('\n')}`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, relatedMessage]);
    } catch (error) {
      console.error('Error fetching related concepts:', error);
    }
  };

  const exportResearchReport = async (query, research) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/research-query/export-html`, {
        query: query,
        results: research
      }, {
        responseType: 'blob'
      });

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `research_report_${query.replace(/\s+/g, '_')}_${Date.now()}.html`);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (error) {
      console.error('Export error:', error);
      alert('Export failed. Please try again.');
    }
  };

  const renderHybridMessage = (message) => {
    const results = message.content;
    const apiCount = results.api_results?.length || 0;
    const validatedCount = results.validated_results?.length || 0;
    const discoveryCount = results.discovery_results?.length || 0;

    return (
      <div key={message.id} className="message-container assistant-message">
        <div className="message-bubble assistant-bubble">
          <div className="message-header">
            <div className="message-icon">🧬</div>
            <div className="message-title">
              Hybrid AI + API Results
              <div className="message-subtitle">
                {apiCount} API • {validatedCount} Validated • {discoveryCount} Discoveries
              </div>
            </div>
            <div className="export-buttons">
              <button 
                onClick={() => downloadResults(message.query, message.searchMode, 'csv')}
                className="download-btn"
                title="Download Enhanced CSV"
              >
                📊
              </button>
              <button 
                onClick={() => downloadResults(message.query, message.searchMode, 'html')}
                className="download-btn"
                title="Export as HTML Report"
              >
                📄
              </button>
            </div>
          </div>

          <div className="results-container">
            {/* API Results */}
            {results.api_results && results.api_results.length > 0 && (
              <div className="result-section">
                <div className="section-header api-header">
                  🔗 Authoritative API Results ({results.api_results.length})
                </div>
                <div className="results-grid">
                  {results.api_results.slice(0, 5).map((result, idx) => (
                    <div key={idx} className="result-card api-card">
                      <div className="result-code">{result.concept_id}</div>
                      <div className="result-name">{result.concept_name}</div>
                      <div className="result-source">{result.source_ontology}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Validated Results */}
            {results.validated_results && results.validated_results.length > 0 && (
              <div className="result-section">
                <div className="section-header validated-header">
                  ✅ AI-Validated Results ({results.validated_results.length})
                </div>
                <div className="results-grid">
                  {results.validated_results.slice(0, 5).map((result, idx) => (
                    <div key={idx} className="result-card validated-card">
                      <div className="result-code">{result.concept_id}</div>
                      <div className="result-name">{result.concept_name}</div>
                      <div className="result-confidence">
                        {result.validation_confidence ? 
                          `${(result.validation_confidence * 100).toFixed(0)}% confidence` :
                          'AI + API confirmed'
                        }
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Discovery Results */}
            {results.discovery_results && results.discovery_results.length > 0 && (
              <div className="result-section">
                <div className="section-header discovery-header">
                  🔍 AI Semantic Discoveries ({results.discovery_results.length})
                </div>
                <div className="results-grid">
                  {results.discovery_results.slice(0, 5).map((result, idx) => (
                    <div key={idx} className="result-card discovery-card">
                      <div className="result-code">{result.concept_id}</div>
                      <div className="result-name">{result.concept_name}</div>
                      <div className="result-note">Requires validation</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="app">
      {/* Header */}
      <div className="header">
        <div className="header-content">
          <h1 className="header-title">🏥 Medical Terminology Assistant</h1>
          <div className="header-controls">
            <div className="mode-toggle-header">
              <button 
                className={`mode-btn ${currentMode === 'chat' ? 'active' : ''}`}
                onClick={() => setCurrentMode('chat')}
              >
                💬 Chat
              </button>
              <button 
                className={`mode-btn ${currentMode === 'research' ? 'active' : ''}`}
                onClick={() => setCurrentMode('research')}
              >
                🔬 Research
              </button>
              <button 
                className={`mode-btn ${currentMode === 'csv' ? 'active' : ''}`}
                onClick={() => setCurrentMode('csv')}
              >
                📊 CSV Upload
              </button>
            </div>
            <div className="header-status">
              {vectorDbStatus && (
                <div className={`status-indicator ${vectorDbStatus.status === 'connected' ? 'connected' : 'disconnected'}`}>
                  <span className="status-dot"></span>
                  {vectorDbStatus.status === 'connected' ? 
                    'BioBERT Active (5.4M concepts)' : 
                    'API Only Mode'
                  }
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="chat-container">
        {currentMode === 'chat' ? (
          <div className="messages">
            {messages.map(renderMessage)}
            {isLoading && (
              <div className="message-container assistant-message">
                <div className="message-bubble assistant-bubble loading">
                  <div className="message-icon">⏳</div>
                  <div className="loading-text">
                    <div className="loading-dots">
                      <span></span><span></span><span></span>
                    </div>
                    Searching medical terminology databases...
                  </div>
                </div>
              </div>
            )}
          </div>
        ) : (
          <CSVUpload 
            onUploadSuccess={handleCSVUploadSuccess}
            onUploadError={handleCSVUploadError}
          />
        )}
      </div>

      {/* Input Area - Show in chat and research modes */}
      {(currentMode === 'chat' || currentMode === 'research') && (
        <div className="input-area">
          <div className="input-container">
            {/* Search Mode Toggle */}
            <div className="mode-toggle">
              <button 
                className={`mode-btn ${searchMode === 'api_only' ? 'active' : ''}`}
                onClick={() => setSearchMode('api_only')}
              >
                🔗 API Only
              </button>
              <button 
                className={`mode-btn ${searchMode === 'hybrid' ? 'active' : ''}`}
                onClick={() => setSearchMode('hybrid')}
                disabled={vectorDbStatus?.status !== 'connected'}
              >
                🧬 API + RAG
              </button>
            </div>

            {/* Input Field */}
            <div className="input-wrapper">
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask about medical conditions, medications, lab tests, or diagnostic codes..."
                className="message-input"
                rows="1"
                disabled={isLoading}
              />
              <button 
                onClick={() => setShowAbbreviations(!showAbbreviations)}
                className="abbr-btn"
                title="Toggle medical abbreviations"
              >
                📖
              </button>
              <button 
                onClick={handleSubmit}
                disabled={isLoading || !inputText.trim()}
                className="send-btn"
              >
                {isLoading ? '⏳' : '➤'}
              </button>
            </div>

            {/* Mode Description */}
            <div className="mode-description">
              {currentMode === 'research' ? 
                '🔬 Comprehensive multi-agent research analysis with detailed medical context' :
                searchMode === 'api_only' ? 
                  '🔗 AI-powered chat with access to 5 medical APIs' :
                  '🧬 Enhanced AI chat with semantic discovery across 5.4M concepts'
              }
            </div>
            
            {/* Abbreviations Helper */}
            {showAbbreviations && abbreviations && (
              <div className="abbreviations-panel">
                <div className="panel-header">
                  <h3>Medical Abbreviations</h3>
                  <button onClick={() => setShowAbbreviations(false)}>✕</button>
                </div>
                <div className="abbreviations-list">
                  {Object.entries(abbreviations).slice(0, 20).map(([abbr, expansions]) => (
                    <div key={abbr} className="abbreviation-item">
                      <strong>{abbr}:</strong> {expansions.join(', ')}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;