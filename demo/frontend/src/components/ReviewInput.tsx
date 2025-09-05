import React, { useState, useEffect } from 'react';
import { Review, ModelListResponse, ModelInfo } from '../types/Review';
import { analyzeSentiment, SentimentAnalysisError, checkSentimentAPIHealth } from '../services/sentimentService';
import { detectTopics, TopicAnalysisError, checkTopicAPIHealth } from '../services/topicService';
import { getAvailableModels, ModelServiceError } from '../services/modelService';
import './ReviewInput.css';

interface ReviewInputProps {
  onAddReview: (review: Review) => void;
}

const ReviewInput: React.FC<ReviewInputProps> = ({ onAddReview }) => {
  const [text, setText] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isBackendHealthy, setIsBackendHealthy] = useState<boolean | null>(null);
  const [isCheckingHealth, setIsCheckingHealth] = useState(false);
  
  // Model selection state
  const [availableModels, setAvailableModels] = useState<ModelListResponse | null>(null);
  const [selectedSentimentModel, setSelectedSentimentModel] = useState<string>('');
  const [selectedTopicModel, setSelectedTopicModel] = useState<string>('');
  const [isLoadingModels, setIsLoadingModels] = useState(false);

  // Check backend health on component mount
  useEffect(() => {
    checkBackendHealth();
    loadAvailableModels();
  }, []);

  const loadAvailableModels = async () => {
    setIsLoadingModels(true);
    try {
      const models = await getAvailableModels();
      setAvailableModels(models);
      
      // Set default selections
      if (models.current_sentiment_model) {
        setSelectedSentimentModel(models.current_sentiment_model);
      }
      if (models.current_topic_model) {
        setSelectedTopicModel(models.current_topic_model);
      }
      
    } catch (error) {
      console.error('Error loading models:', error);
      if (error instanceof ModelServiceError && !error.isNetworkError) {
        // Only set error if it's not a network error (which would be handled by health check)
        setError(`Failed to load available models: ${error.message}`);
      }
    } finally {
      setIsLoadingModels(false);
    }
  };

  const checkBackendHealth = async () => {
    setIsCheckingHealth(true);
    try {
      const [sentimentHealthy, topicHealthy] = await Promise.all([
        checkSentimentAPIHealth(),
        checkTopicAPIHealth()
      ]);
      const isHealthy = sentimentHealthy && topicHealthy;
      setIsBackendHealthy(isHealthy);
      
      // If backend becomes healthy and we don't have models, try to load them
      if (isHealthy && !availableModels) {
        loadAvailableModels();
      }
    } catch {
      setIsBackendHealthy(false);
    } finally {
      setIsCheckingHealth(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!text.trim()) {
      setError('Please enter some text to analyze');
      return;
    }

    if (isBackendHealthy === false) {
      setError('Backend API is not available. Please ensure the backend server is running.');
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    
    try {
      // Make both API calls and wait for both to complete
      const [sentiment, topics] = await Promise.all([
        analyzeSentiment(text, selectedSentimentModel || undefined),
        detectTopics(text, selectedTopicModel || undefined)
      ]);
      
      const newReview: Review = {
        id: Date.now().toString(),
        text: text.trim(),
        timestamp: new Date(),
        sentiment,
        topics
      };
      
      onAddReview(newReview);
      setText('');
      setError(null);
      
    } catch (error) {
      console.error('Error analyzing review:', error);
      
      if (error instanceof SentimentAnalysisError || error instanceof TopicAnalysisError) {
        if (error.isNetworkError) {
          setError(`Network Error: ${error.message}. Please check if the backend server is running and try again.`);
          setIsBackendHealthy(false);
        } else {
          setError(`Analysis Error: ${error.message}`);
        }
      } else {
        setError('An unexpected error occurred during analysis. Please try again.');
      }
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getHealthStatusDisplay = () => {
    if (isCheckingHealth) {
      return <span className="health-status checking">Checking backend...</span>;
    }
    
    if (isBackendHealthy === null) {
      return <span className="health-status unknown">Backend status unknown</span>;
    }
    
    return isBackendHealthy ? (
      <span className="health-status healthy">✓ Backend connected</span>
    ) : (
      <span className="health-status unhealthy">✗ Backend not available</span>
    );
  };

  return (
    <div className="review-input">
      <div className="review-input-header">
        <h2>Write a Product Review</h2>
        <div className="backend-status">
          {getHealthStatusDisplay()}
          <button 
            onClick={checkBackendHealth} 
            className="refresh-status-btn"
            disabled={isCheckingHealth}
            title="Refresh backend status"
          >
            🔄
          </button>
        </div>
      </div>

      {/* Model Selection Section */}
      {isBackendHealthy && availableModels && (
        <div className="model-selection">
          <h3>Select Analysis Models</h3>
          <div className="model-selectors">
            <div className="model-selector">
              <label htmlFor="sentiment-model">Sentiment Analysis Model:</label>
              <select
                id="sentiment-model"
                value={selectedSentimentModel}
                onChange={(e) => setSelectedSentimentModel(e.target.value)}
                disabled={isAnalyzing || isLoadingModels}
                className="model-select"
              >
                {availableModels.sentiment_models.map((model) => (
                  <option key={model.key} value={model.key}>
                    {model.name}
                  </option>
                ))}
              </select>
            </div>
            
            <div className="model-selector">
              <label htmlFor="topic-model">Topic Detection Model:</label>
              <select
                id="topic-model"
                value={selectedTopicModel}
                onChange={(e) => setSelectedTopicModel(e.target.value)}
                disabled={isAnalyzing || isLoadingModels}
                className="model-select"
              >
                {availableModels.topic_models.map((model) => (
                  <option key={model.key} value={model.key}>
                    {model.name} ({model.num_topics} topics)
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>
      )}
      
      <form onSubmit={handleSubmit} className="review-form">
        <textarea
          value={text}
          onChange={(e) => {
            setText(e.target.value);
            if (error && e.target.value.trim()) {
              setError(null); // Clear error when user starts typing
            }
          }}
          placeholder="Share your experience with this product... (Note: Analysis requires backend API)"
          rows={4}
          disabled={isAnalyzing}
          className="review-textarea"
        />
        
        {error && (
          <div className="error-message">
            <span className="error-icon">⚠️</span>
            {error}
          </div>
        )}
        
        <button 
          type="submit" 
          disabled={!text.trim() || isAnalyzing || isBackendHealthy === false}
          className="analyze-button"
        >
          {isAnalyzing ? (
            <>
              <span className="loading-spinner">⏳</span>
              Analyzing with AI...
            </>
          ) : (
            'Analyze with Backend AI'
          )}
        </button>
        
        {isBackendHealthy === false && (
          <div className="backend-warning">
            <p>
              <strong>Backend Required:</strong> This application requires the Python backend server 
              to perform AI analysis. Please ensure the backend is running on the configured port.
            </p>
            <div className="backend-instructions">
              <p>To start the backend:</p>
              <code>cd backend && python app.py</code>
            </div>
          </div>
        )}
      </form>
    </div>
  );
};

export default ReviewInput;