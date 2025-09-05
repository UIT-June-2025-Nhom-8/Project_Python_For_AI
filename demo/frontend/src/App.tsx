import React, { useState, useEffect } from 'react';
import './App.css';
import ReviewInput from './components/ReviewInput';
import ReviewDisplay from './components/ReviewDisplay';
import { Review, ModelListResponse } from './types/Review';
import { getAvailableModels } from './services/modelService';

function App() {
  const [reviews, setReviews] = useState<Review[]>([]);
  const [availableModels, setAvailableModels] = useState<ModelListResponse | null>(null);

  useEffect(() => {
    loadAvailableModels();
  }, []);

  const loadAvailableModels = async () => {
    try {
      const models = await getAvailableModels();
      setAvailableModels(models);
    } catch (error) {
      console.error('Error loading models for header:', error);
      // Don't set error state here as it's not critical for app functionality
    }
  };

  const handleAddReview = (review: Review) => {
    setReviews(prev => [review, ...prev]);
  };

  const clearReviews = () => {
    setReviews([]);
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>🤖 AI Product Review Analyzer</h1>
        <p>Advanced sentiment analysis and topic detection using trained ML models</p>
        <div className="app-description">
          <div className="feature-badges">
            {availableModels ? (
              <>
                <span className="badge">
                  {availableModels.sentiment_models.length} Sentiment Model{availableModels.sentiment_models.length !== 1 ? 's' : ''}
                </span>
                <span className="badge">
                  {availableModels.topic_models.length} Topic Model{availableModels.topic_models.length !== 1 ? 's' : ''}
                </span>
                <span className="badge">Real-time Analysis</span>
              </>
            ) : (
              <>
                <span className="badge">Loading Models...</span>
                <span className="badge">Real-time Analysis</span>
              </>
            )}
          </div>
        </div>
      </header>
      
      <main className="app-main">
        <ReviewInput onAddReview={handleAddReview} />
        <div className="display-section">
          {reviews.length > 0 && (
            <div className="results-controls">
              <button onClick={clearReviews} className="clear-button">
                Clear All Results
              </button>
            </div>
          )}
          <ReviewDisplay reviews={reviews} />
        </div>
      </main>
      
      <footer className="app-footer">
        <p>
          <strong>Note:</strong> This application requires a Python backend server with trained ML models. 
          Ensure the backend is running for full functionality.
        </p>
      </footer>
    </div>
  );
}

export default App;