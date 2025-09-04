import React, { useState } from 'react';
import './App.css';
import ReviewInput from './components/ReviewInput';
import ReviewDisplay from './components/ReviewDisplay';
import { Review } from './types/Review';

function App() {
  const [reviews, setReviews] = useState<Review[]>([]);

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
            <span className="badge">Logistic Regression</span>
            <span className="badge">Gensim LDA</span>
            <span className="badge">Real-time Analysis</span>
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