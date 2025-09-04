import React from 'react';
import { Review } from '../types/Review';
import './ReviewDisplay.css';

interface ReviewDisplayProps {
  reviews: Review[];
}

const ReviewDisplay: React.FC<ReviewDisplayProps> = ({ reviews }) => {
  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'positive': return '#28a745';
      case 'negative': return '#dc3545';
      default: return '#6c757d';
    }
  };

  const getSentimentEmoji = (sentiment: string) => {
    switch (sentiment) {
      case 'positive': return '😊';
      case 'negative': return '😞';
      default: return '🤔';
    }
  };

  const formatConfidence = (confidence: number) => {
    return (confidence * 100).toFixed(1);
  };

  const formatProbability = (prob: number) => {
    return (prob * 100).toFixed(1);
  };

  if (reviews.length === 0) {
    return (
      <div className="review-display">
        <h2>AI Analysis Results</h2>
        <div className="no-reviews">
          <p>No reviews yet. Write your first review to see AI analysis!</p>
          <div className="analysis-info">
            <h4>What you'll see:</h4>
            <ul>
              <li><strong>Sentiment Analysis:</strong> AI-powered emotion detection (Positive/Negative) using trained logistic regression model</li>
              <li><strong>Topic Detection:</strong> Automatic topic discovery using Gensim LDA model</li>
              <li><strong>Confidence Scores:</strong> Probability distributions for all predictions</li>
            </ul>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="review-display">
      <h2>AI Analysis Results</h2>
      <div className="results-summary">
        <span className="review-count">{reviews.length} review{reviews.length !== 1 ? 's' : ''} analyzed</span>
      </div>
      <div className="reviews-list">
        {reviews.map((review) => (
          <div key={review.id} className="review-card">
            <div className="review-header">
              <span className="review-timestamp">
                {review.timestamp.toLocaleString()}
              </span>
              <span className="ai-badge">AI Analyzed</span>
            </div>
            
            <div className="review-text">
              <p>{review.text}</p>
            </div>
            
            <div className="analysis-section">
              <div className="sentiment-analysis">
                <h4>
                  <span className="analysis-icon">🧠</span>
                  Sentiment Analysis (AI Model)
                </h4>
                <div 
                  className="sentiment-result"
                  style={{ backgroundColor: getSentimentColor(review.sentiment.label) }}
                >
                  <span className="sentiment-emoji">
                    {getSentimentEmoji(review.sentiment.label)}
                  </span>
                  <span className="sentiment-label">
                    {review.sentiment.label.toUpperCase()}
                  </span>
                  <span className="sentiment-confidence">
                    {formatConfidence(review.sentiment.confidence)}% confidence
                  </span>
                </div>
                
                <div className="probability-bars">
                  <h5>Probability Distribution:</h5>
                  {Object.entries(review.sentiment.probabilities).map(([emotion, prob]) => (
                    <div key={emotion} className="probability-bar">
                      <label>{emotion.charAt(0).toUpperCase() + emotion.slice(1)}</label>
                      <div className="bar-container">
                        <div 
                          className="bar-fill" 
                          style={{ 
                            width: `${prob * 100}%`,
                            backgroundColor: getSentimentColor(emotion)
                          }}
                        />
                        <span className="bar-value">{formatProbability(prob)}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              
              <div className="topic-analysis">
                <h4>
                  <span className="analysis-icon">📊</span>
                  Topic Detection (LDA Model)
                </h4>
                <div className="dominant-topic">
                  <strong>Primary Topic:</strong> {review.topics.dominant_topic.name}
                  <span className="topic-confidence">
                    ({formatProbability(review.topics.dominant_topic.probability)}% probability)
                  </span>
                </div>
                
                {review.topics.topics && review.topics.topics.length > 0 && (
                  <div className="all-topics">
                    <h5>All Detected Topics:</h5>
                    <div className="topics-list">
                      {review.topics.topics.map((topic) => (
                        <div key={topic.id} className="topic-item">
                          <div className="topic-header">
                            <span className="topic-name">{topic.name}</span>
                            <span className="topic-prob">
                              {formatProbability(topic.probability)}%
                            </span>
                          </div>
                          {topic.words && topic.words.length > 0 && (
                            <div className="topic-words">
                              <strong>Key terms:</strong> {topic.words.join(', ')}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ReviewDisplay;