"""
Simplified Gradient Boosting Classifier for Sentiment Analysis
Uses sklearn's built-in functionality for cleaner, more maintainable code
"""

from sklearn.ensemble import GradientBoostingClassifier
from base_classifier import BaseSentimentClassifier


class SimplifiedGradientBoostingAnalyzer(BaseSentimentClassifier):
    """
    Simplified Gradient Boosting Classifier for sentiment analysis
    Clean implementation using sklearn's built-in functionality
    """
    
    def __init__(self, random_state=42):
        super().__init__(random_state)
        self.model_name = "Gradient Boosting"
    
    def get_default_model(self):
        """
        Return Gradient Boosting with sensible default parameters
        """
        return GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            max_features='sqrt',
            random_state=self.random_state,
            validation_fraction=0.1,
            n_iter_no_change=10,
            tol=1e-4
        )
    
    def get_param_grid(self):
        """
        Return parameter grid for hyperparameter optimization
        """
        return {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5],
            'min_samples_split': [10, 20, 50],
            'subsample': [0.8, 0.9, 1.0]
        }
    
    def get_training_progress(self):
        """
        Get training progress information
        """
        if self.model is None:
            return None
        
        return {
            'train_scores': self.model.train_score_.tolist(),
            'n_estimators_used': len(self.model.train_score_),
            'early_stopping': len(self.model.train_score_) < self.model.n_estimators
        }
    
    def quick_train(self, df, text_column='text', target_column='sentiment', 
                   test_size=0.2, optimize=False):
        """
        Quick training method - prepares data and trains in one step
        """
        print(f"Training {self.model_name} Classifier...")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(
            df, text_column, target_column, test_size
        )
        
        # Train model
        results = self.train(X_train, X_test, y_train, y_test, optimize)
        
        # Show feature importance
        importance_df = self.get_feature_importance()
        if importance_df is not None:
            print(f"\nTop 5 Most Important Features:")
            for idx, row in importance_df.head(5).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Show training progress
        progress = self.get_training_progress()
        if progress:
            print(f"\nTraining Progress:")
            print(f"  Estimators used: {progress['n_estimators_used']}")
            print(f"  Early stopping: {progress['early_stopping']}")
            print(f"  Final train score: {progress['train_scores'][-1]:.4f}")
        
        return results


# Example usage
if __name__ == "__main__":
    # Example with sample data
    import pandas as pd
    
    # Create sample data
    sample_data = pd.DataFrame({
        'text': [
            'I love this product, it is amazing!',
            'This is terrible, worst purchase ever',
            'Pretty good, would recommend',
            'Not bad, could be better',
            'Excellent quality and fast delivery'
        ],
        'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive']
    })
    
    # Train classifier
    gb_classifier = SimplifiedGradientBoostingAnalyzer()
    results = gb_classifier.quick_train(sample_data, optimize=False)
    
    # Make predictions
    test_texts = ['This is great!', 'I hate it']
    predictions = gb_classifier.predict(test_texts)
    
    for text, pred in zip(test_texts, predictions):
        print(f"Text: '{text}' -> {pred['sentiment']} (confidence: {pred['confidence']:.3f})")
