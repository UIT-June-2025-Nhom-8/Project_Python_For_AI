"""
Simplified Random Forest Classifier for Sentiment Analysis
Uses sklearn's built-in functionality for cleaner, more maintainable code
"""

from sklearn.ensemble import RandomForestClassifier
from base_classifier import BaseSentimentClassifier


class SimplifiedRandomForestAnalyzer(BaseSentimentClassifier):
    """
    Simplified Random Forest Classifier for sentiment analysis
    Clean implementation using sklearn's built-in functionality
    """
    
    def __init__(self, random_state=42):
        super().__init__(random_state)
        self.model_name = "Random Forest"
    
    def get_default_model(self):
        """
        Return Random Forest with sensible default parameters
        """
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
    
    def get_param_grid(self):
        """
        Return parameter grid for hyperparameter optimization
        """
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [5, 10, 20],
            'min_samples_leaf': [2, 5, 10],
            'max_features': ['sqrt', 'log2']
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
    rf_classifier = SimplifiedRandomForestAnalyzer()
    results = rf_classifier.quick_train(sample_data, optimize=False)
    
    # Make predictions
    test_texts = ['This is great!', 'I hate it']
    predictions = rf_classifier.predict(test_texts)
    
    for text, pred in zip(test_texts, predictions):
        print(f"Text: '{text}' -> {pred['sentiment']} (confidence: {pred['confidence']:.3f})")
