"""
Simplified Logistic Regression Classifier for Sentiment Analysis
Uses sklearn's built-in functionality for cleaner, more maintainable code
"""

from sklearn.linear_model import LogisticRegression
from base_classifier import BaseSentimentClassifier


class SimplifiedLogisticRegressionAnalyzer(BaseSentimentClassifier):
    """
    Simplified Logistic Regression Classifier for sentiment analysis
    Clean implementation using sklearn's built-in functionality
    """
    
    def __init__(self, random_state=42):
        super().__init__(random_state)
        self.model_name = "Logistic Regression"
    
    def get_default_model(self):
        """
        Return Logistic Regression with sensible default parameters
        """
        return LogisticRegression(
            C=1.0,
            penalty='l2',
            solver='liblinear',
            max_iter=1000,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
    
    def get_param_grid(self):
        """
        Return parameter grid for hyperparameter optimization
        """
        return {
            'C': [0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000, 2000]
        }
    
    def get_feature_coefficients(self, top_n=10):
        """
        Get top positive and negative features based on coefficients
        """
        if self.model is None:
            return None
        
        feature_names = self.vectorizer.get_feature_names_out()
        coefficients = self.model.coef_[0]
        
        # Top positive features
        pos_indices = coefficients.argsort()[-top_n:][::-1]
        positive_features = [(feature_names[i], coefficients[i]) for i in pos_indices]
        
        # Top negative features  
        neg_indices = coefficients.argsort()[:top_n]
        negative_features = [(feature_names[i], coefficients[i]) for i in neg_indices]
        
        return {
            'positive': positive_features,
            'negative': negative_features
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
        
        # Show feature coefficients
        coefficients = self.get_feature_coefficients(5)
        if coefficients:
            print(f"\nTop 5 Positive Features:")
            for feature, coef in coefficients['positive']:
                print(f"  {feature}: {coef:.4f}")
            
            print(f"\nTop 5 Negative Features:")
            for feature, coef in coefficients['negative']:
                print(f"  {feature}: {coef:.4f}")
        
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
    lr_classifier = SimplifiedLogisticRegressionAnalyzer()
    results = lr_classifier.quick_train(sample_data, optimize=False)
    
    # Make predictions
    test_texts = ['This is great!', 'I hate it']
    predictions = lr_classifier.predict(test_texts)
    
    for text, pred in zip(test_texts, predictions):
        print(f"Text: '{text}' -> {pred['sentiment']} (confidence: {pred['confidence']:.3f})")
