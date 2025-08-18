"""
Base Classifier for Sentiment Analysis
Provides common functionality for all sentiment classifiers
"""

import numpy as np
import pandas as pd
import pickle
import os
import time
from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    f1_score, precision_score, recall_score
)


class BaseSentimentClassifier(ABC):
    """
    Abstract base class for sentiment classifiers
    Provides common functionality and standardized interface
    """
    
    def __init__(self, random_state=42):
        self.model = None
        self.vectorizer = None
        self.label_encoder = LabelEncoder()
        self.random_state = random_state
        self.results = {}
        self.training_time = 0
        
    def prepare_data(self, df, text_column='text', target_column='sentiment', 
                     test_size=0.2, max_features=10000):
        """
        Prepare data for training - simplified version
        """
        print(f"Preparing data: {len(df)} samples")
        
        # Handle missing values
        df = df.dropna(subset=[text_column, target_column])
        
        # Vectorize text
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.9
            )
            X = self.vectorizer.fit_transform(df[text_column])
        else:
            X = self.vectorizer.transform(df[text_column])
        
        # Encode labels
        y = self.label_encoder.fit_transform(df[target_column])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}, Features: {X_train.shape[1]}")
        return X_train, X_test, y_train, y_test
    
    @abstractmethod
    def get_param_grid(self):
        """Return parameter grid for hyperparameter optimization"""
        pass
    
    @abstractmethod
    def get_default_model(self):
        """Return model with default parameters"""
        pass
    
    def optimize_hyperparameters(self, X_train, y_train, cv=3, scoring='f1_macro'):
        """
        Optimize hyperparameters using GridSearchCV
        """
        print(f"Optimizing hyperparameters with {cv}-fold CV...")
        
        base_model = self.get_default_model()
        param_grid = self.get_param_grid()
        
        grid_search = GridSearchCV(
            base_model, param_grid, cv=cv, scoring=scoring, 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        print(f"Best score: {grid_search.best_score_:.4f}")
        print(f"Best params: {grid_search.best_params_}")
        
        return grid_search.best_params_
    
    def train(self, X_train, X_test, y_train, y_test, optimize=False):
        """
        Train the model
        """
        start_time = time.time()
        
        if optimize:
            self.optimize_hyperparameters(X_train, y_train)
        else:
            if self.model is None:
                self.model = self.get_default_model()
            self.model.fit(X_train, y_train)
        
        self.training_time = time.time() - start_time
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        self.results = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'test_accuracy': accuracy_score(y_test, test_pred),
            'train_f1': f1_score(y_train, train_pred, average='macro'),
            'test_f1': f1_score(y_test, test_pred, average='macro'),
            'classification_report': classification_report(
                y_test, test_pred, target_names=self.label_encoder.classes_, output_dict=True
            ),
            'confusion_matrix': confusion_matrix(y_test, test_pred),
            'training_time': self.training_time
        }
        
        print(f"Training completed in {self.training_time:.2f}s")
        print(f"Test Accuracy: {self.results['test_accuracy']:.4f}")
        print(f"Test F1-Score: {self.results['test_f1']:.4f}")
        
        return self.results
    
    def predict(self, texts):
        """
        Predict sentiment for new texts
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        if isinstance(texts, str):
            texts = [texts]
        
        X = self.vectorizer.transform(texts)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        results = []
        for pred, prob in zip(predictions, probabilities):
            sentiment = self.label_encoder.inverse_transform([pred])[0]
            confidence = prob.max()
            results.append({
                'sentiment': sentiment,
                'confidence': confidence,
                'probabilities': dict(zip(self.label_encoder.classes_, prob))
            })
        
        return results
    
    def save_model(self, filepath):
        """
        Save the complete model
        """
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'results': self.results,
            'random_state': self.random_state
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a saved model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.label_encoder = model_data['label_encoder']
        self.results = model_data.get('results', {})
        
        print(f"Model loaded from {filepath}")
    
    def get_feature_importance(self, top_n=10):
        """
        Get feature importance if available
        """
        if not hasattr(self.model, 'feature_importances_'):
            return None
        
        feature_names = self.vectorizer.get_feature_names_out()
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def evaluate(self):
        """
        Print evaluation results
        """
        if not self.results:
            print("No results available. Train the model first.")
            return
        
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"Training Accuracy: {self.results['train_accuracy']:.4f}")
        print(f"Test Accuracy: {self.results['test_accuracy']:.4f}")
        print(f"Training F1-Score: {self.results['train_f1']:.4f}")
        print(f"Test F1-Score: {self.results['test_f1']:.4f}")
        print(f"Training Time: {self.results['training_time']:.2f}s")
        
        print("\nClassification Report:")
        report_df = pd.DataFrame(self.results['classification_report']).T
        print(report_df.round(4))
        
        print("\nConfusion Matrix:")
        print(self.results['confusion_matrix'])
        
        # Feature importance if available
        importance_df = self.get_feature_importance()
        if importance_df is not None:
            print(f"\nTop 10 Most Important Features:")
            for idx, row in importance_df.iterrows():
                print(f"{row['feature']}: {row['importance']:.4f}")
        
        print("="*50)
