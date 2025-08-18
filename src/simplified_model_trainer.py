"""
Simplified Model Trainer for Sentiment Analysis
Clean implementation that uses standardized classifiers
"""

import pandas as pd
import json
import os
import time
from datetime import datetime
from simplified_random_forest import SimplifiedRandomForestAnalyzer
from simplified_logistic_regression import SimplifiedLogisticRegressionAnalyzer
from simplified_gradient_boosting import SimplifiedGradientBoostingAnalyzer


class SimplifiedModelTrainer:
    """
    Simplified model trainer that handles multiple classifiers
    Uses standardized interface for clean, maintainable code
    """
    
    def __init__(self, output_dir="reports"):
        self.output_dir = output_dir
        self.results = {}
        self.trained_models = {}
        
        # Initialize all classifiers
        self.classifiers = {
            'random_forest': SimplifiedRandomForestAnalyzer(),
            'logistic_regression': SimplifiedLogisticRegressionAnalyzer(),
            'gradient_boosting': SimplifiedGradientBoostingAnalyzer()
        }
    
    def train_all_models(self, df, text_column='text', target_column='sentiment', 
                        test_size=0.2, optimize_hyperparameters=False):
        """
        Train all classifiers on the same dataset
        """
        print("="*60)
        print("TRAINING ALL SENTIMENT CLASSIFIERS")
        print("="*60)
        
        results = {}
        
        for name, classifier in self.classifiers.items():
            print(f"\n{'-'*20} {name.upper()} {'-'*20}")
            
            try:
                # Train the model
                start_time = time.time()
                model_results = classifier.quick_train(
                    df, text_column, target_column, test_size, optimize_hyperparameters
                )
                
                # Store results
                results[name] = {
                    'model_name': classifier.model_name,
                    'training_time': time.time() - start_time,
                    'test_accuracy': model_results['test_accuracy'],
                    'test_f1': model_results['test_f1'],
                    'train_accuracy': model_results['train_accuracy'],
                    'train_f1': model_results['train_f1'],
                    'classification_report': model_results['classification_report'],
                    'confusion_matrix': model_results['confusion_matrix'].tolist()
                }
                
                # Store trained model
                self.trained_models[name] = classifier
                
                print(f"✅ {name} completed successfully")
                
            except Exception as e:
                print(f"❌ Error training {name}: {str(e)}")
                results[name] = {'error': str(e)}
        
        self.results = results
        return results
    
    def compare_models(self):
        """
        Compare performance of all trained models
        """
        if not self.results:
            print("No results available. Train models first.")
            return
        
        print("\n" + "="*80)
        print("MODEL COMPARISON RESULTS")
        print("="*80)
        
        # Create comparison table
        comparison_data = []
        for name, result in self.results.items():
            if 'error' not in result:
                comparison_data.append({
                    'Model': result['model_name'],
                    'Test Accuracy': f"{result['test_accuracy']:.4f}",
                    'Test F1-Score': f"{result['test_f1']:.4f}",
                    'Training Time': f"{result['training_time']:.2f}s",
                    'Overfitting Gap': f"{result['train_accuracy'] - result['test_accuracy']:.4f}"
                })
        
        # Display as table
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            print(df_comparison.to_string(index=False))
            
            # Find best model
            best_accuracy = max(comparison_data, key=lambda x: float(x['Test Accuracy']))
            best_f1 = max(comparison_data, key=lambda x: float(x['Test F1-Score']))
            
            print(f"\n🏆 Best Accuracy: {best_accuracy['Model']} ({best_accuracy['Test Accuracy']})")
            print(f"🏆 Best F1-Score: {best_f1['Model']} ({best_f1['Test F1-Score']})")
        
        print("="*80)
    
    def save_results(self, filename=None):
        """
        Save training results to JSON file
        """
        if not self.results:
            print("No results to save")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simplified_model_results_{timestamp}.json"
        
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, filename)
        
        # Add metadata
        save_data = {
            'timestamp': datetime.now().isoformat(),
            'model_count': len(self.results),
            'results': self.results
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Results saved to: {filepath}")
        return filepath
    
    def save_models(self, model_dir="output/models"):
        """
        Save all trained models
        """
        if not self.trained_models:
            print("No trained models to save")
            return
        
        os.makedirs(model_dir, exist_ok=True)
        saved_paths = {}
        
        for name, model in self.trained_models.items():
            filepath = os.path.join(model_dir, f"simplified_{name}_model.pkl")
            model.save_model(filepath)
            saved_paths[name] = filepath
        
        print(f"All models saved to: {model_dir}")
        return saved_paths
    
    def predict_sentiment(self, texts, model_name='logistic_regression'):
        """
        Make predictions using a specific trained model
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained or not available")
        
        model = self.trained_models[model_name]
        return model.predict(texts)
    
    def get_model_summary(self):
        """
        Get summary of all trained models
        """
        if not self.results:
            return "No models trained yet"
        
        summary = {
            'total_models': len(self.results),
            'successful_models': len([r for r in self.results.values() if 'error' not in r]),
            'failed_models': len([r for r in self.results.values() if 'error' in r]),
            'best_accuracy': None,
            'best_f1': None
        }
        
        # Find best performing models
        successful_results = [(name, result) for name, result in self.results.items() 
                            if 'error' not in result]
        
        if successful_results:
            best_acc_model = max(successful_results, key=lambda x: x[1]['test_accuracy'])
            best_f1_model = max(successful_results, key=lambda x: x[1]['test_f1'])
            
            summary['best_accuracy'] = {
                'model': best_acc_model[0],
                'score': best_acc_model[1]['test_accuracy']
            }
            summary['best_f1'] = {
                'model': best_f1_model[0],
                'score': best_f1_model[1]['test_f1']
            }
        
        return summary


# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = pd.DataFrame({
        'text': [
            'I love this product, it is amazing!',
            'This is terrible, worst purchase ever',
            'Pretty good, would recommend',
            'Not bad, could be better',
            'Excellent quality and fast delivery',
            'Outstanding service and quality',
            'Completely disappointed with this',
            'Good value for money',
            'Would not recommend this at all',
            'Perfect, exactly what I needed'
        ] * 10,  # Repeat to have more samples
        'sentiment': [
            'positive', 'negative', 'positive', 'negative', 'positive',
            'positive', 'negative', 'positive', 'negative', 'positive'
        ] * 10
    })
    
    # Train all models
    trainer = SimplifiedModelTrainer()
    results = trainer.train_all_models(sample_data, optimize_hyperparameters=False)
    
    # Compare models
    trainer.compare_models()
    
    # Save results
    trainer.save_results()
    
    # Test predictions
    test_texts = ['This is amazing!', 'I hate this product']
    predictions = trainer.predict_sentiment(test_texts, 'logistic_regression')
    
    print(f"\nPrediction Test:")
    for text, pred in zip(test_texts, predictions):
        print(f"'{text}' -> {pred['sentiment']} (confidence: {pred['confidence']:.3f})")
