import pandas as pd
import numpy as np
import json
import time
import os
from datetime import datetime


class ModelTrainer:
    """
    Train multiple models and compare results
    """
    
    def __init__(self, output_dir="reports"):
        """
        Initialize ModelTrainer
        
        Args:
            output_dir (str): Directory to save results
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    
    def run_training_pipeline_with_implemented_classifiers(self, X_train_tfidf, X_test_tfidf, y_train, y_test, 
                                                          optimize_hyperparameters=False, save_results=True):
        """
        Run complete training pipeline using implemented classifier files instead of direct sklearn imports
        
        Args:
            X_train_tfidf: TF-IDF matrix for training data
            X_test_tfidf: TF-IDF matrix for test data  
            y_train: Training labels
            y_test: Test labels
            optimize_hyperparameters (bool): Whether to optimize hyperparameters
            save_results (bool): Whether to save results
            
        Returns:
            dict: Training results
        """
        print("\n" + "="*100)
        print("STARTING MODEL TRAINING PIPELINE WITH IMPLEMENTED CLASSIFIERS")
        print("="*100)
        
        # Import the implemented classifiers
        from logistic_regression_classifier import LogisticRegressionAnalyzer
        from random_forest_classifier import RandomForestAnalyzer
        from gradient_boosting_classifier import GradientBoostingAnalyzer
        
        # Map labels: 1 -> negative (0), 2 -> positive (1) 
        label_mapping = {1: 0, 2: 1}
        y_train_mapped = y_train.map(label_mapping)
        y_test_mapped = y_test.map(label_mapping)
        
        print(f"Using pre-computed TF-IDF matrices:")
        print(f"   - X_train shape: {X_train_tfidf.shape}")
        print(f"   - X_test shape: {X_test_tfidf.shape}")
        print(f"   - y_train shape: {y_train_mapped.shape}")
        print(f"   - y_test shape: {y_test_mapped.shape}")
        
        pipeline_start_time = time.time()
        
        # Dictionary to store results
        all_results = {}
        
        # Train Logistic Regression using implemented classifier
        print(f"\n>>> Training Logistic Regression with implemented classifier...")
        try:
            start_time = time.time()
            
            # Initialize analyzer
            lr_analyzer = LogisticRegressionAnalyzer(optimize_hyperparameters=optimize_hyperparameters)
            
            # Set pre-computed data directly
            lr_analyzer.X_train = X_train_tfidf
            lr_analyzer.X_test = X_test_tfidf
            lr_analyzer.y_train = y_train_mapped
            lr_analyzer.y_test = y_test_mapped
            
            # Initialize and train model
            lr_analyzer.initialize_model()
            results = lr_analyzer.train_model()
            
            training_time = time.time() - start_time
            
            # Format results for consistency
            lr_results = {
                'model_name': 'LogisticRegression_Implemented',
                'train_accuracy': round(results.get('train_accuracy', 0), 4),
                'test_accuracy': round(results.get('test_accuracy', 0), 4),
                'f1_score': round(results.get('test_f1_weighted', 0), 4),
                'training_time_seconds': round(training_time, 2),
                'hyperparameter_optimization': optimize_hyperparameters,
                'classification_report': results.get('classification_report', {}),
                'implementation_source': 'logistic_regression_classifier.py'
            }
            
            all_results['Logistic_Regression_Implemented'] = lr_results
            print(f"Logistic Regression (implemented) training completed in {training_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error training Logistic Regression (implemented): {e}")
            all_results['Logistic_Regression_Implemented'] = {
                'model_name': 'LogisticRegression_Implemented',
                'error': str(e),
                'training_time_seconds': time.time() - start_time
            }
        
        # Train Random Forest using implemented classifier
        print(f"\n>>> Training Random Forest with implemented classifier...")
        try:
            start_time = time.time()
            
            # Initialize analyzer
            rf_analyzer = RandomForestAnalyzer(optimize_hyperparameters=optimize_hyperparameters)
            
            # Set pre-computed data directly
            rf_analyzer.X_train = X_train_tfidf
            rf_analyzer.X_test = X_test_tfidf
            rf_analyzer.y_train = y_train_mapped
            rf_analyzer.y_test = y_test_mapped
            
            # Initialize and train model
            rf_analyzer.initialize_model()
            results = rf_analyzer.train_model()
            
            training_time = time.time() - start_time
            
            # Format results for consistency
            rf_results = {
                'model_name': 'RandomForest_Implemented',
                'train_accuracy': round(results.get('train_accuracy', 0), 4),
                'test_accuracy': round(results.get('test_accuracy', 0), 4),
                'f1_score': round(results.get('test_f1_weighted', 0), 4),
                'training_time_seconds': round(training_time, 2),
                'hyperparameter_optimization': optimize_hyperparameters,
                'classification_report': results.get('classification_report', {}),
                'implementation_source': 'random_forest_classifier.py'
            }
            
            all_results['Random_Forest_Implemented'] = rf_results
            print(f"Random Forest (implemented) training completed in {training_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error training Random Forest (implemented): {e}")
            all_results['Random_Forest_Implemented'] = {
                'model_name': 'RandomForest_Implemented',
                'error': str(e),
                'training_time_seconds': time.time() - start_time
            }
        
        # Train Gradient Boosting using implemented classifier
        print(f"\n>>> Training Gradient Boosting with implemented classifier...")
        try:
            start_time = time.time()
            
            # Initialize analyzer
            gb_analyzer = GradientBoostingAnalyzer()
            
            # Set pre-computed data directly
            gb_analyzer.X_train = X_train_tfidf
            gb_analyzer.X_test = X_test_tfidf
            gb_analyzer.y_train = y_train_mapped
            gb_analyzer.y_test = y_test_mapped
            
            # Initialize and train model
            gb_analyzer.initialize_model()
            results = gb_analyzer.train_model()
            
            training_time = time.time() - start_time
            
            # Format results for consistency
            gb_results = {
                'model_name': 'GradientBoosting_Implemented',
                'train_accuracy': round(results.get('train_accuracy', 0), 4),
                'test_accuracy': round(results.get('test_accuracy', 0), 4),
                'f1_score': round(0, 4),  # GB classifier doesn't return f1_score in standard format
                'training_time_seconds': round(training_time, 2),
                'hyperparameter_optimization': False,  # GB analyzer doesn't support hyperparameter optimization
                'classification_report': results.get('classification_report', {}),
                'implementation_source': 'gradient_boosting_classifier.py'
            }
            
            all_results['Gradient_Boosting_Implemented'] = gb_results
            print(f"Gradient Boosting (implemented) training completed in {training_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error training Gradient Boosting (implemented): {e}")
            all_results['Gradient_Boosting_Implemented'] = {
                'model_name': 'GradientBoosting_Implemented',
                'error': str(e),
                'training_time_seconds': time.time() - start_time
            }
        
        # Calculate total time
        total_pipeline_time = time.time() - pipeline_start_time
        
        # Create summary
        pipeline_summary = self.create_pipeline_summary(all_results, total_pipeline_time)
        
        # Save results
        final_results = {
            'pipeline_summary': pipeline_summary,
            'individual_results': all_results,
            'training_config': {
                'optimize_hyperparameters': optimize_hyperparameters,
                'used_precomputed_tfidf': True,
                'used_implemented_classifiers': True,
                'tfidf_train_shape': X_train_tfidf.shape,
                'tfidf_test_shape': X_test_tfidf.shape,
                'train_samples': len(y_train),
                'test_samples': len(y_test),
                'implementation_note': 'Using custom implemented classifier files instead of direct sklearn imports'
            }
        }
        
        # Print summary
        self.print_summary(final_results)
        
        # Save results
        if save_results:
            filename = f"implemented_classifiers_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.save_results_to_json(final_results, filename)
        
        return final_results
