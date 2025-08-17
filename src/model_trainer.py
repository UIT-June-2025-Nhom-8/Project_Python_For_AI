import pandas as pd
import numpy as np
import json
import time
import os
from datetime import datetime


class ModelTrainer:
    """
    Simplified class to train multiple models and compare results
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
        
    def train_logistic_regression_with_tfidf(self, X_train_tfidf, X_test_tfidf, y_train, y_test, optimize_hyperparameters=True):
        """
        Train Logistic Regression with pre-computed TF-IDF matrix
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import accuracy_score, classification_report, f1_score
        
        print("\n" + "="*60)
        print("TRAINING LOGISTIC REGRESSION WITH PRE-COMPUTED TF-IDF")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # Initialize model
            if optimize_hyperparameters:
                print("Running hyperparameter optimization...")
                param_grid = {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
                lr = LogisticRegression(random_state=42, max_iter=1000)
                grid_search = GridSearchCV(lr, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
                grid_search.fit(X_train_tfidf, y_train)
                model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                print(f"Best parameters: {best_params}")
            else:
                model = LogisticRegression(random_state=42, max_iter=1000)
                model.fit(X_train_tfidf, y_train)
                best_params = None
            
            # Predictions
            y_train_pred = model.predict(X_train_tfidf)
            y_test_pred = model.predict(X_test_tfidf)
            
            # Metrics
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            f1 = f1_score(y_test, y_test_pred, average='weighted')
            
            training_time = time.time() - start_time
            
            print(f"Training Accuracy: {train_acc:.4f}")
            print(f"Test Accuracy: {test_acc:.4f}")
            print(f"F1 Score: {f1:.4f}")
            
            results = {
                'model_name': 'LogisticRegression',
                'train_accuracy': round(train_acc, 4),
                'test_accuracy': round(test_acc, 4),
                'f1_score': round(f1, 4),
                'training_time_seconds': round(training_time, 2),
                'hyperparameter_optimization': optimize_hyperparameters,
                'best_params': best_params,
                'classification_report': classification_report(y_test, y_test_pred, output_dict=True)
            }
            
            print(f"Logistic Regression training completed in {training_time:.2f} seconds")
            return results
            
        except Exception as e:
            print(f"Error training Logistic Regression: {e}")
            return {
                'model_name': 'LogisticRegression',
                'error': str(e),
                'training_time_seconds': time.time() - start_time
            }
    
    def train_random_forest_with_tfidf(self, X_train_tfidf, X_test_tfidf, y_train, y_test, optimize_hyperparameters=False):
        """
        Train Random Forest with pre-computed TF-IDF matrix
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import accuracy_score, classification_report, f1_score
        
        print("\n" + "="*60)
        print("TRAINING RANDOM FOREST WITH PRE-COMPUTED TF-IDF")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # Initialize model
            if optimize_hyperparameters:
                print("Running hyperparameter optimization...")
                param_grid = {
                    'n_estimators': [50, 100],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
                rf = RandomForestClassifier(random_state=42, n_jobs=-1)
                grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
                grid_search.fit(X_train_tfidf, y_train)
                model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                print(f"Best parameters: {best_params}")
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(X_train_tfidf, y_train)
                best_params = None
            
            # Predictions
            y_train_pred = model.predict(X_train_tfidf)
            y_test_pred = model.predict(X_test_tfidf)
            
            # Metrics
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            f1 = f1_score(y_test, y_test_pred, average='weighted')
            
            training_time = time.time() - start_time
            
            print(f"Training Accuracy: {train_acc:.4f}")
            print(f"Test Accuracy: {test_acc:.4f}")
            print(f"F1 Score: {f1:.4f}")
            
            results = {
                'model_name': 'RandomForestClassifier',
                'train_accuracy': round(train_acc, 4),
                'test_accuracy': round(test_acc, 4),
                'f1_score': round(f1, 4),
                'training_time_seconds': round(training_time, 2),
                'hyperparameter_optimization': optimize_hyperparameters,
                'best_params': best_params,
                'classification_report': classification_report(y_test, y_test_pred, output_dict=True)
            }
            
            print(f"Random Forest training completed in {training_time:.2f} seconds")
            return results
            
        except Exception as e:
            print(f"Error training Random Forest: {e}")
            return {
                'model_name': 'RandomForestClassifier',
                'error': str(e),
                'training_time_seconds': time.time() - start_time
            }
    
    def train_gradient_boosting_with_tfidf(self, X_train_tfidf, X_test_tfidf, y_train, y_test):
        """
        Train Gradient Boosting with pre-computed TF-IDF matrix
        """
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import accuracy_score, classification_report, f1_score
        
        print("\n" + "="*60)
        print("TRAINING GRADIENT BOOSTING WITH PRE-COMPUTED TF-IDF")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # Initialize model with optimal parameters
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            )
            
            print("Training Gradient Boosting Classifier...")
            model.fit(X_train_tfidf, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train_tfidf)
            y_test_pred = model.predict(X_test_tfidf)
            
            # Metrics
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            f1 = f1_score(y_test, y_test_pred, average='weighted')
            
            training_time = time.time() - start_time
            
            print(f"Training Accuracy: {train_acc:.4f}")
            print(f"Test Accuracy: {test_acc:.4f}")
            print(f"F1 Score: {f1:.4f}")
            
            results = {
                'model_name': 'GradientBoostingClassifier',
                'train_accuracy': round(train_acc, 4),
                'test_accuracy': round(test_acc, 4),
                'f1_score': round(f1, 4),
                'training_time_seconds': round(training_time, 2),
                'hyperparameter_optimization': False,
                'classification_report': classification_report(y_test, y_test_pred, output_dict=True)
            }
            
            print(f"Gradient Boosting training completed in {training_time:.2f} seconds")
            return results
            
        except Exception as e:
            print(f"Error training Gradient Boosting: {e}")
            return {
                'model_name': 'GradientBoostingClassifier',
                'error': str(e),
                'training_time_seconds': time.time() - start_time
            }

    def create_pipeline_summary(self, all_results, total_time):
        """
        Create summary comparing all models
        
        Args:
            all_results (dict): Results from all models
            total_time (float): Total training time
            
        Returns:
            dict: Pipeline summary
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_pipeline_time_seconds': round(total_time, 2),
            'models_trained': len(all_results),
            'comparison': {}
        }
        
        # Compare accuracy
        accuracies = {}
        training_times = {}
        
        for model_name, results in all_results.items():
            if 'error' not in results:
                accuracies[model_name] = results.get('test_accuracy', 0)
                training_times[model_name] = results.get('training_time_seconds', 0)
        
        if accuracies:
            best_model = max(accuracies, key=accuracies.get)
            fastest_model = min(training_times, key=training_times.get)
            
            summary['comparison'] = {
                'best_accuracy_model': best_model,
                'best_accuracy_score': round(accuracies[best_model], 4),
                'fastest_model': fastest_model,
                'fastest_time': round(training_times[fastest_model], 2),
                'all_accuracies': {k: round(v, 4) for k, v in accuracies.items()},
                'all_training_times': {k: round(v, 2) for k, v in training_times.items()}
            }
        
        return summary
    
    def save_results_to_json(self, results, filename=None):
        """
        Save results to JSON file
        
        Args:
            results (dict): Training results
            filename (str): File name (optional)
            
        Returns:
            str: Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_training_results_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                converted_dict = {}
                for key, value in obj.items():
                    # Convert numpy types in keys to standard Python types
                    if isinstance(key, np.integer):
                        converted_key = int(key)
                    elif isinstance(key, np.floating):
                        converted_key = float(key)
                    elif isinstance(key, np.ndarray):
                        converted_key = str(key.tolist())
                    else:
                        converted_key = key
                    
                    converted_dict[converted_key] = convert_numpy(value)
                return converted_dict
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        # Remove large fields if present
        results_to_save = results.copy()
        if 'individual_results' in results_to_save:
            for model_name, model_result in results_to_save['individual_results'].items():
                if isinstance(model_result, dict):
                    for field in ['coefficients', 'feature_names', 'feature_importance']:
                        if field in model_result:
                            del model_result[field]

        # Convert results
        serializable_results = convert_numpy(results_to_save)

        # Save file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {filepath}")
        return filepath
    
    def print_summary(self, results):
        """
        Print summary of results to console
        
        Args:
            results (dict): Training results
        """
        print("\n" + "="*80)
        print("TRAINING PIPELINE SUMMARY")
        print("="*80)
        
        summary = results.get('pipeline_summary', {})
        
        print(f"Timestamp: {summary.get('timestamp', 'N/A')}")
        print(f"Total Pipeline Time: {summary.get('total_pipeline_time_seconds', 0)} seconds")
        print(f"Models Trained: {summary.get('models_trained', 0)}")
        
        comparison = summary.get('comparison', {})
        if comparison:
            print(f"\nBest Model (Accuracy): {comparison.get('best_accuracy_model', 'N/A')}")
            print(f"Best Accuracy Score: {comparison.get('best_accuracy_score', 0)}")
            print(f"Fastest Model: {comparison.get('fastest_model', 'N/A')}")
            print(f"Fastest Training Time: {comparison.get('fastest_time', 0)} seconds")
            
            print(f"\nAll Model Accuracies:")
            for model, acc in comparison.get('all_accuracies', {}).items():
                print(f"   {model}: {acc}")
                
            print(f"\nAll Training Times:")
            for model, time in comparison.get('all_training_times', {}).items():
                print(f"   {model}: {time} seconds")
        
        print("="*80)
    
    def run_training_pipeline_with_tfidf(self, X_train_tfidf, X_test_tfidf, y_train, y_test, 
                                         optimize_hyperparameters=False, save_results=True):
        """
        Run complete training pipeline with pre-computed TF-IDF matrices
        
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
        print("STARTING MODEL TRAINING PIPELINE WITH PRE-COMPUTED TF-IDF")
        print("="*100)
        
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
        
        # Train each model with pre-computed TF-IDF
        models_to_train = [
            ('Logistic_Regression', lambda: self.train_logistic_regression_with_tfidf(
                X_train_tfidf, X_test_tfidf, y_train_mapped, y_test_mapped, optimize_hyperparameters)),
            ('Random_Forest', lambda: self.train_random_forest_with_tfidf(
                X_train_tfidf, X_test_tfidf, y_train_mapped, y_test_mapped, optimize_hyperparameters)),
            ('Gradient_Boosting', lambda: self.train_gradient_boosting_with_tfidf(
                X_train_tfidf, X_test_tfidf, y_train_mapped, y_test_mapped))
        ]
        
        for model_name, train_func in models_to_train:
            print(f"\n>>> Training {model_name} with pre-computed TF-IDF...")
            try:
                results = train_func()
                all_results[model_name] = results
            except Exception as e:
                print(f"Failed to train {model_name}: {e}")
                all_results[model_name] = {
                    'model_name': model_name,
                    'error': str(e),
                    'training_time_seconds': 0
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
                'tfidf_train_shape': X_train_tfidf.shape,
                'tfidf_test_shape': X_test_tfidf.shape,
                'train_samples': len(y_train),
                'test_samples': len(y_test)
            }
        }
        
        # Print summary
        self.print_summary(final_results)
        
        # Save results
        if save_results:
            self.save_results_to_json(final_results)
        
        return final_results
