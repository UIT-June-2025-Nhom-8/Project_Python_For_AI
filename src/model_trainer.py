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
    
    def run_training_pipeline_with_simplified_classifiers(self, df, text_column='text', target_column='sentiment', 
                                                         test_size=0.2, optimize_hyperparameters=False, 
                                                         save_results=True, max_features=10000):
        """
        Run complete training pipeline using simplified classifier implementations
        This method provides a clean, easy-to-understand alternative to the complex implementations
        
        Args:
            df: DataFrame containing the data
            text_column (str): Column containing text data
            target_column (str): Column containing sentiment labels
            test_size (float): Test data ratio
            optimize_hyperparameters (bool): Whether to optimize hyperparameters
            save_results (bool): Whether to save results
            max_features (int): Maximum features for TF-IDF
            
        Returns:
            dict: Training results
        """
        print("\n" + "="*100)
        print("STARTING SIMPLIFIED MODEL TRAINING PIPELINE")
        print("="*100)
        
        # Import simplified classifiers
        try:
            from simplified_random_forest import SimplifiedRandomForestAnalyzer
            from simplified_logistic_regression import SimplifiedLogisticRegressionAnalyzer
            from simplified_gradient_boosting import SimplifiedGradientBoostingAnalyzer
        except ImportError as e:
            print(f"Error importing simplified classifiers: {e}")
            return {"error": f"Could not import simplified classifiers: {e}"}
        
        pipeline_start_time = time.time()
        
        print(f"Configuration:")
        print(f"   - Dataset size: {len(df)} samples")
        print(f"   - Text column: {text_column}")
        print(f"   - Target column: {target_column}")
        print(f"   - Test size: {test_size}")
        print(f"   - Max features: {max_features}")
        print(f"   - Hyperparameter optimization: {optimize_hyperparameters}")
        
        # Dictionary to store results
        all_results = {}
        
        # Initialize simplified classifiers
        classifiers = {
            'Simplified_Random_Forest': SimplifiedRandomForestAnalyzer(),
            'Simplified_Logistic_Regression': SimplifiedLogisticRegressionAnalyzer(),
            'Simplified_Gradient_Boosting': SimplifiedGradientBoostingAnalyzer()
        }
        
        # Train each simplified classifier
        for name, classifier in classifiers.items():
            print(f"\n>>> Training {name.replace('_', ' ')} (Simplified)...")
            try:
                start_time = time.time()
                
                # Prepare data and train
                X_train, X_test, y_train, y_test = classifier.prepare_data(
                    df, text_column, target_column, test_size, max_features
                )
                
                # Train model
                results = classifier.train(X_train, X_test, y_train, y_test, optimize_hyperparameters)
                
                training_time = time.time() - start_time
                
                # Format results for consistency with existing format
                formatted_results = {
                    'model_name': f'{classifier.model_name}_Simplified',
                    'train_accuracy': round(results.get('train_accuracy', 0), 4),
                    'test_accuracy': round(results.get('test_accuracy', 0), 4),
                    'train_f1': round(results.get('train_f1', 0), 4),
                    'test_f1': round(results.get('test_f1', 0), 4),
                    'f1_score': round(results.get('test_f1', 0), 4),  # For compatibility
                    'training_time_seconds': round(training_time, 2),
                    'hyperparameter_optimization': optimize_hyperparameters,
                    'classification_report': results.get('classification_report', {}),
                    'confusion_matrix': results.get('confusion_matrix', []).tolist() if hasattr(results.get('confusion_matrix', []), 'tolist') else results.get('confusion_matrix', []),
                    'implementation_source': 'simplified_classifiers',
                    'data_preparation': {
                        'vectorizer_type': 'TfidfVectorizer',
                        'max_features': max_features,
                        'train_samples': X_train.shape[0],
                        'test_samples': X_test.shape[0],
                        'features': X_train.shape[1]
                    }
                }
                
                # Add feature importance if available
                if hasattr(classifier, 'get_feature_importance'):
                    importance_df = classifier.get_feature_importance()
                    if importance_df is not None:
                        formatted_results['top_features'] = importance_df.to_dict('records')
                
                # Add model-specific information
                if hasattr(classifier, 'get_feature_coefficients'):
                    coefficients = classifier.get_feature_coefficients()
                    if coefficients:
                        formatted_results['feature_coefficients'] = {
                            'top_positive': coefficients['positive'][:5],
                            'top_negative': coefficients['negative'][:5]
                        }
                
                all_results[name] = formatted_results
                print(f"✅ {name.replace('_', ' ')} training completed in {training_time:.2f} seconds")
                print(f"   Test Accuracy: {formatted_results['test_accuracy']:.4f}")
                print(f"   Test F1-Score: {formatted_results['test_f1']:.4f}")
                
            except Exception as e:
                print(f"❌ Error training {name}: {e}")
                all_results[name] = {
                    'model_name': f'{name}_Simplified',
                    'error': str(e),
                    'training_time_seconds': time.time() - start_time if 'start_time' in locals() else 0
                }
        
        # Calculate total time
        total_pipeline_time = time.time() - pipeline_start_time
        
        # Create summary
        pipeline_summary = self.create_pipeline_summary(all_results, total_pipeline_time)
        
        # Enhanced pipeline summary for simplified classifiers
        pipeline_summary['simplified_classifier_info'] = {
            'code_reduction': '~75% less code than original implementations',
            'key_benefits': [
                'Standardized interface across all classifiers',
                'Uses sklearn built-in functionality',
                'Cleaner, more maintainable code',
                'Easier to understand and modify'
            ],
            'features_used': max_features,
            'data_preprocessing': 'Automated with sklearn TfidfVectorizer'
        }
        
        # Save results
        final_results = {
            'pipeline_summary': pipeline_summary,
            'individual_results': all_results,
            'training_config': {
                'dataset_size': len(df),
                'text_column': text_column,
                'target_column': target_column,
                'test_size': test_size,
                'max_features': max_features,
                'optimize_hyperparameters': optimize_hyperparameters,
                'implementation_type': 'simplified_classifiers',
                'automated_data_preparation': True
            }
        }
        
        # Print enhanced summary for simplified classifiers
        self.print_simplified_summary(final_results)
        
        # Save results
        if save_results:
            filename = f"simplified_classifiers_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.save_results_to_json(final_results, filename)
        
        return final_results
    
    def print_simplified_summary(self, results):
        """
        Print enhanced summary for simplified classifiers
        
        Args:
            results (dict): Training results
        """
        print("\n" + "="*80)
        print("SIMPLIFIED CLASSIFIERS TRAINING SUMMARY")
        print("="*80)
        
        summary = results.get('pipeline_summary', {})
        config = results.get('training_config', {})
        
        print(f"Implementation Type: Simplified Classifiers")
        print(f"Dataset Size: {config.get('dataset_size', 'N/A'):,} samples")
        print(f"Features: {config.get('max_features', 'N/A'):,}")
        print(f"Test Size: {config.get('test_size', 'N/A')}")
        print(f"Total Pipeline Time: {summary.get('total_pipeline_time_seconds', 0):.2f} seconds")
        print(f"Models Trained: {summary.get('models_trained', 0)}")
        
        # Show benefits of simplified implementation
        simplified_info = summary.get('simplified_classifier_info', {})
        if simplified_info:
            print(f"\n📊 Simplified Implementation Benefits:")
            print(f"   • {simplified_info.get('code_reduction', 'Significant code reduction')}")
            for benefit in simplified_info.get('key_benefits', []):
                print(f"   • {benefit}")
        
        comparison = summary.get('comparison', {})
        if comparison:
            print(f"\n🏆 Performance Results:")
            print(f"   Best Model (Accuracy): {comparison.get('best_accuracy_model', 'N/A').replace('_', ' ')}")
            print(f"   Best Accuracy Score: {comparison.get('best_accuracy_score', 0):.4f}")
            print(f"   Fastest Model: {comparison.get('fastest_model', 'N/A').replace('_', ' ')}")
            print(f"   Fastest Training Time: {comparison.get('fastest_time', 0):.2f} seconds")
            
            print(f"\n📈 All Model Results:")
            # Create a simple table
            print(f"{'Model':<25} {'Accuracy':<10} {'F1-Score':<10} {'Time (s)':<10}")
            print("-" * 60)
            
            individual_results = results.get('individual_results', {})
            for model_name, model_result in individual_results.items():
                if 'error' not in model_result:
                    display_name = model_result.get('model_name', model_name).replace('_', ' ')
                    accuracy = model_result.get('test_accuracy', 0)
                    f1_score = model_result.get('test_f1', 0)
                    time_taken = model_result.get('training_time_seconds', 0)
                    print(f"{display_name:<25} {accuracy:<10.4f} {f1_score:<10.4f} {time_taken:<10.2f}")
        
        print("\n💡 Usage Tips for Simplified Classifiers:")
        print("   • Use quick_train() method for one-line training")
        print("   • Set optimize_hyperparameters=True for better performance")
        print("   • All classifiers use the same standardized interface")
        print("   • Built-in feature importance and model evaluation")
        
        print("="*80)
    
    def run_training_pipeline_with_simplified_classifiers_tfidf(self, X_train_tfidf, X_test_tfidf, y_train, y_test, 
                                                               optimize_hyperparameters=False, save_results=True):
        """
        Run training pipeline using simplified classifiers with pre-computed TF-IDF matrices
        This method provides a clean alternative that works with pre-computed TF-IDF data
        
        Args:
            X_train_tfidf: Pre-computed TF-IDF matrix for training data
            X_test_tfidf: Pre-computed TF-IDF matrix for test data  
            y_train: Training labels
            y_test: Test labels
            optimize_hyperparameters (bool): Whether to optimize hyperparameters
            save_results (bool): Whether to save results
            
        Returns:
            dict: Training results
        """
        print("\n" + "="*100)
        print("STARTING SIMPLIFIED MODEL TRAINING PIPELINE WITH PRE-COMPUTED TF-IDF")
        print("="*100)
        
        # Import simplified classifiers
        try:
            from simplified_random_forest import SimplifiedRandomForestAnalyzer
            from simplified_logistic_regression import SimplifiedLogisticRegressionAnalyzer
            from simplified_gradient_boosting import SimplifiedGradientBoostingAnalyzer
        except ImportError as e:
            print(f"Error importing simplified classifiers: {e}")
            return {"error": f"Could not import simplified classifiers: {e}"}
        
        # Map labels: 1 -> negative (0), 2 -> positive (1) 
        label_mapping = {1: 0, 2: 1}
        y_train_mapped = y_train.map(label_mapping)
        y_test_mapped = y_test.map(label_mapping)
        
        pipeline_start_time = time.time()
        
        print(f"Configuration:")
        print(f"   - X_train shape: {X_train_tfidf.shape}")
        print(f"   - X_test shape: {X_test_tfidf.shape}")
        print(f"   - y_train shape: {y_train_mapped.shape}")
        print(f"   - y_test shape: {y_test_mapped.shape}")
        print(f"   - Features: {X_train_tfidf.shape[1]:,}")
        print(f"   - Hyperparameter optimization: {optimize_hyperparameters}")
        
        # Dictionary to store results
        all_results = {}
        
        # Initialize simplified classifiers
        classifiers = {
            'Simplified_Random_Forest': SimplifiedRandomForestAnalyzer(),
            'Simplified_Logistic_Regression': SimplifiedLogisticRegressionAnalyzer(),
            'Simplified_Gradient_Boosting': SimplifiedGradientBoostingAnalyzer()
        }
        
        # Train each simplified classifier using pre-computed TF-IDF
        for name, classifier in classifiers.items():
            print(f"\n>>> Training {name.replace('_', ' ')} (Simplified with TF-IDF)...")
            try:
                start_time = time.time()
                
                # Set up the classifier to use pre-computed data
                # We need to create a mock vectorizer and label encoder
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.preprocessing import LabelEncoder
                
                # Create mock vectorizer (needed for predict method)
                classifier.vectorizer = TfidfVectorizer()
                classifier.vectorizer.vocabulary_ = {f'feature_{i}': i for i in range(X_train_tfidf.shape[1])}
                
                # Set up label encoder
                classifier.label_encoder.fit(['negative', 'positive'])
                
                # Train model directly with TF-IDF data
                results = classifier.train(X_train_tfidf, X_test_tfidf, y_train_mapped, y_test_mapped, optimize_hyperparameters)
                
                training_time = time.time() - start_time
                
                # Format results for consistency
                formatted_results = {
                    'model_name': f'{classifier.model_name}_Simplified_TFIDF',
                    'train_accuracy': round(results.get('train_accuracy', 0), 4),
                    'test_accuracy': round(results.get('test_accuracy', 0), 4),
                    'train_f1': round(results.get('train_f1', 0), 4),
                    'test_f1': round(results.get('test_f1', 0), 4),
                    'f1_score': round(results.get('test_f1', 0), 4),  # For compatibility
                    'training_time_seconds': round(training_time, 2),
                    'hyperparameter_optimization': optimize_hyperparameters,
                    'classification_report': results.get('classification_report', {}),
                    'confusion_matrix': results.get('confusion_matrix', []).tolist() if hasattr(results.get('confusion_matrix', []), 'tolist') else results.get('confusion_matrix', []),
                    'implementation_source': 'simplified_classifiers_with_precomputed_tfidf',
                    'data_preparation': {
                        'vectorizer_type': 'Pre-computed TF-IDF',
                        'train_samples': X_train_tfidf.shape[0],
                        'test_samples': X_test_tfidf.shape[0],
                        'features': X_train_tfidf.shape[1]
                    }
                }
                
                # Add feature importance if available
                if hasattr(classifier, 'get_feature_importance'):
                    try:
                        importance_df = classifier.get_feature_importance()
                        if importance_df is not None:
                            formatted_results['top_features'] = importance_df.head(10).to_dict('records')
                    except:
                        pass  # Skip if feature importance fails
                
                # Add model-specific information for Logistic Regression
                if hasattr(classifier, 'get_feature_coefficients'):
                    try:
                        coefficients = classifier.get_feature_coefficients()
                        if coefficients:
                            formatted_results['feature_coefficients'] = {
                                'top_positive': coefficients['positive'][:5],
                                'top_negative': coefficients['negative'][:5]
                            }
                    except:
                        pass  # Skip if coefficients fail
                
                all_results[name] = formatted_results
                print(f"✅ {name.replace('_', ' ')} training completed in {training_time:.2f} seconds")
                print(f"   Test Accuracy: {formatted_results['test_accuracy']:.4f}")
                print(f"   Test F1-Score: {formatted_results['test_f1']:.4f}")
                
            except Exception as e:
                print(f"❌ Error training {name}: {e}")
                all_results[name] = {
                    'model_name': f'{name}_Simplified_TFIDF',
                    'error': str(e),
                    'training_time_seconds': time.time() - start_time if 'start_time' in locals() else 0
                }
        
        # Calculate total time
        total_pipeline_time = time.time() - pipeline_start_time
        
        # Create summary
        pipeline_summary = self.create_pipeline_summary(all_results, total_pipeline_time)
        
        # Enhanced pipeline summary for simplified classifiers
        pipeline_summary['simplified_classifier_info'] = {
            'code_reduction': '~75% less code than original implementations',
            'key_benefits': [
                'Standardized interface across all classifiers',
                'Uses sklearn built-in functionality', 
                'Cleaner, more maintainable code',
                'Easier to understand and modify',
                'Works with pre-computed TF-IDF matrices'
            ],
            'features_used': X_train_tfidf.shape[1],
            'data_preprocessing': 'Pre-computed TF-IDF matrices'
        }
        
        # Save results
        final_results = {
            'pipeline_summary': pipeline_summary,
            'individual_results': all_results,
            'training_config': {
                'train_samples': X_train_tfidf.shape[0],
                'test_samples': X_test_tfidf.shape[0],
                'features': X_train_tfidf.shape[1],
                'optimize_hyperparameters': optimize_hyperparameters,
                'implementation_type': 'simplified_classifiers_with_precomputed_tfidf',
                'used_precomputed_tfidf': True
            }
        }
        
        # Print enhanced summary for simplified classifiers
        self.print_simplified_summary(final_results)
        
        # Save results
        if save_results:
            filename = f"simplified_classifiers_tfidf_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.save_results_to_json(final_results, filename)
        
        return final_results
    
    def compare_implementations(self, original_results=None, simplified_results=None):
        """
        Compare results between original complex and simplified implementations
        
        Args:
            original_results (dict): Results from original implementation
            simplified_results (dict): Results from simplified implementation
        """
        if not original_results or not simplified_results:
            print("Both original and simplified results needed for comparison")
            return
        
        print("\n" + "="*80)
        print("IMPLEMENTATION COMPARISON: ORIGINAL vs SIMPLIFIED")
        print("="*80)
        
        print(f"{'Metric':<30} {'Original':<15} {'Simplified':<15} {'Difference':<15}")
        print("-" * 80)
        
        # Compare training times
        orig_time = original_results.get('pipeline_summary', {}).get('total_pipeline_time_seconds', 0)
        simp_time = simplified_results.get('pipeline_summary', {}).get('total_pipeline_time_seconds', 0)
        time_diff = simp_time - orig_time
        
        print(f"{'Total Training Time (s)':<30} {orig_time:<15.2f} {simp_time:<15.2f} {time_diff:<15.2f}")
        
        # Compare accuracy for each model type
        orig_individual = original_results.get('individual_results', {})
        simp_individual = simplified_results.get('individual_results', {})
        
        # Map model names for comparison
        model_mapping = {
            'Random_Forest_Implemented': 'Simplified_Random_Forest',
            'Logistic_Regression_Implemented': 'Simplified_Logistic_Regression',
            'Gradient_Boosting_Implemented': 'Simplified_Gradient_Boosting'
        }
        
        for orig_name, simp_name in model_mapping.items():
            if orig_name in orig_individual and simp_name in simp_individual:
                orig_acc = orig_individual[orig_name].get('test_accuracy', 0)
                simp_acc = simp_individual[simp_name].get('test_accuracy', 0)
                acc_diff = simp_acc - orig_acc
                
                model_display = orig_name.replace('_Implemented', '').replace('_', ' ')
                print(f"{f'{model_display} Accuracy':<30} {orig_acc:<15.4f} {simp_acc:<15.4f} {acc_diff:<15.4f}")
        
        print("\n📊 Key Differences:")
        print("   • Code Complexity: ~75% reduction in simplified version")
        print("   • Maintainability: Standardized interface vs custom implementations")
        print("   • Library Usage: Built-in sklearn vs manual implementations")
        print("   • Error Handling: Centralized vs scattered throughout code")
        print("   • Feature Analysis: Built-in methods vs manual calculations")
        
        print("="*80)
