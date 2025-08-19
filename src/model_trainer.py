import pandas as pd
import numpy as np
import os
import json
import time
from datetime import datetime


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder để handle numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types"""
    if isinstance(obj, dict):
        return {str(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


# Import các classifier đã đơn giản hóa
from logistic_regression_classifier import LogisticRegressionAnalyzer
from random_forest_classifier import RandomForestAnalyzer
from gradient_boosting_classifier import GradientBoostingAnalyzer


class ModelTrainer:
    """
    Đơn giản hóa ModelTrainer để train và so sánh các models
    """

    def __init__(self, output_dir="reports"):
        """
        Khởi tạo ModelTrainer

        Args:
            output_dir (str): Thư mục lưu kết quả
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Định nghĩa các models có thể train
        self.available_models = {
            'logistic_regression': LogisticRegressionAnalyzer,
            'random_forest': RandomForestAnalyzer,
            'gradient_boosting': GradientBoostingAnalyzer
        }
    
    def run_training_pipeline_with_configs(self, train_df, test_df, model_configs, save_results=True):
        """
        Run training pipeline với configs từ JSON file
        
        Args:
            train_df (pd.DataFrame): Training data với 'sentiment' column
            test_df (pd.DataFrame): Test data với 'sentiment' column  
            model_configs (dict): Configs loaded từ JSON
            save_results (bool): Có lưu kết quả không
            
        Returns:
            dict: Training results
        """
        print(f"\n{'='*80}")
        print("MODEL TRAINING WITH JSON CONFIGURATIONS")
        print(f"{'='*80}")
        
        config_info = model_configs.get('config_info', {})
        print(f"Using: {config_info.get('name', 'Unknown Config')}")
        print(f"Expected time: {config_info.get('expected_training_time', 'Unknown')}")
        
        # Use separate train and test sets
        print(f"Training dataset: {train_df.shape}")
        print(f"Test dataset: {test_df.shape}")
        
        start_time = time.time()
        all_results = {}
        
        # Train each model with its specific config
        for model_name in ['logistic_regression', 'random_forest', 'gradient_boosting']:
            if model_name in model_configs:
                print(f"\n{'-'*60}")
                print(f"TRAINING {model_name.upper()}")
                print(f"{'-'*60}")
                
                try:
                    # Get model config
                    model_config = model_configs[model_name]
                    
                    # Extract all parameters
                    model_params = model_config.get('model_params', {})
                    tfidf_params = model_config.get('tfidf_params', {})
                    training_params = model_config.get('training_params', {})
                    
                    # Combine all params for model initialization
                    all_params = {}
                    all_params.update(model_params)
                    
                    # Add TF-IDF max_features and training test_size
                    if 'max_features' in tfidf_params:
                        all_params['max_features'] = tfidf_params['max_features']
                    if 'test_size' in training_params:
                        all_params['test_size'] = training_params['test_size']
                    
                    print(f"Initializing model with: {all_params}")
                    
                    # Create model with all parameters
                    model_class = self.available_models[model_name]
                    model = model_class(**all_params)
                    
                    # Update TF-IDF parameters
                    if tfidf_params:
                        model.update_tfidf_params(tfidf_params)
                        print(f"TF-IDF parameters updated")
                    
                    # Train model with separate train and test sets
                    result = model.train_and_evaluate(
                        train_df=train_df,
                        test_df=test_df,
                        text_column='normalized_input',
                        target_column='label'
                    )
                    
                    # Add config info to result
                    result['model_config'] = model_config
                    result['config_name'] = config_info.get('name', 'Unknown')
                    
                    all_results[model_name] = result
                    print(f"✅ {model_name} completed successfully")
                    
                except Exception as e:
                    print(f"❌ Error training {model_name}: {e}")
                    all_results[model_name] = {
                        'error': str(e),
                        'model_config': model_configs.get(model_name, {})
                    }
        
        total_time = time.time() - start_time
        
        # Compile final results
        final_results = {
            'config_info': config_info,
            'training_summary': {
                'total_time_seconds': round(total_time, 2),
                'total_time_minutes': round(total_time / 60, 2),
                'train_dataset_size': len(train_df),
                'test_dataset_size': len(test_df),
                'timestamp': datetime.now().isoformat()
            },
            'model_results': all_results
        }
        
        # Save results if requested
        if save_results:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            config_name = config_info.get('name', 'unknown').lower().replace(' ', '_')
            filename = f"{config_name}_results_{timestamp}.json"
            output_path = os.path.join(self.output_dir, filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                # Convert numpy types before serializing
                safe_results = convert_numpy_types(final_results)
                json.dump(safe_results, f, indent=2, ensure_ascii=False)
            
            print(f"\n✅ Results saved to: {output_path}")
        
        # Print summary
        print(f"\n{'='*80}")
        print("TRAINING PIPELINE COMPLETED")
        print(f"{'='*80}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Models trained: {len([r for r in all_results.values() if 'error' not in r])}/{len(all_results)}")
        
        print(f"\nModel Performance Summary:")
        print(f"{'-'*50}")
        for model_name, result in all_results.items():
            if 'error' not in result and 'test_accuracy' in result:
                acc = result['test_accuracy']
                time_taken = result.get('training_time', 0)
                print(f"{model_name:20s}: {acc:.4f} ({time_taken:.1f}s)")
            else:
                print(f"{model_name:20s}: ERROR")
        
        return final_results