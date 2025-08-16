import pandas as pd
import numpy as np
import json
import time
import os
from datetime import datetime

# Import các classifier
from logistic_regression_classifier import LogisticRegressionAnalyzer
from random_forest_classifier import RandomForestAnalyzer
from gradient_boosting_classifier import GradientBoostingAnalyzer


class ModelTrainer:
    """
    Class để train nhiều models và so sánh kết quả
    """
    
    def __init__(self, output_dir="reports"):
        """
        Khởi tạo ModelTrainer
        
        Args:
            output_dir (str): Thư mục lưu kết quả
        """
        self.output_dir = output_dir
        self.results = {}
        self.training_history = []
        
        # Tạo thư mục output nếu chưa có
        os.makedirs(output_dir, exist_ok=True)
        
    def prepare_data_for_models(self, train_df, test_df):
        """
        Chuẩn bị dữ liệu chung cho tất cả models
        
        Args:
            train_df (pd.DataFrame): Training data
            test_df (pd.DataFrame): Test data
            
        Returns:
            tuple: Combined dataframe và thông tin
        """
        print("\n=== PREPARING DATA FOR MODEL TRAINING ===")
        
        # Thêm cột sentiment cho train data (chuyển đổi label)
        train_df = train_df.copy()
        test_df = test_df.copy()
        
        # Map labels: 1 -> negative (0), 2 -> positive (1) 
        label_mapping = {1: 0, 2: 1}  # 1=negative, 2=positive
        train_df['sentiment'] = train_df['label'].map(label_mapping)
        test_df['sentiment'] = test_df['label'].map(label_mapping)
        
        # Combine train and test để split lại theo cách chuẩn
        train_df['dataset_type'] = 'train'
        test_df['dataset_type'] = 'test'
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        
        # Sử dụng cột input đã được tiền xử lý
        text_column = 'input'  # Sử dụng input thay vì normalized_input
        
        print(f"Combined dataset shape: {combined_df.shape}")
        print(f"Text column: {text_column}")
        print(f"Sentiment distribution: {combined_df['sentiment'].value_counts().to_dict()}")
        
        return combined_df, text_column
        
    def train_logistic_regression(self, df, text_column, optimize_hyperparameters=False):
        """
        Train Logistic Regression model
        
        Args:
            df (pd.DataFrame): Dataset
            text_column (str): Text column name
            optimize_hyperparameters (bool): Có optimize hyperparameters không
            
        Returns:
            dict: Kết quả training
        """
        print("\n" + "="*60)
        print("TRAINING LOGISTIC REGRESSION")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # Khởi tạo analyzer
            lr_analyzer = LogisticRegressionAnalyzer(optimize_hyperparameters=optimize_hyperparameters)
            
            # Chuẩn bị dữ liệu
            lr_analyzer.prepare_data(df, text_column, test_size=0.2, random_state=42)
            
            # Khởi tạo model (sẽ optimize nếu cần)
            lr_analyzer.initialize_model()
            
            # Train model
            results = lr_analyzer.train_model()
            
            training_time = time.time() - start_time
            
            # Thêm thông tin bổ sung
            results.update({
                'training_time_seconds': round(training_time, 2),
                'hyperparameter_optimization': optimize_hyperparameters,
                'best_params': getattr(lr_analyzer, 'best_params', None)
            })
            
            print(f"Logistic Regression training completed in {training_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            print(f"Error training Logistic Regression: {e}")
            return {
                'model_name': 'LogisticRegression',
                'error': str(e),
                'training_time_seconds': time.time() - start_time
            }
    
    def train_random_forest(self, df, text_column, optimize_hyperparameters=False):
        """
        Train Random Forest model
        
        Args:
            df (pd.DataFrame): Dataset
            text_column (str): Text column name
            optimize_hyperparameters (bool): Có optimize hyperparameters không
            
        Returns:
            dict: Kết quả training
        """
        print("\n" + "="*60)
        print("TRAINING RANDOM FOREST")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # Khởi tạo analyzer
            rf_analyzer = RandomForestAnalyzer(optimize_hyperparameters=optimize_hyperparameters)
            
            # Chuẩn bị dữ liệu
            rf_analyzer.prepare_data(df, text_column, test_size=0.2, random_state=42)
            
            # Khởi tạo model (sẽ optimize nếu cần)
            rf_analyzer.initialize_model()
            
            # Train model
            results = rf_analyzer.train_model()
            
            training_time = time.time() - start_time
            
            # Thêm thông tin bổ sung
            results.update({
                'training_time_seconds': round(training_time, 2),
                'hyperparameter_optimization': optimize_hyperparameters,
                'best_params': getattr(rf_analyzer, 'best_params', None)
            })
            
            print(f"Random Forest training completed in {training_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            print(f"Error training Random Forest: {e}")
            return {
                'model_name': 'RandomForestClassifier',
                'error': str(e),
                'training_time_seconds': time.time() - start_time
            }
    
    def train_gradient_boosting(self, df, text_column):
        """
        Train Gradient Boosting model
        
        Args:
            df (pd.DataFrame): Dataset
            text_column (str): Text column name
            
        Returns:
            dict: Kết quả training
        """
        print("\n" + "="*60)
        print("TRAINING GRADIENT BOOSTING")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # Khởi tạo analyzer
            gb_analyzer = GradientBoostingAnalyzer()
            
            # Chuẩn bị dữ liệu
            gb_analyzer.prepare_data(df, text_column, test_size=0.2, random_state=42)
            
            # Khởi tạo model với parameters mặc định
            gb_analyzer.initialize_model()
            
            # Train model
            results = gb_analyzer.train_model()
            
            training_time = time.time() - start_time
            
            # Thêm thông tin bổ sung
            results.update({
                'training_time_seconds': round(training_time, 2),
                'hyperparameter_optimization': False
            })
            
            print(f"Gradient Boosting training completed in {training_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            print(f"Error training Gradient Boosting: {e}")
            return {
                'model_name': 'GradientBoostingClassifier',
                'error': str(e),
                'training_time_seconds': time.time() - start_time
            }
    
    def train_all_models(self, train_df, test_df, optimize_hyperparameters=False):
        """
        Train tất cả models và so sánh kết quả
        
        Args:
            train_df (pd.DataFrame): Training data
            test_df (pd.DataFrame): Test data
            optimize_hyperparameters (bool): Có optimize hyperparameters không
            
        Returns:
            dict: Kết quả của tất cả models
        """
        print("\n" + "="*80)
        print("TRAINING ALL MODELS PIPELINE")
        print("="*80)
        
        pipeline_start_time = time.time()
        
        # Chuẩn bị dữ liệu
        combined_df, text_column = self.prepare_data_for_models(train_df, test_df)
        
        # Dictionary lưu kết quả
        all_results = {}
        
        # Train từng model
        models_to_train = [
            ('Logistic_Regression', lambda: self.train_logistic_regression(
                combined_df, text_column, optimize_hyperparameters)),
            ('Random_Forest', lambda: self.train_random_forest(
                combined_df, text_column, optimize_hyperparameters)),
            ('Gradient_Boosting', lambda: self.train_gradient_boosting(
                combined_df, text_column))
        ]
        
        for model_name, train_func in models_to_train:
            print(f"\n>>> Training {model_name}...")
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
        
        # Tính tổng thời gian
        total_pipeline_time = time.time() - pipeline_start_time
        
        # Tạo summary
        pipeline_summary = self.create_pipeline_summary(all_results, total_pipeline_time)
        
        # Lưu kết quả
        final_results = {
            'pipeline_summary': pipeline_summary,
            'individual_results': all_results,
            'training_config': {
                'optimize_hyperparameters': optimize_hyperparameters,
                'text_column': text_column,
                'train_samples': len(train_df),
                'test_samples': len(test_df),
                'total_samples': len(combined_df)
            }
        }
        
        return final_results
    
    def create_pipeline_summary(self, all_results, total_time):
        """
        Tạo summary so sánh các models
        
        Args:
            all_results (dict): Kết quả của tất cả models
            total_time (float): Tổng thời gian training
            
        Returns:
            dict: Pipeline summary
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_pipeline_time_seconds': round(total_time, 2),
            'models_trained': len(all_results),
            'comparison': {}
        }
        
        # So sánh accuracy
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
        Lưu kết quả vào file JSON
        
        Args:
            results (dict): Kết quả training
            filename (str): Tên file (optional)
            
        Returns:
            str: Đường dẫn file đã lưu
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_training_results_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Chuyển đổi numpy arrays thành lists để serialize JSON
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                # Convert both keys and values, handling numpy types in keys
                converted_dict = {}
                for key, value in obj.items():
                    # Convert numpy types in keys to standard Python types
                    if isinstance(key, np.integer):
                        converted_key = int(key)
                    elif isinstance(key, np.floating):
                        converted_key = float(key)
                    elif isinstance(key, np.ndarray):
                        converted_key = str(key.tolist())  # Convert arrays to string representation
                    else:
                        converted_key = key
                    
                    converted_dict[converted_key] = convert_numpy(value)
                return converted_dict
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        # Convert results
        serializable_results = convert_numpy(results)
        
        # Lưu file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {filepath}")
        return filepath
    
    def print_summary(self, results):
        """
        In summary kết quả ra console
        
        Args:
            results (dict): Kết quả training
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
    
    def run_training_pipeline(self, train_df, test_df, optimize_hyperparameters=False, save_results=True):
        """
        Chạy toàn bộ pipeline training
        
        Args:
            train_df (pd.DataFrame): Training data
            test_df (pd.DataFrame): Test data
            optimize_hyperparameters (bool): Có optimize hyperparameters không
            save_results (bool): Có lưu kết quả không
            
        Returns:
            dict: Kết quả training
        """
        print("\n" + "="*100)
        print("STARTING MODEL TRAINING PIPELINE")
        print("="*100)
        
        # Train tất cả models
        results = self.train_all_models(train_df, test_df, optimize_hyperparameters)
        
        # In summary
        self.print_summary(results)
        
        # Lưu kết quả
        if save_results:
            self.save_results_to_json(results)
        
        return results
