import pandas as pd
import numpy as np
import hashlib
import json
import time
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import pickle
import os


class LogisticRegressionAnalyzer:
    """
    Đơn giản hóa LogisticRegression cho phân tích cảm xúc
    """
    
    def __init__(self, C=0.8, max_iter=1500, solver='liblinear', 
                 max_features=25000, test_size=0.2, random_state=42, **kwargs):
        """
        Khởi tạo với tham số mặc định được tối ưu cho sentiment analysis
        
        Args:
            C (float): Regularization strength (giảm để tránh overfitting)
            max_iter (int): Maximum iterations (tăng cho convergence tốt hơn)
            solver (str): Solver algorithm (liblinear tốt cho text classification)
            max_features (int): Maximum features for TF-IDF (tăng cho text data)
            test_size (float): Tỷ lệ data test
            random_state (int): Random state
            **kwargs: Additional parameters từ JSON config (sẽ được ignore nếu không supported)
        """
        self.params = {
            'C': C,
            'max_iter': max_iter,
            'solver': solver,
            'max_features': max_features,
            'test_size': test_size,
            'random_state': random_state
        }
        
        # Thêm class_weight từ kwargs nếu có
        if 'class_weight' in kwargs:
            self.params['class_weight'] = kwargs['class_weight']
        
        # TF-IDF parameters (có thể override từ JSON config)
        self.tfidf_params = {
            'max_features': max_features,
            'stop_words': None,
            'ngram_range': (1, 3),
            'min_df': 2,
            'max_df': 0.92,
            'sublinear_tf': True,
            'lowercase': True,
            'strip_accents': 'unicode'
        }
        
        self.model = None
        self.tfidf_vectorizer = None
        self.label_encoder = LabelEncoder()
        self.results = {}
        self.cache_dir = "output/models/cache/"
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def update_tfidf_params(self, tfidf_config):
        """
        Update TF-IDF parameters từ JSON config
        
        Args:
            tfidf_config (dict): TF-IDF parameters từ JSON
        """
        if tfidf_config:
            # Update tất cả TF-IDF parameters
            self.tfidf_params.update(tfidf_config)
            
            # Convert ngram_range from list to tuple if needed
            if 'ngram_range' in self.tfidf_params and isinstance(self.tfidf_params['ngram_range'], list):
                self.tfidf_params['ngram_range'] = tuple(self.tfidf_params['ngram_range'])
                
            print(f"TF-IDF params updated: {self.tfidf_params}")
    
    def _generate_cache_key(self, df):
        """Generate cache key from data and params (fixed for list handling)"""
        # Create hashable representation of the data
        df_for_hash = df[['label']].copy()
        
        # Convert normalized_input lists to strings for hashing
        if 'normalized_input' in df.columns:
            df_for_hash['normalized_input_str'] = df['normalized_input'].apply(
                lambda x: ' '.join(x) if isinstance(x, list) else str(x)
            )
        
        # Hash the data
        try:
            data_hash = hashlib.md5(
                pd.util.hash_pandas_object(df_for_hash).values
            ).hexdigest()[:10]
        except Exception as e:
            # Fallback: hash string representation
            data_str = str(df_for_hash.values)
            data_hash = hashlib.md5(data_str.encode()).hexdigest()[:10]
        
        # Hash the parameters
        params_str = json.dumps(self.params, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:10]
        
        return f"{self.__class__.__name__.lower()}_{data_hash}_{params_hash}"
    
    def _get_cache_path(self, cache_key):
        """Lấy đường dẫn cache file"""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def _load_cached_model(self, cache_key):
        """Load model từ cache nếu có"""
        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                self.model = cached_data['model']
                self.tfidf_vectorizer = cached_data['tfidf_vectorizer']
                self.label_encoder = cached_data['label_encoder']
                self.results = cached_data['results']
                
                print(f"✓ Loaded cached model: {cache_key}")
                return True
            except Exception as e:
                print(f"Cache load failed: {e}")
                return False
        return False
    
    def _save_model_to_cache(self, cache_key):
        """Lưu model vào cache"""
        cache_path = self._get_cache_path(cache_key)
        try:
            cached_data = {
                'model': self.model,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'label_encoder': self.label_encoder,
                'results': self.results,
                'params': self.params
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f)
            
            print(f"✓ Model cached: {cache_key}")
        except Exception as e:
            print(f"Cache save failed: {e}")

    def train_and_evaluate(self, train_df, test_df, text_column='normalized_input', target_column='label'):
        """
        Main method: train model hoặc load từ cache
        
        Args:
            train_df (pd.DataFrame): Training data đã preprocessed
            test_df (pd.DataFrame): Test data đã preprocessed
            text_column (str): Tên cột text
            target_column (str): Tên cột target
            
        Returns:
            dict: Kết quả training/evaluation
        """
        print("\n=== LOGISTIC REGRESSION CLASSIFIER ===")
        
        # Generate cache key from training data only
        cache_key = self._generate_cache_key(train_df)
        
        # Thử load từ cache
        if self._load_cached_model(cache_key):
            return self.results
        
        # Nếu không có cache, train từ đầu
        print("Training new model...")
        start_time = time.time()
        
        # Prepare data
        self._prepare_data(train_df, test_df, text_column, target_column)
        
        # Initialize and train model
        self._initialize_model()
        self._train_model()
        
        training_time = time.time() - start_time
        self.results['training_time'] = round(training_time, 2)
        
        # Lưu vào cache
        self._save_model_to_cache(cache_key)
        
        return self.results
    
    def _prepare_data(self, train_df, test_df, text_column, target_column):
        """Chuẩn bị dữ liệu từ preprocessed tokens hoặc raw text"""
        
        # Convert tokens to text if using normalized_input
        if text_column == 'normalized_input':
            # Convert list of tokens back to text for TF-IDF
            train_texts = train_df[text_column].apply(
                lambda x: ' '.join(x) if isinstance(x, list) else str(x)
            )
            test_texts = test_df[text_column].apply(
                lambda x: ' '.join(x) if isinstance(x, list) else str(x)
            )
            print(f"✅ Using preprocessed tokens from '{text_column}'")
        else:
            # Use raw text directly
            train_texts = train_df[text_column].fillna('')
            test_texts = test_df[text_column].fillna('')
            print(f"✅ Using raw text from '{text_column}'")
        
        # Tạo TF-IDF vectorizer với params có thể được config từ JSON
        self.tfidf_vectorizer = TfidfVectorizer(**self.tfidf_params)
        
        # Fit vectorizer trên training data và transform cả train và test
        self.X_train = self.tfidf_vectorizer.fit_transform(train_texts)
        self.X_test = self.tfidf_vectorizer.transform(test_texts)
        
        # Encode labels
        self.y_train = self.label_encoder.fit_transform(train_df[target_column])
        self.y_test = self.label_encoder.transform(test_df[target_column])
        
        print(f"Data prepared: Train {self.X_train.shape}, Test {self.X_test.shape}")
        print(f"Label classes: {self.label_encoder.classes_}")
    
    def _initialize_model(self):
        """Khởi tạo model với params"""
        model_params = {
            'C': self.params['C'],
            'max_iter': self.params['max_iter'],
            'solver': self.params['solver'],
            'random_state': self.params['random_state']
        }
        
        # Thêm class_weight nếu có
        if 'class_weight' in self.params:
            model_params['class_weight'] = self.params['class_weight']
            
        self.model = LogisticRegression(**model_params)
    
    def _train_model(self):
        """Train model và tính metrics"""
        # Train
        self.model.fit(self.X_train, self.y_train)
        
        # Predict
        train_pred = self.model.predict(self.X_train)
        test_pred = self.model.predict(self.X_test)
        
        # Metrics
        train_acc = accuracy_score(self.y_train, train_pred)
        test_acc = accuracy_score(self.y_test, test_pred)
        f1_macro = f1_score(self.y_test, test_pred, average='macro')
        f1_weighted = f1_score(self.y_test, test_pred, average='weighted')
        
        # Classification report
        report = classification_report(
            self.y_test, test_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        # Lưu kết quả
        self.results = {
            'model_name': 'LogisticRegression',
            'params': self.params,
            'train_accuracy': round(train_acc, 4),
            'test_accuracy': round(test_acc, 4),
            'overfitting_score': round(train_acc - test_acc, 4),  # Thêm overfitting score
            'f1_macro': round(f1_macro, 4),
            'f1_weighted': round(f1_weighted, 4),
            'classification_report': report,
            'confusion_matrix': confusion_matrix(self.y_test, test_pred).tolist()
        }
        
        # In kết quả
        self._print_results()
    
    def _print_results(self):
        """Print training results to console"""
        print(f"\n{'='*50}")
        print(f"{self.results['model_name']} Results:")
        print(f"{'='*50}")
        print(f"Training Accuracy: {self.results['train_accuracy']:.4f}")
        print(f"Test Accuracy: {self.results['test_accuracy']:.4f}")
        print(f"Overfitting Score: {self.results['overfitting_score']:.4f}")
        print(f"F1-Macro: {self.results['f1_macro']:.4f}")
        print(f"F1-Weighted: {self.results['f1_weighted']:.4f}")
        
        if 'training_time' in self.results:
            print(f"Training Time: {self.results['training_time']:.2f} seconds")
        
        # Print per-class metrics
        report = self.results['classification_report']
        print(f"\nPer-Class Performance:")
        for class_name in self.label_encoder.classes_:
            if str(class_name) in report:
                class_metrics = report[str(class_name)]
                print(f"  {class_name}: Precision={class_metrics['precision']:.3f}, "
                      f"Recall={class_metrics['recall']:.3f}, F1={class_metrics['f1-score']:.3f}")
    
    def predict(self, texts):
        """Predict sentiment for new texts"""
        if self.model is None or self.tfidf_vectorizer is None:
            raise ValueError("Model not trained. Call train_and_evaluate() first.")
        
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        # Transform texts using trained vectorizer
        X_new = self.tfidf_vectorizer.transform(texts)
        
        # Predict
        predictions = self.model.predict(X_new)
        
        # Convert back to original labels
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        # Get prediction probabilities if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_new)
            return predicted_labels, probabilities
        else:
            return predicted_labels
    
    def get_results(self):
        """Get complete training results"""
        if not self.results:
            raise ValueError("No results available. Train the model first.")
        
        return self.results.copy()
