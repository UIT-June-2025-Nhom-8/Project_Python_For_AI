import pandas as pd
import numpy as np
import os
import pickle
import time
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from scipy.sparse import hstack


class LogisticRegressionAnalyzer:
    """
    Simplified Logistic Regression Classifier optimized for binary sentiment analysis
    Focuses on: fast training, good generalization, interpretable coefficients
    """
    
    def __init__(self):
        """
        Initialize Logistic Regression Analyzer optimized for sentiment classification
        """
        self.model = None
        
        # TF-IDF Vectorizer - optimized for Logistic Regression and sentiment analysis
        # LogReg works well with moderate feature counts and proper regularization
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=35000,     # Optimal for LogReg - not too many features
            stop_words=None,   # Remove common words
            ngram_range=(1, 2),     # Unigrams + bigrams (sufficient for sentiment)
            min_df=3,              # Filter rare words (reduce noise)
            max_df=0.85,           # Remove very common words
            sublinear_tf=True,     # Log scaling helps LogReg
            analyzer='word',
            lowercase=True,
            strip_accents='unicode'
        )
        
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()  # Important for LogReg
        
        # Data storage
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.results = {}
        self.training_time = 0
        self.numerical_features = []
        self.training_time = 0
        
    def prepare_data(self, df, text_column, test_size=0.2, random_state=42):
        """
        Chuẩn bị dữ liệu cho machine learning với feature engineering
        
        Args:
            df (pd.DataFrame): DataFrame chứa dữ liệu
            text_column (str): Tên cột chứa text cần phân tích
            test_size (float): Tỷ lệ dữ liệu test
            random_state (int): Random state cho reproducibility
        """
        print("Chuẩn bị dữ liệu cho Logistic Regression...")
        
        # Kiểm tra và xử lý missing values
        df[text_column] = df[text_column].fillna('')
        
        # Vectorize text data
        print("Creating TF-IDF features optimized for Logistic Regression...")
        X_text = self.tfidf_vectorizer.fit_transform(df[text_column])
        print(f"TF-IDF matrix shape: {X_text.shape}")
        
        # Thêm numerical features nếu có
        numerical_features = []
        for col in df.columns:
            if (col.endswith('_count') or col.endswith('_length') or 
                col.startswith('has_') or col in ['exclamation_count', 'question_count', 
                                                'uppercase_count', 'negation_count']):
                numerical_features.append(col)
        
        if numerical_features:
            print(f"Adding {len(numerical_features)} numerical features")
            
            # Chuẩn hóa numerical features - quan trọng cho LogReg
            numerical_data = self.scaler.fit_transform(df[numerical_features])
            
            # Kết hợp text features và numerical features
            X = hstack([X_text, numerical_data])
            self.numerical_features = numerical_features
        else:
            print("No additional numerical features found")
            X = X_text
        
        # Encode labels
        y = self.label_encoder.fit_transform(df['sentiment'])
        
        # Split data với stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")
        print(f"Feature dimensions: {X.shape[1]} features")
        print(f"Class distribution in training: {np.bincount(self.y_train)}")
        
    def get_hyperparameter_grid(self):
        """
        Định nghĩa hyperparameters để tối ưu cho Logistic Regression
        
        Returns:
            dict: Dictionary chứa các hyperparameters để test
        """
        param_grid = {
            'C': [0.1, 1.0, 10.0, 100.0],  # Regularization strength
            'penalty': ['l1', 'l2', 'elasticnet'],  # Regularization type
            'solver': ['liblinear', 'saga'],  # Optimization algorithm
            'max_iter': [1000, 2000, 5000],  # Maximum iterations
            'class_weight': [None, 'balanced'],  # Handle class imbalance
            'l1_ratio': [0.1, 0.5, 0.9]  # For elasticnet penalty
        }
        
        return param_grid
    
    def optimize_hyperparameters(self, cv_folds=5, n_jobs=-1):
        """
        Tối ưu hyperparameters sử dụng GridSearchCV
        
        Args:
            cv_folds (int): Số folds cho cross-validation
            n_jobs (int): Số parallel jobs (-1 để dùng tất cả cores)
            
        Returns:
            dict: Best parameters found
        """
        if self.X_train is None:
            raise ValueError("Chưa chuẩn bị dữ liệu. Gọi prepare_data() trước.")
        
        print("Starting hyperparameter optimization for Logistic Regression...")
        print("This process optimizes regularization and solver parameters...")
        
        # Khởi tạo base model
        base_model = LogisticRegression(
            random_state=42,
            n_jobs=n_jobs
        )
        
        # Get parameter grid - cần handle elasticnet special case
        param_combinations = []
        
        # L1 and L2 regularization
        for C in [0.1, 1.0, 10.0, 100.0]:
            for penalty in ['l1', 'l2']:
                for solver in ['liblinear', 'saga']:
                    for max_iter in [1000, 2000]:
                        for class_weight in [None, 'balanced']:
                            if penalty == 'l1' and solver == 'liblinear':
                                param_combinations.append({
                                    'C': C, 'penalty': penalty, 'solver': solver,
                                    'max_iter': max_iter, 'class_weight': class_weight
                                })
                            elif penalty == 'l2':
                                param_combinations.append({
                                    'C': C, 'penalty': penalty, 'solver': solver,
                                    'max_iter': max_iter, 'class_weight': class_weight
                                })
        
        # Elasticnet regularization (only with saga solver)
        for C in [0.1, 1.0, 10.0]:
            for l1_ratio in [0.1, 0.5, 0.9]:
                for max_iter in [1000, 2000]:
                    for class_weight in [None, 'balanced']:
                        param_combinations.append({
                            'C': C, 'penalty': 'elasticnet', 'solver': 'saga',
                            'max_iter': max_iter, 'class_weight': class_weight,
                            'l1_ratio': l1_ratio
                        })
        
        print(f"Testing {len(param_combinations)} parameter combinations...")
        
        # Manual grid search to handle elasticnet properly
        best_score = -1
        best_params = None
        
        from sklearn.model_selection import cross_val_score
        
        for i, params in enumerate(param_combinations[:50]):  # Limit to first 50 for speed
            try:
                model = LogisticRegression(random_state=42, **params)
                scores = cross_val_score(model, self.X_train, self.y_train, 
                                       cv=cv_folds, scoring='f1_macro', n_jobs=2)
                mean_score = scores.mean()
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params.copy()
                    
                if i % 10 == 0:
                    print(f"Processed {i+1}/{min(50, len(param_combinations))} combinations...")
                    
            except Exception as e:
                continue  # Skip invalid parameter combinations
        
        # Lưu best parameters
        self.best_params = best_params
        self.best_score = best_score
        
        print(f"\nHyperparameter optimization completed!")
        print(f"Best cross-validation F1-score: {self.best_score:.4f}")
        print(f"Best parameters: {self.best_params}")
        
        return self.best_params
    
    def initialize_model(self, **kwargs):
        """
        Khởi tạo LogisticRegression với parameters tối ưu
        
        Args:
            **kwargs: Custom parameters (sẽ override optimized parameters)
        """
        if self.optimize_hyperparameters and self.best_params is None:
            self.optimize_hyperparameters()
            params = self.best_params.copy()
        else:
            # Default optimized parameters based on common best practices
            params = {
                'C': 1.0,
                'penalty': 'l2',
                'solver': 'liblinear',
                'max_iter': 1000,
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1
            }
        
        # Override với custom parameters
        params.update(kwargs)
        
        self.model = LogisticRegression(**params)
        print(f"Initialized Logistic Regression with parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
    def train_model(self):
        """
        Huấn luyện model Logistic Regression
        
        Returns:
            dict: Kết quả huấn luyện chi tiết
        """
        if self.model is None:
            self.initialize_model()
            
        if self.X_train is None:
            raise ValueError("Chưa chuẩn bị dữ liệu. Gọi prepare_data() trước.")
            
        print("Bắt đầu huấn luyện Logistic Regression...")
        print(f"Training on {self.X_train.shape[0]} samples with {self.X_train.shape[1]} features")
        
        # Training với time tracking
        start_time = time.time()
        self.model.fit(self.X_train, self.y_train)
        self.training_time = time.time() - start_time
        
        print(f"Training completed in {self.training_time:.2f} seconds")
        
        # Predictions
        print("Making predictions...")
        train_pred = self.model.predict(self.X_train)
        test_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        train_acc = accuracy_score(self.y_train, train_pred)
        test_acc = accuracy_score(self.y_test, test_pred)
        
        # F1-scores cho từng class
        train_f1_macro = f1_score(self.y_train, train_pred, average='macro')
        test_f1_macro = f1_score(self.y_test, test_pred, average='macro')
        train_f1_weighted = f1_score(self.y_train, train_pred, average='weighted')
        test_f1_weighted = f1_score(self.y_test, test_pred, average='weighted')
        
        # Classification report chi tiết
        report = classification_report(
            self.y_test, test_pred, 
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, test_pred)
        
        # Feature coefficients (weights)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        if self.numerical_features:
            feature_names = np.concatenate([feature_names, self.numerical_features])
        
        coefficients = self.model.coef_[0] if len(self.model.coef_.shape) > 1 else self.model.coef_
        
        # Lưu kết quả
        self.results = {
            'model_name': 'LogisticRegression',
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_f1_macro': train_f1_macro,
            'test_f1_macro': test_f1_macro,
            'train_f1_weighted': train_f1_weighted,
            'test_f1_weighted': test_f1_weighted,
            'classification_report': report,
            'confusion_matrix': cm,
            'coefficients': coefficients,
            'feature_names': feature_names,
            'training_time': self.training_time,
            'n_iterations': getattr(self.model, 'n_iter_', [0])[0] if hasattr(self.model, 'n_iter_') else 'N/A'
        }
        
        print(f"Huấn luyện hoàn thành!")
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test F1-Macro: {test_f1_macro:.4f}")
        print(f"Training Time: {self.training_time:.2f} seconds")
        
        return self.results
    
    def evaluate_model(self):
        """
        Đánh giá chi tiết model
        
        Returns:
            tuple: (classification_report, confusion_matrix)
        """
        if not self.results:
            raise ValueError("Chưa huấn luyện model. Gọi train_model() trước.")
            
        print("\n=== ĐÁNH GIÁ LOGISTIC REGRESSION CLASSIFIER ===")
        print(f"Training Accuracy: {self.results['train_accuracy']:.4f}")
        print(f"Test Accuracy: {self.results['test_accuracy']:.4f}")
        print(f"Test F1-Macro Score: {self.results['test_f1_macro']:.4f}")
        print(f"Test F1-Weighted Score: {self.results['test_f1_weighted']:.4f}")
        print(f"Training Time: {self.results['training_time']:.2f} seconds")
        print(f"Convergence Iterations: {self.results['n_iterations']}")
        
        print("\nClassification Report:")
        report_df = pd.DataFrame(self.results['classification_report']).transpose()
        print(report_df)
        
        print(f"\nConfusion Matrix:")
        print(self.results['confusion_matrix'])
        
        # Top positive and negative coefficients
        if self.results['coefficients'] is not None:
            print("\nTop 10 Most Positive Features:")
            pos_indices = np.argsort(self.results['coefficients'])[-10:][::-1]
            for i, idx in enumerate(pos_indices):
                if idx < len(self.results['feature_names']):
                    coef = self.results['coefficients'][idx]
                    feature = self.results['feature_names'][idx]
                    print(f"{i+1:2d}. {feature}: {coef:.4f}")
            
            print("\nTop 10 Most Negative Features:")
            neg_indices = np.argsort(self.results['coefficients'])[:10]
            for i, idx in enumerate(neg_indices):
                if idx < len(self.results['feature_names']):
                    coef = self.results['coefficients'][idx]
                    feature = self.results['feature_names'][idx]
                    print(f"{i+1:2d}. {feature}: {coef:.4f}")
        
        return self.results['classification_report'], self.results['confusion_matrix']
    
    def predict(self, text_data):
        """
        Dự đoán cảm xúc cho dữ liệu mới
        
        Args:
            text_data (list or str): Dữ liệu text cần dự đoán
            
        Returns:
            list: Dự đoán cảm xúc
        """
        if self.model is None:
            raise ValueError("Model chưa được huấn luyện. Gọi train_model() trước.")
            
        if isinstance(text_data, str):
            text_data = [text_data]
            
        # Vectorize dữ liệu mới
        X_new = self.tfidf_vectorizer.transform(text_data)
        
        # Dự đoán
        predictions = self.model.predict(X_new)
        
        # Chuyển đổi về label gốc
        sentiment_predictions = self.label_encoder.inverse_transform(predictions)
        
        return sentiment_predictions.tolist()
    
    def predict_proba(self, text_data):
        """
        Dự đoán xác suất cho từng class
        
        Args:
            text_data (list or str): Dữ liệu text cần dự đoán
            
        Returns:
            numpy.ndarray: Ma trận xác suất
        """
        if self.model is None:
            raise ValueError("Model chưa được huấn luyện. Gọi train_model() trước.")
            
        if isinstance(text_data, str):
            text_data = [text_data]
            
        # Vectorize dữ liệu mới
        X_new = self.tfidf_vectorizer.transform(text_data)
        
        # Dự đoán xác suất
        probabilities = self.model.predict_proba(X_new)
        
        return probabilities
    
    def save_model(self, model_path='models/logistic_regression_model.pkl'):
        """
        Lưu model và các thành phần liên quan
        
        Args:
            model_path (str): Đường dẫn lưu model
        """
        if self.model is None:
            raise ValueError("Model chưa được huấn luyện. Gọi train_model() trước.")
            
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Lưu toàn bộ analyzer
        model_data = {
            'model': self.model,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'results': self.results,
            'best_params': self.best_params,
            'numerical_features': self.numerical_features
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
            
        print(f"Model đã được lưu tại: {model_path}")
    
    def load_model(self, model_path='models/logistic_regression_model.pkl'):
        """
        Tải model đã lưu
        
        Args:
            model_path (str): Đường dẫn đến model đã lưu
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy file model: {model_path}")
            
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        self.model = model_data['model']
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.label_encoder = model_data['label_encoder']
        self.scaler = model_data['scaler']
        self.results = model_data['results']
        self.best_params = model_data.get('best_params', None)
        self.numerical_features = model_data.get('numerical_features', [])
        
        print(f"Model đã được tải từ: {model_path}")
    
    def get_model_info(self):
        """
        Lấy thông tin về model
        
        Returns:
            dict: Thông tin chi tiết về model
        """
        if self.model is None:
            raise ValueError("Model chưa được huấn luyện. Gọi train_model() trước.")
            
        info = {
            "model_type": "LogisticRegression",
            "C": self.model.C,
            "penalty": self.model.penalty,
            "solver": self.model.solver,
            "max_iter": self.model.max_iter,
            "class_weight": str(self.model.class_weight),
            "n_features": self.model.n_features_in_ if hasattr(self.model, 'n_features_in_') else 'N/A',
            "classes": self.model.classes_.tolist() if hasattr(self.model, 'classes_') else 'N/A'
        }
        
        if self.results:
            info.update({
                "train_accuracy": self.results['train_accuracy'],
                "test_accuracy": self.results['test_accuracy'],
                "test_f1_macro": self.results['test_f1_macro'],
                "training_time": self.results['training_time'],
                "n_iterations": self.results['n_iterations']
            })
            
        return info
