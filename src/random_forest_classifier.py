import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
import pickle
import os
import time


class RandomForestAnalyzer:
    """
    Random Forest Classifier được tối ưu hóa với hyperparameter tuning và feature engineering nâng cao
    """
    
    def __init__(self, optimize_hyperparameters=False):
        """
        Khởi tạo với tùy chọn tối ưu hyperparameters
        
        Args:
            optimize_hyperparameters (bool): Có chạy GridSearch để tối ưu parameters hay không
        """
        self.model = None
        self.best_params = None
        self.optimize_hyperparameters = optimize_hyperparameters
        
        # TF-IDF Vectorizer với cấu hình tối ưu
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=50000,  # Tăng từ 10k lên 50k
            stop_words='english',
            ngram_range=(1, 3),  # Sử dụng unigrams, bigrams, và trigrams
            min_df=2,           # Loại bỏ từ xuất hiện ít hơn 2 lần
            max_df=0.95,        # Loại bỏ từ xuất hiện quá nhiều
            sublinear_tf=True,  # Áp dụng scaling logarithmic
            analyzer='word',
            lowercase=True
        )
        
        self.label_encoder = LabelEncoder()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.results = {}
        self.training_time = 0
        
    def prepare_data(self, df, text_column, test_size=0.2, random_state=42):
        """
        Chuẩn bị dữ liệu với feature engineering nâng cao
        
        Args:
            df (pd.DataFrame): DataFrame chứa dữ liệu
            text_column (str): Tên cột chứa text cần phân tích
            test_size (float): Tỷ lệ dữ liệu test
            random_state (int): Random state cho reproducibility
        """
        print("Preparing data with enhanced feature engineering...")
        
        # Kiểm tra và xử lý missing values
        df[text_column] = df[text_column].fillna('')
        
        # Feature engineering: Kết hợp text features
        text_features = []
        if text_column in df.columns:
            text_features.append(text_column)
        
        # Nếu có processed_text thì ưu tiên sử dụng
        if 'processed_text' in df.columns:
            text_to_use = 'processed_text'
        else:
            text_to_use = text_column
        
        print(f"Using text column: {text_to_use}")
        print(f"Sample text: {df[text_to_use].iloc[0][:100]}...")
        
        # Vectorize text data với cấu hình tối ưu
        print("Vectorizing text with enhanced TF-IDF...")
        X_text = self.tfidf_vectorizer.fit_transform(df[text_to_use])
        print(f"TF-IDF matrix shape: {X_text.shape}")
        
        # Thêm numerical features nếu có
        numerical_features = []
        for col in df.columns:
            if col.endswith('_count') or col.endswith('_length') or col.startswith('has_'):
                if col in df.columns and df[col].dtype in ['int64', 'float64']:
                    numerical_features.append(col)
        
        if numerical_features:
            print(f"Adding {len(numerical_features)} numerical features")
            from scipy.sparse import hstack
            from sklearn.preprocessing import StandardScaler
            
            # Chuẩn hóa numerical features
            scaler = StandardScaler()
            numerical_data = scaler.fit_transform(df[numerical_features])
            
            # Kết hợp text features và numerical features
            X = hstack([X_text, numerical_data])
            self.scaler = scaler
            self.numerical_features = numerical_features
        else:
            print("No numerical features found, using only text features")
            X = X_text
            self.scaler = None
            self.numerical_features = []
        
        # Encode labels
        y = self.label_encoder.fit_transform(df['sentiment'])
        
        # Split data with stratification để đảm bảo phân bố đều
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")
        print(f"Feature dimensions: {X.shape[1]} features")
        print(f"Class distribution in training: {np.bincount(self.y_train)}")
        
    def get_optimized_hyperparameters(self):
        """
        Định nghĩa các hyperparameters để tối ưu
        
        Returns:
            dict: Dictionary chứa các hyperparameters để test
        """
        # Tối ưu cho tốc độ và hiệu suất trên MacBook Air 8GB RAM
        param_grid = {
            'n_estimators': [200, 300, 400],  # Tăng số trees
            'max_depth': [15, 20, 25, None],   # Tăng độ sâu
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.3, 0.5],  # Thử các cách chọn features
            'bootstrap': [True, False],
            'class_weight': [None, 'balanced', 'balanced_subsample']  # Xử lý imbalanced data
        }
        
        return param_grid
    
    def optimize_hyperparameters(self, cv_folds=3, n_jobs=-1):
        """
        Tối ưu hyperparameters sử dụng GridSearchCV
        
        Args:
            cv_folds (int): Số folds cho cross-validation
            n_jobs (int): Số parallel jobs (-1 để dùng tất cả cores)
            
        Returns:
            dict: Best parameters found
        """
        if self.X_train is None:
            raise ValueError("Chưa chuẩn bị dữ liệu. Hãy gọi prepare_data() trước.")
        
        print("Starting hyperparameter optimization...")
        print("This may take several minutes on MacBook Air 8GB RAM...")
        
        # Khởi tạo base model
        base_model = RandomForestClassifier(
            random_state=42,
            n_jobs=n_jobs,
            warm_start=False  # Tắt warm_start cho GridSearch
        )
        
        # Get parameter grid
        param_grid = self.get_optimized_hyperparameters()
        
        # GridSearchCV với tối ưu cho memory
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv_folds,
            scoring='f1_macro',  # Sử dụng F1-macro cho balanced evaluation
            n_jobs=min(n_jobs, 2),  # Giới hạn n_jobs để tránh out of memory
            verbose=1,
            pre_dispatch='2*n_jobs',  # Giới hạn số jobs pre-dispatch
            error_score='raise'
        )
        
        print(f"Testing {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * len(param_grid['max_features']) * len(param_grid['bootstrap']) * len(param_grid['class_weight'])} parameter combinations...")
        
        # Fit GridSearch
        start_time = time.time()
        grid_search.fit(self.X_train, self.y_train)
        optimization_time = time.time() - start_time
        
        # Lưu best parameters
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        
        print(f"\nHyperparameter optimization completed in {optimization_time:.2f} seconds")
        print(f"Best cross-validation F1-score: {self.best_score:.4f}")
        print(f"Best parameters: {self.best_params}")
        
        return self.best_params
    
    def initialize_model(self, **kwargs):
        """
        Khởi tạo RandomForestClassifier với parameters tối ưu
        
        Args:
            **kwargs: Custom parameters (sẽ override optimized parameters)
        """
        if self.optimize_hyperparameters and self.best_params is None:
            print("Running hyperparameter optimization...")
            self.optimize_hyperparameters()
            params = self.best_params.copy()
        else:
            # Default optimized parameters
            params = {
                'n_estimators': 300,
                'max_depth': 20,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'bootstrap': True,
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1,
                'warm_start': False
            }
        
        # Override với custom parameters
        params.update(kwargs)
        
        self.model = RandomForestClassifier(**params)
        print(f"Initialized optimized RandomForestClassifier with parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
    def train_model(self):
        """
        Huấn luyện model với enhanced training process
        
        Returns:
            dict: Kết quả huấn luyện chi tiết
        """
        if self.model is None:
            self.initialize_model()
            
        if self.X_train is None:
            raise ValueError("Chưa chuẩn bị dữ liệu. Hãy gọi prepare_data() trước.")
            
        print("Training optimized RandomForestClassifier...")
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
        
        # Feature importances
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        if self.numerical_features:
            feature_names = np.concatenate([feature_names, self.numerical_features])
        
        # Lưu kết quả
        self.results = {
            'model_name': 'OptimizedRandomForestClassifier',
            'training_time': self.training_time,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_f1_macro': train_f1_macro,
            'test_f1_macro': test_f1_macro,
            'train_f1_weighted': train_f1_weighted,
            'test_f1_weighted': test_f1_weighted,
            'overfitting_gap': train_acc - test_acc,
            'classification_report': report,
            'confusion_matrix': cm,
            'feature_importance': self.model.feature_importances_,
            'feature_names': feature_names,
            'best_params': self.best_params if hasattr(self, 'best_params') else None,
            'n_features': self.X_train.shape[1],
            'n_samples': self.X_train.shape[0]
        }
        
        print(f"\n=== TRAINING RESULTS ===")
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Overfitting Gap: {train_acc - test_acc:.4f}")
        print(f"F1-Score (Macro): {test_f1_macro:.4f}")
        print(f"F1-Score (Weighted): {test_f1_weighted:.4f}")
        print(f"Training Time: {self.training_time:.2f} seconds")
        
        return self.results
    
    def evaluate_model_detailed(self):
        """
        Đánh giá chi tiết model với các metrics nâng cao
        """
        if not self.results:
            raise ValueError("Chưa huấn luyện model. Hãy gọi train_model() trước.")
            
        print("\n=== DETAILED MODEL EVALUATION ===")
        
        # Basic metrics
        print(f"Model: {self.results['model_name']}")
        print(f"Training Time: {self.results['training_time']:.2f}s")
        print(f"Features: {self.results['n_features']}")
        print(f"Training Samples: {self.results['n_samples']}")
        
        print(f"\nAccuracy Metrics:")
        print(f"  Training: {self.results['train_accuracy']:.4f}")
        print(f"  Test: {self.results['test_accuracy']:.4f}")
        print(f"  Overfitting Gap: {self.results['overfitting_gap']:.4f}")
        
        print(f"\nF1-Score Metrics:")
        print(f"  Macro F1 (Test): {self.results['test_f1_macro']:.4f}")
        print(f"  Weighted F1 (Test): {self.results['test_f1_weighted']:.4f}")
        
        # Per-class metrics
        print(f"\nPer-Class Performance:")
        report = self.results['classification_report']
        for class_name in self.label_encoder.classes_:
            if class_name in report:
                metrics = report[class_name]
                print(f"  {class_name}:")
                print(f"    Precision: {metrics['precision']:.4f}")
                print(f"    Recall: {metrics['recall']:.4f}")
                print(f"    F1-Score: {metrics['f1-score']:.4f}")
                print(f"    Support: {int(metrics['support'])}")
        
        # Confusion Matrix Analysis
        cm = self.results['confusion_matrix']
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], cm[0,1], cm[1,0], cm[1,1])
        
        print(f"\nConfusion Matrix Analysis:")
        print(f"  True Negatives: {tn}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
        print(f"  True Positives: {tp}")
        
        # Error rates
        total = tn + fp + fn + tp
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        print(f"  False Positive Rate: {fpr:.4f}")
        print(f"  False Negative Rate: {fnr:.4f}")
        print(f"  Error Rate: {(fp + fn)/total:.4f}")
        
        # Feature importance analysis
        if self.results['feature_importance'] is not None:
            print(f"\nTop 15 Most Important Features:")
            feature_importance_df = pd.DataFrame({
                'feature': self.results['feature_names'],
                'importance': self.results['feature_importance']
            }).sort_values('importance', ascending=False)
            
            for i, (_, row) in enumerate(feature_importance_df.head(15).iterrows()):
                print(f"  {i+1:2d}. {row['feature'][:30]:30s}: {row['importance']:.6f}")
        
        return self.results
    
    def cross_validate_model(self, cv_folds=5, scoring='f1_macro'):
        """
        Cross-validation để đánh giá độ ổn định của model
        
        Args:
            cv_folds (int): Số folds cho cross-validation
            scoring (str): Metric để đánh giá
            
        Returns:
            dict: Kết quả cross-validation
        """
        if self.model is None:
            self.initialize_model()
            
        if self.X_train is None:
            raise ValueError("Chưa chuẩn bị dữ liệu. Hãy gọi prepare_data() trước.")
        
        print(f"Running {cv_folds}-fold cross-validation with {scoring} scoring...")
        
        # Combine train and test for cross-validation
        from scipy.sparse import vstack
        X_combined = vstack([self.X_train, self.X_test])
        y_combined = np.concatenate([self.y_train, self.y_test])
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_combined, y_combined, 
            cv=cv_folds, scoring=scoring, n_jobs=-1
        )
        
        cv_results = {
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_min': cv_scores.min(),
            'cv_max': cv_scores.max()
        }
        
        print(f"Cross-Validation Results ({scoring}):")
        print(f"  Individual scores: {cv_scores}")
        print(f"  Mean: {cv_results['cv_mean']:.4f}")
        print(f"  Std: {cv_results['cv_std']:.4f}")
        print(f"  Min: {cv_results['cv_min']:.4f}")
        print(f"  Max: {cv_results['cv_max']:.4f}")
        
        return cv_results
    
    def predict_with_confidence(self, text_data):
        """
        Dự đoán với confidence scores
        
        Args:
            text_data (list or str): Text data để dự đoán
            
        Returns:
            list: List of tuples (prediction, confidence)
        """
        if self.model is None:
            raise ValueError("Chưa huấn luyện model.")
            
        if isinstance(text_data, str):
            text_data = [text_data]
        
        # Vectorize new data
        X_text_new = self.tfidf_vectorizer.transform(text_data)
        
        # Add numerical features if available
        if hasattr(self, 'scaler') and self.scaler is not None:
            # Tạo dummy numerical features (tất cả = 0)
            numerical_dummy = np.zeros((len(text_data), len(self.numerical_features)))
            numerical_scaled = self.scaler.transform(numerical_dummy)
            
            from scipy.sparse import hstack
            X_new = hstack([X_text_new, numerical_scaled])
        else:
            X_new = X_text_new
        
        # Predictions và probabilities
        predictions = self.model.predict(X_new)
        probabilities = self.model.predict_proba(X_new)
        
        # Convert predictions back to labels
        prediction_labels = self.label_encoder.inverse_transform(predictions)
        
        # Get confidence (max probability)
        confidences = np.max(probabilities, axis=1)
        
        results = []
        for i, (pred, conf) in enumerate(zip(prediction_labels, confidences)):
            results.append({
                'text': text_data[i][:50] + '...' if len(text_data[i]) > 50 else text_data[i],
                'prediction': pred,
                'confidence': conf,
                'probabilities': {
                    class_name: prob for class_name, prob 
                    in zip(self.label_encoder.classes_, probabilities[i])
                }
            })
        
        return results
    
    def save_model(self, model_path='models/optimized_random_forest_model.pkl'):
        """
        Lưu optimized model
        """
        if self.model is None:
            raise ValueError("Chưa huấn luyện model.")
            
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler if hasattr(self, 'scaler') else None,
            'numerical_features': self.numerical_features if hasattr(self, 'numerical_features') else [],
            'results': self.results,
            'best_params': self.best_params if hasattr(self, 'best_params') else None
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
            
        print(f"Optimized model saved to: {model_path}")
        
    def load_model(self, model_path='models/optimized_random_forest_model.pkl'):
        """
        Load optimized model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        self.model = model_data['model']
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.label_encoder = model_data['label_encoder']
        self.scaler = model_data.get('scaler')
        self.numerical_features = model_data.get('numerical_features', [])
        self.results = model_data['results']
        self.best_params = model_data.get('best_params')
        
        print(f"Optimized model loaded from: {model_path}")
