import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os


class RandomForestAnalyzer:
    """
    Lớp phân tích cảm xúc sử dụng RandomForestClassifier
    """
    
    def __init__(self):
        """Khởi tạo các thành phần của RandomForestAnalyzer"""
        self.model = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        self.label_encoder = LabelEncoder()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.results = {}
        
    def prepare_data(self, df, text_column, test_size=0.2, random_state=42):
        """
        Chuẩn bị dữ liệu cho machine learning
        
        Args:
            df (pd.DataFrame): DataFrame chứa dữ liệu
            text_column (str): Tên cột chứa text cần phân tích
            test_size (float): Tỷ lệ dữ liệu test
            random_state (int): Random state cho reproducibility
        """
        print("Chuẩn bị dữ liệu cho RandomForest...")
        
        # Vectorize text data
        X = self.tfidf_vectorizer.fit_transform(df[text_column].fillna(''))
        
        # Encode labels
        y = self.label_encoder.fit_transform(df['sentiment'])
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")
        
    def initialize_model(self, **kwargs):
        """
        Khởi tạo RandomForestClassifier với các tham số tùy chỉnh
        
        Args:
            **kwargs: Các tham số cho RandomForestClassifier
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(kwargs)
        
        self.model = RandomForestClassifier(**default_params)
        print(f"Khởi tạo RandomForestClassifier với tham số: {default_params}")
        
    def train_model(self):
        """
        Huấn luyện model RandomForestClassifier
        
        Returns:
            dict: Kết quả huấn luyện bao gồm accuracy và metrics
        """
        if self.model is None:
            self.initialize_model()
            
        if self.X_train is None:
            raise ValueError("Chưa chuẩn bị dữ liệu. Hãy gọi prepare_data() trước.")
            
        print("Bắt đầu huấn luyện RandomForestClassifier...")
        
        # Huấn luyện model
        self.model.fit(self.X_train, self.y_train)
        
        # Dự đoán
        train_pred = self.model.predict(self.X_train)
        test_pred = self.model.predict(self.X_test)
        
        # Tính accuracy
        train_acc = accuracy_score(self.y_train, train_pred)
        test_acc = accuracy_score(self.y_test, test_pred)
        
        # Tạo classification report
        report = classification_report(
            self.y_test, test_pred, 
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, test_pred)
        
        # Lưu kết quả
        self.results = {
            'model_name': 'RandomForestClassifier',
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'classification_report': report,
            'confusion_matrix': cm,
            'feature_importance': self.model.feature_importances_ if hasattr(self.model, 'feature_importances_') else None
        }
        
        print(f"Huấn luyện hoàn thành!")
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        
        return self.results
    
    def evaluate_model(self):
        """
        Đánh giá chi tiết model
        
        Returns:
            tuple: (classification_report, confusion_matrix)
        """
        if not self.results:
            raise ValueError("Chưa huấn luyện model. Hãy gọi train_model() trước.")
            
        print("\n=== ĐÁNH GIÁ RANDOM FOREST CLASSIFIER ===")
        print(f"Training Accuracy: {self.results['train_accuracy']:.4f}")
        print(f"Test Accuracy: {self.results['test_accuracy']:.4f}")
        
        print("\nClassification Report:")
        report_df = pd.DataFrame(self.results['classification_report']).transpose()
        print(report_df)
        
        print(f"\nConfusion Matrix:")
        print(self.results['confusion_matrix'])
        
        if self.results['feature_importance'] is not None:
            print(f"\nTop 10 Feature Importances:")
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.results['feature_importance']
            }).sort_values('importance', ascending=False)
            print(importance_df.head(10))
        
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
            raise ValueError("Chưa huấn luyện model.")
            
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
            raise ValueError("Chưa huấn luyện model.")
            
        if isinstance(text_data, str):
            text_data = [text_data]
            
        # Vectorize dữ liệu mới
        X_new = self.tfidf_vectorizer.transform(text_data)
        
        # Dự đoán xác suất
        probabilities = self.model.predict_proba(X_new)
        
        return probabilities
    
    def save_model(self, model_path='models/random_forest_model.pkl'):
        """
        Lưu model và các thành phần liên quan
        
        Args:
            model_path (str): Đường dẫn lưu model
        """
        if self.model is None:
            raise ValueError("Chưa huấn luyện model.")
            
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Lưu toàn bộ analyzer
        model_data = {
            'model': self.model,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'label_encoder': self.label_encoder,
            'results': self.results
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
            
        print(f"Model đã được lưu tại: {model_path}")
    
    def load_model(self, model_path='models/random_forest_model.pkl'):
        """
        Tải model đã lưu
        
        Args:
            model_path (str): Đường dẫn đến model đã lưu
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy model tại: {model_path}")
            
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        self.model = model_data['model']
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.label_encoder = model_data['label_encoder']
        self.results = model_data['results']
        
        print(f"Model đã được tải từ: {model_path}")
    
    def get_model_info(self):
        """
        Lấy thông tin về model
        
        Returns:
            dict: Thông tin chi tiết về model
        """
        if self.model is None:
            return {"status": "Model chưa được huấn luyện"}
            
        info = {
            "model_type": "RandomForestClassifier",
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
            "min_samples_split": self.model.min_samples_split,
            "min_samples_leaf": self.model.min_samples_leaf,
            "max_features": self.model.max_features,
            "random_state": self.model.random_state,
            "n_jobs": self.model.n_jobs,
        }
        
        if hasattr(self.model, 'n_features_'):
            info["n_features"] = self.model.n_features_
            
        if hasattr(self.model, 'n_classes_'):
            info["n_classes"] = self.model.n_classes_
        
        if self.results:
            info.update({
                "train_accuracy": self.results['train_accuracy'],
                "test_accuracy": self.results['test_accuracy']
            })
            
        return info
    
    def get_feature_importance(self, top_n=20):
        """
        Lấy feature importance từ Random Forest
        
        Args:
            top_n (int): Số lượng features quan trọng nhất
            
        Returns:
            pd.DataFrame: DataFrame chứa feature importance
        """
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model chưa được huấn luyện hoặc không có feature importance.")
            
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def train_with_cross_validation(self, cv=5):
        """
        Huấn luyện model với cross-validation
        
        Args:
            cv (int): Số folds cho cross-validation
            
        Returns:
            dict: Kết quả cross-validation
        """
        if self.model is None:
            self.initialize_model()
            
        if self.X_train is None:
            raise ValueError("Chưa chuẩn bị dữ liệu. Hãy gọi prepare_data() trước.")
            
        from sklearn.model_selection import cross_val_score
        
        print(f"Bắt đầu cross-validation với {cv} folds...")
        
        # Combine training and test data for CV
        X_combined = np.vstack([self.X_train.toarray(), self.X_test.toarray()])
        y_combined = np.hstack([self.y_train, self.y_test])
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.model, X_combined, y_combined, cv=cv, scoring='accuracy')
        
        # Train on full data
        self.model.fit(self.X_train, self.y_train)
        
        # Final predictions
        train_pred = self.model.predict(self.X_train)
        test_pred = self.model.predict(self.X_test)
        
        train_acc = accuracy_score(self.y_train, train_pred)
        test_acc = accuracy_score(self.y_test, test_pred)
        
        # Tạo classification report
        report = classification_report(
            self.y_test, test_pred, 
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, test_pred)
        
        # Lưu kết quả
        self.results = {
            'model_name': 'RandomForestClassifier (with CV)',
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': report,
            'confusion_matrix': cm,
            'feature_importance': self.model.feature_importances_
        }
        
        print(f"Cross-Validation Scores: {cv_scores}")
        print(f"CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        
        return self.results
