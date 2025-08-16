import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os


class GradientBoostingAnalyzer:
    """
    Lớp phân tích cảm xúc sử dụng GradientBoostingClassifier
    """
    
    def __init__(self):
        """Khởi tạo các thành phần của GradientBoostingAnalyzer"""
        self.model = None
        # TF-IDF Vectorizer với cấu hình tối ưu
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=30000,  # Tăng từ 10k lên 30k
            stop_words='english',
            ngram_range=(1, 2),  # Sử dụng unigrams và bigrams
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
        
    def prepare_data(self, df, text_column, test_size=0.2, random_state=42):
        """
        Chuẩn bị dữ liệu cho machine learning
        
        Args:
            df (pd.DataFrame): DataFrame chứa dữ liệu
            text_column (str): Tên cột chứa text cần phân tích
            test_size (float): Tỷ lệ dữ liệu test
            random_state (int): Random state cho reproducibility
        """
        print("Chuẩn bị dữ liệu cho GradientBoosting...")
        
        # Vectorize text data
        print("Creating enhanced TF-IDF features...")
        X = self.tfidf_vectorizer.fit_transform(df[text_column].fillna(''))
        
        # Thêm numerical features nếu có
        numerical_features = []
        for col in df.columns:
            if (col.endswith('_count') or col.endswith('_length') or 
                col.startswith('has_') or col in ['exclamation_count', 'question_count', 
                                                'uppercase_count', 'negation_count']):
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
            X = hstack([X, numerical_data])
            self.scaler = scaler
            self.numerical_features = numerical_features
        else:
            X = X
            self.scaler = None
            self.numerical_features = []
        
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
        Khởi tạo GradientBoostingClassifier với các tham số tùy chỉnh
        
        Args:
            **kwargs: Các tham số cho GradientBoostingClassifier
        """
        default_params = {
            'n_estimators': 200,        # Tăng từ 100
            'learning_rate': 0.1,       # Giữ nguyên
            'max_depth': 8,             # Tăng từ 3
            'subsample': 0.8,           # Giảm overfitting
            'min_samples_split': 10,    # Tăng để giảm overfitting
            'min_samples_leaf': 5,      # Tăng để giảm overfitting
            'random_state': 42
        }
        default_params.update(kwargs)
        
        self.model = GradientBoostingClassifier(**default_params)
        print(f"Khởi tạo GradientBoostingClassifier với tham số: {default_params}")
        
    def train_model(self):
        """
        Huấn luyện model GradientBoostingClassifier
        
        Returns:
            dict: Kết quả huấn luyện bao gồm accuracy và metrics
        """
        if self.model is None:
            self.initialize_model()
            
        if self.X_train is None:
            raise ValueError("Chưa chuẩn bị dữ liệu. Hãy gọi prepare_data() trước.")
            
        print("Bắt đầu huấn luyện GradientBoostingClassifier...")
        
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
            'model_name': 'GradientBoostingClassifier',
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
            
        print("\n=== ĐÁNH GIÁ GRADIENT BOOSTING CLASSIFIER ===")
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
    
    def save_model(self, model_path='models/gradient_boosting_model.pkl'):
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
    
    def load_model(self, model_path='models/gradient_boosting_model.pkl'):
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
            "model_type": "GradientBoostingClassifier",
            "n_estimators": self.model.n_estimators,
            "learning_rate": self.model.learning_rate,
            "max_depth": self.model.max_depth,
            "random_state": self.model.random_state,
            "n_features": self.model.n_features_ if hasattr(self.model, 'n_features_') else None,
            "n_classes": self.model.n_classes_ if hasattr(self.model, 'n_classes_') else None,
        }
        
        if self.results:
            info.update({
                "train_accuracy": self.results['train_accuracy'],
                "test_accuracy": self.results['test_accuracy']
            })
            
        return info
