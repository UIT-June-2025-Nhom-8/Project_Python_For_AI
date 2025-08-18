import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from scipy.sparse import hstack
import pickle
import os
import time


class RandomForestAnalyzer:
    """
    Simplified Random Forest Classifier optimized for binary sentiment analysis (Positive/Negative)
    Focuses on: reducing overfitting, increasing accuracy, clear code structure
    """
    
    def __init__(self):
        """
        Initialize Random Forest Analyzer with optimized settings for sentiment classification
        """
        self.model = None
        
        # TF-IDF Vectorizer - optimized for sentiment analysis
        # Reduced features to prevent overfitting, focused on important sentiment words
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=15000,     # Reduced from 35k to prevent overfitting
            stop_words='english',   # Remove common words
            ngram_range=(1, 2),     # Unigrams + bigrams (trigrams removed to reduce complexity)
            min_df=3,              # Increased to filter rare words (reduce noise)
            max_df=0.9,            # Remove very common words
            sublinear_tf=True,     # Log scaling for better performance
            analyzer='word',
            lowercase=True,
            strip_accents='unicode' # Handle accented characters
        )
        
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()  # For numerical features
        
        # Data storage
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.results = {}
        self.training_time = 0
        self.numerical_features = []
        
    def prepare_data(self, df, text_column, test_size=0.2, random_state=42):
        """
        Prepare data for sentiment classification with anti-overfitting measures
        
        Args:
            df (pd.DataFrame): DataFrame containing the data
            text_column (str): Name of text column for analysis
            test_size (float): Test data ratio
            random_state (int): Random state for reproducibility
        """
        print("🔄 Preparing data for binary sentiment classification...")
        
        # 1. Handle missing values
        df[text_column] = df[text_column].fillna('')
        print(f"✅ Cleaned missing values in text column")
        
        # 2. Choose text column (prefer processed_text if available)
        text_to_use = 'processed_text' if 'processed_text' in df.columns else text_column
        print(f"📝 Using text column: {text_to_use}")
        print(f"📋 Sample text: {df[text_to_use].iloc[0][:100]}...")
        
        # 3. Create TF-IDF features (main features for sentiment analysis)
        print("🔍 Creating TF-IDF features optimized for sentiment...")
        X_text = self.tfidf_vectorizer.fit_transform(df[text_to_use])
        print(f"📊 TF-IDF matrix shape: {X_text.shape}")
        
        # 4. Add simple numerical features (if available)
        # Only include features that are likely to help with sentiment
        numerical_features = []
        potential_features = ['text_length', 'exclamation_count', 'question_count', 
                            'uppercase_count', 'negation_count', 'word_count']
        
        for col in potential_features:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                numerical_features.append(col)
        
        # 5. Combine features if numerical features exist
        if numerical_features:
            print(f"➕ Adding {len(numerical_features)} numerical features: {numerical_features}")
            
            # Scale numerical features to prevent dominance
            numerical_data = self.scaler.fit_transform(df[numerical_features])
            
            # Combine text and numerical features
            X = hstack([X_text, numerical_data])
            self.numerical_features = numerical_features
        else:
            print("📝 Using only text features (no additional numerical features found)")
            X = X_text
            self.numerical_features = []
        
        # 6. Encode sentiment labels (binary classification)
        y = self.label_encoder.fit_transform(df['sentiment'])
        print(f"🏷️  Sentiment classes: {self.label_encoder.classes_}")
        
        # 7. Split data with stratification (maintain class balance)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # 8. Display data summary
        print(f"✅ Data preparation completed!")
        print(f"📊 Training set: {self.X_train.shape[0]} samples")
        print(f"📊 Test set: {self.X_test.shape[0]} samples")
        print(f"📊 Total features: {X.shape[1]}")
        print(f"📊 Class distribution in training: {dict(zip(self.label_encoder.classes_, np.bincount(self.y_train)))}")
        
        # 9. Check for potential overfitting risks
        if X.shape[1] > self.X_train.shape[0] * 0.1:
            print("⚠️  Warning: High feature-to-sample ratio detected. Model tuned to prevent overfitting.")
        
        return True
        
    def initialize_model(self):
        """
        Initialize Random Forest with anti-overfitting parameters
        Optimized specifically for binary sentiment classification
        """
        # Anti-overfitting parameters based on best practices for sentiment analysis
        self.model = RandomForestClassifier(
            # Tree parameters - prevent overfitting
            n_estimators=100,           # Reduced from 300 (fewer trees = less overfitting)
            max_depth=10,               # Limited depth to prevent memorization
            min_samples_split=20,       # Increased minimum samples to split (more robust)
            min_samples_leaf=10,        # Increased minimum samples per leaf (smoother decisions)
            
            # Feature selection - reduce model complexity
            max_features='sqrt',        # Use sqrt of features (prevents relying on too many features)
            
            # Sampling parameters - add randomness to prevent overfitting
            bootstrap=True,             # Use bootstrap sampling
            class_weight='balanced',    # Handle class imbalance automatically
            
            # Performance parameters
            random_state=42,            # Reproducible results
            n_jobs=-1,                  # Use all CPU cores
            warm_start=False            # Fresh training each time
        )
        
        print("🌲 Initialized Random Forest with anti-overfitting parameters:")
        print(f"   📊 Trees: {self.model.n_estimators}")
        print(f"   📏 Max depth: {self.model.max_depth}")
        print(f"   🔢 Min samples split: {self.model.min_samples_split}")
        print(f"   🍃 Min samples leaf: {self.model.min_samples_leaf}")
        print(f"   🎯 Max features: {self.model.max_features}")
        print(f"   ⚖️  Class weight: {self.model.class_weight}")
        
        return self.model
        
    def train_model(self):
        """
        Train Random Forest model with overfitting monitoring
        
        Returns:
            dict: Detailed training results with overfitting analysis
        """
        if self.model is None:
            self.initialize_model()
            
        if self.X_train is None:
            raise ValueError("❌ Data not prepared. Call prepare_data() first.")
            
        print("🚀 Training Random Forest for sentiment classification...")
        print(f"📊 Training samples: {self.X_train.shape[0]}")
        print(f"📊 Features: {self.X_train.shape[1]}")
        
        # Train model with timing
        start_time = time.time()
        self.model.fit(self.X_train, self.y_train)
        self.training_time = time.time() - start_time
        
        print(f"✅ Training completed in {self.training_time:.2f} seconds")
        
        # Make predictions for evaluation
        print("🔍 Evaluating model performance...")
        train_pred = self.model.predict(self.X_train)
        test_pred = self.model.predict(self.X_test)
        
        # Calculate comprehensive metrics
        train_acc = accuracy_score(self.y_train, train_pred)
        test_acc = accuracy_score(self.y_test, test_pred)
        
        # F1 scores (important for sentiment analysis)
        train_f1 = f1_score(self.y_train, train_pred, average='weighted')
        test_f1 = f1_score(self.y_test, test_pred, average='weighted')
        train_f1_macro = f1_score(self.y_train, train_pred, average='macro')
        test_f1_macro = f1_score(self.y_test, test_pred, average='macro')
        
        # Overfitting analysis
        overfitting_gap = train_acc - test_acc
        f1_overfitting_gap = train_f1 - test_f1
        
        # Classification report for detailed analysis
        report = classification_report(
            self.y_test, test_pred, 
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, test_pred)
        
        # Feature importance analysis
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        if self.numerical_features:
            feature_names = np.concatenate([feature_names, self.numerical_features])
        
        # Store comprehensive results
        self.results = {
            'model_name': 'Optimized_RandomForest_SentimentAnalysis',
            'training_time': self.training_time,
            
            # Accuracy metrics
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'accuracy_overfitting_gap': overfitting_gap,
            
            # F1 metrics (more important for sentiment analysis)
            'train_f1_weighted': train_f1,
            'test_f1_weighted': test_f1,
            'train_f1_macro': train_f1_macro,
            'test_f1_macro': test_f1_macro,
            'f1_overfitting_gap': f1_overfitting_gap,
            
            # Detailed analysis
            'classification_report': report,
            'confusion_matrix': cm,
            'feature_importances': self.model.feature_importances_,
            'feature_names': feature_names,
            
            # Model info
            'n_features': self.X_train.shape[1],
            'n_samples': self.X_train.shape[0],
            'model_params': self.model.get_params()
        }
        
        # Display results with overfitting analysis
        print(f"\n🎯 TRAINING RESULTS:")
        print(f"📈 Training Accuracy: {train_acc:.4f}")
        print(f"📊 Test Accuracy: {test_acc:.4f}")
        print(f"📉 Overfitting Gap: {overfitting_gap:.4f}", end="")
        
        # Overfitting warning
        if overfitting_gap > 0.1:
            print(" ⚠️  HIGH - Consider reducing model complexity")
        elif overfitting_gap > 0.05:
            print(" ⚡ MODERATE - Model is acceptable")
        else:
            print(" ✅ LOW - Excellent generalization")
            
        print(f"🎯 F1-Score (Weighted): {test_f1:.4f}")
        print(f"🎯 F1-Score (Macro): {test_f1_macro:.4f}")
        print(f"⏱️  Training Time: {self.training_time:.2f}s")
        
        return self.results
    
    def evaluate_model(self):
        """
        Comprehensive model evaluation with focus on sentiment analysis metrics
        """
        if not self.results:
            raise ValueError("❌ Model not trained. Call train_model() first.")
            
        print("\n📊 DETAILED SENTIMENT ANALYSIS EVALUATION")
        print("=" * 50)
        
        # Basic model info
        print(f"🤖 Model: {self.results['model_name']}")
        print(f"⏱️  Training Time: {self.results['training_time']:.2f}s")
        print(f"📊 Features: {self.results['n_features']:,}")
        print(f"📊 Training Samples: {self.results['n_samples']:,}")
        
        # Performance metrics
        print(f"\n🎯 PERFORMANCE METRICS:")
        print(f"   📈 Training Accuracy: {self.results['train_accuracy']:.4f}")
        print(f"   📊 Test Accuracy: {self.results['test_accuracy']:.4f}")
        print(f"   📉 Overfitting Gap: {self.results['accuracy_overfitting_gap']:.4f}")
        
        # Overfitting assessment
        if self.results['accuracy_overfitting_gap'] > 0.1:
            print("   ⚠️  HIGH overfitting - consider simpler model")
        elif self.results['accuracy_overfitting_gap'] > 0.05:
            print("   ⚡ MODERATE overfitting - acceptable for production")
        else:
            print("   ✅ LOW overfitting - excellent generalization")
        
        # F1 scores (critical for sentiment analysis)
        print(f"\n🎯 F1-SCORE ANALYSIS:")
        print(f"   🎯 Weighted F1 (Test): {self.results['test_f1_weighted']:.4f}")
        print(f"   🎯 Macro F1 (Test): {self.results['test_f1_macro']:.4f}")
        print(f"   📉 F1 Overfitting Gap: {self.results['f1_overfitting_gap']:.4f}")
        
        # Per-class performance (Positive vs Negative)
        print(f"\n📊 PER-CLASS PERFORMANCE:")
        report = self.results['classification_report']
        for class_name in self.label_encoder.classes_:
            if class_name in report:
                metrics = report[class_name]
                print(f"   {class_name.upper()}:")
                print(f"     🎯 Precision: {metrics['precision']:.4f}")
                print(f"     🔍 Recall: {metrics['recall']:.4f}")
                print(f"     ⚖️  F1-Score: {metrics['f1-score']:.4f}")
                print(f"     📊 Support: {int(metrics['support'])}")
        
        # Confusion matrix analysis
        cm = self.results['confusion_matrix']
        print(f"\n🔍 CONFUSION MATRIX ANALYSIS:")
        
        if cm.size == 4:  # Binary classification
            tn, fp, fn, tp = cm.ravel()
            print(f"   ✅ True Negatives: {tn}")
            print(f"   ❌ False Positives: {fp}")
            print(f"   ❌ False Negatives: {fn}")
            print(f"   ✅ True Positives: {tp}")
            
            # Calculate error rates
            total = tn + fp + fn + tp
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            print(f"   📊 False Positive Rate: {fpr:.4f}")
            print(f"   📊 False Negative Rate: {fnr:.4f}")
            print(f"   📊 Overall Error Rate: {(fp + fn)/total:.4f}")
        
        # Top sentiment indicators (most important features)
        print(f"\n🔥 TOP 10 SENTIMENT INDICATORS:")
        if self.results['feature_importances'] is not None:
            importance_df = pd.DataFrame({
                'feature': self.results['feature_names'],
                'importance': self.results['feature_importances']
            }).sort_values('importance', ascending=False)
            
            for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
                feature_name = row['feature'][:25]  # Truncate long feature names
                print(f"   {i+1:2d}. {feature_name:25s}: {row['importance']:.6f}")
        
        print("=" * 50)
        return self.results
    
    def predict_sentiment(self, text_data):
        """
        Predict sentiment for new text data with confidence scores
        
        Args:
            text_data (str or list): Text to analyze
            
        Returns:
            list: Predictions with confidence scores
        """
        if self.model is None:
            raise ValueError("❌ Model not trained. Call train_model() first.")
            
        # Ensure input is list
        if isinstance(text_data, str):
            text_data = [text_data]
        
        print(f"🔍 Analyzing sentiment for {len(text_data)} text(s)...")
        
        # Vectorize new text
        X_text_new = self.tfidf_vectorizer.transform(text_data)
        
        # Add numerical features if model was trained with them
        if self.numerical_features:
            # Create dummy numerical features (zeros for new data)
            numerical_dummy = np.zeros((len(text_data), len(self.numerical_features)))
            numerical_scaled = self.scaler.transform(numerical_dummy)
            X_new = hstack([X_text_new, numerical_scaled])
        else:
            X_new = X_text_new
        
        # Get predictions and probabilities
        predictions = self.model.predict(X_new)
        probabilities = self.model.predict_proba(X_new)
        
        # Convert to sentiment labels
        sentiment_labels = self.label_encoder.inverse_transform(predictions)
        
        # Prepare results with confidence
        results = []
        for i, (pred, probs) in enumerate(zip(sentiment_labels, probabilities)):
            confidence = np.max(probs)  # Highest probability
            
            # Create probability dict for both classes
            prob_dict = {}
            for j, class_name in enumerate(self.label_encoder.classes_):
                prob_dict[class_name] = probs[j]
            
            results.append({
                'text_preview': text_data[i][:50] + '...' if len(text_data[i]) > 50 else text_data[i],
                'predicted_sentiment': pred,
                'confidence': confidence,
                'probabilities': prob_dict
            })
        
        return results
    
    def save_model(self, model_path='../output/models/random_forest_sentiment_model.pkl'):
        """
        Save the trained model and all components
        
        Args:
            model_path (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("❌ No model to save. Train model first.")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Package all model components
        model_package = {
            'model': self.model,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler if hasattr(self, 'scaler') else None,
            'numerical_features': self.numerical_features,
            'results': self.results,
            'model_type': 'RandomForestSentimentAnalyzer'
        }
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model_package, f)
            
        print(f"✅ Model saved successfully to: {model_path}")
        print(f"📊 Model accuracy: {self.results.get('test_accuracy', 'Unknown'):.4f}")
        
    def load_model(self, model_path='../output/models/random_forest_sentiment_model.pkl'):
        """
        Load a previously saved model
        
        Args:
            model_path (str): Path to the saved model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model file not found: {model_path}")
            
        # Load model package
        with open(model_path, 'rb') as f:
            model_package = pickle.load(f)
            
        # Restore all components
        self.model = model_package['model']
        self.tfidf_vectorizer = model_package['tfidf_vectorizer']
        self.label_encoder = model_package['label_encoder']
        self.scaler = model_package.get('scaler')
        self.numerical_features = model_package.get('numerical_features', [])
        self.results = model_package.get('results', {})
        
        print(f"✅ Model loaded successfully from: {model_path}")
        print(f"🤖 Model type: {model_package.get('model_type', 'Unknown')}")
        if self.results:
            print(f"📊 Previous accuracy: {self.results.get('test_accuracy', 'Unknown'):.4f}")
    
    def get_model_summary(self):
        """
        Get a summary of the current model state
        
        Returns:
            dict: Model summary information
        """
        if self.model is None:
            return {"status": "❌ Model not initialized"}
            
        summary = {
            "model_status": "✅ Ready" if self.results else "⚠️ Trained but not evaluated",
            "model_type": "Random Forest Classifier",
            "sentiment_classes": self.label_encoder.classes_.tolist() if hasattr(self.label_encoder, 'classes_') else [],
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
            "features_used": len(self.numerical_features) if self.numerical_features else 0,
        }
        
        # Add performance metrics if available
        if self.results:
            summary.update({
                "test_accuracy": f"{self.results.get('test_accuracy', 0):.4f}",
                "f1_score": f"{self.results.get('test_f1_weighted', 0):.4f}",
                "overfitting_gap": f"{self.results.get('accuracy_overfitting_gap', 0):.4f}",
                "training_time": f"{self.results.get('training_time', 0):.2f}s"
            })
            
        return summary
