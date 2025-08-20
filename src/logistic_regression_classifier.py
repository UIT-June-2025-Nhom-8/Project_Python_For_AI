import pandas as pd
import numpy as np
import json
import time
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
import pickle
import os


class LogisticRegressionAnalyzer:
    """
    Simplified LogisticRegression for sentiment analysis
    """

    def __init__(self):

        self.model_params = {
            "C": 0.8,
            "max_iter": 1500,
            "solver": "liblinear",
            "random_state": 42,
            "class_weight": "balanced",
            # Logistic Regression doesn't have model max_features parameter
        }

        self.tfidf_params = {
            "max_features": 5000,  # TF-IDF max_features: number of words in vocabulary
            "stop_words": None,
            "ngram_range": (1, 3),
            "min_df": 2,
            "max_df": 0.92,
            "sublinear_tf": True,
            "lowercase": True,
            "strip_accents": "unicode",
        }

        self.model = None
        self.tfidf_vectorizer = None
        self.label_encoder = LabelEncoder()
        self.results = {}
        self.models_dir = "output/models/"
        os.makedirs(self.models_dir, exist_ok=True)

    def update_tfidf_params(self, tfidf_config):
        if tfidf_config:
            # Update tất cả TF-IDF parameters
            self.tfidf_params.update(tfidf_config)

            # Convert ngram_range from list to tuple if needed
            if "ngram_range" in self.tfidf_params and isinstance(
                self.tfidf_params["ngram_range"], list
            ):
                self.tfidf_params["ngram_range"] = tuple(
                    self.tfidf_params["ngram_range"]
                )

            print(f"TF-IDF params updated: {self.tfidf_params}")

    def update_model_params(self, model_config):
        if model_config:
            # Update tất cả model parameters
            self.model_params.update(model_config)

            print(f"Model params updated: {self.model_params}")

    def _save_model(self, model_name_suffix=""):
        """Save trained model to file"""
        if self.model is None or self.tfidf_vectorizer is None:
            print("❌ No trained model to save")
            return

        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"logistic_regression_{timestamp}{model_name_suffix}.pkl"
            filepath = os.path.join(self.models_dir, filename)

            model_data = {
                "model": self.model,
                "tfidf_vectorizer": self.tfidf_vectorizer,
                "label_encoder": self.label_encoder,
                "results": self.results,
                "model_params": self.model_params,
                "tfidf_params": self.tfidf_params,
                "timestamp": timestamp,
            }

            with open(filepath, "wb") as f:
                pickle.dump(model_data, f)

            print(f"✅ Model saved to: {filepath}")

        except Exception as e:
            print(f"❌ Failed to save model: {e}")

    def train_and_evaluate(
        self, train_df, test_df, text_column="normalized_input", target_column="label"
    ):
        """
        Main method: train model

        Args:
            train_df (pd.DataFrame): Training data đã preprocessed
            test_df (pd.DataFrame): Test data đã preprocessed
            text_column (str): Tên cột text
            target_column (str): Tên cột target

        Returns:
            dict: Training/evaluation results
        """
        print("\n=== LOGISTIC REGRESSION CLASSIFIER ===")

        # Train new model
        print("Training new model...")
        start_time = time.time()

        # Prepare data
        self._prepare_data(train_df, test_df, text_column, target_column)

        # Initialize and train model
        self._initialize_model()
        self._train_model()

        training_time = time.time() - start_time
        self.results["training_time"] = round(training_time, 2)

        # Save model
        self._save_model()

        return self.results

    def _prepare_data(self, train_df, test_df, text_column, target_column):
        """Prepare data from preprocessed tokens or raw text"""

        # Convert tokens to text if using normalized_input
        if text_column == "normalized_input":
            # Convert list of tokens back to text for TF-IDF
            train_texts = train_df[text_column].apply(
                lambda x: " ".join(x) if isinstance(x, list) else str(x)
            )
            test_texts = test_df[text_column].apply(
                lambda x: " ".join(x) if isinstance(x, list) else str(x)
            )
            print(f"✅ Using preprocessed tokens from '{text_column}'")
        else:
            # Use raw text directly
            train_texts = train_df[text_column].fillna("")
            test_texts = test_df[text_column].fillna("")
            print(f"✅ Using raw text from '{text_column}'")

        # Create TF-IDF vectorizer with params configurable from JSON
        self.tfidf_vectorizer = TfidfVectorizer(**self.tfidf_params)

        # Fit vectorizer on training data and transform both train and test
        self.X_train = self.tfidf_vectorizer.fit_transform(train_texts)
        self.X_test = self.tfidf_vectorizer.transform(test_texts)

        # Encode labels
        self.y_train = self.label_encoder.fit_transform(train_df[target_column])
        self.y_test = self.label_encoder.transform(test_df[target_column])

        print(f"Data prepared: Train {self.X_train.shape}, Test {self.X_test.shape}")
        print(f"Label classes: {self.label_encoder.classes_}")

    def _initialize_model(self):
        """Initialize model with params"""
        model_params = {
            "C": self.model_params["C"],
            "max_iter": self.model_params["max_iter"],
            "solver": self.model_params["solver"],
            "random_state": self.model_params["random_state"],
            "class_weight": self.model_params.get(
                "class_weight", None
            ),  # Add class_weight if available
        }

        self.model = LogisticRegression(**model_params)

    def _train_model(self):
        """Train model and calculate metrics"""
        # Train
        self.model.fit(self.X_train, self.y_train)

        # Predict
        train_pred = self.model.predict(self.X_train)
        test_pred = self.model.predict(self.X_test)

        # Metrics
        train_acc = accuracy_score(self.y_train, train_pred)
        test_acc = accuracy_score(self.y_test, test_pred)
        f1_macro = f1_score(self.y_test, test_pred, average="macro")
        f1_weighted = f1_score(self.y_test, test_pred, average="weighted")

        # Classification report
        report = classification_report(
            self.y_test,
            test_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True,
        )

        # Save results
        self.results = {
            "model_name": "LogisticRegression",
            "model_params": self.model_params,
            "tfidf_params": self.tfidf_params,
            "train_accuracy": round(train_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "overfitting_score": round(
                train_acc - test_acc, 4
            ),  # Thêm overfitting score
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
            "classification_report": report,
            "confusion_matrix": confusion_matrix(self.y_test, test_pred).tolist(),
        }

        # Print results
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

        if "training_time" in self.results:
            print(f"Training Time: {self.results['training_time']:.2f} seconds")

        # Print per-class metrics
        report = self.results["classification_report"]
        print(f"\nPer-Class Performance:")
        for class_name in self.label_encoder.classes_:
            if str(class_name) in report:
                class_metrics = report[str(class_name)]
                print(
                    f"  {class_name}: Precision={class_metrics['precision']:.3f}, "
                    f"Recall={class_metrics['recall']:.3f}, F1={class_metrics['f1-score']:.3f}"
                )

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
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X_new)
            return predicted_labels, probabilities
        else:
            return predicted_labels

    def get_results(self):
        """Get complete training results"""
        if not self.results:
            raise ValueError("No results available. Train the model first.")

        return self.results.copy()
