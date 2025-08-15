import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class TFIDFVectorizer:
    def __init__(self, max_features=10000, min_df=2, max_df=0.8, ngram_range=(1, 2)):
        """
        Initialize TF-IDF Vectorizer.

        Args:
            max_features (int): Maximum number of features
            min_df (int): Minimum frequency of words in corpus
            max_df (float): Maximum frequency of words in corpus (ratio)
            ngram_range (tuple): N-gram range (1, 1) for unigram, (1, 2) for unigram + bigram
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            stop_words="english",
        )
        self.is_fitted = False

    def preprocess_tokens_to_text(self, tokens):
        """
        Convert token list to text string.

        Args:
            tokens (list): List of tokens

        Returns:
            str: Concatenated text string
        """
        if isinstance(tokens, list):
            return " ".join(tokens)
        else:
            return str(tokens)

    def fit(self, text_data):
        """
        Train TF-IDF vectorizer on text data.

        Args:
            text_data (pd.Series or list): Text data for training

        Returns:
            self: Return this object to enable method chaining
        """
        # Convert tokens to text if needed
        if isinstance(text_data, pd.Series):
            processed_text = text_data.apply(self.preprocess_tokens_to_text)
        else:
            processed_text = [
                self.preprocess_tokens_to_text(text) for text in text_data
            ]

        print("Training TF-IDF Vectorizer...")
        self.vectorizer.fit(processed_text)
        self.is_fitted = True
        print(
            f"Completed! Number of features: {len(self.vectorizer.get_feature_names_out())}"
        )
        return self

    def transform(self, text_data):
        """
        Transform text data into TF-IDF matrix.

        Args:
            text_data (pd.Series or list): Text data to be transformed

        Returns:
            scipy.sparse.matrix: Sparse TF-IDF matrix
        """
        if not self.is_fitted:
            raise ValueError(
                "Vectorizer has not been trained. Please call fit() method first."
            )

        # Convert tokens to text if needed
        if isinstance(text_data, pd.Series):
            processed_text = text_data.apply(self.preprocess_tokens_to_text)
        else:
            processed_text = [
                self.preprocess_tokens_to_text(text) for text in text_data
            ]

        print("Vectorizing data...")
        tfidf_matrix = self.vectorizer.transform(processed_text)
        print(f"Completed! Matrix shape: {tfidf_matrix.shape}")
        return tfidf_matrix

    def fit_transform(self, text_data):
        """
        Train and transform data in one step.

        Args:
            text_data (pd.Series or list): Text data

        Returns:
            scipy.sparse.matrix: Sparse TF-IDF matrix
        """
        return self.fit(text_data).transform(text_data)

    def get_feature_names(self):
        """
        Get list of feature names.

        Returns:
            list: List of feature names
        """
        if not self.is_fitted:
            raise ValueError(
                "Vectorizer has not been trained. Please call fit() method first."
            )

        return self.vectorizer.get_feature_names_out().tolist()

    def get_top_features(self, tfidf_matrix, top_n=20):
        """
        Get top N features with highest TF-IDF scores.

        Args:
            tfidf_matrix (scipy.sparse.matrix): TF-IDF matrix
            top_n (int): Number of top features to retrieve

        Returns:
            list: List of tuples (feature_name, avg_tfidf_score)
        """
        if not self.is_fitted:
            raise ValueError(
                "Vectorizer has not been trained. Please call fit() method first."
            )

        # Calculate average TF-IDF score for each feature
        mean_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
        feature_names = self.get_feature_names()

        # Create list of (feature, score) and sort
        feature_scores = list(zip(feature_names, mean_scores))
        feature_scores.sort(key=lambda x: x[1], reverse=True)

        return feature_scores[:top_n]

    def save_vectorizer(self, filepath):
        """
        Save the trained vectorizer.

        Args:
            filepath (str): File path to save
        """
        import joblib

        if not self.is_fitted:
            raise ValueError(
                "Vectorizer has not been trained. Please call fit() method first."
            )

        joblib.dump(self.vectorizer, filepath)
        print(f"Vectorizer saved to: {filepath}")

    def load_vectorizer(self, filepath):
        """
        Load the trained vectorizer.

        Args:
            filepath (str): File path to load from
        """
        import joblib

        self.vectorizer = joblib.load(filepath)
        self.is_fitted = True
        print(f"Vectorizer loaded from: {filepath}")
        print(f"Number of features: {len(self.vectorizer.get_feature_names_out())}")
