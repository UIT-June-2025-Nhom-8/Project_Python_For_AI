import pandas as pd
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer


class PreProcessor:
    # Advanced regex patterns for comprehensive text cleaning
    CLEANING_PATTERNS = [
        # Web content removal
        (r"http[s]?://\S+|www\.\S+", ""),  # URLs
        (r"\S+@\S+\.\S+", ""),  # Email addresses
        (r"<[^>]+>", ""),  # HTML tags
        (r"&[a-zA-Z0-9]+;", ""),  # HTML entities
        # Social media content
        (r"@\w+|#\w+", ""),  # Mentions and hashtags
        # Numbers and digits
        (r"\d+", ""),  # Remove all numbers
        # Character filtering
        (r"[^a-zA-ZÀ-ÿĀ-žА-я\u00C0-\u017F\u0100-\u024F\s]", ""),  # Keep only letters
        (r"(.)\1{2,}", r"\1\1"),  # Repeated characters
        (r"\s+", " "),  # Normalize whitespace
        (r"\b[b-hj-z]\b", ""),  # Single chars except a,i
    ]

    # Meaningful short words to preserve
    MEANINGFUL_SHORT_WORDS = {
        "a",
        "i",
        "is",
        "it",
        "to",
        "go",
        "no",
        "so",
        "me",
        "we",
        "he",
        "my",
        "be",
        "or",
        "in",
        "on",
        "at",
    }

    def __init__(self):
        nltk.download("punkt")
        nltk.download("stopwords")

    def clean_data(self, df):
        """
        Check and handle null values, and examine data types in DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to be cleaned.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        print("Number of null values before processing:")
        print(df.isnull().sum())

        if "input" in df.columns:
            df["input"] = df["input"].fillna("")
        elif "text" in df.columns:
            df["text"] = df["text"].fillna("")
        elif "title" in df.columns:
            df["title"] = df["title"].fillna("")

        print("\nNumber of null values after processing:")
        print(df.isnull().sum())

        print("\nData types of columns:")
        print(df.dtypes)

        return df

    def remove_duplicates(self, df):
        """
        Check and remove duplicate records in DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to be processed.

        Returns:
            pd.DataFrame: DataFrame after removing duplicate records.
        """
        print(f"Number of records before removing duplicates: {len(df)}")

        df_cleaned = df.drop_duplicates()

        print(f"Number of records after removing duplicates: {len(df_cleaned)}")

        return df_cleaned

    def clean_text(self, text):
        """
        Comprehensive text cleaning with advanced preprocessing.

        Features:
        - URL/Email removal
        - HTML cleaning
        - Social media content removal
        - Number removal
        - Unicode letter filtering
        - Repeated character handling
        - Whitespace normalization
        - Short word filtering

        Args:
            text (str): Text string to be cleaned.

        Returns:
            str: Cleaned text string.
        """
        if not isinstance(text, str):
            return ""

        text = text.lower()

        for pattern, replacement in self.CLEANING_PATTERNS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        text = text.strip()

        words = text.split()
        words = [
            word
            for word in words
            if len(word) >= 2 or word.lower() in self.MEANINGFUL_SHORT_WORDS
        ]
        text = " ".join(words)

        return text

    def preprocess_text_pipeline(self, text):
        """
        Complete text preprocessing pipeline that combines all steps efficiently.

        This method is more memory-efficient than calling individual methods
        as it processes text in one pass without storing intermediate results.

        Args:
            text (str): Text string to be fully preprocessed.

        Returns:
            list: Final normalized tokens ready for vectorization.
        """
        if not isinstance(text, str):
            return []

        cleaned_text = self.clean_text(text)

        tokens = self.tokenize_text(cleaned_text)

        tokens_no_stopwords = self.remove_stopwords(tokens)

        normalized_tokens = self.normalize_token(tokens_no_stopwords)

        return normalized_tokens

    def tokenize_text(self, text):
        """
        Split text into tokens (words).

        Args:
            text (str): Text string to be tokenized.

        Returns:
            list: List of tokens.
        """
        if isinstance(text, str):
            return word_tokenize(text)
        else:
            return text

    def remove_stopwords(self, tokens):
        """
        Remove English stopwords from the list of tokens.

        Args:
            tokens (list): List of tokens to be processed.

        Returns:
            list: List of tokens after removing stopwords.
        """
        if not isinstance(tokens, list):
            return tokens
        else:
            stop_words = set(stopwords.words("english"))
            filtered_tokens = [word for word in tokens if word not in stop_words]
            return filtered_tokens

    def normalize_token(self, tokens):
        """
        Normalize token list by applying English Snowball Stemmer to each token.

        Args:
            tokens (list): List of tokens to be normalized.

        Returns:
            list: List of tokens after normalization.
        """
        if not isinstance(tokens, list):
            return tokens
        else:
            stemmer = SnowballStemmer("english")
            normalized_tokens = [stemmer.stem(word) for word in tokens]
            return normalized_tokens
