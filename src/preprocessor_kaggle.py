"""
Kaggle Text Preprocessor for Sentiment Analysis - STRICT MODE

Purpose: Preprocess text data for sentiment analysis on Kaggle environment.
Approach: Single source of truth, fail-fast, no fallbacks, predictable behavior.
"""

import pandas as pd
import re
from typing import List, Set


class KagglePreProcessor:
    """
    Kaggle text preprocessor for sentiment analysis - STRICT MODE.
    Uses direct NLTK access with pre-initialized components.
    """
    
    # Advanced regex patterns for sentiment analysis
    CLEANING_PATTERNS = [
        # Web content removal
        (r"http[s]?://\S+|www\.\S+", ""),  # URLs
        (r"\S+@\S+\.\S+", ""),  # Email addresses
        (r"<[^>]+>", ""),  # HTML tags
        (r"&[a-zA-Z0-9]+;", ""),  # HTML entities
        
        # Social media content (preserve some sentiment indicators)
        (r"@\w+", ""),  # Mentions (but keep hashtags as they might indicate sentiment)
        
        # Emoji and special characters (convert some common ones)
        (r":\)|:-\)|:\(|:-\(", " emoticon "),  # Basic emoticons
        (r"[!]{2,}", "!"),  # Multiple exclamation marks -> single
        (r"[?]{2,}", "?"),  # Multiple question marks -> single
        
        # Numbers - keep ratings but remove other numbers
        (r"\b[1-5]\s*(?:star|stars|out of 5|/5)\b", " rating "),  # Convert ratings to tokens
        (r"\$\d+(?:\.\d{2})?", " price "),  # Convert prices to tokens
        (r"\b\d{1,2}(?:\.\d)?/10\b", " rating "),  # x/10 ratings
        (r"\b(?:\d+\s*(?:star|stars)?)\b", " rating "),  # Star ratings
        (r"\d+", ""),  # Remove other numbers
        
        # Character filtering (more permissive for sentiment words)
        (r"[^a-zA-ZÀ-ÿĀ-žА-я\u00C0-\u017F\u0100-\u024F\s!?]", ""),  # Keep letters and basic punctuation
        (r"(.)\1{2,}", r"\1\1"),  # Repeated characters (but allow double for emphasis like "sooo good")
        (r"\s+", " "),  # Normalize whitespace
    ]

    def __init__(self):
        """Initialize preprocessor - STRICT MODE."""
        self._sentiment_stopwords = None
        self._lemmatizer = None
        self._tokenizer = None  # Will store NLTK tokenizer
        self._is_initialized = False
    
    def initialize(self, sentiment_stopwords: Set[str]) -> None:
        """Initialize preprocessor components. Fail fast if any step fails."""
        if self._is_initialized:
            return

        self._sentiment_stopwords = sentiment_stopwords

        # Setup lemmatizer
        try:
            from nltk.stem import WordNetLemmatizer
            self._lemmatizer = WordNetLemmatizer()
        except Exception as e:
            raise RuntimeError(f"❌ Failed to initialize lemmatizer: {e}")
        
        # Setup tokenizer
        try:
            from nltk.tokenize import word_tokenize
            self._tokenizer = word_tokenize
        except Exception as e:
            raise RuntimeError(f"❌ Failed to initialize tokenizer: {e}")
        
        self._is_initialized = True
    
    def clean_text_basic(self, text: str) -> str:
        """
        Advanced text cleaning with comprehensive patterns.
        Enhanced version matching local PreProcessor quality.
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Apply advanced cleaning patterns (same as local PreProcessor)
        for pattern, replacement in self.CLEANING_PATTERNS:
            text = re.sub(pattern, replacement, text)
        
        # Handle contractions carefully (preserve negation)
        contractions = {
            "won't": "will not",
            "can't": "cannot", 
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Final cleanup
        text = text.strip()
        
        # Filter words but preserve sentiment-relevant short words (matching local PreProcessor)
        words = text.split()
        words = [
            word
            for word in words
            if len(word) >= 2 or word.lower() in self.MEANINGFUL_SHORT_WORDS
        ]
        text = " ".join(words)
        
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text using NLTK word_tokenize - STRICT MODE."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        if not text.strip():
            return []
        
        # STRICT MODE: Must have NLTK tokenizer initialized
        if not self._tokenizer:
            raise RuntimeError("NLTK tokenizer not initialized. Call initialize() first.")
        
        try:
            return self._tokenizer(text)
        except Exception as e:
            raise RuntimeError(f"Tokenization failed: {e}")
    
    def remove_stopwords(self, words: List[str]) -> List[str]:
        """Remove sentiment stopwords from word list."""
        if not self._is_initialized:
            self.initialize()
        
        return [word for word in words if word not in self._sentiment_stopwords]
    
    def lemmatize_words(self, words: List[str]) -> List[str]:
        """Apply lemmatization to words - STRICT MODE."""
        if not isinstance(words, list):
            raise ValueError("Input must be a list of strings")
            
        if not self._lemmatizer:
            raise RuntimeError("Lemmatizer not initialized")
        
        try:
            return [self._lemmatizer.lemmatize(word) for word in words]
        except Exception as e:
            raise RuntimeError(f"Lemmatization failed: {e}")
    
    def preprocess_for_sentiment(self, text: str) -> str:
        """
        Complete preprocessing pipeline optimized for sentiment analysis - STRICT MODE.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            str: Preprocessed text ready for sentiment analysis
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        # Ensure initialization
        if not self._is_initialized:
            self.initialize()
        
        # Advanced cleaning (uses comprehensive patterns)
        cleaned_text = self.clean_text_basic(text)
        
        if not cleaned_text:
            return ""
        
        # STRICT tokenization (NLTK required)
        words = self.tokenize_text(cleaned_text)
        
        # Remove stopwords (preserves negation)
        words = self.remove_stopwords(words)
        
        # Apply lemmatization (REQUIRED)
        words = self.lemmatize_words(words)
        
        # Join words back into text
        return " ".join(words)
        
    def clean_data(self, df: pd.DataFrame, text_column: str = 'input') -> pd.DataFrame:
        """Clean DataFrame by removing null values and empty texts."""
        df_clean = df.copy()
        df_clean = df_clean.dropna(subset=[text_column])
        df_clean = df_clean[df_clean[text_column].str.strip() != '']
        return df_clean
    
    def remove_duplicates(self, df: pd.DataFrame, text_column: str = 'input') -> pd.DataFrame:
        """Remove duplicate texts from DataFrame."""
        return df.drop_duplicates(subset=[text_column])
    
    def get_preprocessing_stats(self) -> dict:
        """Get preprocessing statistics and configuration."""
        return {
            'initialized': self._is_initialized,
            'lemmatization_enabled': True,
            'lemmatizer_available': self._lemmatizer is not None,
            'tokenizer_available': self._tokenizer is not None,
            'stopwords_count': len(self._sentiment_stopwords) if self._sentiment_stopwords else 0,
            'cleaning_patterns_count': len(self.CLEANING_PATTERNS),
            'meaningful_short_words_count': len(self.MEANINGFUL_SHORT_WORDS)
        }


def create_kaggle_preprocessor() -> KagglePreProcessor:
    """Create and initialize a Kaggle preprocessor."""
    preprocessor = KagglePreProcessor()
    preprocessor.initialize()
    return preprocessor
