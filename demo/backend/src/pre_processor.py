import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from stopwords_config import SENTIMENT_STOPWORDS


class PreProcessor:
    # Optimized regex patterns for sentiment analysis
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

    # Enhanced meaningful short words for sentiment analysis
    MEANINGFUL_SHORT_WORDS = {
        "a", "i", "is", "it", "to", "go", "no", "so", "me", "we", "he", "my", "be", "or", "in", "on", "at",
        # Sentiment-critical short words
        "ok", "wow", "bad", "sad", "mad", "glad", "good", "nice", "poor", "rich", "big", "old", "new",
        "hot", "cold", "dry", "wet", "fun", "cool", "cute", "ugly", "slow", "fast", "easy", "hard"
    }

    def __init__(self, use_lemmatization=True):
        """
        Initialize PreProcessor with sentiment analysis optimizations
        
        Args:
            use_lemmatization (bool): Use lemmatization instead of stemming for better sentiment preservation
        """
        nltk.download("punkt")
        nltk.download("stopwords") 
        nltk.download("wordnet")
        nltk.download("omw-1.4")
        
        self.use_lemmatization = use_lemmatization
        if use_lemmatization:
            self.lemmatizer = WordNetLemmatizer()
        else:
            self.stemmer = SnowballStemmer("english")

    def clean_data(self, df):
        """
        Clean and prepare DataFrame for sentiment analysis.
        Handles Amazon Reviews format and combines title + text into input.

        Args:
            df (pd.DataFrame): DataFrame to be cleaned.

        Returns:
            pd.DataFrame: Cleaned DataFrame with 'input' and 'label' columns.
        """
        print("Number of null values before processing:")
        print(df.isnull().sum())
        print(f"Original data shape: {df.shape}")

        # Handle different data formats
        if "input" in df.columns:
            # Already has input column
            df["input"] = df["input"].fillna("").astype(str)
        elif "text" in df.columns and "title" in df.columns:
            # Amazon Reviews format: combine title and text
            df["title"] = df["title"].fillna("").astype(str)
            df["text"] = df["text"].fillna("").astype(str)
            # Combine title and text with space separator
            df["input"] = df["title"] + " " + df["text"]
        elif "text" in df.columns:
            # Only text column
            df["text"] = df["text"].fillna("").astype(str)
            df["input"] = df["text"]
        elif "title" in df.columns:
            # Only title column
            df["title"] = df["title"].fillna("").astype(str)
            df["input"] = df["title"]
        else:
            raise ValueError("No valid text column found (input, text, or title)")

        # Ensure we have label column
        if "label" not in df.columns:
            raise ValueError("No label column found")

        # Clean input text: remove empty entries
        initial_count = len(df)
        df = df[df["input"].str.strip() != ""]
        final_count = len(df)
        
        print(f"\nRemoved {initial_count - final_count} empty text records")
        print(f"Final data shape: {df.shape}")

        print("\nNumber of null values after processing:")
        print(df.isnull().sum())

        # Keep only essential columns
        return df[["input", "label"]].reset_index(drop=True)

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
        Comprehensive text cleaning optimized for sentiment analysis.

        Features:
        - URL/Email removal
        - HTML cleaning
        - Emoji and emoticon handling
        - Rating and price normalization
        - Sentiment-aware character filtering
        - Negation preservation
        - Emphasis preservation (repeated chars)

        Args:
            text (str): Text string to be cleaned.

        Returns:
            str: Cleaned text string optimized for sentiment analysis.
        """
        if not isinstance(text, str):
            return ""

        text = text.lower()

        # Apply cleaning patterns first (without negation handling here)
        for pattern, replacement in self.CLEANING_PATTERNS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        text = text.strip()

        # Filter words but preserve sentiment-relevant short words
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
        Complete text preprocessing pipeline optimized for sentiment analysis.

        This method combines all preprocessing steps efficiently for sentiment classification:
        - Text cleaning with sentiment preservation
        - Tokenization
        - Sentiment-aware stopword removal  
        - Lemmatization/stemming

        Args:
            text (str): Text string to be fully preprocessed.

        Returns:
            list: Final normalized tokens ready for sentiment classification.
        """
        if not isinstance(text, str):
            return []

        # Step 1: Clean text while preserving sentiment indicators
        cleaned_text = self.clean_text(text)

        # Step 2: Tokenize
        tokens = self.tokenize_text(cleaned_text)

        # Step 3: Remove stopwords (but keep sentiment-critical words)
        tokens_no_stopwords = self.remove_stopwords(tokens)

        # Step 4: Normalize tokens (lemmatization preferred for sentiment)
        normalized_tokens = self.normalize_token(tokens_no_stopwords)

        return normalized_tokens

    def tokenize_text(self, text):
        """
        Split text into tokens using NLTK word_tokenize with error handling.

        Args:
            text (str): Text string to be tokenized.

        Returns:
            list: List of tokens.
        """
        if not isinstance(text, str) or not text.strip():
            return []
        
        try:
            tokens = word_tokenize(text.lower())
            # Filter out punctuation-only tokens and very short tokens
            tokens = [token for token in tokens if token.isalnum() and len(token) >= 2]
            return tokens
        except Exception as e:
            print(f"Tokenization error for text: {text[:50]}... Error: {e}")
            # Fallback to simple split
            return [word for word in text.lower().split() if word.isalnum() and len(word) >= 2]

    def remove_stopwords(self, tokens):
        """
        Remove stopwords using sentiment-optimized stopword list.
        Preserves negation words and sentiment-critical terms.

        Args:
            tokens (list): List of tokens to be processed.

        Returns:
            list: List of tokens after removing non-essential stopwords.
        """
        if not isinstance(tokens, list):
            return []
        
        # Use sentiment-optimized stopwords that preserve negation and emotion words
        filtered_tokens = [token for token in tokens if token.lower() not in SENTIMENT_STOPWORDS]
        return filtered_tokens

    def normalize_token(self, tokens):
        """
        Normalize tokens using lemmatization (preferred) or stemming.
        Lemmatization preserves word meaning better for sentiment analysis.

        Args:
            tokens (list): List of tokens to be normalized.

        Returns:
            list: List of normalized tokens.
        """
        if not isinstance(tokens, list):
            return []
        
        normalized_tokens = []
        for token in tokens:
            # Handle negation tokens specially (don't lemmatize/stem them)
            if token.startswith('not_'):
                normalized_tokens.append(token)
            elif token.isalpha() and len(token) >= 2:
                if self.use_lemmatization:
                    # Lemmatization preserves word meaning - better for sentiment
                    normalized = self.lemmatizer.lemmatize(token.lower())
                else:
                    # Stemming - faster but may lose meaning
                    normalized = self.stemmer.stem(token.lower())
                
                if len(normalized) >= 2:
                    normalized_tokens.append(normalized)
        
        return normalized_tokens
    
    def preprocess_for_sentiment(self, text, preserve_negation=True):
        """
        Special preprocessing method optimized specifically for sentiment analysis.
        
        Args:
            text (str): Input text to preprocess
            preserve_negation (bool): Whether to preserve negation structure
            
        Returns:
            list: Processed tokens ready for sentiment classification
        """
        if not isinstance(text, str) or not text.strip():
            return []
            
        # Clean text with sentiment preservation
        cleaned_text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize_text(cleaned_text)
        
        # Handle negation explicitly if requested
        if preserve_negation:
            tokens = self._handle_negation_tokens(tokens)
        
        # Remove stopwords but keep sentiment-critical words
        tokens = self.remove_stopwords(tokens)
        
        # Normalize (lemmatize preferred for sentiment)
        tokens = self.normalize_token(tokens)
        
        return tokens
    
    def preprocess_dataframe(self, df, preserve_negation=True):
        """
        Complete preprocessing pipeline for DataFrame.
        Processes raw data from CSV to preprocessed format ready for modeling.
        
        Args:
            df (pd.DataFrame): Raw DataFrame (can have title, text columns)
            preserve_negation (bool): Whether to preserve negation in preprocessing
            
        Returns:
            pd.DataFrame: DataFrame with 'input', 'label', and 'normalized_input' columns
        """
        print("🔄 Starting DataFrame preprocessing for sentiment analysis...")
        
        # Step 1: Clean data and handle different formats
        cleaned_df = self.clean_data(df)
        
        # Step 2: Remove duplicates
        cleaned_df = self.remove_duplicates(cleaned_df)
        
        # Step 3: Apply sentiment-optimized preprocessing to each text
        print("🔄 Applying sentiment preprocessing to texts...")
        cleaned_df = cleaned_df.assign(
            normalized_input=cleaned_df["input"].apply(
                lambda x: self.preprocess_for_sentiment(x, preserve_negation=preserve_negation)
            )
        )
        
        # Step 4: Filter out empty results
        initial_count = len(cleaned_df)
        cleaned_df = cleaned_df[cleaned_df['normalized_input'].apply(len) > 0]
        final_count = len(cleaned_df)
        
        if initial_count != final_count:
            print(f"⚠️  Removed {initial_count - final_count} texts that resulted in empty tokens")
        
        print(f"✅ Preprocessing complete! Final dataset: {len(cleaned_df)} samples")
        print(f"📊 Average tokens per sample: {cleaned_df['normalized_input'].apply(len).mean():.2f}")
        
        return cleaned_df.reset_index(drop=True)
    
    def _handle_negation_tokens(self, tokens):
        """
        Handle negation by combining negation words with following tokens.
        e.g., ["not", "good"] -> ["not_good"]
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Tokens with negation handling
        """
        negation_words = {"not", "no", "never", "none", "neither", "nor", "nothing", "nowhere", 
                         "dont", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
                         "won't", "wouldn't", "haven't", "hasn't", "hadn't", "can't", "couldn't",
                         "shouldn't", "mustn't", "needn't"}
        
        if not isinstance(tokens, list):
            return []
            
        result = []
        i = 0
        
        while i < len(tokens):
            current_token = tokens[i].lower()
            
            # Check if current token is a negation word
            if current_token in negation_words and i + 1 < len(tokens):
                # Combine negation with next meaningful word
                next_token = tokens[i+1].lower()
                if next_token.isalpha() and len(next_token) >= 2:
                    combined = f"not_{next_token}"
                    result.append(combined)
                    i += 2  # Skip both current and next token
                else:
                    result.append(tokens[i])
                    i += 1
            else:
                result.append(tokens[i])
                i += 1
                
        return result
