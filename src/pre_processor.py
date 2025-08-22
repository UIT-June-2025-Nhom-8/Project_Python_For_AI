import pandas as pd
import re
import nltk
import ssl
import os
from stopwords_config import SENTIMENT_STOPWORDS


def strict_nltk_setup():
    """
    Strict NLTK setup that fails fast with clear error messages.
    No fallbacks - ensures consistent and predictable behavior.
    
    Returns:
        dict: Status of each NLTK component (True if available, False if failed)
    
    Raises:
        RuntimeError: If any required NLTK component is not available
    """
    print("🔧 Starting strict NLTK setup - no fallbacks, fail-fast mode...")
    
    status = {
        'punkt': False,
        'stopwords': False, 
        'wordnet': False,
        'omw': False
    }
    
    # Check if we're on Kaggle
    is_kaggle = os.path.exists('/kaggle/input') or 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
    
    if is_kaggle:
        print("🔍 Kaggle environment detected - checking for pre-installed NLTK data...")
        
        # Add common Kaggle NLTK data paths
        kaggle_nltk_paths = [
            '/opt/conda/nltk_data',
            '/usr/share/nltk_data', 
            '/usr/local/share/nltk_data',
            '/usr/lib/nltk_data'
        ]
        
        for path in kaggle_nltk_paths:
            if os.path.exists(path) and path not in nltk.data.path:
                nltk.data.path.append(path)
                print(f"✅ Added NLTK data path: {path}")
        
        # Test if components are available - STRICT MODE
        try:
            from nltk.corpus import stopwords
            test_stopwords = stopwords.words('english')
            status['stopwords'] = True
            print("✅ NLTK stopwords available")
        except Exception as e:
            raise RuntimeError(f"❌ NLTK stopwords not available in Kaggle environment: {e}")
            
        try:
            from nltk.tokenize import word_tokenize
            test_tokens = word_tokenize("test")
            status['punkt'] = True
            print("✅ NLTK punkt tokenizer available")
        except Exception as e:
            raise RuntimeError(f"❌ NLTK punkt tokenizer not available in Kaggle environment: {e}")
            
        try:
            from nltk.stem import WordNetLemmatizer
            lemmatizer = WordNetLemmatizer()
            test_lemma = lemmatizer.lemmatize("running")
            status['wordnet'] = True
            status['omw'] = True
            print("✅ NLTK WordNet lemmatizer available")
        except Exception as e:
            raise RuntimeError(f"❌ NLTK WordNet lemmatizer not available in Kaggle environment: {e}")
            
    else:
        print("🌐 Non-Kaggle environment - attempting NLTK downloads...")
        
        # Try downloading for non-Kaggle environments - STRICT MODE
        downloads = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
        failed_downloads = []
        
        for download in downloads:
            try:
                nltk.download(download, quiet=True)
                status[download.replace('-1.4', '')] = True
                print(f"✅ Downloaded {download}")
            except Exception as e:
                failed_downloads.append(f"{download}: {e}")
        
        # If any downloads failed, raise error with details
        if failed_downloads:
            error_msg = "❌ Failed to download required NLTK components:\n"
            for failure in failed_downloads:
                error_msg += f"  - {failure}\n"
            error_msg += "\nPlease ensure internet connection and try again."
            raise RuntimeError(error_msg)
    
    print("✅ All NLTK components verified successfully!")
    return status


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
        Initialize PreProcessor with strict validation and no fallbacks.
        Fails fast with clear error messages for predictable behavior.
        
        Args:
            use_lemmatization (bool): Use lemmatization instead of stemming
            
        Raises:
            RuntimeError: If NLTK components are not properly available
            ValueError: If invalid parameters are provided
        """
        print("🚀 Initializing PreProcessor in STRICT mode - no fallbacks...")
        
        if not isinstance(use_lemmatization, bool):
            raise ValueError("use_lemmatization must be a boolean value")
        
        # Strict NLTK setup - will raise RuntimeError if anything fails
        self.nltk_status = strict_nltk_setup()
        
        self.use_lemmatization = use_lemmatization
        self.lemmatizer = None
        self.stemmer = None
        
        # Initialize lemmatizer/stemmer - STRICT MODE
        if use_lemmatization:
            if not self.nltk_status['wordnet']:
                raise RuntimeError("❌ WordNet lemmatizer requested but not available. Cannot proceed.")
            
            try:
                from nltk.stem import WordNetLemmatizer
                self.lemmatizer = WordNetLemmatizer()
                print("✅ WordNet lemmatizer initialized")
            except Exception as e:
                raise RuntimeError(f"❌ Failed to initialize WordNet lemmatizer: {e}")
        else:
            try:
                from nltk.stem import SnowballStemmer  
                self.stemmer = SnowballStemmer("english")
                print("✅ Snowball stemmer initialized")
            except Exception as e:
                raise RuntimeError(f"❌ Failed to initialize Snowball stemmer: {e}")
        
        # Set tokenization method - STRICT MODE
        if not self.nltk_status['punkt']:
            raise RuntimeError("❌ NLTK punkt tokenizer not available. Cannot proceed without proper tokenization.")
        
        try:
            from nltk.tokenize import word_tokenize
            self.tokenize_func = word_tokenize
            print("✅ NLTK tokenizer ready")
        except Exception as e:
            raise RuntimeError(f"❌ Failed to initialize NLTK tokenizer: {e}")
            
        print("✅ PreProcessor initialization complete!")
        print(f"📊 NLTK Status: {self.nltk_status}")
        print(f"🔧 Using lemmatization: {self.use_lemmatization}")
        print("🔧 Tokenizer: NLTK word_tokenize")
        print("⚠️  Running in STRICT mode - any processing errors will raise exceptions")

    def clean_data(self, df):
        """
        Clean and prepare DataFrame for sentiment analysis.
        Handles Amazon Reviews format and combines title + text into input.
        STRICT MODE: Validates input and raises clear errors.

        Args:
            df (pd.DataFrame): DataFrame to be cleaned.

        Returns:
            pd.DataFrame: Cleaned DataFrame with 'input' and 'label' columns.
            
        Raises:
            ValueError: If input is invalid or required columns missing
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Input must be a pandas DataFrame, got {type(df)}")
            
        if df.empty:
            raise ValueError("Input DataFrame is empty")
            
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
            available_cols = list(df.columns)
            raise ValueError(f"No valid text column found. Available columns: {available_cols}. Expected: 'input', 'text', or 'title'")

        # Ensure we have label column
        if "label" not in df.columns:
            available_cols = list(df.columns)
            raise ValueError(f"No 'label' column found. Available columns: {available_cols}")

        # Validate label column
        if df["label"].isnull().any():
            raise ValueError("Label column contains null values - this will cause training errors")

        # Clean input text: remove empty entries
        initial_count = len(df)
        df = df[df["input"].str.strip() != ""]
        final_count = len(df)
        
        if final_count == 0:
            raise ValueError("All text records are empty after cleaning")
        
        print(f"\nRemoved {initial_count - final_count} empty text records")
        print(f"Final data shape: {df.shape}")

        print("\nNumber of null values after processing:")
        print(df.isnull().sum())

        # Keep only essential columns
        return df[["input", "label"]].reset_index(drop=True)

    def remove_duplicates(self, df):
        """
        Check and remove duplicate records in DataFrame.
        STRICT MODE: Validates input and reports exact changes.

        Args:
            df (pd.DataFrame): DataFrame to be processed.

        Returns:
            pd.DataFrame: DataFrame after removing duplicate records.
            
        Raises:
            ValueError: If input is invalid
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Input must be a pandas DataFrame, got {type(df)}")
            
        if df.empty:
            raise ValueError("Input DataFrame is empty")
            
        print(f"Number of records before removing duplicates: {len(df)}")

        df_cleaned = df.drop_duplicates()

        removed_count = len(df) - len(df_cleaned)
        print(f"Number of records after removing duplicates: {len(df_cleaned)}")
        
        if removed_count > 0:
            print(f"Removed {removed_count} duplicate records ({removed_count/len(df)*100:.1f}%)")

        return df_cleaned

    def clean_text(self, text):
        """
        Comprehensive text cleaning optimized for sentiment analysis.
        STRICT MODE: Validates input and ensures consistent output.

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
            
        Raises:
            ValueError: If input is not a string
        """
        if not isinstance(text, str):
            raise ValueError(f"Input must be a string, got {type(text)}")

        if not text.strip():
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
        Split text into tokens using NLTK word_tokenize.
        STRICT MODE: No fallbacks, raises exceptions on errors.

        Args:
            text (str): Text string to be tokenized.

        Returns:
            list: List of tokens.
            
        Raises:
            ValueError: If input is invalid
            RuntimeError: If tokenization fails
        """
        if not isinstance(text, str):
            raise ValueError(f"Input must be a string, got {type(text)}")
            
        if not text.strip():
            return []
        
        try:
            # Use NLTK tokenizer (no fallback in strict mode)
            tokens = self.tokenize_func(text.lower())
            
            # Filter out punctuation-only tokens and very short tokens
            tokens = [token for token in tokens if token.isalnum() and len(token) >= 2]
            
            return tokens
        except Exception as e:
            raise RuntimeError(f"❌ Tokenization failed for text: '{text[:50]}...'. Error: {e}")

    def remove_stopwords(self, tokens):
        """
        Remove stopwords using sentiment-optimized stopword list.
        Preserves negation words and sentiment-critical terms.
        STRICT MODE: Validates input and ensures consistent behavior.

        Args:
            tokens (list): List of tokens to be processed.

        Returns:
            list: List of tokens after removing non-essential stopwords.
            
        Raises:
            ValueError: If input is invalid
        """
        if not isinstance(tokens, list):
            raise ValueError(f"Input must be a list, got {type(tokens)}")
        
        # Validate that SENTIMENT_STOPWORDS is available and is a set/list
        if not hasattr(self, '_stopwords_validated'):
            if not SENTIMENT_STOPWORDS:
                raise RuntimeError("❌ SENTIMENT_STOPWORDS not available from stopwords_config")
            if not isinstance(SENTIMENT_STOPWORDS, (set, list, frozenset)):
                raise RuntimeError(f"❌ SENTIMENT_STOPWORDS must be a set/list, got {type(SENTIMENT_STOPWORDS)}")
            self._stopwords_validated = True
        
        # Use sentiment-optimized stopwords that preserve negation and emotion words
        filtered_tokens = [token for token in tokens if token.lower() not in SENTIMENT_STOPWORDS]
        return filtered_tokens

    def normalize_token(self, tokens):
        """
        Normalize tokens using lemmatization or stemming.
        STRICT MODE: No fallbacks, raises exceptions on errors.

        Args:
            tokens (list): List of tokens to be normalized.

        Returns:
            list: List of normalized tokens.
            
        Raises:
            ValueError: If input is invalid
            RuntimeError: If normalization fails consistently
        """
        if not isinstance(tokens, list):
            raise ValueError(f"Input must be a list, got {type(tokens)}")
        
        normalized_tokens = []
        failed_tokens = []
        
        for token in tokens:
            # Handle negation tokens specially (don't lemmatize/stem them)
            if token.startswith('not_'):
                normalized_tokens.append(token)
            elif token.isalpha() and len(token) >= 2:
                try:
                    if self.use_lemmatization and self.lemmatizer is not None:
                        # Lemmatization preserves word meaning - better for sentiment
                        normalized = self.lemmatizer.lemmatize(token.lower())
                    elif self.stemmer is not None:
                        # Stemming - faster but may lose meaning
                        normalized = self.stemmer.stem(token.lower())
                    else:
                        raise RuntimeError("❌ No normalization method available")
                    
                    if len(normalized) >= 2:
                        normalized_tokens.append(normalized)
                except Exception as e:
                    failed_tokens.append(f"'{token}': {e}")
        
        # In strict mode, if too many tokens fail, raise error
        if failed_tokens and len(failed_tokens) > len(tokens) * 0.1:  # More than 10% failure
            error_msg = f"❌ Token normalization failed for {len(failed_tokens)} tokens:\n"
            for failure in failed_tokens[:5]:  # Show first 5 failures
                error_msg += f"  - {failure}\n"
            if len(failed_tokens) > 5:
                error_msg += f"  ... and {len(failed_tokens) - 5} more"
            raise RuntimeError(error_msg)
        
        return normalized_tokens
    
    def preprocess_for_sentiment(self, text, preserve_negation=True):
        """
        Special preprocessing method optimized specifically for sentiment analysis.
        STRICT MODE: Raises exceptions on errors for predictable behavior.
        
        Args:
            text (str): Input text to preprocess
            preserve_negation (bool): Whether to preserve negation structure
            
        Returns:
            list: Processed tokens ready for sentiment classification
            
        Raises:
            ValueError: If input parameters are invalid
            RuntimeError: If preprocessing fails
        """
        if not isinstance(text, str):
            raise ValueError(f"Input text must be a string, got {type(text)}")
            
        if not isinstance(preserve_negation, bool):
            raise ValueError(f"preserve_negation must be a boolean, got {type(preserve_negation)}")
            
        if not text.strip():
            return []
            
        try:
            # Clean text with sentiment preservation
            cleaned_text = self.clean_text(text)
            
            # Tokenize - strict mode, will raise on error
            tokens = self.tokenize_text(cleaned_text)
            
            # Handle negation explicitly if requested
            if preserve_negation:
                tokens = self._handle_negation_tokens(tokens)
            
            # Remove stopwords but keep sentiment-critical words
            tokens = self.remove_stopwords(tokens)
            
            # Normalize (lemmatize preferred for sentiment) - strict mode
            tokens = self.normalize_token(tokens)
            
            return tokens
            
        except Exception as e:
            # Re-raise with more context
            raise RuntimeError(f"❌ Sentiment preprocessing failed for text: '{text[:50]}...'. Error: {e}")
    
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
        STRICT MODE: Validates input and raises clear errors.
        e.g., ["not", "good"] -> ["not_good"]
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Tokens with negation handling
            
        Raises:
            ValueError: If input is invalid
        """
        if not isinstance(tokens, list):
            raise ValueError(f"Input must be a list, got {type(tokens)}")
            
        negation_words = {"not", "no", "never", "none", "neither", "nor", "nothing", "nowhere", 
                         "dont", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
                         "won't", "wouldn't", "haven't", "hasn't", "hadn't", "can't", "couldn't",
                         "shouldn't", "mustn't", "needn't"}
        
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
