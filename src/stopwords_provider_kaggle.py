"""
Kaggle Stopwords Configuration - Optimized for Sentiment Analysis

Purpose: Provide optimal stopwords for Amazon reviews sentiment analysis (positive/negative).
Approach: Single optimized stopwords set, simple and effective.
Includes automatic download of base stopwords if not available.
"""

import os
from typing import Set

OUTPUT_STOPWORDS_CUSTOM_FILE = '/kaggle/working/custom_sentiment_stopwords.txt'

class StopwordsProviderKaggle:
    """
    Optimized stopwords manager for Amazon reviews sentiment analysis.
    Provides one perfect stopwords set for positive/negative classification.
    Saves/loads from custom file for faster subsequent runs.
    """

    def __init__(self, output_custom_file=OUTPUT_STOPWORDS_CUSTOM_FILE):
        self._sentiment_stopwords: Set[str] = None
        self._is_initialized = False
        self.output_custom_file = output_custom_file

    def _download_base_stopwords(self) -> str:
        """
        Download base stopwords from alvations/nltk-corpora using kagglehub.
        Uses kagglehub's built-in caching - fast if already downloaded.
        Returns the path to the stopwords file.
        """
        print("🔍 Getting base stopwords from alvations/nltk-corpora...")
        
        try:
            import kagglehub
        except ImportError:
            raise RuntimeError(
                "❌ kagglehub not available. Install with: pip install kagglehub"
            )
        
        # Download the dataset (uses cache if already downloaded)
        dataset_path = kagglehub.dataset_download("alvations/nltk-corpora")
        
        # Find the stopwords file
        stopwords_file = f"{dataset_path}/corpora/stopwords/english"
        
        if not os.path.exists(stopwords_file):
            raise RuntimeError(f"❌ Stopwords file not found in dataset: {stopwords_file}")
        
        print(f"✅ Base stopwords available at: {stopwords_file}")
        return stopwords_file

    def _load_base_stopwords_from_file(self) -> Set[str]:
        """
        Load stopwords directly from kagglehub downloaded NLTK data.
        Automatically downloads if needed (or uses cache if available).
        """
        # Get base stopwords (kagglehub handles caching automatically)
        stopwords_file = self._download_base_stopwords()
        
        try:
            with open(stopwords_file, 'r', encoding='utf-8') as f:
                stopwords_list = [line.strip() for line in f if line.strip()]
            
            return set(stopwords_list)
            
        except Exception as e:
            raise RuntimeError(f"❌ Failed to read stopwords file: {e}")
    
    def _save_custom_stopwords(self, stopwords: Set[str]) -> None:
        """Save custom stopwords to file for faster loading next time."""
        try:
            with open(self.output_custom_file, 'w', encoding='utf-8') as f:
                for word in sorted(stopwords):  # Sort for consistent file
                    f.write(f"{word}\n")
            print(f"✅ Saved custom stopwords to: {self.output_custom_file}")
        except Exception as e:
            print(f"⚠️  Warning: Could not save custom stopwords: {e}")
    
    def _load_custom_stopwords(self) -> Set[str]:
        """Load custom stopwords from file if it exists."""
        if not os.path.exists(self.output_custom_file):
            return None
        
        try:
            with open(self.output_custom_file, 'r', encoding='utf-8') as f:
                stopwords_list = [line.strip() for line in f if line.strip()]
            print(f"✅ Loaded custom stopwords from: {self.output_custom_file}")
            return set(stopwords_list)
        except Exception as e:
            print(f"⚠️  Warning: Could not load custom stopwords: {e}")
            return None
    
    def _create_optimal_sentiment_stopwords(self) -> Set[str]:
        """
        Create optimal stopwords for Amazon reviews sentiment analysis.
        Removes common words but preserves all sentiment-critical words.
        """
        # Load base NLTK stopwords
        base_stopwords = self._load_base_stopwords_from_file()
        
        # Critical words to PRESERVE for sentiment analysis (remove from stopwords)
        sentiment_critical_words = set([
            # Negation words - ESSENTIAL for sentiment
            "not", "no", "never", "none", "neither", "nor", "nothing", "nowhere",
            "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't",
            "won't", "wouldn't", "don't", "doesn't", "didn't", "can't", "couldn't",
            "shouldn't", "mustn't", "needn't",
            
            # Contrast/adversative words - Important for sentiment shifts
            "but", "however", "although", "though", "yet", "still", "despite",
            
            # Intensity/degree words - Show sentiment strength
            "very", "really", "quite", "pretty", "too", "much", "more", "most", "less",
            "extremely", "incredibly", "absolutely", "totally", "completely",
            
            # Basic sentiment words - Core sentiment indicators
            "good", "bad", "best", "worst", "better", "worse", "great", "terrible",
            "awful", "amazing", "excellent", "poor", "fantastic", "horrible",
            "wonderful", "disappointing", "perfect", "useless",
            
            # Emotion words - Direct sentiment
            "love", "hate", "like", "dislike", "enjoy", "disappointed", "satisfied",
            "happy", "sad", "angry", "pleased", "frustrated", "impressed",
            
            # Quality/value words for products
            "cheap", "expensive", "worth", "waste", "recommend", "avoid",

            # Additional descriptive words
            "ok", "wow", "mad", "glad", "nice", "rich", "big", "old", "new",
            "hot", "cold", "dry", "wet", "fun", "cool", "cute", "ugly", "slow", "fast", "easy", "hard"
        ])
        
        # Remove sentiment-critical words from stopwords
        optimal_stopwords = base_stopwords - sentiment_critical_words
        
        # Add some extra common words that don't affect sentiment
        extra_stopwords = {
            "said", "say", "says", "told", "tell", "asked", "ask",  # Speech verbs
            "came", "come", "went", "go", "put", "take", "make", "get",  # Action verbs
            "one", "two", "first", "second", "last", "next",  # Numbers/order
            "many", "few", "several", "some", "any", "each", "every",  # Quantifiers
            "here", "there", "where", "when", "why", "how",  # Question words
            "may", "might", "could", "would", "should", "must",  # Modals (keep negations)
            "since", "until", "while", "during", "after", "before"  # Time prepositions
        }
        
        optimal_stopwords.update(extra_stopwords)
        
        return optimal_stopwords
    
    def initialize(self) -> None:
        """
        Initialize optimal stopwords for sentiment analysis.
        Automatically handles all necessary downloads and setup.
        """
        if self._is_initialized:
            return
        
        # Try to load from custom file first (much faster)
        custom_stopwords = self._load_custom_stopwords()
        
        if custom_stopwords is not None:
            # Use cached custom stopwords
            self._sentiment_stopwords = custom_stopwords
            print("🚀 Using cached custom stopwords (fast mode)")
        else:
            # Generate optimal stopwords and save for next time
            print("🔧 Generating optimal stopwords...")
            # This will automatically download base stopwords if needed
            self._sentiment_stopwords = self._create_optimal_sentiment_stopwords()
            
            # Save for next time
            self._save_custom_stopwords(self._sentiment_stopwords)
        
        self._is_initialized = True
    
    def get_stopwords(self) -> Set[str]:
        """
        Get the optimal stopwords for Amazon reviews sentiment analysis.
        
        Automatically handles all setup:
        1. Checks for cached custom stopwords (fastest)
        2. If not available, checks for base stopwords
        3. If base stopwords missing, downloads them automatically
        4. Generates and caches custom optimized stopwords
        
        Returns:
            Set[str]: Optimized stopwords set for sentiment analysis
        """
        if not self._is_initialized:
            self.initialize()
        return self._sentiment_stopwords.copy()