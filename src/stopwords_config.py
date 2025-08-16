"""
Stopwords configuration module for text processing

This module provides centralized stopwords management for both:
- PreProcessor (NLTK-based stopwords removal)
- TextAnalyzer (WordCloud stopwords filtering)
"""

import nltk
from nltk.corpus import stopwords
import ssl


def download_nltk_data():
    """Download required NLTK data safely"""
    try:
        nltk.download("stopwords", quiet=True)
    except Exception as e:
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
            nltk.download("stopwords", quiet=True)
        except Exception as ssl_error:
            print(f"Warning: Could not download NLTK stopwords: {ssl_error}")
            return False
    return True


download_nltk_data()


def get_nltk_stopwords():
    """
    Get NLTK English stopwords

    Returns:
        set: Set of English stopwords from NLTK
    """
    try:
        return set(stopwords.words("english"))
    except Exception as e:
        print(f"Warning: Could not load NLTK stopwords: {e}")
        return get_basic_stopwords()


def get_basic_stopwords():
    """
    Get basic English stopwords (fallback when NLTK is not available)

    Returns:
        set: Set of basic English stopwords
    """
    return {
        "the",
        "and",
        "a",
        "i",
        "to",
        "of",
        "is",
        "this",
        "it",
        "in",
        "that",
        "for",
        "you",
        "with",
        "on",
        "are",
        "as",
        "was",
        "have",
        "but",
        "not",
        "be",
        "or",
        "my",
        "so",
        "can",
        "will",
        "if",
        "from",
        "would",
        "has",
        "had",
        "do",
        "get",
        "an",
        "all",
        "at",
        "me",
        "just",
        "one",
        "out",
        "up",
        "very",
        "much",
        "more",
        "only",
        "also",
        "there",
        "well",
        "really",
        "than",
        "like",
        "about",
        "when",
        "what",
        "how",
        "some",
        "time",
        "even",
        "your",
        "no",
        "any",
        "way",
        "see",
        "could",
        "go",
        "were",
        "been",
        "did",
        "make",
        "know",
        "back",
        "now",
        "may",
        "too",
        "still",
        "she",
        "he",
        "we",
        "they",
        "them",
        "their",
        "said",
        "each",
        "which",
        "she",
        "do",
        "its",
        "had",
        "hi",
        "him",
        "has",
        "his",
        "her",
        "here",
        "hers",
        "herself",
        "himself",
        "how",
        "however",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "let",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "nor",
        "of",
        "on",
        "once",
        "only",
        "or",
        "other",
        "ought",
        "our",
        "ours",
        "ourselves",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    }


def get_extended_stopwords():
    """
    Get extended stopwords including NLTK + additional domain-specific words

    Returns:
        set: Extended set of stopwords for better filtering
    """
    nltk_stops = get_nltk_stopwords()
    additional_stops = {
        # Common contractions and informal words
        "i'm",
        "you're",
        "he's",
        "she's",
        "it's",
        "we're",
        "they're",
        "i've",
        "you've",
        "we've",
        "they've",
        "i'd",
        "you'd",
        "he'd",
        "she'd",
        "we'd",
        "they'd",
        "i'll",
        "you'll",
        "he'll",
        "she'll",
        "we'll",
        "they'll",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "haven't",
        "hasn't",
        "hadn't",
        "won't",
        "wouldn't",
        "don't",
        "doesn't",
        "didn't",
        "can't",
        "couldn't",
        "shouldn't",
        "mustn't",
        "needn't",
        "daren't",
        "mayn't",
        "oughtn't",
        # Common filler words
        "yeah",
        "yes",
        "no",
        "okay",
        "ok",
        "oh",
        "ah",
        "um",
        "hmm",
        "well",
        "like",
        "just",
        "really",
        "quite",
        "pretty",
        "sort",
        "kind",
        "thing",
        "stuff",
        "things",
        "something",
        "anything",
        "nothing",
        "everything",
        "somewhere",
        "anywhere",
        "nowhere",
        "everywhere",
        "someone",
        "anyone",
        "everyone",
        "nobody",
        # Single characters and numbers (cleaned by regex but just in case)
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
    }

    return nltk_stops.union(additional_stops)


NLTK_STOPWORDS = get_nltk_stopwords()
BASIC_STOPWORDS = get_basic_stopwords()
EXTENDED_STOPWORDS = get_extended_stopwords()

DEFAULT_PREPROCESSING_STOPWORDS = NLTK_STOPWORDS  # For PreProcessor
DEFAULT_WORDCLOUD_STOPWORDS = EXTENDED_STOPWORDS  # For WordCloud (more comprehensive)

__all__ = [
    "get_nltk_stopwords",
    "get_basic_stopwords",
    "get_extended_stopwords",
    "NLTK_STOPWORDS",
    "BASIC_STOPWORDS",
    "EXTENDED_STOPWORDS",
    "DEFAULT_PREPROCESSING_STOPWORDS",
    "DEFAULT_WORDCLOUD_STOPWORDS",
]
