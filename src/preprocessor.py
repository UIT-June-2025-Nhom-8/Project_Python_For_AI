import pandas as pd
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# For data exploration (Part 2)
import numpy as np

# Optional: For basic EDA without visualization (Part 5 logic only)
from collections import Counter


class PreProcessor:
    """
    Text preprocessing class for sentiment analysis pipeline.
    
    This class provides methods to clean data, preprocess text, and perform sentiment analysis
    on DataFrame objects passed as parameters (does not read files directly).
    """
    
    def __init__(self):
        # Download required NLTK data
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        
        # Initialize components for text preprocessing
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.analyzer = SentimentIntensityAnalyzer()

    def clean_data(self, df):
        """
        Clean DataFrame by handling null values and checking data types.
        Does not read from files - processes DataFrame passed as parameter.
        Based on notebook Part 3 - Data cleaning
        
        Args:
            df (pd.DataFrame): Input DataFrame to clean
            
        Returns:
            pd.DataFrame: Cleaned DataFrame with nulls handled
        """
        print("Checking for missing values in each column:")
        print(df.isnull().sum())
        
        # Remove columns that have any NaN value
        df = df.dropna(axis=1)
        
        print("Remaining columns after removing columns with NaN:")
        print(df.isnull().sum())
        
        # Handle text columns specifically
        if "text" in df.columns:
            df["text"] = df["text"].fillna("")
        if "title" in df.columns:
            df["title"] = df["title"].fillna("")
            
        return df

    def remove_duplicates(self, df):
        """
        Check and remove duplicate records from DataFrame.
        Processes DataFrame passed as parameter (does not read from files).
        Based on notebook Part 3 - Data cleaning
        
        Args:
            df (pd.DataFrame): Input DataFrame to process
            
        Returns:
            pd.DataFrame: DataFrame after removing duplicates
        """
        print(f"Number of records before removing duplicates: {len(df)}")
        
        # Remove duplicate records
        df_cleaned = df.drop_duplicates()
        
        print(f"Number of records after removing duplicates: {len(df_cleaned)}")
        
        return df_cleaned

    def clean_text(self, text):
        """
        Clean text by removing URLs, handles, punctuation and special characters.
        Based on notebook Part 4 - Text Preprocessing
        
        Args:
            text (str): Text string to clean
            
        Returns:
            str: Cleaned text string
        """
        if isinstance(text, str):
            # Convert text to lowercase
            text = text.lower()

            # Remove URLs (http, https, and www links)
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

            # Remove markdown-style links [text](link)
            text = re.sub(r'\[.*?\]\(.*?\)', '', text)

            # Remove handles (@username mentions)
            text = re.sub(r'@\w+', '', text)

            # Remove punctuation and special characters
            text = text.translate(str.maketrans('', '', string.punctuation))

            return text
        else:
            return text

    def tokenize_text(self, text):
        """
        Tokenize text into individual words.
        Based on notebook Part 4 - Text Preprocessing
        
        Args:
            text (str): Text string to tokenize
            
        Returns:
            list: List of tokens
        """
        if isinstance(text, str):
            return word_tokenize(text)
        else:
            return text

    def remove_stopwords(self, tokens):
        """
        Remove English stopwords from token list.
        Based on notebook Part 4 - Text Preprocessing
        
        Args:
            tokens (list): List of tokens to process
            
        Returns:
            list: List of tokens after removing stopwords
        """
        if isinstance(tokens, list):
            return [word for word in tokens if word not in self.stop_words]
        else:
            return tokens

    def stem_tokens(self, tokens):
        """
        Apply stemming to tokens using Porter Stemmer.
        Based on notebook Part 4 - Text Preprocessing
        
        Args:
            tokens (list): List of tokens to stem
            
        Returns:
            list: List of stemmed tokens
        """
        if isinstance(tokens, list):
            return [self.stemmer.stem(token) for token in tokens]
        else:
            return tokens
    
    def get_sentiment(self, text):
        """
        Apply VADER sentiment analysis to text.
        Based on notebook Part 4.1 - Sentiment Analysis with Vader
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Dictionary containing sentiment scores
        """
        return self.analyzer.polarity_scores(text)
    
    def classify_sentiment(self, compound_score):
        """
        Classify sentiment based on compound score.
        Based on notebook Part 4.1 - Sentiment Analysis with Vader
        
        Args:
            compound_score (float): Compound score from VADER
            
        Returns:
            str: Sentiment classification (Positive, Negative, or Neutral)
        """
        if compound_score >= 0.05:
            return 'Positive'
        elif compound_score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'
    
    def add_sentiment_analysis(self, df, text_column='text'):
        """
        Add sentiment analysis columns to DataFrame.
        Processes DataFrame passed as parameter (does not read from files).
        Based on notebook Part 4.1 - Sentiment Analysis with Vader
        
        Args:
            df (pd.DataFrame): Input DataFrame to process
            text_column (str): Name of text column to analyze
            
        Returns:
            pd.DataFrame: DataFrame with added sentiment columns
        """
        if text_column in df.columns:
            # Apply VADER sentiment analysis
            df['vader_scores'] = df[text_column].apply(self.get_sentiment)
            
            # Extract compound score
            df['compound'] = df['vader_scores'].apply(lambda score_dict: score_dict['compound'])
            
            # Classify sentiment
            df['sentiment'] = df['compound'].apply(self.classify_sentiment)
            
        return df
    
    def process_full_pipeline(self, df, text_column='text'):
        """
        Complete preprocessing pipeline combining all steps.
        Processes DataFrame passed as parameter (does not read from files).
        Based on notebook sections 1-5
        
        Args:
            df (pd.DataFrame): Input DataFrame to process
            text_column (str): Name of text column to process
            
        Returns:
            pd.DataFrame: Fully processed DataFrame
        """
        # Step 1: Clean data (Part 3)
        print("Step 1: Cleaning data...")
        df = self.clean_data(df)
        
        # Step 2: Remove duplicates (Part 3)
        print("Step 2: Removing duplicates...")
        df = self.remove_duplicates(df)
        
        # Step 3: Text preprocessing (Part 4)
        print("Step 3: Text preprocessing...")
        if text_column in df.columns:
            # Clean text
            df['cleaned_text'] = df[text_column].apply(self.clean_text)
            
            # Tokenize text
            df['tokenized_text'] = df['cleaned_text'].apply(self.tokenize_text)
            
            # Remove stopwords
            df['no_stopwords'] = df['tokenized_text'].apply(self.remove_stopwords)
            
            # Apply stemming
            df['stemmed_text'] = df['no_stopwords'].apply(self.stem_tokens)
            
        # Step 4: Sentiment analysis (Part 4.1)
        print("Step 4: Sentiment analysis...")
        df = self.add_sentiment_analysis(df, 'cleaned_text')
        
        print("✅ Preprocessing pipeline completed successfully!")
        
        return df
