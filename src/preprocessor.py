import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
import numpy as np


class PreProcessor:
    """
    Preprocessor cải tiến với các kỹ thuật tối ưu hóa hiệu suất
    """
    
    def __init__(self):
        # Download required NLTK data
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)
        nltk.download("averaged_perceptron_tagger", quiet=True)
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        
        # Sử dụng Lemmatizer thay vì Stemmer cho kết quả tốt hơn
        self.lemmatizer = WordNetLemmatizer()
        
        # Custom stopwords - loại bỏ các từ có thể mang tính sentiment
        basic_stopwords = set(stopwords.words('english'))
        sentiment_words = {'not', 'no', 'never', 'nothing', 'neither', 'nowhere', 
                          'none', 'hardly', 'scarcely', 'barely', 'dont', 'doesnt',
                          'didnt', 'wasnt', 'werent', 'havent', 'hasnt', 'hadnt',
                          'wont', 'wouldnt', 'shouldnt', 'couldnt', 'cant', 'cannot',
                          'mustnt', 'mightnt', 'neednt'}
        self.stop_words = basic_stopwords - sentiment_words
        
        self.analyzer = SentimentIntensityAnalyzer()
        
        # Negation patterns để xử lý phủ định tốt hơn
        self.negation_patterns = [
            (r'\bnot\s+(\w+)', r'not_\1'),
            (r'\bno\s+(\w+)', r'no_\1'),
            (r'\bnever\s+(\w+)', r'never_\1'),
            (r'\bnothing\s+(\w+)', r'nothing_\1'),
            (r'\bdont\s+(\w+)', r'dont_\1'),
            (r'\bdoesnt\s+(\w+)', r'doesnt_\1'),
            (r'\bwont\s+(\w+)', r'wont_\1'),
            (r'\bcant\s+(\w+)', r'cant_\1'),
        ]
        
        # Các từ viết tắt thường gặp
        self.contractions = {
            "ain't": "am not",
            "aren't": "are not", 
            "can't": "cannot",
            "couldn't": "could not",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he would",
            "he'll": "he will",
            "he's": "he is",
            "i'd": "i would",
            "i'll": "i will",
            "i'm": "i am",
            "i've": "i have",
            "isn't": "is not",
            "it'd": "it would",
            "it'll": "it will",
            "it's": "it is",
            "let's": "let us",
            "shouldn't": "should not",
            "that's": "that is",
            "there's": "there is",
            "they'd": "they would",
            "they'll": "they will",
            "they're": "they are",
            "they've": "they have",
            "we'd": "we would",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what's": "what is",
            "where's": "where is",
            "who's": "who is",
            "won't": "will not",
            "wouldn't": "would not",
            "you'd": "you would",
            "you'll": "you will",
            "you're": "you are",
            "you've": "you have"
        }

    def expand_contractions(self, text):
        """
        Mở rộng các từ viết tắt
        """
        if not isinstance(text, str):
            return ""
            
        text = text.lower()
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)
        return text

    def handle_negations(self, text):
        """
        Xử lý phủ định bằng cách kết hợp với từ tiếp theo
        """
        if not isinstance(text, str):
            return ""
            
        for pattern, replacement in self.negation_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def clean_data(self, df):
        """
        Clean DataFrame - fail fast nếu structure không đúng
        """
        print("Checking for missing values in each column:")
        print(df.isnull().sum())
        
        # Remove columns that have any NaN value
        df = df.dropna(axis=1)
        
        print("Remaining columns after removing columns with NaN:")
        print(df.isnull().sum())
        
        # Handle text columns - fail fast
        df["text"] = df["text"].fillna("")
        df["title"] = df["title"].fillna("")
            
        return df

    def remove_duplicates(self, df, text_column):
        """
        Loại bỏ duplicates - fail fast nếu columns không tồn tại
        """
        print(f"Records before removing duplicates: {len(df)}")
        
        # Fail fast - không cần if-else
        if "sentiment" in df.columns:
            df_cleaned = df.drop_duplicates(subset=[text_column, "sentiment"])
        else:
            df_cleaned = df.drop_duplicates(subset=[text_column])
        
        print(f"Records after removing duplicates: {len(df_cleaned)}")
        print(f"Removed {len(df) - len(df_cleaned)} duplicate records")
        return df_cleaned

    def advanced_text_cleaning(self, text):
        """
        Làm sạch text với các kỹ thuật nâng cao
        """
        if not isinstance(text, str):
            return ""
            
        # Chuyển về lowercase
        text = text.lower()
        
        # Mở rộng contractions
        text = self.expand_contractions(text)
        
        # Xử lý HTML entities
        text = text.replace('&amp;', 'and')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&nbsp;', ' ')
        
        # Loại bỏ URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Loại bỏ markdown links
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)
        
        # Loại bỏ mentions và hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Loại bỏ số điện thoại và email
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Xử lý negations trước khi tokenize
        text = self.handle_negations(text)
        
        # Loại bỏ ký tự đặc biệt nhưng giữ lại dấu chấm câu quan trọng
        text = re.sub(r'[^\w\s!?.,;:]', '', text)
        
        # Chuẩn hóa whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text

    def smart_tokenize(self, text):
        """
        Tokenization thông minh - fail fast nếu input sai
        """
        # Tokenize
        tokens = word_tokenize(text)
        
        # Filter tokens
        tokens = [token for token in tokens 
                 if not all(c in string.punctuation for c in token)
                 and len(token) > 1 
                 and not token.isdigit()]
        
        return tokens

    def enhanced_remove_stopwords(self, tokens):
        """
        Loại bỏ stopwords - expect list input
        """
        return [token for token in tokens if token not in self.stop_words]

    def advanced_lemmatization(self, tokens):
        """
        Lemmatization với POS tagging - expect list input
        """
        pos_tags = nltk.pos_tag(tokens)
        
        lemmatized_tokens = []
        for word, pos in pos_tags:
            # Convert POS tag to WordNet format
            wordnet_pos = ('v' if pos.startswith('V') else
                          'n' if pos.startswith('N') else
                          'r' if pos.startswith('R') else
                          'a' if pos.startswith('J') else 'n')
                
            lemmatized_word = self.lemmatizer.lemmatize(word, pos=wordnet_pos)
            lemmatized_tokens.append(lemmatized_word)
            
        return lemmatized_tokens

    def extract_advanced_features(self, df, text_column='cleaned_text'):
        """
        Trích xuất features nâng cao - expect text_column tồn tại
        """
        print("Extracting advanced features...")
        
        # Fail fast - expect column to exist
        df['text_length'] = df[text_column].str.len()
        df['word_count'] = df[text_column].str.split().str.len()
        df['uppercase_count'] = df[text_column].str.count(r'[A-Z]')
        df['exclamation_count'] = df[text_column].str.count(r'!')
        df['question_count'] = df[text_column].str.count(r'\?')
        
        # Positive/negative word features
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
                         'perfect', 'love', 'best', 'awesome', 'brilliant', 'outstanding']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 
                         'disgusting', 'pathetic', 'useless', 'disappointing', 'boring']
        
        for word in positive_words:
            df[f'has_{word}'] = df[text_column].str.contains(word, case=False).astype(int)
        
        for word in negative_words:
            df[f'has_{word}'] = df[text_column].str.contains(word, case=False).astype(int)
        
        # Negation count
        negation_words = ['not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 
                         'neither', 'hardly', 'scarcely', 'barely']
        df['negation_count'] = df[text_column].apply(
            lambda x: sum(1 for word in negation_words if word in x.lower().split())
        )
        
        feature_count = len([col for col in df.columns if col.startswith('has_') or col.endswith('_count') or col.endswith('_length')])
        print(f"Added {feature_count} advanced features")
        
        return df

    def get_enhanced_sentiment(self, text):
        """
        Phân tích sentiment với VADER và các features bổ sung
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            return {
                'compound': 0.0,
                'pos': 0.0,
                'neu': 1.0,
                'neg': 0.0,
                'enhanced_compound': 0.0
            }
            
        # VADER sentiment
        vader_scores = self.analyzer.polarity_scores(text)
        
        # Điều chỉnh dựa trên các features bổ sung
        enhancement_factor = 1.0
        
        # Tăng cường nếu có nhiều dấu chấm than
        exclamation_count = text.count('!')
        if exclamation_count > 0:
            enhancement_factor += 0.1 * min(exclamation_count, 3)
        
        # Tăng cường nếu có từ viết hoa
        uppercase_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        if uppercase_ratio > 0.1:
            enhancement_factor += 0.2
        
        # Điều chỉnh compound score
        enhanced_compound = vader_scores['compound'] * enhancement_factor
        enhanced_compound = max(-1.0, min(1.0, enhanced_compound))  # Giới hạn trong [-1, 1]
        
        return {
            'compound': vader_scores['compound'],
            'pos': vader_scores['pos'],
            'neu': vader_scores['neu'],
            'neg': vader_scores['neg'],
            'enhanced_compound': enhanced_compound
        }

    def classify_enhanced_sentiment(self, enhanced_compound_score):
        """
        Phân loại sentiment với threshold được tối ưu
        """
        # Sử dụng threshold thấp hơn để phân loại chính xác hơn
        if enhanced_compound_score >= 0.1:
            return 'Positive'
        elif enhanced_compound_score <= -0.1:
            return 'Negative'
        else:
            return 'Neutral'

    def create_combined_text(self, df):
        """
        Tạo combined_text từ title và text columns - fail fast
        """
        print("Creating combined text from title and text columns...")
        
        # Fill NaN values - expect these columns to exist
        df['title'] = df['title'].fillna("")
        df['text'] = df['text'].fillna("")
        
        # Combine title and text
        df['combined_text'] = df['title'].astype(str) + " . " + df['text'].astype(str)
        
        # Clean empty cases
        df['combined_text'] = df['combined_text'].str.replace(r'^\s*\.\s*', '', regex=True)
        df['combined_text'] = df['combined_text'].str.replace(r'\s*\.\s*$', '', regex=True) 
        df['combined_text'] = df['combined_text'].str.strip()
        
        # Replace empty strings with text or title if available
        mask_empty = df['combined_text'].str.len() == 0
        df.loc[mask_empty, 'combined_text'] = df.loc[mask_empty, 'text'].fillna(df.loc[mask_empty, 'title']).fillna("")
        
        print(f"Combined text created. Average length: {df['combined_text'].str.len().mean():.1f} characters")
        
        return df

    def enhanced_preprocessing_pipeline(self, df, use_advanced_features=True):
        """
        Pipeline preprocessing đơn giản và trực tiếp - fail fast
        Expect columns: ['label', 'title', 'text', 'sentiment']
        """
        print("=== ENHANCED PREPROCESSING PIPELINE ===")
        print(f"Input DataFrame shape: {df.shape}")
        print(f"Input columns: {list(df.columns)}")
        
        # Step 0: Create combined text - expect title and text columns
        print("Step 0: Creating combined text...")
        df = self.create_combined_text(df)
        
        # Step 1: Clean data
        print("Step 1: Enhanced data cleaning...")
        df = self.clean_data(df)
        
        # Step 2: Remove duplicates using combined_text
        print("Step 2: Smart duplicate removal...")
        df = self.remove_duplicates(df, 'combined_text')
        
        # Step 3: Advanced text preprocessing
        print("Step 3: Advanced text preprocessing...")
        df['cleaned_text'] = df['combined_text'].apply(self.advanced_text_cleaning)
        
        # Step 4: Smart tokenization
        print("Step 4: Smart tokenization...")
        df['tokenized_text'] = df['cleaned_text'].apply(self.smart_tokenize)
        
        # Step 5: Enhanced stopword removal
        print("Step 5: Enhanced stopword removal...")
        df['no_stopwords'] = df['tokenized_text'].apply(self.enhanced_remove_stopwords)
        
        # Step 6: Advanced lemmatization
        print("Step 6: Advanced lemmatization...")
        df['lemmatized_text'] = df['no_stopwords'].apply(self.advanced_lemmatization)
        
        # Step 7: Reconstruct processed text
        df['processed_text'] = df['lemmatized_text'].apply(lambda x: ' '.join(x))
        
        # Step 8: Enhanced sentiment analysis
        print("Step 8: Enhanced sentiment analysis...")
        df['enhanced_sentiment_scores'] = df['cleaned_text'].apply(self.get_enhanced_sentiment)
        df['enhanced_compound'] = df['enhanced_sentiment_scores'].apply(lambda x: x['enhanced_compound'])
        df['enhanced_sentiment'] = df['enhanced_compound'].apply(self.classify_enhanced_sentiment)
        
        # Step 9: Extract advanced features
        if use_advanced_features:
            print("Step 9: Advanced feature extraction...")
            df = self.extract_advanced_features(df, 'cleaned_text')
        
        # Final statistics
        print("\n=== PREPROCESSING STATISTICS ===")
        print(f"Final dataset shape: {df.shape}")
        print(f"Text processing columns created: ['cleaned_text', 'processed_text', 'enhanced_sentiment']")
        
        print("Enhanced sentiment distribution:")
        print(df['enhanced_sentiment'].value_counts().to_dict())
        
        print("Original sentiment distribution:")  
        print(df['sentiment'].value_counts().to_dict())
        
        avg_words_original = df['combined_text'].str.split().str.len().mean()
        avg_words_processed = df['processed_text'].str.split().str.len().mean()
        print(f"Average words per text - Original: {avg_words_original:.1f}, Processed: {avg_words_processed:.1f}")
        
        # Data quality checks
        print("\n=== DATA QUALITY CHECKS ===")
        empty_processed = (df['processed_text'].str.len() == 0).sum()
        empty_cleaned = (df['cleaned_text'].str.len() == 0).sum()
        print(f"Empty processed texts: {empty_processed}/{len(df)} ({empty_processed/len(df)*100:.1f}%)")
        print(f"Empty cleaned texts: {empty_cleaned}/{len(df)} ({empty_cleaned/len(df)*100:.1f}%)")
        
        print("✅ Enhanced preprocessing completed successfully!")
        
        return df

    def get_feature_statistics(self, df):
        """
        Thống kê về các features đã tạo
        """
        stats = {}
        
        feature_columns = [col for col in df.columns if 
                          col.endswith('_count') or col.endswith('_length') or col.startswith('has_')]
        
        for col in feature_columns:
            if col in df.columns:
                stats[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max()
                }
        
        return stats
