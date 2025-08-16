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
        if isinstance(text, str):
            text = text.lower()
            for contraction, expansion in self.contractions.items():
                text = text.replace(contraction, expansion)
        return text

    def handle_negations(self, text):
        """
        Xử lý phủ định bằng cách kết hợp với từ tiếp theo
        """
        if isinstance(text, str):
            for pattern, replacement in self.negation_patterns:
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def clean_data(self, df):
        """
        Clean DataFrame với các bước tối ưu hóa
        """
        print("Cleaning data with enhanced methods...")
        
        # Kiểm tra missing values
        missing_before = df.isnull().sum().sum()
        print(f"Missing values before cleaning: {missing_before}")
        
        # Xử lý missing values thông minh hơn
        if "text" in df.columns:
            df["text"] = df["text"].fillna("")
        if "title" in df.columns:
            df["title"] = df["title"].fillna("")
            
        # Kết hợp title và text nếu có cả hai
        if "title" in df.columns and "text" in df.columns:
            df["combined_text"] = df["title"].astype(str) + " " + df["text"].astype(str)
        elif "text" in df.columns:
            df["combined_text"] = df["text"].astype(str)
        else:
            df["combined_text"] = ""
            
        # Loại bỏ các dòng có text quá ngắn (có thể là noise)
        df = df[df["combined_text"].str.len() > 10].reset_index(drop=True)
        
        missing_after = df.isnull().sum().sum()
        print(f"Missing values after cleaning: {missing_after}")
        print(f"Rows removed due to short text: {len(df)}")
        
        return df

    def remove_duplicates(self, df):
        """
        Loại bỏ duplicates với logic tối ưu
        """
        print(f"Records before removing duplicates: {len(df)}")
        
        # Loại bỏ duplicates dựa trên cả text và sentiment để tránh mất thông tin
        if "combined_text" in df.columns and "sentiment" in df.columns:
            df_cleaned = df.drop_duplicates(subset=["combined_text", "sentiment"])
        else:
            df_cleaned = df.drop_duplicates()
        
        print(f"Records after removing duplicates: {len(df_cleaned)}")
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
        Tokenization thông minh với xử lý các trường hợp đặc biệt
        """
        if not isinstance(text, str):
            return []
            
        # Tokenize
        tokens = word_tokenize(text)
        
        # Loại bỏ tokens chỉ chứa punctuation
        tokens = [token for token in tokens if not all(c in string.punctuation for c in token)]
        
        # Loại bỏ tokens quá ngắn (có thể là noise)
        tokens = [token for token in tokens if len(token) > 1]
        
        # Loại bỏ tokens chỉ chứa số
        tokens = [token for token in tokens if not token.isdigit()]
        
        return tokens

    def enhanced_remove_stopwords(self, tokens):
        """
        Loại bỏ stopwords với logic cải tiến
        """
        if not isinstance(tokens, list):
            return []
            
        # Giữ lại một số stopwords có thể quan trọng cho sentiment
        filtered_tokens = []
        for token in tokens:
            if token not in self.stop_words:
                filtered_tokens.append(token)
                
        return filtered_tokens

    def advanced_lemmatization(self, tokens):
        """
        Lemmatization với POS tagging để có kết quả chính xác hơn
        """
        if not isinstance(tokens, list):
            return []
            
        # POS tagging để lemmatize chính xác hơn
        pos_tags = nltk.pos_tag(tokens)
        
        lemmatized_tokens = []
        for word, pos in pos_tags:
            # Chuyển đổi POS tag sang định dạng WordNet
            if pos.startswith('V'):
                wordnet_pos = 'v'  # verb
            elif pos.startswith('N'):
                wordnet_pos = 'n'  # noun
            elif pos.startswith('R'):
                wordnet_pos = 'r'  # adverb
            elif pos.startswith('J'):
                wordnet_pos = 'a'  # adjective
            else:
                wordnet_pos = 'n'  # default noun
                
            lemmatized_word = self.lemmatizer.lemmatize(word, pos=wordnet_pos)
            lemmatized_tokens.append(lemmatized_word)
            
        return lemmatized_tokens

    def extract_advanced_features(self, df, text_column='combined_text'):
        """
        Trích xuất các features nâng cao cho sentiment analysis
        """
        print("Extracting advanced features...")
        
        if text_column not in df.columns:
            return df
            
        # 1. Độ dài text
        df['text_length'] = df[text_column].str.len()
        df['word_count'] = df[text_column].str.split().str.len()
        
        # 2. Số lượng từ viết hoa (có thể biểu hiện cảm xúc mạnh)
        df['uppercase_count'] = df[text_column].str.count(r'[A-Z]')
        
        # 3. Số lượng dấu chấm than và dấu hỏi
        df['exclamation_count'] = df[text_column].str.count(r'!')
        df['question_count'] = df[text_column].str.count(r'\?')
        
        # 4. Tỷ lệ từ tích cực/tiêu cực
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
                         'perfect', 'love', 'best', 'awesome', 'brilliant', 'outstanding']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 
                         'disgusting', 'pathetic', 'useless', 'disappointing', 'boring']
        
        for word in positive_words:
            df[f'has_{word}'] = df[text_column].str.contains(word, case=False).astype(int)
        
        for word in negative_words:
            df[f'has_{word}'] = df[text_column].str.contains(word, case=False).astype(int)
        
        # 5. Số lượng từ phủ định
        negation_words = ['not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 
                         'neither', 'hardly', 'scarcely', 'barely']
        df['negation_count'] = df[text_column].apply(
            lambda x: sum(1 for word in negation_words if word in x.lower().split())
        )
        
        print(f"Extracted {df.shape[1] - len([col for col in df.columns if col.startswith('has_') or col.endswith('_count') or col.endswith('_length')])} base features")
        print(f"Added {len([col for col in df.columns if col.startswith('has_') or col.endswith('_count') or col.endswith('_length')])} advanced features")
        
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

    def enhanced_preprocessing_pipeline(self, df, text_column='text', use_advanced_features=True):
        """
        Pipeline preprocessing hoàn chỉnh với các cải tiến
        """
        print("=== ENHANCED PREPROCESSING PIPELINE ===")
        
        # Step 1: Clean data
        print("Step 1: Enhanced data cleaning...")
        df = self.clean_data(df)
        
        # Step 2: Remove duplicates
        print("Step 2: Smart duplicate removal...")
        df = self.remove_duplicates(df)
        
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
        df['processed_text'] = df['lemmatized_text'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
        
        # Step 8: Enhanced sentiment analysis
        print("Step 7: Enhanced sentiment analysis...")
        df['enhanced_sentiment_scores'] = df['cleaned_text'].apply(self.get_enhanced_sentiment)
        df['enhanced_compound'] = df['enhanced_sentiment_scores'].apply(lambda x: x['enhanced_compound'])
        df['enhanced_sentiment'] = df['enhanced_compound'].apply(self.classify_enhanced_sentiment)
        
        # Step 9: Extract advanced features (optional)
        if use_advanced_features:
            print("Step 8: Advanced feature extraction...")
            df = self.extract_advanced_features(df, 'cleaned_text')
        
        # Thống kê cuối
        print("\n=== PREPROCESSING STATISTICS ===")
        print(f"Final dataset size: {len(df)} samples")
        if 'enhanced_sentiment' in df.columns:
            print("Sentiment distribution:")
            print(df['enhanced_sentiment'].value_counts().to_dict())
        
        avg_words = df['processed_text'].str.split().str.len().mean()
        print(f"Average words per text after preprocessing: {avg_words:.1f}")
        
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
