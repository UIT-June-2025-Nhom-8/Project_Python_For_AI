# gensim_lda.py
"""
Gensim LDA Implementation for Amazon Reviews
Sử dụng PreProcessor methods (trừ normalize_token)
Thay thế bằng Lemmatization cho LDA
"""

import pandas as pd
import numpy as np
from gensim import corpora, models
from gensim.models import LdaModel, CoherenceModel, Phrases
import nltk
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import warnings
warnings.filterwarnings('ignore')
from wordcloud import WordCloud
import os

# Import PreProcessor từ file của bạn
from pre_processor import PreProcessor

# Download required NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


class GensimLDA:
    """
    LDA Topic Modeling using Gensim
    Sử dụng PreProcessor cho cleaning nhưng thay stemming bằng lemmatization
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.preprocessor = PreProcessor()  # Sử dụng PreProcessor có sẵn
        self.lemmatizer = WordNetLemmatizer()
        self.dictionary = None
        self.corpus = None
        self.lda_model = None
        self.texts = None
        self.bigram = None
        self.trigram = None
        
    def preprocess_for_lda(self, text):
        """
        Preprocessing pipeline cho LDA sử dụng PreProcessor methods
        
        Args:
            text (str): Raw text input
            
        Returns:
            list: Processed tokens with lemmatization
        """
        # Step 1: Clean text using PreProcessor
        cleaned_text = self.preprocessor.clean_text(text)
        
        # Step 2: Tokenize using PreProcessor
        tokens = self.preprocessor.tokenize_text(cleaned_text)
        
        # Step 3: Remove stopwords using PreProcessor
        tokens_no_stop = self.preprocessor.remove_stopwords(tokens)
        
        # Step 4: Lemmatization (thay vì dùng normalize_token với stemming)
        lemmatized_tokens = [self.lemmatizer.lemmatize(token.lower()) for token in tokens_no_stop]
        
        # Step 5: Filter tokens (chỉ giữ từ có độ dài > 2)
        final_tokens = [token for token in lemmatized_tokens if len(token) > 2]
        
        return final_tokens
    
    def prepare_texts_for_lda(self, df, text_column='lda_input', sample_size=None):
        """
        Prepare texts from dataframe cho LDA
        
        Args:
            df: DataFrame với text data
            text_column: Tên column chứa text
            sample_size: Số lượng samples (None = dùng tất cả)
            
        Returns:
            list: Preprocessed texts
        """
        # Sample nếu cần
        if sample_size and sample_size < len(df):
            print(f"Sampling {sample_size} documents for LDA...")
            df_sample = df.sample(n=sample_size, random_state=self.random_state)
        else:
            df_sample = df
            
        print(f"Preprocessing {len(df_sample)} documents for LDA...")
        
        # Lấy trực tiếp các tokens đã được tiền xử lý
        processed_texts = []
        for doc_tokens in df_sample[text_column]: # <-- Duyệt qua các list tokens
            if doc_tokens:  # Chỉ giữ documents có tokens (sau tiền xử lý ban đầu)
                processed_texts.append(doc_tokens)
        
        print(f"Documents after preprocessing: {len(processed_texts)}")
        print(f"Average tokens per document: {np.mean([len(doc) for doc in processed_texts]):.2f}")
        
        return processed_texts
    
    def create_bigrams_trigrams(self, texts):
        """
        Detect và create bigrams/trigrams
        
        Args:
            texts: List of tokenized texts
            
        Returns:
            Texts với bigrams và trigrams
        """
        print("\nDetecting bigrams and trigrams...")
        
        # Build bigram và trigram models
        bigram = Phrases(texts, min_count=5, threshold=50)
        trigram = Phrases(bigram[texts], threshold=50)
        
        self.bigram = bigram
        self.trigram = trigram
        
        # Apply to texts
        texts_bigrams = [bigram[doc] for doc in texts]
        texts_trigrams = [trigram[bigram[doc]] for doc in texts]
        
        # Print sample bigrams
        bigram_phrases = list(bigram.export_phrases())[:10]
        if bigram_phrases:
            print(f"Sample bigrams found: {bigram_phrases[:5]}")
        
        return texts_trigrams
    
    def prepare_corpus(self, texts):
        """
        Create dictionary và corpus cho LDA
        
        Args:
            texts: Preprocessed texts với bigrams/trigrams
            
        Returns:
            tuple: (dictionary, corpus)
        """
        print("\nCreating dictionary and corpus...")
        
        # Store texts for coherence calculation
        self.texts = texts
        
        # Create dictionary
        self.dictionary = corpora.Dictionary(texts)
        print(f"Dictionary size before filtering: {len(self.dictionary)}")
        
        # Filter extremes
        self.dictionary.filter_extremes(
            no_below=10,      # Min document frequency
            no_above=0.5,    # Max document proportion (80%)
            keep_n=10000      # Keep top 10000 words
        )
        print(f"Dictionary size after filtering: {len(self.dictionary)}")
        
        # Create corpus
        self.corpus = [self.dictionary.doc2bow(text) for text in texts]
        print(f"Corpus size: {len(self.corpus)} documents")
        
        return self.dictionary, self.corpus
    
    def train_lda(self, num_topics=16, passes=10, iterations=100):
        """
        Train LDA model
        
        Args:
            num_topics: Number of topics (default=16)
            passes: Number of passes through corpus
            iterations: Max iterations per document
            
        Returns:
            LDA model
        """
        print(f"\nTraining LDA with {num_topics} topics...")
        
        self.lda_model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=num_topics,
            random_state=self.random_state,
            passes=passes,
            iterations=iterations,
            alpha='auto',
            eta='auto',
            per_word_topics=True
        )
        
        print("LDA training completed!")
        return self.lda_model
    
    def calculate_metrics(self):
        """
        Calculate Coherence Score và Perplexity
        
        Returns:
            tuple: (coherence_score, perplexity)
        """
        print("\nCalculating metrics...")
        
        # Coherence Score
        coherence_model = CoherenceModel(
            model=self.lda_model,
            texts=self.texts,
            dictionary=self.dictionary,
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        
        # Perplexity
        perplexity = self.lda_model.log_perplexity(self.corpus)
        
        print(f"Coherence Score: {coherence_score:.4f}")
        print(f"Perplexity: {perplexity:.4f}")
        
        return coherence_score, perplexity
    
    def find_optimal_topics(self, min_topics=10, max_topics=20):
        """
        Find optimal number of topics
        
        Args:
            min_topics: Min topics to test
            max_topics: Max topics to test
            
        Returns:
            Optimal number of topics
        """
        print(f"\n=== FINDING OPTIMAL TOPICS ({min_topics}-{max_topics}) ===")
        
        coherence_scores = []
        perplexity_scores = []
        models = []
        
        for num_topics in range(min_topics, max_topics + 1):
            print(f"\nTesting {num_topics} topics...")
            
            # Train model
            model = LdaModel(
                corpus=self.corpus,
                id2word=self.dictionary,
                num_topics=num_topics,
                random_state=self.random_state,
                passes=10,
                iterations=100,
                alpha='auto',
                eta='auto'
            )
            models.append(model)
            
            # Calculate coherence
            coherence_model = CoherenceModel(
                model=model,
                texts=self.texts,
                dictionary=self.dictionary,
                coherence='c_v'
            )
            coherence = coherence_model.get_coherence()
            coherence_scores.append(coherence)
            
            # Calculate perplexity
            perplexity = model.log_perplexity(self.corpus)
            perplexity_scores.append(perplexity)
            
            print(f"  Coherence: {coherence:.4f}, Perplexity: {perplexity:.4f}")
        
        # Find optimal
        optimal_idx = np.argmax(coherence_scores)
        optimal_topics = min_topics + optimal_idx
        
        print("\n| Number of Topics | Coherence Score | Perplexity Score |")
        print("|------------------|-----------------|------------------|")
        for i, num_topics in enumerate(range(min_topics, max_topics + 1)):
            print(f"| {num_topics:<16} | {coherence_scores[i]:<15.4f} | {perplexity_scores[i]:<16.4f} |")

        print(f"\n=== RESULTS ===")
        print(f"Optimal topics: {optimal_topics}")
        print(f"Best coherence: {coherence_scores[optimal_idx]:.4f}")
        
        # Plot results
        self.plot_metrics(range(min_topics, max_topics + 1), 
                         coherence_scores, perplexity_scores)
        
        # Set best model
        self.lda_model = models[optimal_idx]
        
        return optimal_topics, coherence_scores[optimal_idx], perplexity_scores[optimal_idx]
    
    def plot_metrics(self, topic_range, coherence_scores, perplexity_scores):
        """
        Plot coherence và perplexity
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Coherence plot
        ax1.plot(topic_range, coherence_scores, 'b-o', linewidth=2)
        optimal_idx = np.argmax(coherence_scores)
        ax1.axvline(x=list(topic_range)[optimal_idx], color='r', linestyle='--', 
                   label=f'Optimal: {list(topic_range)[optimal_idx]}')
        ax1.set_xlabel('Number of Topics')
        ax1.set_ylabel('Coherence Score')
        ax1.set_title('Coherence Score Analysis')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Perplexity plot  
        ax2.plot(topic_range, perplexity_scores, 'g-s', linewidth=2)
        ax2.axvline(x=list(topic_range)[optimal_idx], color='r', linestyle='--',
                   label=f'Optimal: {list(topic_range)[optimal_idx]}')
        ax2.set_xlabel('Number of Topics')
        ax2.set_ylabel('Perplexity')
        ax2.set_title('Perplexity Analysis')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def print_topics(self, num_words=10):
        """
        Print discovered topics
        """
        print("\n=== DISCOVERED TOPICS ===")
        for idx, topic in self.lda_model.print_topics(num_words=num_words):
            print(f"\nTopic {idx}:")
            words = []
            for word_weight in topic.split(' + '):
                weight, word = word_weight.split('*')
                word = word.strip('"')
                words.append(f"{word} ({float(weight):.3f})")
            print("  " + ", ".join(words))
    
    def create_pyldavis(self, save_html=True):
        """
        Create interactive pyLDAvis visualization
        
        Args:
            save_html: Whether to save as HTML file
            
        Returns:
            pyLDAvis prepared data
        """
        print("\nCreating pyLDAvis visualization...")
        
        vis = gensimvis.prepare(
            self.lda_model,
            self.corpus,
            self.dictionary,
            sort_topics=True
        )
        
        if save_html:
            pyLDAvis.save_html(vis, 'reports/gensim_lda__pyLDAvis_visualization.html')
            print("Visualization saved as 'gensim_lda__pyLDAvis_visualization.html'")
        
        return vis
    
    def get_document_topics(self, doc_idx):
        """
        Get topic distribution for a document
        """
        return self.lda_model.get_document_topics(self.corpus[doc_idx])
        
    def plot_wordclouds(self, num_words=30):
        topics = self.lda_model.show_topics(num_topics=-1, num_words=num_words, formatted=False)
        
        for topic_no, words in topics:
            plt.figure(figsize=(6,6))
            wordcloud = WordCloud(background_color='white').generate_from_frequencies(dict(words))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.title(f"Topic {topic_no}")
            plt.show()

    def plot_top_words(self, num_words=10):
        topics = self.lda_model.show_topics(num_topics=-1, num_words=num_words, formatted=False)
        
        for topic_no, words in topics:
            plt.figure(figsize=(8,4))
            words_list, weights = zip(*words)
            plt.bar(words_list, weights)
            plt.title(f"Topic {topic_no}")
            plt.xticks(rotation=45)
            plt.show()    

    def save_model_components(self, path="models/gensim_lda_model"):
        """
        Saves the LDA model, dictionary, and bigram/trigram models.
        """
        os.makedirs(path, exist_ok=True) # Create directory if it doesn't exist
        print(f"LDA model saving...")

        if self.lda_model:
            self.lda_model.save(os.path.join(path, "lda_model"))
            print(f"LDA model saved to {os.path.join(path, 'lda_model')}")
        if self.dictionary:
            self.dictionary.save(os.path.join(path, "dictionary.gensim"))
            print(f"Dictionary saved to {os.path.join(path, 'dictionary.gensim')}")
        if self.bigram:
            self.bigram.save(os.path.join(path, "bigram_model"))
            print(f"Bigram model saved to {os.path.join(path, 'bigram_model')}")
        if self.trigram:
            self.trigram.save(os.path.join(path, "trigram_model"))
            print(f"Trigram model saved to {os.path.join(path, 'trigram_model')}")

    def load_model_components(self, path="models/gensim_lda_model"):
        """
        Loads the LDA model, dictionary, and bigram/trigram models.
        """
        try:
            self.lda_model = LdaModel.load(os.path.join(path, "lda_model"))
            print(f"LDA model loaded from {os.path.join(path, 'lda_model')}")
        except FileNotFoundError:
            print(f"LDA model not found at {os.path.join(path, 'lda_model')}")
            self.lda_model = None

        try:
            self.dictionary = corpora.Dictionary.load(os.path.join(path, "dictionary.gensim"))
            print(f"Dictionary loaded from {os.path.join(path, 'dictionary.gensim')}")
        except FileNotFoundError:
            print(f"Dictionary not found at {os.path.join(path, 'dictionary.gensim')}")
            self.dictionary = None

        try:
            self.bigram = Phrases.load(os.path.join(path, "bigram_model"))
            print(f"Bigram model loaded from {os.path.join(path, 'bigram_model')}")
        except FileNotFoundError:
            print(f"Bigram model not found at {os.path.join(path, 'bigram_model')}")
            self.bigram = None

        try:
            self.trigram = Phrases.load(os.path.join(path, "trigram_model"))
            print(f"Trigram model loaded from {os.path.join(path, 'trigram_model')}")
        except FileNotFoundError:
            print(f"Trigram model not found at {os.path.join(path, 'trigram_model')}")
            self.trigram = None


def run_lda_analysis(train_df, sample_size=50000, find_optimal=False, fixed_topics=15):
    """
    Main function to run complete LDA analysis
    
    Args:
        train_df: DataFrame with 'input' column
        sample_size: Number of samples to analyze
        
    Returns:
        tuple: (lda_model_object, metrics)
    """
    print("="*60)
    print("GENSIM LDA TOPIC MODELING")
    print("="*60)
    
    # Initialize LDA
    lda = GensimLDA(random_state=42)
    
    # Step 1: Preprocess texts (using PreProcessor methods + lemmatization)
    processed_texts = lda.prepare_texts_for_lda(
        train_df, 
        text_column='lda_input',
        sample_size=sample_size
    )
    
    # Step 2: Create bigrams/trigrams
    texts_with_ngrams = lda.create_bigrams_trigrams(processed_texts)
    
    # Step 3: Prepare corpus
    dictionary, corpus = lda.prepare_corpus(texts_with_ngrams)
    
    # Step 4: Find optimal topics (10-20)
    if find_optimal:
        print("\nFinding optimal number of topics...")
        optimal_topics, best_coherence, best_perplexity = lda.find_optimal_topics(
            min_topics=10,
            max_topics=20
        )
        num_topics = optimal_topics
    else:
        print(f"\nUsing fixed number of topics: {fixed_topics}")
        num_topics = fixed_topics
        print(f"\n=== TRAINING WITH {num_topics} TOPICS ===")
        lda.train_lda(num_topics=num_topics, passes=10, iterations=100)    

    coherence, perplexity = lda.calculate_metrics()
    
    # Step 5: Print topics
    lda.print_topics(num_words=10)
    
    # Step 6: Create visualization
    try:
        lda.create_pyldavis(save_html=True)
    except:
        print("Note: pyLDAvis requires display environment")
    
    lda.plot_top_words(num_words=10)
    lda.plot_wordclouds(num_words=30)

    lda.save_model_components(path="models/gensim_lda_model")

    # Final metrics
    print("\n" + "="*60)
    print("FINAL METRICS")
    print("="*60)
    print(f"Number of Topics: { num_topics}")
    print(f"Coherence Score: {coherence:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    print("="*60)
    
    metrics = {
        'num_topics': num_topics,
        'coherence_score': coherence,
        'perplexity': perplexity,
        'dictionary_size': len(dictionary),
        'corpus_size': len(corpus)
    }
    
    return lda, metrics
