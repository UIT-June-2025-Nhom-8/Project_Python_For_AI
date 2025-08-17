import pandas as pd
import os
import sys

# from kaggle_data_loader import KaggleDataLoader
from local_data_loader import LocalDataLoader as KaggleDataLoader
CONFIG = {
    "train_size": 100000,
    "test_size": 10000,
    "tfidf_max_features": 5000,
    "tfidf_min_df": 2,
    "tfidf_max_df": 0.8,
    "ngram_range": (1, 2),
}

print("=== AMAZON REVIEWS DATA PROCESSING PIPELINE ===")
print(f"Configuration: {CONFIG}")

print("\n=== INITIALIZING DATA LOADER ===")
data_loader = KaggleDataLoader(CONFIG)
train_df, test_df = data_loader.prepare_dataframes()

from pre_processor import PreProcessor

preprocessor = PreProcessor()

print("\n=== TEXT PREPROCESSING ===")
print("Processing training data...")
train_df = preprocessor.clean_data(train_df.copy())
train_df = preprocessor.remove_duplicates(train_df)
# Use efficient pipeline method that combines cleaning, tokenization, stopword removal and normalization
train_df = train_df.assign(
    normalized_input=train_df["input"].apply(preprocessor.preprocess_text_pipeline)
)

print("Processing test data...")
test_df = preprocessor.clean_data(test_df.copy())
test_df = preprocessor.remove_duplicates(test_df)
# Use efficient pipeline method that combines cleaning, tokenization, stopword removal and normalization
test_df = test_df.assign(
    normalized_input=test_df["input"].apply(preprocessor.preprocess_text_pipeline)
)

print("\n=== POST-PREPROCESSING VALIDATION ===")
train_empty = (
    train_df["normalized_input"]
    .apply(lambda x: len(x) if isinstance(x, list) else 0)
    .eq(0)
    .sum()
)
test_empty = (
    test_df["normalized_input"]
    .apply(lambda x: len(x) if isinstance(x, list) else 0)
    .eq(0)
    .sum()
)

print(f"Training data quality:")
print(f"   - Final shape: {train_df.shape}")
print(f"   - Empty normalized_input: {train_empty}")
print(
    f"   - Average tokens per document: {train_df['normalized_input'].apply(len).mean():.2f}"
)

print(f"Test data quality:")
print(f"   - Final shape: {test_df.shape}")
print(f"   - Empty normalized_input: {test_empty}")
print(
    f"   - Average tokens per document: {test_df['normalized_input'].apply(len).mean():.2f}"
)

print(f"\nFinal columns: {list(train_df.columns)}")
print("\nSample processed data:")
print(train_df.head(3))
print("\n" + "=" * 50)
print(test_df.head(3))

from text_analyzer import TextAnalyzer

print("\n=== TEXT ANALYSIS BEFORE TF-IDF VECTORIZATION ===")
text_analyzer = TextAnalyzer()

print("\n1. TRAINING DATA ANALYSIS")
train_analysis = text_analyzer.analyze_text_statistics(train_df, "input")

print("\n2. WORD CLOUD GENERATION")
try:
    text_analyzer.generate_wordcloud(
        train_df, "input", figsize=(12, 6), save_path="src/images/wordcloud_train.png"
    )
except Exception as e:
    print(f"   Could not generate word cloud: {e}")

print("\n3. DATASET COMPARISON")
comparison_results = text_analyzer.compare_datasets(train_df, test_df, "input")

print("\n4. WORD FREQUENCY ANALYSIS")
word_freq_report = text_analyzer.get_word_frequency_report(min_frequency=5)
if not word_freq_report.empty:
    print("\nTop 15 words with frequency >= 5:")
    print(word_freq_report.head(15).to_string(index=False))

from tf_idf_vectorizer import TFIDFVectorizer

print("\n=== TF-IDF VECTORIZATION ===")
tfidf_vectorizer = TFIDFVectorizer(
    max_features=CONFIG["tfidf_max_features"],
    min_df=CONFIG["tfidf_min_df"],
    max_df=CONFIG["tfidf_max_df"],
    ngram_range=CONFIG["ngram_range"],
)
print(f"TF-IDF Configuration: {CONFIG}")

print("\nTraining TF-IDF Vectorizer...")
X_train_tfidf = tfidf_vectorizer.fit_transform(train_df["normalized_input"])

print("Transforming test data...")
X_test_tfidf = tfidf_vectorizer.transform(test_df["normalized_input"])

print(f"\n=== TF-IDF MATRIX ANALYSIS ===")
print(f"Matrix Information:")
print(f"   Train shape: {X_train_tfidf.shape}")
print(f"   Test shape: {X_test_tfidf.shape}")
print(
    f"   Sparsity: {(1 - X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1])):.4f}"
)
print(f"   Memory usage: ~{X_train_tfidf.data.nbytes / (1024**2):.2f} MB")

print(f"\nTop 10 Most Important TF-IDF Features:")
try:
    top_features = tfidf_vectorizer.get_top_features(X_train_tfidf, top_n=10)
    for i, (feature, score) in enumerate(top_features, 1):
        print(f"   {i:2d}. {feature:20s} -> {score:.4f}")
except Exception as e:
    print(f"   Could not extract top features: {e}")

print(f"\nSaving model...")
try:
    model_path = "output/models/tfidf_vectorizer.pkl"
    tfidf_vectorizer.save_vectorizer(model_path)
    print(f"   Model saved to: {model_path}")
except Exception as e:
    print(f"   Error saving model: {e}")

print(f"\n" + "=" * 60)
print(f"PIPELINE COMPLETION SUMMARY")
print(f"=" * 60)
print(f"Dataset Information:")
print(f"   - Train samples: {len(train_df):,}")
print(f"   - Test samples: {len(test_df):,}")
print(f"   - Features: {X_train_tfidf.shape[1]:,}")

print(f"\nText Analysis Summary:")
if text_analyzer.analysis_results:
    corpus_stats = text_analyzer.analysis_results["corpus_statistics"]
    word_stats = text_analyzer.analysis_results["word_analysis"]
    print(f"   - Vocabulary size: {corpus_stats['vocabulary_size']:,}")
    print(f"   - Total words: {corpus_stats['total_word_occurrences']:,}")
    print(f"   - Average word length: {word_stats['average_word_length']} characters")
    print(
        f"   - Most frequent word: '{word_stats['most_frequent_word'][0]}' ({word_stats['most_frequent_word'][1]:,} times)"
    )

print(f"\nLabel Distribution:")
train_labels = train_df["label"].value_counts()
test_labels = test_df["label"].value_counts()
print(f"   Train: {dict(train_labels)}")
print(f"   Test:  {dict(test_labels)}")

print(f"\nData Ready for training model:")
print(f"   - X_train_tfidf: {X_train_tfidf.shape}")
print(f"   - X_test_tfidf: {X_test_tfidf.shape}")
print(f"   - y_train: {train_df['label'].shape}")
print(f"   - y_test: {test_df['label'].shape}")
print(f"=" * 60)

# Import và khởi tạo ModelTrainer
from model_trainer import ModelTrainer

print(f"\n=== STARTING MODEL TRAINING PIPELINE ===")
model_trainer = ModelTrainer(output_dir="reports")

# Chạy training pipeline với tất cả models
print("Running training pipeline for all models...")
training_results = model_trainer.run_training_pipeline(
    train_df=train_df, 
    test_df=test_df, 
    optimize_hyperparameters=False,  # Set True nếu muốn tối ưu hyperparameters (tốn thời gian)
    save_results=True
)

print(f"\n" + "="*100)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print(f"="*100)
print("Check the 'reports/' directory for detailed JSON results.")
print(f"="*100)

##### LDA TOPIC MODELING #####
# === LDA GRID-SEARCH OVER N_TOPICS ===
from lda_model import LDATopicModeler
import pandas as pd
import numpy as np
import time

def _tokens_to_text_list(series_or_list):
    """
    Join list-of-tokens -> string cho CountVectorizer.
    """
    if isinstance(series_or_list, pd.Series):
        series_or_list = series_or_list.tolist()
    out = []
    for x in series_or_list:
        out.append(" ".join(x) if isinstance(x, list) else str(x))
    return out

def run_lda_experiments(
    n_topics_list,
    train_tokens,   # Series/list các token đã preprocess
    test_tokens,    # Series/list các token đã preprocess
    base_params=None,       # tham số chung cho LDA
    vectorizer_params=None  # tham số CountVectorizer
):
    """
    Chạy LDA cho nhiều giá trị n_topics và trả về DataFrame so sánh.
    Tính train/test Perplexity, Log-likelihood, Log-perplexity và Coherence (c_v nếu có gensim).
    """
    train_texts = _tokens_to_text_list(train_tokens)
    test_texts  = _tokens_to_text_list(test_tokens)

    results = []
    for k in n_topics_list:
        print(f"\n>>> Running LDA with n_topics={k}")
        params = dict(
            n_topics=k,
            max_iter=20,              # có thể tăng 30–50 nếu thời gian cho phép
            learning_method="online", # thường hội tụ tốt trên tập lớn
            random_state=42,
            evaluate_every=-1
        )
        if base_params:
            params.update(base_params)

        # Khởi tạo model
        lda = LDATopicModeler(
            n_topics=params["n_topics"],
            max_features=None,        # vocab tối đa (override bằng vectorizer_params nếu muốn)
            min_df=5,
            max_df=0.8,
            random_state=params["random_state"],
            max_iter=params["max_iter"],
            learning_method=params["learning_method"],
            evaluate_every=params["evaluate_every"],
        )

        # Điều chỉnh CountVectorizer (nếu có)
        if vectorizer_params:
            lda.vectorizer.set_params(**vectorizer_params)

        t0 = time.time()
        # Fit + transform train
        doc_topic_train = lda.fit_transform(train_texts)
        # Transform test
        doc_topic_test  = lda.transform(test_texts)
        fit_secs = time.time() - t0

        # Tạo BOW để tính metric (API sklearn cần X)
        X_train_bow = lda.vectorizer.transform(train_texts)
        X_test_bow  = lda.vectorizer.transform(test_texts)

        # Perplexity & Log-likelihood (sklearn)
        train_perp = lda.model.perplexity(X_train_bow)
        test_perp  = lda.model.perplexity(X_test_bow)
        train_ll   = lda.model.score(X_train_bow)   # tổng log-likelihood
        test_ll    = lda.model.score(X_test_bow)

        # Log-perplexity (per-word): -loglik / total_word_count
        # Lưu ý: sum() trên sparse trả về tổng số đếm từ
        n_words_train = float(X_train_bow.sum())
        n_words_test  = float(X_test_bow.sum())
        train_log_perp = -train_ll / n_words_train if n_words_train > 0 else np.nan
        test_log_perp  = -test_ll  / n_words_test  if n_words_test  > 0 else np.nan

        # Coherence (c_v) nếu có gensim
        coherence_cv = np.nan
        try:
            from gensim.models.coherencemodel import CoherenceModel
            # Lấy top words mỗi topic để tính coherence
            top_words = lda.get_top_words_per_topic(top_n=10)
            topics_words = [[w for (w, _) in topic] for topic in top_words]
            # texts cho gensim cần list-of-tokens
            texts_tokens_for_coh = train_tokens.tolist() if isinstance(train_tokens, pd.Series) else train_tokens
            cm = CoherenceModel(
                topics=topics_words,
                texts=texts_tokens_for_coh,
                coherence="c_v"
            )
            coherence_cv = cm.get_coherence()
        except Exception as e:
            print(f"(Skip coherence for n_topics={k}: {e})")

        results.append({
            "n_topics": k,
            "test_perplexity": test_perp,
            "train_perplexity": train_perp,
            "test_log_perplexity": test_log_perp,
            "train_log_perplexity": train_log_perp,
            "test_log_likelihood": test_ll,
            "train_log_likelihood": train_ll,
            "coherence_c_v": coherence_cv,
            "fit_seconds": fit_secs,
            "vocab_size": len(lda.vectorizer.get_feature_names_out()),
            "n_train_docs": X_train_bow.shape[0]
        })

    df = pd.DataFrame(results)
    # Sắp xếp theo test_perplexity tăng dần (tốt nhất ở trên)
    df = df.sort_values(by=["test_perplexity"], ascending=True).reset_index(drop=True)
    return df

# Danh sách n_topics cần thử
n_topics_list = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# Tùy chọn: tham số CountVectorizer để cải thiện topic
vectorizer_params = {
    "max_features": 20000,  # tăng vocab cho tập lớn
    "min_df": 5,            # bỏ từ quá hiếm
    "max_df": 0.7           # bỏ từ quá phổ biến
}

# Tùy chọn: tham số LDA chung
base_params = {
    "max_iter": 20,
    "learning_method": "online",
    "random_state": 42,
    "evaluate_every": -1
}

print("\n=== LDA GRID SEARCH START ===")
lda_grid_df = run_lda_experiments(
    n_topics_list=n_topics_list,
    train_tokens=train_df["normalized_input"],
    test_tokens=test_df["normalized_input"],
    base_params=base_params,
    vectorizer_params=vectorizer_params
)

print("\n=== LDA GRID SEARCH RESULTS (sorted by lowest Test Perplexity) ===")
cols = [
    "n_topics",
    "test_perplexity", "train_perplexity",
    "test_log_perplexity", "train_log_perplexity",
    "test_log_likelihood", "train_log_likelihood",
    "coherence_c_v",
    "fit_seconds", "vocab_size", "n_train_docs"
]
print(lda_grid_df[cols].to_string(index=False))
