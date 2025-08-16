import pandas as pd
import os
import sys

# from kaggle_data_loader import KaggleDataLoader
from local_data_loader import LocalDataLoader as KaggleDataLoader
CONFIG = {
    "train_size": 100_000,
    "test_size": 10000,
    "tfidf_max_features": 5000,
    "tfidf_min_df": 2,
    "tfidf_max_df": 0.8,
    "ngram_range": (1, 2),
}

# OUTPUT_REPORT = "/kaggle/working/reports"
OUTPUT_REPORT = "reports"

print("=== AMAZON REVIEWS DATA PROCESSING PIPELINE ===")
print(f"Configuration: {CONFIG}")


print("\n=== INITIALIZING DATA LOADER ===")
data_loader = KaggleDataLoader(CONFIG)
train_df, test_df = data_loader.prepare_dataframes()

# ===== BALANCE NEGATIVE AND POSITIVE IN TRAIN SET =====
print("\n=== BALANCING TRAIN DATA (NEGATIVE vs POSITIVE) ===")
target_train_size = CONFIG["train_size"]
if "label" in train_df.columns:
    neg_df = train_df[train_df["label"] == 1]
    pos_df = train_df[train_df["label"] == 2]
    n_each = target_train_size // 2
    # Nếu thiếu thì lấy tối đa có thể
    neg_sample = neg_df.sample(n=min(n_each, len(neg_df)), random_state=42)
    pos_sample = pos_df.sample(n=min(n_each, len(pos_df)), random_state=42)
    # Nếu thiếu số lượng, bổ sung từ class còn lại
    total = len(neg_sample) + len(pos_sample)
    if total < target_train_size:
        # Ưu tiên bổ sung từ class còn lại nếu còn dư
        if len(neg_df) > len(neg_sample):
            extra = min(target_train_size - total, len(neg_df) - len(neg_sample))
            neg_sample = pd.concat([neg_sample, neg_df.drop(neg_sample.index).sample(n=extra, random_state=43)])
        elif len(pos_df) > len(pos_sample):
            extra = min(target_train_size - total, len(pos_df) - len(pos_sample))
            pos_sample = pd.concat([pos_sample, pos_df.drop(pos_sample.index).sample(n=extra, random_state=44)])
    train_df = pd.concat([neg_sample, pos_sample]).sample(frac=1, random_state=99).reset_index(drop=True)
    print(f"Train set balanced: negative={sum(train_df['label']==1)}, positive={sum(train_df['label']==2)}, total={len(train_df)}")
else:
    print("Warning: 'label' column not found in train_df, skipping balancing step.")

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

print("\n=== MEMORY OPTIMIZATION ===")
print("Dropping original 'input' column to save memory...")
print(f"Before: Train memory usage ~{train_df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
# Drop original input column to save memory since we have normalized_input
train_df = train_df.drop('input', axis=1)

print(f"After: Train memory usage ~{train_df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

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
# Convert normalized_input (list of tokens) back to text for analysis
train_df_analysis = train_df.copy()
train_df_analysis['input'] = train_df_analysis['normalized_input'].apply(
    lambda tokens: ' '.join(tokens) if isinstance(tokens, list) else str(tokens)
)
train_analysis = text_analyzer.analyze_text_statistics(train_df_analysis, "input")

print("\n2. WORD CLOUD GENERATION")
try:
    text_analyzer.generate_wordcloud(
        train_df_analysis, "input", figsize=(12, 6), save_path="src/images/wordcloud_train.png"
    )
except Exception as e:
    print(f"   Could not generate word cloud: {e}")

print("\n3. DATASET COMPARISON")
test_df_analysis = test_df.copy()
test_df_analysis['input'] = test_df_analysis['normalized_input'].apply(
    lambda tokens: ' '.join(tokens) if isinstance(tokens, list) else str(tokens)
)
comparison_results = text_analyzer.compare_datasets(train_df_analysis, test_df_analysis, "input")

print("\n4. WORD FREQUENCY ANALYSIS")
word_freq_report = text_analyzer.get_word_frequency_report(min_frequency=5)
if not word_freq_report.empty:
    print("\nTop 15 words with frequency >= 5:")
    print(word_freq_report.head(15).to_string(index=False))

# Clean up temporary analysis dataframes to save memory
del train_df_analysis, test_df_analysis

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
model_trainer = ModelTrainer(output_dir=OUTPUT_REPORT)

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
