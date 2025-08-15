import pandas as pd
import os
import sys

# Configuration
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

# Get the downloaded path from kagglehub
# If kritanjalijain_amazon_reviews_path is not defined, try to re-download or find the path
kritanjalijain_amazon_reviews_path = None
try:
    if kritanjalijain_amazon_reviews_path is None:
        raise NameError
except NameError:
    import kagglehub

    kritanjalijain_amazon_reviews_path = kagglehub.dataset_download(
        "kritanjalijain/amazon-reviews"
    )
    print(f"KaggleHub download path: {kritanjalijain_amazon_reviews_path}")


# Construct the full paths to the train and test CSV files within the downloaded directory
train_csv_path = os.path.join(kritanjalijain_amazon_reviews_path, "train.csv")
test_csv_path = os.path.join(kritanjalijain_amazon_reviews_path, "test.csv")


# Load the dataframes with error handling
print("\n=== LOADING DATA ===")
try:
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    print(f"Successfully loaded data:")
    print(f"   - Train: {train_df.shape}")
    print(f"   - Test: {test_df.shape}")
except FileNotFoundError as e:
    print(f"Error: Dataset files not found!")
    print(f"   Expected paths: {train_csv_path}, {test_csv_path}")
    sys.exit(1)
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit(1)

# Rename columns
train_df.columns = ["label", "title", "text"]
test_df.columns = ["label", "title", "text"]

# Data validation
print("\n=== DATA VALIDATION ===")
print(f"Train data info:")
print(f"   - Shape: {train_df.shape}")
print(f"   - Null values: {train_df.isnull().sum().sum()}")
print(f"   - Label distribution: {train_df['label'].value_counts().to_dict()}")

print(f"Test data info:")
print(f"   - Shape: {test_df.shape}")
print(f"   - Null values: {test_df.isnull().sum().sum()}")
print(f"   - Label distribution: {test_df['label'].value_counts().to_dict()}")

# Create copies of the limited dataframes using configuration
train_df = train_df.iloc[: CONFIG["train_size"]].copy()
test_df = test_df.iloc[: CONFIG["test_size"]].copy()
print(f"\nLimited datasets to train: {len(train_df)}, test: {len(test_df)}")

# Combine title and text columns into input column
train_df.loc[:, "input"] = train_df["title"] + " " + train_df["text"]
test_df.loc[:, "input"] = test_df["title"] + " " + test_df["text"]

# Drop original title and text columns
train_df = train_df.drop(["title", "text"], axis=1)
test_df = test_df.drop(["title", "text"], axis=1)

# Import the PreProcessor class
from pre_processor import PreProcessor

# Instantiate the PreProcessor
preprocessor = PreProcessor()

print("\n=== TEXT PREPROCESSING ===")
# Process training data using efficient pipeline method
print("Processing training data...")
train_df = preprocessor.clean_data(train_df.copy())
train_df = preprocessor.remove_duplicates(train_df)
# Use efficient pipeline method that combines cleaning, tokenization, stopword removal and normalization
train_df = train_df.assign(
    normalized_input=train_df["input"].apply(preprocessor.preprocess_text_pipeline)
)

# Process test data using efficient pipeline method
print("Processing test data...")
test_df = preprocessor.clean_data(test_df.copy())
test_df = preprocessor.remove_duplicates(test_df)
# Use efficient pipeline method that combines cleaning, tokenization, stopword removal and normalization
test_df = test_df.assign(
    normalized_input=test_df["input"].apply(preprocessor.preprocess_text_pipeline)
)

# Data quality check after preprocessing
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

# Display sample results
print(f"\nFinal columns: {list(train_df.columns)}")
print("\nSample processed data:")
print(train_df.head(3))
print("\n" + "=" * 50)
print(test_df.head(3))

# Import TF-IDF Vectorizer
from tf_idf_vectorizer import TFIDFVectorizer

# Initialize TF-IDF Vectorizer with configuration
print("\n=== TF-IDF VECTORIZATION ===")
tfidf_vectorizer = TFIDFVectorizer(
    max_features=CONFIG["tfidf_max_features"],
    min_df=CONFIG["tfidf_min_df"],
    max_df=CONFIG["tfidf_max_df"],
    ngram_range=CONFIG["ngram_range"],
)
print(f"TF-IDF Configuration: {CONFIG}")

# Vectorize data
print("\nTraining TF-IDF Vectorizer...")
X_train_tfidf = tfidf_vectorizer.fit_transform(train_df["normalized_input"])

print("Transforming test data...")
X_test_tfidf = tfidf_vectorizer.transform(test_df["normalized_input"])

# Display comprehensive TF-IDF information
print(f"\n=== TF-IDF MATRIX ANALYSIS ===")
print(f"Matrix Information:")
print(f"   Train shape: {X_train_tfidf.shape}")
print(f"   Test shape: {X_test_tfidf.shape}")
print(
    f"   Sparsity: {(1 - X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1])):.4f}"
)
print(f"   Memory usage: ~{X_train_tfidf.data.nbytes / (1024**2):.2f} MB")

# Display top features
print(f"\nTop 10 Most Important TF-IDF Features:")
try:
    top_features = tfidf_vectorizer.get_top_features(X_train_tfidf, top_n=10)
    for i, (feature, score) in enumerate(top_features, 1):
        print(f"   {i:2d}. {feature:20s} -> {score:.4f}")
except Exception as e:
    print(f"   Could not extract top features: {e}")

# Save vectorizer
print(f"\nSaving model...")
try:
    model_path = "tfidf_vectorizer.pkl"
    tfidf_vectorizer.save_vectorizer(model_path)
    print(f"   Model saved to: {model_path}")
except Exception as e:
    print(f"   Error saving model: {e}")

# Final summary
print(f"\n" + "=" * 60)
print(f"PIPELINE COMPLETION SUMMARY")
print(f"=" * 60)
print(f"Dataset Information:")
print(f"   - Train samples: {len(train_df):,}")
print(f"   - Test samples: {len(test_df):,}")
print(f"   - Features: {X_train_tfidf.shape[1]:,}")

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
