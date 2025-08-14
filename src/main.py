import pandas as pd
import os

# Get the downloaded path from kagglehub (assuming the variable name is correct from previous execution)
# If kritanjalijain_amazon_reviews_path is not defined, try to re-download or find the path
try:
    kritanjalijain_amazon_reviews_path
except NameError:
    import kagglehub

    kritanjalijain_amazon_reviews_path = kagglehub.dataset_download(
        "kritanjalijain/amazon-reviews"
    )
    print(f"KaggleHub download path: {kritanjalijain_amazon_reviews_path}")


# Construct the full paths to the train and test CSV files within the downloaded directory
train_csv_path = os.path.join(kritanjalijain_amazon_reviews_path, "train.csv")
test_csv_path = os.path.join(kritanjalijain_amazon_reviews_path, "test.csv")


# Load the dataframes again with the correct path
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

# Rename columns
train_df.columns = ["label", "title", "text"]
test_df.columns = ["label", "title", "text"]

# Limit the dataframes as in the previous successful execution
train_df = train_df.head(100000)
test_df = test_df.head(10000)

# Import the PreProcessor class
from pre_processor import PreProcessor

# Instantiate the PreProcessor
preprocessor = PreProcessor()

# Apply the preprocessing methods using the PreProcessor instance
print("--- Xử lý train_df ---")
train_df = preprocessor.clean_data(train_df)
train_df = preprocessor.remove_duplicates(train_df)
train_df["cleaned_title"] = train_df["title"].apply(preprocessor.clean_text)
train_df["cleaned_text"] = train_df["text"].apply(preprocessor.clean_text)
train_df["tokenized_title"] = train_df["cleaned_title"].apply(
    preprocessor.tokenize_text
)
train_df["tokenized_text"] = train_df["cleaned_text"].apply(preprocessor.tokenize_text)
train_df["no_stopwords_title"] = train_df["tokenized_title"].apply(
    preprocessor.remove_stopwords
)
train_df["no_stopwords_text"] = train_df["tokenized_text"].apply(
    preprocessor.remove_stopwords
)
train_df["normalized_title"] = train_df["no_stopwords_title"].apply(
    preprocessor.normalize_token
)
train_df["normalized_text"] = train_df["no_stopwords_text"].apply(
    preprocessor.normalize_token
)

print("\n--- Xử lý test_df ---")
test_df = preprocessor.clean_data(test_df)
test_df = preprocessor.remove_duplicates(test_df)
test_df["cleaned_title"] = test_df["title"].apply(preprocessor.clean_text)
test_df["cleaned_text"] = test_df["text"].apply(preprocessor.clean_text)
test_df["tokenized_title"] = test_df["cleaned_title"].apply(preprocessor.tokenize_text)
test_df["tokenized_text"] = test_df["cleaned_text"].apply(preprocessor.tokenize_text)
test_df["no_stopwords_title"] = test_df["tokenized_title"].apply(
    preprocessor.remove_stopwords
)
test_df["no_stopwords_text"] = test_df["tokenized_text"].apply(
    preprocessor.remove_stopwords
)
test_df["normalized_title"] = test_df["no_stopwords_title"].apply(
    preprocessor.normalize_token
)
test_df["normalized_text"] = test_df["no_stopwords_text"].apply(
    preprocessor.normalize_token
)

display(train_df.head())
display(test_df.head())
