"""
Simple NLTK Stopwords Downloader for Kaggle

Purpose: Download specific stopwords/english file using kagglehub.
Dataset: alvations/nltk-corpora (most reliable)
"""

import os
import shutil

STOPWORDS_FILE_OUTPUT = "/kaggle/working/nltk_data/stopwords"


def download_stopwords_english() -> str:
    """
    Download specific stopwords/english file from alvations/nltk-corpora.
    Simple and direct - no fallbacks.
    """
    print("🔍 Downloading stopwords/english from alvations/nltk-corpora...")
    
    import kagglehub
    
    # Download the best dataset
    dataset_path = kagglehub.dataset_download("alvations/nltk-corpora")
    
    # Find the stopwords file
    source_file = f"{dataset_path}/corpora/stopwords/english"
    
    if not os.path.exists(source_file):
        raise RuntimeError(f"❌ Stopwords file not found in dataset: {source_file}")
    
    # Create target directory
    target_dir = os.path.dirname(STOPWORDS_FILE_OUTPUT)
    os.makedirs(target_dir, exist_ok=True)
    
    # Copy the file
    shutil.copy2(source_file, STOPWORDS_FILE_OUTPUT)
    
    print(f"✅ Downloaded stopwords/english to: {STOPWORDS_FILE_OUTPUT}")
    return STOPWORDS_FILE_OUTPUT


def setup_nltk_stopwords(stopwords_file_output: str=STOPWORDS_FILE_OUTPUT) -> str:
    """
    Setup: Download stopwords/english and save for later use.
    
    Returns: Path to stopwords file
    """
    print("🚀 Setting up NLTK stopwords...")
    
    # Check if already exists
    if os.path.exists(STOPWORDS_FILE_OUTPUT):
        print(f"✅ Stopwords already exist: {STOPWORDS_FILE_OUTPUT}")
        return STOPWORDS_FILE_OUTPUT
    
    # Download specific file
    return download_stopwords_english()


if __name__ == "__main__":
    # Simple setup
    setup_nltk_stopwords()
    print("🎯 Done! Ready for stopwords_config_kaggle to process.")
