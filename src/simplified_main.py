"""
Simplified Main Pipeline for Amazon Reviews Sentiment Analysis
Clean, easy-to-understand implementation using standard libraries
"""

import pandas as pd
import os
from simplified_model_trainer import SimplifiedModelTrainer


def load_data(data_path="data/amazon_reviews"):
    """
    Load Amazon reviews data
    """
    print("Loading Amazon Reviews data...")
    
    train_file = os.path.join(data_path, "train.csv")
    test_file = os.path.join(data_path, "test.csv")
    
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print(f"Data files not found in {data_path}")
        print("Please ensure train.csv and test.csv are available")
        return None, None
    
    # Load data
    train_df = pd.read_csv(train_file, names=['label', 'title', 'text'])
    test_df = pd.read_csv(test_file, names=['label', 'title', 'text'])
    
    print(f"Loaded {len(train_df)} training samples and {len(test_df)} test samples")
    return train_df, test_df


def preprocess_data(df, sample_size=None):
    """
    Simple data preprocessing
    """
    print("Preprocessing data...")
    
    # Sample data if requested (for faster testing)
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size} records for faster processing")
    
    # Clean data
    df = df.dropna().copy()
    
    # Combine title and text
    df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    
    # Convert labels (1=negative, 2=positive) to text
    df['sentiment'] = df['label'].map({1: 'negative', 2: 'positive'})
    
    # Remove any rows with missing sentiment
    df = df.dropna(subset=['sentiment', 'full_text'])
    
    print(f"After preprocessing: {len(df)} samples")
    print(f"Label distribution: {df['sentiment'].value_counts().to_dict()}")
    
    return df


def main():
    """
    Main pipeline for sentiment analysis
    """
    print("="*80)
    print("SIMPLIFIED AMAZON REVIEWS SENTIMENT ANALYSIS PIPELINE")
    print("="*80)
    
    # Configuration
    CONFIG = {
        "sample_size": 50000,  # Use smaller sample for faster processing
        "test_size": 0.2,
        "optimize_hyperparameters": False  # Set to True for better performance (slower)
    }
    
    print(f"Configuration: {CONFIG}")
    
    # Step 1: Load data
    train_df, test_df = load_data()
    if train_df is None:
        return
    
    # Step 2: Preprocess data
    # Combine train and test for consistent preprocessing
    all_data = pd.concat([train_df, test_df], ignore_index=True)
    processed_data = preprocess_data(all_data, CONFIG["sample_size"])
    
    # Step 3: Train models
    print(f"\n{'='*60}")
    print("TRAINING SENTIMENT ANALYSIS MODELS")
    print(f"{'='*60}")
    
    trainer = SimplifiedModelTrainer(output_dir="reports")
    
    # Train all models
    results = trainer.train_all_models(
        processed_data, 
        text_column='full_text',
        target_column='sentiment',
        test_size=CONFIG["test_size"],
        optimize_hyperparameters=CONFIG["optimize_hyperparameters"]
    )
    
    # Step 4: Compare models
    trainer.compare_models()
    
    # Step 5: Save results and models
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")
    
    results_file = trainer.save_results()
    model_paths = trainer.save_models()
    
    # Step 6: Test predictions
    print(f"\n{'='*60}")
    print("TESTING PREDICTIONS")
    print(f"{'='*60}")
    
    test_texts = [
        "This product is amazing, I love it!",
        "Terrible quality, waste of money",
        "Good value for the price",
        "Would not recommend this to anyone",
        "Perfect, exactly what I needed"
    ]
    
    # Test with best model (logistic regression usually performs well)
    try:
        predictions = trainer.predict_sentiment(test_texts, 'logistic_regression')
        
        print("Sample Predictions (Logistic Regression):")
        for text, pred in zip(test_texts, predictions):
            print(f"  Text: '{text}'")
            print(f"  Sentiment: {pred['sentiment']} (confidence: {pred['confidence']:.3f})")
            print()
            
    except Exception as e:
        print(f"Error making predictions: {e}")
    
    # Step 7: Summary
    print(f"{'='*80}")
    print("PIPELINE COMPLETION SUMMARY")
    print(f"{'='*80}")
    
    summary = trainer.get_model_summary()
    print(f"Total models trained: {summary['total_models']}")
    print(f"Successful models: {summary['successful_models']}")
    print(f"Failed models: {summary['failed_models']}")
    
    if summary['best_accuracy']:
        print(f"Best accuracy: {summary['best_accuracy']['model']} "
              f"({summary['best_accuracy']['score']:.4f})")
    
    if summary['best_f1']:
        print(f"Best F1-score: {summary['best_f1']['model']} "
              f"({summary['best_f1']['score']:.4f})")
    
    print(f"\nResults saved to: {results_file}")
    print(f"Models saved to: output/models/")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
