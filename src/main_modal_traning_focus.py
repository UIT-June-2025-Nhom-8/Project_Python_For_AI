def main():
    # from kaggle_data_loader import KaggleDataLoader

    from local_data_loader import LocalDataLoader as KaggleDataLoader
    from config_loader import load_json_config
    from pathlib import Path

    # Load configuration with proper path
    script_dir = Path(__file__).parent
    config_path = script_dir / "configs" / "accuracy_optimized_config.json"
    config = load_json_config(str(config_path))

    CONFIG = {
        "train_size": config.get("dataset_config", {}).get("train_size", 1000),
        "test_size": config.get("dataset_config", {}).get("test_size", 100),
    }

    print("=== AMAZON REVIEWS DATA PROCESSING PIPELINE ===")
    print(f"Configuration: {CONFIG}")

    print("\n=== INITIALIZING DATA LOADER ===")
    data_loader = KaggleDataLoader(CONFIG)
    train_df, test_df = data_loader.prepare_dataframes()

    from pre_processor import PreProcessor

    # Initialize preprocessor with sentiment optimization
    preprocessor = PreProcessor(use_lemmatization=True)

    print("\n=== TEXT PREPROCESSING ===")
    print("Processing training data...")
    train_df = preprocessor.clean_data(train_df.copy())
    train_df = preprocessor.remove_duplicates(train_df)

    # Use efficient pipeline method that combines cleaning, tokenization, stopword removal and normalization
    train_df = train_df.assign(
        normalized_input=train_df["input"].apply(
            lambda x: preprocessor.preprocess_for_sentiment(x, preserve_negation=True)
        )
    )

    # Import and initialize ModelTrainer
    from model_trainer import ModelTrainer

    print(f"\n=== STARTING CONFIGURATION-DRIVEN TRAINING PIPELINE ===")

    # Initialize ModelTrainer
    model_trainer = ModelTrainer(output_dir="reports")

    # Optimize for MAXIMUM ACCURACY (accept longer training time) - Fallback params
    # Run training pipeline with configuration
    if config:
        print("Using JSON configuration-driven training")
        training_results = model_trainer.run_training_pipeline_with_configs(
            train_df=train_df, test_df=test_df, model_configs=config, save_results=True
        )
    else:
        print("No config available")

    print(f"\n" + "=" * 100)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"=" * 100)
    print("Check the 'reports/' directory for detailed JSON results.")
    print(f"=" * 100)

if __name__ == "__main__":
    main()
