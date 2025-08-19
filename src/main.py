from lda_utils import run_lda_experiments, plot_coherence_and_perplexity


def main():
    # from kaggle_data_loader import KaggleDataLoader
    from local_data_loader import LocalDataLoader as KaggleDataLoader
    from config_loader import load_json_config
    
    # Load configuration
    config = load_json_config('./configs/accuracy_optimized_config.json')

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

    print("\n=== TEXT PREPROCESSING (SENTIMENT OPTIMIZED) ===")
    print("Processing training data with sentiment-aware preprocessing...")
    train_df = preprocessor.clean_data(train_df.copy())
    train_df = preprocessor.remove_duplicates(train_df)

    # Create copies of the train_df for gensim LDA processing
    train_df_gensimLDA = train_df.copy()

    # Use sentiment-optimized preprocessing pipeline
    train_df = train_df.assign(
        normalized_input=train_df["input"].apply(
            lambda x: preprocessor.preprocess_for_sentiment(x, preserve_negation=True)
        )
    )

    print("Processing test data with sentiment-aware preprocessing...")
    test_df = preprocessor.clean_data(test_df.copy())
    test_df = preprocessor.remove_duplicates(test_df)
    # Use sentiment-optimized preprocessing pipeline
    test_df = test_df.assign(
        normalized_input=test_df["input"].apply(
            lambda x: preprocessor.preprocess_for_sentiment(x, preserve_negation=True)
        )
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
            train_df,
            "input",
            figsize=(12, 6),
            save_path="src/images/wordcloud_train.png",
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
        print(
            f"   - Average word length: {word_stats['average_word_length']} characters"
        )
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

    print(f"\n=== STARTING CONFIGURATION-DRIVEN TRAINING PIPELINE ===")

    # Initialize ModelTrainer
    model_trainer = ModelTrainer(output_dir="reports")
    
    # Tối ưu hóa cho MAXIMUM ACCURACY (chấp nhận training time lâu hơn) - Fallback params
    # Run training pipeline with configuration
    if config:
        print("Using JSON configuration-driven training")
        training_results = model_trainer.run_training_pipeline_with_configs(
            train_df=train_df,
            test_df=test_df,
            model_configs=config,
            save_results=True
        )
    else:
        print("No config available")
        

    print(f"\n" + "=" * 100)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"=" * 100)
    print("Check the 'reports/' directory for detailed JSON results.")
    print(f"=" * 100)

    # Temporarily comment out LDA section for testing
    """
    print(f"\n=== TOPIC MODEL TRAINING GENSIM LDA PIPELINE ===")
    from gensim_lda import GensimLDA, run_lda_analysis

    gensimLDA = GensimLDA()

    print("\n=== TEXT PREPROCESSING ===")
    # Process training data using efficient pipeline method
    print("Processing training data...")
    # Use efficient pipeline method that combines cleaning, tokenization, stopword removal and lemmatization
    train_df_gensimLDA = train_df_gensimLDA.assign(
        lda_input=train_df["input"].apply(gensimLDA.preprocess_for_lda)
    )

    # Data quality check after preprocessing
    print("\n=== POST-PREPROCESSING VALIDATION ===")
    train_df_gensimLDA_empty = (
        train_df_gensimLDA["lda_input"]
        .apply(lambda x: len(x) if isinstance(x, list) else 0)
        .eq(0)
        .sum()
    )

    print(f"Training data quality:")
    print(f"   - Final shape: {train_df_gensimLDA.shape}")
    print(f"   - Empty lda_input: {train_df_gensimLDA_empty}")
    print(
        f"   - Average tokens per document: {train_df_gensimLDA['lda_input'].apply(len).mean():.2f}"
    )

    # Run LDA Topic Modeling
    print("\n" + "=" * 60)
    print("=== GENSIM LDA TOPIC MODELING ===")
    print("=" * 60)

    # Analyze với 50k samples
    lda, lda_metrics = run_lda_analysis(
        train_df=train_df_gensimLDA,
        sample_size=CONFIG["train_size"],
        find_optimal=False,
        fixed_topics=11,
    )

    # In kết quả vào final summary

    print(f"\nLDA Topic Modeling Results:")
    print("=" * 60)
    print(f"Training:")
    print(f"   - Number of topics: {lda_metrics['num_topics']}")
    print(f"   - Coherence Score: {lda_metrics['coherence_score']:.4f}")
    print(f"   - Perplexity: {lda_metrics['perplexity']:.4f}")
    print(f"   - Dictionary size: {lda_metrics['dictionary_size']}")
    print(f"   - Corpus size: {lda_metrics['corpus_size']}")
    print(f"=" * 60)

    ####### SKLEARN LDA TOPIC MODELING ########
    print(f"\n=== SKLEARN LDA TOPIC MODELING ===")

    # Danh sách n_topics cần thử
    n_topics_list = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    # Tùy chọn: tham số CountVectorizer để cải thiện topic
    vectorizer_params = {
        "max_features": 20000,  # tăng vocab cho tập lớn
        "min_df": 5,  # bỏ từ quá hiếm
        "max_df": 0.7,  # bỏ từ quá phổ biến
    }

    # tham số LDA chung
    base_params = {
        "max_iter": 20,
        "learning_method": "online",
        "random_state": 42,
        "evaluate_every": -1,
    }

    print("\n=== LDA GRID SEARCH START ===")
    lda_grid_df = run_lda_experiments(
        n_topics_list=n_topics_list,
        train_tokens=train_df["normalized_input"],
        test_tokens=test_df["normalized_input"],
        base_params=base_params,
        vectorizer_params=vectorizer_params,
    )

    print("\n=== LDA GRID SEARCH RESULTS (sorted by lowest Test Perplexity) ===")
    cols = [
        "n_topics",
        "test_perplexity",
        "train_perplexity",
        "test_log_perplexity",
        "train_log_perplexity",
        "test_log_likelihood",
        "train_log_likelihood",
        "coherence_c_v",
        "fit_seconds",
        "vocab_size",
        "n_train_docs",
    ]
    print(lda_grid_df[cols].to_string(index=False))

    print("\n=== PLOTTING PERPLEXITY RESULTS ===")

    plot_coherence_and_perplexity(
        lda_grid_df,
        save_path=None,
        show=True,
    )


if __name__ == "__main__":
    main()
