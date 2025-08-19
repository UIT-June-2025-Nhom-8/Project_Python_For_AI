import re
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from stopwords_config import DEFAULT_WORDCLOUD_STOPWORDS


class TextAnalyzer:
    """
    Class for analyzing text data before TF-IDF vectorization

    Features:
    - Word frequency analysis
    - Top word identification
    - Word cloud generation
    - Average word length calculation
    - Text statistics and insights
    """

    def __init__(self):
        self.word_count = {}
        self.total_words = 0
        self.total_sentences = 0
        self.analysis_results = {}

    def build_corpus_word_count(self, dataset, text_column="input"):
        """
        Build word count dictionary for entire corpus

        Args:
            dataset (pd.DataFrame): Dataset containing text data
            text_column (str): Column name containing text data

        Returns:
            dict: Complete word count dictionary
        """
        print("Building corpus word count...")
        self.word_count = {}

        for sentence in dataset[text_column]:
            if isinstance(sentence, str):
                for word in sentence.split():
                    if word in self.word_count:
                        self.word_count[word] += 1
                    else:
                        self.word_count[word] = 1

        self.total_words = sum(self.word_count.values())
        self.total_sentences = len(dataset)

        print(f"   Total unique words: {len(self.word_count):,}")
        print(f"   Total word occurrences: {self.total_words:,}")
        print(f"   Total sentences: {self.total_sentences:,}")

        return self.word_count

    def get_top_words(self, top_n=10):
        """
        Get top N most frequent words

        Args:
            top_n (int): Number of top words to return

        Returns:
            dict: Dictionary of top words with their counts
        """
        if not self.word_count:
            print(
                "Warning: Word count not built yet. Call build_corpus_word_count() first."
            )
            return {}

        top_words = dict(
            sorted(self.word_count.items(), key=lambda x: x[1], reverse=True)[:top_n]
        )

        print(f"\nTop {top_n} most frequent words:")
        for i, (word, count) in enumerate(top_words.items(), 1):
            percentage = (count / self.total_words) * 100
            print(f"   {i:2d}. {word:15s} -> {count:6,} times ({percentage:.2f}%)")

        return top_words

    def generate_wordcloud(
        self,
        dataset,
        text_column="input",
        figsize=(12, 6),
        remove_numbers=True,
        save_path=None,
    ):
        """
        Generate and display word cloud from text data

        Args:
            dataset (pd.DataFrame): Dataset containing text data
            text_column (str): Column name containing text data
            figsize (tuple): Figure size for the plot
            remove_numbers (bool): Whether to remove numbers from text
            save_path (str): Path to save the word cloud image
        """
        print("Generating word cloud...")

        joined_sentences = ""
        for sentence in dataset[text_column]:
            if isinstance(sentence, str):
                if remove_numbers:
                    cleaned_sentence = re.sub(r"\d+", "", sentence)
                else:
                    cleaned_sentence = sentence
                joined_sentences += " " + cleaned_sentence

        if not joined_sentences.strip():
            print("   Warning: No text data available for word cloud generation")
            return None

        try:
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color="white",
                max_words=100,
                colormap="viridis",
                stopwords=DEFAULT_WORDCLOUD_STOPWORDS,
                relative_scaling=True,
                min_font_size=10,
            ).generate(joined_sentences)

            plt.figure(figsize=figsize)
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.title(
                "Word Cloud - Most Frequent Words", fontsize=16, fontweight="bold"
            )

            if save_path:
                plt.savefig(save_path, bbox_inches="tight", dpi=300)
                print(f"   Word cloud saved to: {save_path}")

            plt.show()
            print("   Word cloud generated successfully")

            return wordcloud

        except Exception as e:
            print(f"   Error generating word cloud: {e}")
            return None

    def generate_sentiment_wordclouds(
        self,
        dataset,
        text_column="input",
        label_column="label",
        figsize=(15, 8),
        remove_numbers=True,
        save_paths=None,
    ):
        """
        Generate separate word clouds for different sentiment labels

        Args:
            dataset (pd.DataFrame): Dataset containing text data and labels
            text_column (str): Column name containing text data
            label_column (str): Column name containing sentiment labels
            figsize (tuple): Figure size for the combined plot
            remove_numbers (bool): Whether to remove numbers from text
            save_paths (dict): Dictionary with paths to save word clouds {1: "negative_path", 2: "positive_path"}

        Returns:
            dict: Dictionary containing word cloud objects for each sentiment
        """
        print("Generating sentiment-specific word clouds...")

        # Check if label column exists
        if label_column not in dataset.columns:
            print(f"   Warning: Label column '{label_column}' not found in dataset")
            return None

        # Get unique sentiment labels
        unique_labels = sorted(dataset[label_column].unique())
        print(f"   Found sentiment labels: {unique_labels}")

        # Define sentiment mapping
        sentiment_mapping = {1: "Negative", 2: "Positive"}
        colormap_mapping = {1: "Reds", 2: "Greens"}

        wordclouds = {}

        # Create subplot figure
        fig, axes = plt.subplots(1, len(unique_labels), figsize=figsize)
        if len(unique_labels) == 1:
            axes = [axes]

        for idx, label in enumerate(unique_labels):
            # Filter data by sentiment label
            sentiment_data = dataset[dataset[label_column] == label]
            print(
                f"   Processing {sentiment_mapping.get(label, f'Label-{label}')} sentiment: {len(sentiment_data):,} samples"
            )

            # Join sentences for this sentiment
            joined_sentences = ""
            for sentence in sentiment_data[text_column]:
                if isinstance(sentence, str):
                    if remove_numbers:
                        cleaned_sentence = re.sub(r"\d+", "", sentence)
                    else:
                        cleaned_sentence = sentence
                    joined_sentences += " " + cleaned_sentence

            if not joined_sentences.strip():
                print(
                    f"   Warning: No text data available for {sentiment_mapping.get(label, f'Label-{label}')} sentiment"
                )
                continue

            try:
                # Generate word cloud for this sentiment
                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color="white",
                    max_words=100,
                    colormap=colormap_mapping.get(label, "viridis"),
                    stopwords=DEFAULT_WORDCLOUD_STOPWORDS,
                    relative_scaling=True,
                    min_font_size=10,
                ).generate(joined_sentences)

                wordclouds[label] = wordcloud

                # Plot word cloud
                axes[idx].imshow(wordcloud, interpolation="bilinear")
                axes[idx].axis("off")
                axes[idx].set_title(
                    f"{sentiment_mapping.get(label, f'Label-{label}')} Sentiment\n({len(sentiment_data):,} samples)",
                    fontsize=14,
                    fontweight="bold",
                )

                # Save individual word cloud if path provided
                if save_paths and label in save_paths:
                    plt.figure(figsize=(10, 6))
                    plt.imshow(wordcloud, interpolation="bilinear")
                    plt.axis("off")
                    plt.title(
                        f"{sentiment_mapping.get(label, f'Label-{label}')} Sentiment Word Cloud",
                        fontsize=16,
                        fontweight="bold",
                    )
                    plt.savefig(save_paths[label], bbox_inches="tight", dpi=300)
                    plt.close()
                    print(
                        f"   {sentiment_mapping.get(label, f'Label-{label}')} word cloud saved to: {save_paths[label]}"
                    )

            except Exception as e:
                print(
                    f"   Error generating word cloud for {sentiment_mapping.get(label, f'Label-{label}')} sentiment: {e}"
                )
                continue

        # Adjust layout and show combined plot
        plt.tight_layout()
        plt.suptitle(
            "Sentiment-Specific Word Clouds", fontsize=16, fontweight="bold", y=1.02
        )
        plt.show()

        print("   Sentiment word clouds generated successfully")
        return wordclouds

    def generate_dataset_comparison_wordclouds(
        self,
        train_dataset,
        test_dataset,
        text_column="input",
        label_column="label",
        figsize=(20, 12),
        remove_numbers=True,
        save_paths=None,
    ):
        """
        Generate word clouds comparing train/test datasets by sentiment

        Args:
            train_dataset (pd.DataFrame): Training dataset
            test_dataset (pd.DataFrame): Test dataset
            text_column (str): Column name containing text data
            label_column (str): Column name containing sentiment labels
            figsize (tuple): Figure size for the combined plot
            remove_numbers (bool): Whether to remove numbers from text
            save_paths (dict): Dictionary with paths to save word clouds
                              {"train_negative": "path", "train_positive": "path",
                              "test_negative": "path", "test_positive": "path"}

        Returns:
            dict: Dictionary containing all word cloud objects
        """
        print("Generating train/test sentiment comparison word clouds...")

        # Check if label column exists
        for dataset_name, dataset in [("train", train_dataset), ("test", test_dataset)]:
            if label_column not in dataset.columns:
                print(
                    f"   Warning: Label column '{label_column}' not found in {dataset_name} dataset"
                )
                return None

        # Get unique sentiment labels from both datasets
        all_labels = sorted(
            set(train_dataset[label_column].unique())
            | set(test_dataset[label_column].unique())
        )
        print(f"   Found sentiment labels: {all_labels}")

        # Define sentiment mapping
        sentiment_mapping = {1: "Negative", 2: "Positive"}
        colormap_mapping = {1: "Reds", 2: "Greens"}

        # Create subplot figure (2 rows: train/test, N columns: sentiments)
        fig, axes = plt.subplots(2, len(all_labels), figsize=figsize)
        if len(all_labels) == 1:
            axes = axes.reshape(-1, 1)

        all_wordclouds = {}

        datasets = [("Train", train_dataset), ("Test", test_dataset)]

        for row_idx, (dataset_name, dataset) in enumerate(datasets):
            print(f"\n   Processing {dataset_name} Dataset:")

            for col_idx, label in enumerate(all_labels):
                # Filter data by sentiment label
                sentiment_data = dataset[dataset[label_column] == label]
                sentiment_name = sentiment_mapping.get(label, f"Label-{label}")

                print(
                    f"     {sentiment_name} sentiment: {len(sentiment_data):,} samples"
                )

                # Join sentences for this sentiment
                joined_sentences = ""
                for sentence in sentiment_data[text_column]:
                    if isinstance(sentence, str):
                        if remove_numbers:
                            cleaned_sentence = re.sub(r"\d+", "", sentence)
                        else:
                            cleaned_sentence = sentence
                        joined_sentences += " " + cleaned_sentence

                if not joined_sentences.strip():
                    print(
                        f"     Warning: No text data available for {sentiment_name} sentiment in {dataset_name}"
                    )
                    axes[row_idx, col_idx].text(
                        0.5,
                        0.5,
                        "No Data Available",
                        ha="center",
                        va="center",
                        transform=axes[row_idx, col_idx].transAxes,
                    )
                    axes[row_idx, col_idx].axis("off")
                    continue

                try:
                    # Generate word cloud for this sentiment and dataset
                    wordcloud = WordCloud(
                        width=800,
                        height=400,
                        background_color="white",
                        max_words=100,
                        colormap=colormap_mapping.get(label, "viridis"),
                        stopwords=DEFAULT_WORDCLOUD_STOPWORDS,
                        relative_scaling=True,
                        min_font_size=10,
                    ).generate(joined_sentences)

                    # Store word cloud
                    key = f"{dataset_name.lower()}_{sentiment_name.lower()}"
                    all_wordclouds[key] = wordcloud

                    # Plot word cloud
                    axes[row_idx, col_idx].imshow(wordcloud, interpolation="bilinear")
                    axes[row_idx, col_idx].axis("off")
                    axes[row_idx, col_idx].set_title(
                        f"{dataset_name} - {sentiment_name}\n({len(sentiment_data):,} samples)",
                        fontsize=12,
                        fontweight="bold",
                    )

                    # Save individual word cloud if path provided
                    if save_paths and key in save_paths:
                        plt.figure(figsize=(10, 6))
                        plt.imshow(wordcloud, interpolation="bilinear")
                        plt.axis("off")
                        plt.title(
                            f"{dataset_name} Dataset - {sentiment_name} Sentiment",
                            fontsize=16,
                            fontweight="bold",
                        )
                        plt.savefig(save_paths[key], bbox_inches="tight", dpi=300)
                        plt.close()
                        print(
                            f"     {dataset_name} {sentiment_name} word cloud saved to: {save_paths[key]}"
                        )

                except Exception as e:
                    print(
                        f"     Error generating word cloud for {sentiment_name} sentiment in {dataset_name}: {e}"
                    )
                    axes[row_idx, col_idx].text(
                        0.5,
                        0.5,
                        "Generation Error",
                        ha="center",
                        va="center",
                        transform=axes[row_idx, col_idx].transAxes,
                    )
                    axes[row_idx, col_idx].axis("off")
                    continue

        # Adjust layout and show combined plot
        plt.tight_layout()
        plt.suptitle(
            "Train vs Test Dataset - Sentiment Word Clouds Comparison",
            fontsize=18,
            fontweight="bold",
            y=0.98,
        )
        plt.show()

        print("\n   Dataset comparison word clouds generated successfully")
        return all_wordclouds

    def calculate_average_word_length(self, dataset, text_column="input"):
        """
        Calculate average word length across the dataset

        Args:
            dataset (pd.DataFrame): Dataset containing text data
            text_column (str): Column name containing text data

        Returns:
            float: Average word length
        """
        print("Calculating average word length...")

        total_length = 0
        total_word_count = 0

        for sentence in dataset[text_column]:
            if isinstance(sentence, str):
                words = sentence.split()
                total_length += sum(len(word) for word in words)
                total_word_count += len(words)

        if total_word_count == 0:
            print("   Warning: No words found in dataset")
            return 0.0

        avg_word_length = round(total_length / total_word_count, 2)

        print(f"   Total characters: {total_length:,}")
        print(f"   Total words: {total_word_count:,}")
        print(f"   Average word length: {avg_word_length} characters")

        return avg_word_length

    def analyze_text_statistics(self, dataset, text_column="input"):
        """
        Comprehensive text analysis including all statistics

        Args:
            dataset (pd.DataFrame): Dataset containing text data
            text_column (str): Column name containing text data

        Returns:
            dict: Complete analysis results
        """
        print(f"\n=== COMPREHENSIVE TEXT ANALYSIS ===")
        print(f"Analyzing {len(dataset):,} text samples...")

        word_count = self.build_corpus_word_count(dataset, text_column)

        top_10_words = self.get_top_words(10)

        avg_word_length = self.calculate_average_word_length(dataset, text_column)

        sentence_lengths = dataset[text_column].str.len()
        word_counts_per_sentence = dataset[text_column].str.split().str.len()

        self.analysis_results = {
            "corpus_statistics": {
                "total_sentences": len(dataset),
                "total_unique_words": len(word_count),
                "total_word_occurrences": self.total_words,
                "vocabulary_size": len(word_count),
            },
            "word_analysis": {
                "top_10_words": top_10_words,
                "average_word_length": avg_word_length,
                "most_frequent_word": (
                    max(word_count.items(), key=lambda x: x[1])
                    if word_count
                    else ("", 0)
                ),
            },
            "sentence_statistics": {
                "average_sentence_length": round(sentence_lengths.mean(), 2),
                "median_sentence_length": sentence_lengths.median(),
                "max_sentence_length": sentence_lengths.max(),
                "min_sentence_length": sentence_lengths.min(),
                "average_words_per_sentence": round(word_counts_per_sentence.mean(), 2),
            },
            "distribution_analysis": {
                "words_appearing_once": sum(
                    1 for count in word_count.values() if count == 1
                ),
                "words_appearing_more_than_10": sum(
                    1 for count in word_count.values() if count > 10
                ),
                "words_appearing_more_than_100": sum(
                    1 for count in word_count.values() if count > 100
                ),
            },
        }

        print(f"\n=== ANALYSIS SUMMARY ===")
        print(f"Corpus Statistics:")
        for key, value in self.analysis_results["corpus_statistics"].items():
            print(f"   {key.replace('_', ' ').title()}: {value:,}")

        print(f"\nWord Analysis:")
        print(f"   Average word length: {avg_word_length} characters")
        print(
            f"   Most frequent word: '{self.analysis_results['word_analysis']['most_frequent_word'][0]}' ({self.analysis_results['word_analysis']['most_frequent_word'][1]:,} times)"
        )

        print(f"\nSentence Statistics:")
        for key, value in self.analysis_results["sentence_statistics"].items():
            print(f"   {key.replace('_', ' ').title()}: {value}")

        print(f"\nDistribution Analysis:")
        for key, value in self.analysis_results["distribution_analysis"].items():
            print(f"   {key.replace('_', ' ').title()}: {value:,}")

        return self.analysis_results

    def get_word_frequency_report(self, min_frequency=1):
        """
        Generate detailed word frequency report

        Args:
            min_frequency (int): Minimum frequency threshold for words to include

        Returns:
            pd.DataFrame: Word frequency report as DataFrame
        """
        if not self.word_count:
            print(
                "Warning: Word count not built yet. Call build_corpus_word_count() first."
            )
            return pd.DataFrame()

        filtered_words = {
            word: count
            for word, count in self.word_count.items()
            if count >= min_frequency
        }

        word_freq_df = pd.DataFrame(
            [
                {
                    "word": word,
                    "frequency": count,
                    "percentage": (count / self.total_words) * 100,
                }
                for word, count in sorted(
                    filtered_words.items(), key=lambda x: x[1], reverse=True
                )
            ]
        )

        print(f"\nWord Frequency Report (min frequency: {min_frequency}):")
        print(f"   Words included: {len(word_freq_df):,}")
        if not word_freq_df.empty:
            print(
                f"   Coverage: {word_freq_df['percentage'].sum():.2f}% of total words"
            )

        return word_freq_df

    def compare_datasets(self, train_dataset, test_dataset, text_column="input"):
        """
        Compare text statistics between training and test datasets

        Args:
            train_dataset (pd.DataFrame): Training dataset
            test_dataset (pd.DataFrame): Test dataset
            text_column (str): Column name containing text data

        Returns:
            dict: Comparison results
        """
        print(f"\n=== DATASET COMPARISON ===")

        print("Analyzing training dataset...")
        train_results = self.analyze_text_statistics(train_dataset, text_column)

        print("\nAnalyzing test dataset...")
        test_analyzer = TextAnalyzer()
        test_results = test_analyzer.analyze_text_statistics(test_dataset, text_column)

        comparison = {
            "train_stats": train_results,
            "test_stats": test_results,
            "comparison": {
                "vocabulary_size_ratio": test_results["corpus_statistics"][
                    "vocabulary_size"
                ]
                / train_results["corpus_statistics"]["vocabulary_size"],
                "avg_word_length_diff": test_results["word_analysis"][
                    "average_word_length"
                ]
                - train_results["word_analysis"]["average_word_length"],
                "avg_sentence_length_diff": test_results["sentence_statistics"][
                    "average_sentence_length"
                ]
                - train_results["sentence_statistics"]["average_sentence_length"],
            },
        }

        print(f"\n=== DATASET COMPARISON SUMMARY ===")
        print(
            f"Vocabulary size ratio (test/train): {comparison['comparison']['vocabulary_size_ratio']:.3f}"
        )
        print(
            f"Average word length difference: {comparison['comparison']['avg_word_length_diff']:.2f} characters"
        )
        print(
            f"Average sentence length difference: {comparison['comparison']['avg_sentence_length_diff']:.2f} characters"
        )

        return comparison
