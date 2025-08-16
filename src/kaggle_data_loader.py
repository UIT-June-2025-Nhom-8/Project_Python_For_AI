import os
import sys
import pandas as pd
import kagglehub


class KaggleDataLoader:
    """
    Class for loading and preparing Kaggle Amazon reviews dataset

    Features:
    - Downloads dataset from Kaggle
    - Loads CSV data with error handling
    - Validates data structure and labels
    - Applies size limits from configuration
    - Combines title and text columns into unified input
    - Delegates data quality processing to PreProcessor
    """

    def __init__(self, config):
        self.config = config
        self.train_df = None
        self.test_df = None
        self.dataset_path = None

    def download_dataset(self):
        """Download Amazon reviews dataset from Kaggle using kritanjalijain dataset"""
        print("Downloading Kaggle Amazon reviews dataset...")

        try:
            if self.dataset_path is None:
                self.dataset_path = kagglehub.dataset_download(
                    "kritanjalijain/amazon-reviews"
                )
            print(f"KaggleHub download path: {self.dataset_path}")
            return self.dataset_path
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            sys.exit(1)

    def load_csv_data(self):
        """Load CSV data from the downloaded dataset"""
        if self.dataset_path is None:
            self.download_dataset()

        train_csv_path = os.path.join(self.dataset_path, "train.csv")
        test_csv_path = os.path.join(self.dataset_path, "test.csv")

        print("\n=== LOADING DATA ===")
        try:
            self.train_df = pd.read_csv(train_csv_path)
            self.test_df = pd.read_csv(test_csv_path)
            print(f"Successfully loaded data:")
            print(f"   - Train: {self.train_df.shape}")
            print(f"   - Test: {self.test_df.shape}")
        except FileNotFoundError as e:
            print(f"Error: Dataset files not found!")
            print(f"   Expected paths: {train_csv_path}, {test_csv_path}")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)

        return self.train_df, self.test_df

    def prepare_dataframes(self):
        """Prepare and validate dataframes with streamlined processing"""
        if self.train_df is None or self.test_df is None:
            self.load_csv_data()

        self.train_df.columns = ["label", "title", "text"]
        self.test_df.columns = ["label", "title", "text"]

        self.validate_data()

        self.apply_size_limits()

        self.clean_and_combine_data()

        return self.train_df, self.test_df

    def validate_data(self):
        """Validate loaded data and perform initial quality checks"""
        print("\n=== DATA VALIDATION ===")

        print(f"Initial Train data info:")
        print(f"   - Shape: {self.train_df.shape}")
        print(f"Initial Test data info:")
        print(f"   - Shape: {self.test_df.shape}")

        print(f"\nInitial label distribution:")
        print(f"   Training: {self.train_df['label'].value_counts().to_dict()}")
        print(f"   Test: {self.test_df['label'].value_counts().to_dict()}")

        train_labels = set(self.train_df["label"].unique())
        test_labels = set(self.test_df["label"].unique())
        expected_labels = {1, 2}  # Binary sentiment: 1=negative, 2=positive

        train_invalid = train_labels - expected_labels
        test_invalid = test_labels - expected_labels

        if train_invalid or test_invalid:
            print(f"Warning: Unexpected labels found")
            if train_invalid:
                print(f"   Training unexpected labels: {train_invalid}")
            if test_invalid:
                print(f"   Test unexpected labels: {test_invalid}")
        else:
            print("All labels are within expected range [1, 2]")

        print("Initial data validation completed")

    def clean_and_combine_data(self):
        """Combine title/text columns and perform basic data preparation"""
        print("\n=== DATA COMBINATION ===")

        self.train_df["title"] = self.train_df["title"].fillna("")
        self.train_df["text"] = self.train_df["text"].fillna("")
        self.test_df["title"] = self.test_df["title"].fillna("")
        self.test_df["text"] = self.test_df["text"].fillna("")

        print("Analyzing content availability...")
        train_title_empty = (self.train_df["title"].str.strip() == "").sum()
        train_text_empty = (self.train_df["text"].str.strip() == "").sum()
        train_both_empty = (
            (self.train_df["title"].str.strip() == "")
            & (self.train_df["text"].str.strip() == "")
        ).sum()

        test_title_empty = (self.test_df["title"].str.strip() == "").sum()
        test_text_empty = (self.test_df["text"].str.strip() == "").sum()
        test_both_empty = (
            (self.test_df["title"].str.strip() == "")
            & (self.test_df["text"].str.strip() == "")
        ).sum()

        print(
            f"   Training - Empty titles: {train_title_empty}, Empty texts: {train_text_empty}, Both empty: {train_both_empty}"
        )
        print(
            f"   Test - Empty titles: {test_title_empty}, Empty texts: {test_text_empty}, Both empty: {test_both_empty}"
        )

        def smart_combine(title, text):
            title_clean = str(title).strip()
            text_clean = str(text).strip()

            if title_clean and text_clean:
                return f"{title_clean} {text_clean}"
            elif title_clean:
                return title_clean
            elif text_clean:
                return text_clean
            else:
                return ""

        print("Combining title and text columns...")
        self.train_df["input"] = self.train_df.apply(
            lambda row: smart_combine(row["title"], row["text"]), axis=1
        )
        self.test_df["input"] = self.test_df.apply(
            lambda row: smart_combine(row["title"], row["text"]), axis=1
        )

        self.train_df = self.train_df.drop(["title", "text"], axis=1).reset_index(
            drop=True
        )
        self.test_df = self.test_df.drop(["title", "text"], axis=1).reset_index(
            drop=True
        )

        print(f"Data combination completed:")
        print(f"   Training: {self.train_df.shape}")
        print(f"   Test: {self.test_df.shape}")
        print(
            f"   Average input length - Train: {self.train_df['input'].str.len().mean():.1f}, Test: {self.test_df['input'].str.len().mean():.1f}"
        )

    def get_data_cleaning_report(self):
        """Generate a comprehensive report of data loading and preparation operations"""
        report = {
            "data_validation": {
                "description": "Initial validation of data structure and label consistency",
                "method": "Shape inspection and label range validation",
                "impact": "Identifies structural issues before processing",
            },
            "size_limiting": {
                "description": "Applied configuration-based size limits to datasets",
                "method": "DataFrame.iloc slicing with config parameters",
                "impact": "Controls dataset size for computational efficiency",
            },
            "column_combination": {
                "description": "Intelligent title-text combination preserving partial content",
                "method": "Conditional logic handling empty/null values in either column",
                "impact": "Creates unified input column for text processing",
            },
            "data_structure": {
                "description": "Standardized column structure for downstream processing",
                "method": "Column renaming and dropping original separate columns",
                "impact": "Prepares data for PreProcessor.clean_data() pipeline",
            },
            "processing_delegation": {
                "description": "Null values, duplicates, and data quality handled by PreProcessor",
                "method": "Separation of concerns - data loading vs data cleaning",
                "impact": "Clear responsibility boundaries between components",
            },
        }
        return report

    def apply_size_limits(self):
        """Apply size limits from configuration"""
        print(f"\n=== APPLYING SIZE LIMITS ===")
        original_train_size = len(self.train_df)
        original_test_size = len(self.test_df)

        self.train_df = self.train_df.iloc[: self.config["train_size"]].copy()
        self.test_df = self.test_df.iloc[: self.config["test_size"]].copy()

        print(f"Size limits applied:")
        print(f"   Training: {original_train_size} -> {len(self.train_df)} samples")
        print(f"   Test: {original_test_size} -> {len(self.test_df)} samples")

    def get_data_summary(self):
        """Get comprehensive summary of loaded and processed data"""
        if self.train_df is None or self.test_df is None:
            return "No data loaded"

        summary = {
            "train_shape": self.train_df.shape,
            "test_shape": self.test_df.shape,
            "train_columns": list(self.train_df.columns),
            "test_columns": list(self.test_df.columns),
            "train_label_distribution": self.train_df["label"].value_counts().to_dict(),
            "test_label_distribution": self.test_df["label"].value_counts().to_dict(),
            "data_quality": {
                "ready_for_preprocessing": True,
                "note": "Null values and duplicates will be handled by PreProcessor",
            },
            "processing_status": {
                "structure_validated": True,
                "size_limits_applied": True,
                "columns_combined": "input" in self.train_df.columns,
                "ready_for_preprocessing": True,
            },
        }

        return summary
