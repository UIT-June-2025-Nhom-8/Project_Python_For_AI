import pandas as pd


class LocalDataLoader:
    """Simple data loader from /data/amazon_reviews"""

    def __init__(self, config):
        self.config = config

    def prepare_dataframes(self, data_root=".."):
        """Load data from local and prepare for pipeline"""
        # Load data
        train_df = pd.read_csv(f"{data_root}/data/amazon_reviews/train.csv")
        test_df = pd.read_csv(f"{data_root}/data/amazon_reviews/test.csv")

        # Set column names
        train_df.columns = ["label", "title", "text"]
        test_df.columns = ["label", "title", "text"]

        # Apply size limits according to config
        train_df = train_df.iloc[: self.config["train_size"]]
        test_df = test_df.iloc[: self.config["test_size"]]

        # Combine title and text into input
        train_df["input"] = (
            train_df["title"].fillna("") + " " + train_df["text"].fillna("")
        ).str.strip()
        test_df["input"] = (
            test_df["title"].fillna("") + " " + test_df["text"].fillna("")
        ).str.strip()

        # Keep only label and input
        train_df = train_df[["label", "input"]].reset_index(drop=True)
        test_df = test_df[["label", "input"]].reset_index(drop=True)

        print(f"Loaded: Train {train_df.shape}, Test {test_df.shape}")
        return train_df, test_df
