import pandas as pd


class LocalDataLoader:
    """Đơn giản load data từ /data/amazon_reviews"""

    def __init__(self, config):
        self.config = config


    def prepare_dataframes(self, data_root = ".."):
        """Load data từ local và chuẩn bị cho pipeline"""
        # Load data
        train_df = pd.read_csv(f"{data_root}/amazon_reviews/train.csv")
        test_df = pd.read_csv(f"{data_root}/amazon_reviews/test.csv")

        # Đặt tên cột
        train_df.columns = ["label", "title", "text"]
        test_df.columns = ["label", "title", "text"]
        
        # Giới hạn kích thước theo config
        train_df = train_df.iloc[:self.config["train_size"]]
        test_df = test_df.iloc[:self.config["test_size"]]
        
        # Kết hợp title và text thành input
        train_df["input"] = (train_df["title"].fillna("") + " " + train_df["text"].fillna("")).str.strip()
        test_df["input"] = (test_df["title"].fillna("") + " " + test_df["text"].fillna("")).str.strip()
        
        # Chỉ giữ lại label và input
        train_df = train_df[["label", "input"]].reset_index(drop=True)
        test_df = test_df[["label", "input"]].reset_index(drop=True)
        
        print(f"Loaded: Train {train_df.shape}, Test {test_df.shape}")
        return train_df, test_df
