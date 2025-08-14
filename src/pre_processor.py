import pandas as pd
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer


class PreProcessor:
    def __init__(self):
        nltk.download("punkt_tab")
        nltk.download("stopwords")

    def clean_data(self, df):
        """
        Kiểm tra và xử lý giá trị null, đồng thời kiểm tra loại dữ liệu trong DataFrame.

        Args:
            df (pd.DataFrame): DataFrame cần làm sạch.

        Returns:
            pd.DataFrame: DataFrame đã được làm sạch.
        """
        print("Số lượng giá trị null trước khi xử lý:")
        print(df.isnull().sum())

        # Điền giá trị NaN trong cột 'text' bằng chuỗi rỗng
        if "text" in df.columns:
            df["text"] = df["text"].fillna("")

        if "title" in df.columns:
            df["title"] = df["title"].fillna("")

        print("\nSố lượng giá trị null sau khi xử lý:")
        print(df.isnull().sum())

        # Kiểm tra loại dữ liệu
        print("\nLoại dữ liệu của các cột:")
        print(df.dtypes)

        return df

    def remove_duplicates(self, df):
        """
        Kiểm tra và loại bỏ các bản ghi trùng lặp trong DataFrame.

        Args:
            df (pd.DataFrame): DataFrame cần xử lý.

        Returns:
            pd.DataFrame: DataFrame sau khi đã loại bỏ các bản ghi trùng lặp.
        """
        print(f"Số lượng bản ghi trước khi loại bỏ trùng lặp: {len(df)}")

        # Kiểm tra và loại bỏ các bản ghi trùng lặp
        df_cleaned = df.drop_duplicates()

        print(f"Số lượng bản ghi sau khi loại bỏ trùng lặp: {len(df_cleaned)}")

        return df_cleaned

    def clean_text(self, text):
        """
        Làm sạch văn bản bằng cách:
        - Loại bỏ các ký tự đặc biệt và số.
        - Chuyển đổi sang chữ thường.
        - Loại bỏ dấu câu.
        - Loại bỏ khoảng trắng thừa.
        - Thay thế một hoặc nhiều khoảng trắng bằng một khoảng trắng duy nhất.
        - Loại bỏ khoảng trắng ở đầu/cuối.

        Args:
            text (str): Chuỗi văn bản cần làm sạch.

        Returns:
            str: Chuỗi văn bản đã được làm sạch.
        """
        # Loại bỏ các ký tự đặc biệt và số
        # Sử dụng regex để giữ lại chỉ các chữ cái (tiếng Anh và tiếng Việt có dấu) và dấu cách
        text = re.sub(r"[^A-Za-zÀ-ú ]+", "", text)
        # Chuyển đổi sang chữ thường
        text = text.lower()
        # loại bỏ dấu câu
        text = text.translate(str.maketrans("", "", string.punctuation))
        # Loại bỏ khoảng trắng thừa
        # Thay thế một hoặc nhiều khoảng trắng bằng một khoảng trắng duy nhất và loại bỏ khoảng trắng ở đầu/cuối
        text = re.sub(r"\s+", " ", text).strip()
        return text  # Trả về chuỗi văn bản đã được làm sạch

    def tokenize_text(self, text):
        """
        Tách văn bản thành các token (từ).

        Args:
            text (str): Chuỗi văn bản cần tách token.

        Returns:
            list: Danh sách các token.
        """
        # Sử dụng word_tokenize của NLTK để tách văn bản thành các token
        if isinstance(text, str):
            return word_tokenize(text)
        else:
            return text

    def remove_stopwords(self, tokens):
        """
        Loại bỏ các từ dừng (stopwords) tiếng Anh khỏi danh sách tokens.

        Args:
            filtered_tokens (list): Danh sách các token cần xử lý.

        Returns:
            list: Danh sách các token sau khi đã loại bỏ stopwords.
        """
        if not isinstance(tokens, list):
            return tokens
        else:
            # Tải danh sách các từ dừng tiếng Anh từ NLTK
            stop_words = set(stopwords.words("english"))
            # Lọc bỏ các từ dừng khỏi danh sách tokens
            filtered_tokens = [word for word in tokens if word not in stop_words]
            # Trả về danh sách các token đã lọc
            return filtered_tokens

    def normalize_token(self, tokens):
        """
        Chuẩn hóa danh sách tokens bằng cách áp dụng Snowball Stemmer tiếng Anh cho từng token.

        Args:
            tokens (list): Danh sách các token cần chuẩn hóa.

        Returns:
            list: Danh sách các token sau khi đã chuẩn hóa.
        """
        if not isinstance(tokens, list):
            return tokens
        else:
            # Khởi tạo Snowball Stemmer cho tiếng Anh
            stemmer = SnowballStemmer("english")
            # Áp dụng stemmer cho từng token trong danh sách và trả về danh sách mới
            normalized_tokens = [stemmer.stem(word) for word in tokens]
            return normalized_tokens
