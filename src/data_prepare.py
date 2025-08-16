import pandas as pd
import os
import sys
from pathlib import Path

# Thêm import kagglehub để download data
try:
    import kagglehub
except ImportError:
    print("⚠️  kagglehub chưa được cài đặt. Cài đặt bằng: pip install kagglehub")


class DataPrepare:
    """
    Class để chuẩn bị dữ liệu cho dự án phân tích cảm xúc Amazon Reviews
    Hỗ trợ pull data từ Kaggle hoặc sử dụng data local
    """
    
    def __init__(self, run_on_kaggle=False):
        """
        Khởi tạo DataPrepare
        
        Args:
            run_on_kaggle (bool): True nếu chạy trên Kaggle, False nếu chạy local
        """
        self.run_on_kaggle = run_on_kaggle
        self.base_path = "/kaggle/input" if run_on_kaggle else "../data"
        self.working_path = "/kaggle/working" if run_on_kaggle else "../data"
        
    def download_amazon_reviews_from_kaggle(self):
        """
        Download Amazon Reviews dataset từ Kaggle sử dụng kagglehub
        
        Returns:
            tuple: (train_path, test_path) hoặc (None, None) nếu fail
        """
        print("📥 Đang download Amazon Reviews dataset từ Kaggle...")
        
        try:
            import kagglehub
            
            # Download dataset sử dụng kagglehub
            kritanjalijain_amazon_reviews_path = kagglehub.dataset_download(
                "kritanjalijain/amazon-reviews"
            )
            print(f"KaggleHub download path: {kritanjalijain_amazon_reviews_path}")
            
            # Tìm các file train và test
            train_path = os.path.join(kritanjalijain_amazon_reviews_path, "train.csv")
            test_path = os.path.join(kritanjalijain_amazon_reviews_path, "test.csv")
            
            print(f"🔍 Kiểm tra files:")
            print(f"  - Train: {train_path} {'✅' if os.path.exists(train_path) else '❌'}")
            print(f"  - Test: {test_path} {'✅' if os.path.exists(test_path) else '❌'}")
            
            if os.path.exists(train_path) and os.path.exists(test_path):
                print("✅ Download thành công từ Kaggle!")
                return train_path, test_path
            else:
                print("❌ Không tìm thấy file sau khi download")
                # List files trong thư mục để debug
                if os.path.exists(kritanjalijain_amazon_reviews_path):
                    files = os.listdir(kritanjalijain_amazon_reviews_path)
                    print(f"📁 Files có trong {kritanjalijain_amazon_reviews_path}: {files}")
                return None, None
                
        except ImportError:
            print("❌ kagglehub package không được cài đặt. Cài đặt bằng: pip install kagglehub")
            return None, None
        except Exception as e:
            print(f"❌ Lỗi download từ Kaggle: {e}")
            print(f"❌ Chi tiết lỗi: {type(e).__name__}: {str(e)}")
            return None, None
    
    def get_local_data_paths(self):
        """
        Lấy đường dẫn data local
        
        Returns:
            tuple: (train_path, test_path)
        """
        train_path = os.path.join(self.base_path, "amazon_reviews", "train.csv")
        test_path = os.path.join(self.base_path, "amazon_reviews", "test.csv")
        
        return train_path, test_path
    
    def prepare_data(self):
        """
        Chuẩn bị data chính - method chính để gọi từ main
        
        Returns:
            tuple: (train_path, test_path, data_info)
        """
        print("🔍 Đang chuẩn bị dữ liệu...")
        
        if self.run_on_kaggle:
            print("🌐 Chạy trên Kaggle - download data từ Kaggle...")
            train_path, test_path = self.download_amazon_reviews_from_kaggle()
            
            if train_path and test_path and os.path.exists(train_path) and os.path.exists(test_path):
                data_info = {
                    "source": "kaggle_download",
                    "train_samples": len(pd.read_csv(train_path)),
                    "test_samples": len(pd.read_csv(test_path))
                }
                return train_path, test_path, data_info
            else:
                raise RuntimeError("❌ KHÔNG THỂ DOWNLOAD DATA TỪ KAGGLE! " +
                                 "Kiểm tra lại: \n" +
                                 "1. Kaggle API credentials\n" +
                                 "2. Kết nối internet\n" + 
                                 "3. Dataset có tồn tại không\n" +
                                 "4. Quyền truy cập dataset")
        else:
            print("💻 Chạy local - sử dụng data từ thư mục data/amazon_reviews")
            train_path, test_path = self.get_local_data_paths()
            
            if os.path.exists(train_path) and os.path.exists(test_path):
                data_info = {
                    "source": "local_files",
                    "train_samples": len(pd.read_csv(train_path)),
                    "test_samples": len(pd.read_csv(test_path))
                }
                return train_path, test_path, data_info
            else:
                raise FileNotFoundError("❌ KHÔNG TÌM THẤY DATA LOCAL! " +
                                      f"Cần có files: \n" +
                                      f"- {train_path}\n" +
                                      f"- {test_path}")
    
    def prepare_data_strict(self):
        """
        Alias cho prepare_data để tương thích với code cũ
        
        Returns:
            tuple: (train_path, test_path, data_info)
        """
        return self.prepare_data()
    
    def get_data_info(self, train_path, test_path):
        """
        Lấy thông tin chi tiết về dataset
        
        Args:
            train_path (str): Đường dẫn file train
            test_path (str): Đường dẫn file test
            
        Returns:
            dict: Thông tin dataset
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            # Set column names nếu chưa có
            if train_df.columns.tolist() == [0, 1, 2]:
                train_df.columns = ["label", "title", "text"]
                test_df.columns = ["label", "title", "text"]
            
            info = {
                "train_samples": len(train_df),
                "test_samples": len(test_df),
                "train_columns": train_df.columns.tolist(),
                "test_columns": test_df.columns.tolist(),
                "train_labels": train_df['label'].value_counts().to_dict() if 'label' in train_df.columns else {},
                "test_labels": test_df['label'].value_counts().to_dict() if 'label' in test_df.columns else {},
                "train_path": train_path,
                "test_path": test_path
            }
            
            return info
            
        except Exception as e:
            print(f"❌ Lỗi đọc thông tin data: {e}")
            return {"error": str(e)}
