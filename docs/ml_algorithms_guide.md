# Machine Learning Algorithms cho Sentiment Analysis

Dự án này triển khai các thuật toán machine learning riêng biệt để phân tích cảm xúc trên bộ dữ liệu Amazon Reviews.

## 📁 Cấu trúc thư mục

```
src/
├── main.py                           # File chính chạy toàn bộ pipeline
├── preprocessor.py                   # Xử lý tiền dữ liệu
├── sentiment_analyzer.py             # VADER sentiment analysis
├── topic_modeling.py                 # Topic modeling (LDA, LSA)
├── gradient_boosting_classifier.py   # GradientBoosting classifier
├── lgbm_classifier.py               # LightGBM classifier  
├── random_forest_classifier.py      # RandomForest classifier
└── demo_ml_algorithms.py            # Demo test các thuật toán
```

## 🛠️ Cài đặt

1. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

2. Cài đặt LightGBM (nếu cần):
```bash
pip install lightgbm
```

## 🚀 Cách sử dụng

### 1. Chạy toàn bộ pipeline
```bash
cd src
python main.py
```

### 2. Test từng thuật toán riêng biệt
```bash
cd src
python demo_ml_algorithms.py
```

### 3. Sử dụng từng classifier riêng

#### GradientBoostingClassifier
```python
from gradient_boosting_classifier import GradientBoostingAnalyzer

# Khởi tạo
gb_analyzer = GradientBoostingAnalyzer()

# Chuẩn bị dữ liệu
gb_analyzer.prepare_data(df, 'text_column')

# Huấn luyện
results = gb_analyzer.train_model()

# Đánh giá
gb_analyzer.evaluate_model()

# Dự đoán
predictions = gb_analyzer.predict(["Great product!", "Bad quality"])

# Lưu model
gb_analyzer.save_model('models/my_gb_model.pkl')
```

#### LGBMClassifier
```python
from lgbm_classifier import LGBMAnalyzer

# Khởi tạo
lgbm_analyzer = LGBMAnalyzer()

# Chuẩn bị dữ liệu
lgbm_analyzer.prepare_data(df, 'text_column')

# Huấn luyện với early stopping
results = lgbm_analyzer.train_with_early_stopping()

# Đánh giá
lgbm_analyzer.evaluate_model()

# Dự đoán
predictions = lgbm_analyzer.predict(["Excellent!", "Terrible"])
```

#### RandomForestClassifier
```python
from random_forest_classifier import RandomForestAnalyzer

# Khởi tạo
rf_analyzer = RandomForestAnalyzer()

# Chuẩn bị dữ liệu
rf_analyzer.prepare_data(df, 'text_column')

# Huấn luyện với cross-validation
results = rf_analyzer.train_with_cross_validation(cv=5)

# Xem feature importance
importance = rf_analyzer.get_feature_importance(top_n=20)
print(importance)

# Dự đoán xác suất
probabilities = rf_analyzer.predict_proba(["Amazing product!"])
```

## 🔧 Tùy chỉnh tham số

### GradientBoostingClassifier
```python
gb_analyzer.initialize_model(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
```

### LGBMClassifier
```python
lgbm_analyzer.initialize_model(
    num_leaves=50,
    learning_rate=0.05,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    n_estimators=150
)
```

### RandomForestClassifier
```python
rf_analyzer.initialize_model(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt'
)
```

## 📊 Kết quả mẫu

Sau khi chạy, bạn sẽ thấy kết quả so sánh như:

```
=== SO SÁNH KẾT QUẢ CÁC MODELS ===
Model              Train Accuracy    Test Accuracy
GradientBoosting   0.9823           0.9156
LightGBM           0.9756           0.9187  
RandomForest       0.9891           0.9123

🏆 Model tốt nhất: LightGBM (Test Accuracy: 0.9187)
```

## 📁 Models được lưu

Tất cả models được lưu trong thư mục `models/`:
- `gradient_boosting_model.pkl`
- `lightgbm_model.pkl`  
- `random_forest_model.pkl`

## 🔄 Load model đã lưu

```python
# Load GradientBoosting model
gb_analyzer = GradientBoostingAnalyzer()
gb_analyzer.load_model('models/gradient_boosting_model.pkl')

# Sử dụng ngay để dự đoán
prediction = gb_analyzer.predict(["This product is awesome!"])
```

## ⚙️ Các tính năng chính

### 1. GradientBoostingAnalyzer
- ✅ Huấn luyện và đánh giá model
- ✅ Dự đoán và xác suất
- ✅ Lưu/tải model
- ✅ Feature importance
- ✅ Thông tin chi tiết model

### 2. LGBMAnalyzer  
- ✅ Huấn luyện với early stopping
- ✅ Tối ưu hóa tự động
- ✅ Feature importance
- ✅ Cross-validation support

### 3. RandomForestAnalyzer
- ✅ Cross-validation training  
- ✅ Feature importance analysis
- ✅ Parallel processing
- ✅ Robust prediction

## 🐛 Xử lý lỗi

### LightGBM không cài đặt được:
```bash
# macOS
brew install libomp
pip install lightgbm

# Linux
apt-get install libgomp1
pip install lightgbm

# Windows
pip install lightgbm
```

### Lỗi memory:
- Giảm `max_features` trong TfidfVectorizer
- Giảm kích thước dataset
- Sử dụng `n_jobs=1` thay vì `-1`

## 🎯 Lưu ý quan trọng

1. **Dữ liệu cần có cột 'sentiment'** với các giá trị: 'Positive', 'Negative', 'Neutral'
2. **Text cần được tiền xử lý** trước khi đưa vào model
3. **Models được lưu bao gồm cả vectorizer và label encoder**
4. **Mỗi analyzer hoạt động độc lập** và có thể sử dụng riêng biệt

## 🔍 Monitoring và Debug

Mỗi analyzer cung cấp thông tin chi tiết:
```python
# Xem thông tin model
model_info = analyzer.get_model_info()
print(model_info)

# Xem kết quả huấn luyện
print(analyzer.results)
```

## 📈 Mở rộng

Để thêm thuật toán mới, tạo file analyzer mới theo pattern:
1. Kế thừa các phương thức cơ bản
2. Implement `initialize_model()`, `train_model()`, `evaluate_model()`  
3. Thêm vào `main.py` để chạy cùng pipeline

---

**Tác giả**: Nhóm 8 - UIT June 2025  
**Liên hệ**: [GitHub Repository](https://github.com/UIT-June-2025-Nhom-8/Project_Python_For_AI)
