# Phân Tích Hiệu Suất Mô Hình: Tại Sao Độ Chính Xác Không Vượt Qua 0.9 với Dataset Lớn?

## Tổng Quan Vấn Đề

Dựa trên phân tích chi tiết codebase và kết quả training, đã phát hiện ra một vấn đề quan trọng: **độ chính xác mô hình giảm đáng kể khi tăng kích thước dataset từ 50,000 lên 500,000 dòng, không thể vượt qua ngưỡng 0.9**.

### Kết Quả Hiện Tại (Dataset 500,000 dòng)
- **Logistic Regression**: 0.8905 accuracy (tốt nhất)
- **Random Forest**: 0.8378 accuracy  
- **Gradient Boosting**: 0.8517 accuracy

### So Sánh với Kết Quả Trước Đó (Dataset nhỏ hơn)
Theo `docs/overview.md`, với dataset nhỏ hơn đã đạt được:
- **Gradient Boosting**: 93.13% accuracy
- **SVM**: Khoảng 90%+ accuracy
- Các mô hình khác cũng đạt performance tốt hơn

---

## Các Điểm Nghi Ngờ và Phân Tích Chi Tiết

### 1. **Data Quality Issues (Chất Lượng Dữ Liệu)**

#### 🔍 **Nghi ngờ chính:**
- Dataset lớn (500K) có thể chứa nhiều **noise** và **low-quality samples** hơn
- Dữ liệu có thể bị **corrupted** hoặc **inconsistent labeling**

#### 📊 **Evidence từ code:**
```python
# Trong main.py
CONFIG = {
    "train_size": 500_000,  # Dataset lớn
    "test_size": 10000,
    # ... 
}
```

```python
# Validation logic trong data_loader cho thấy có checks cho unexpected labels
if train_invalid or test_invalid:
    print(f"Warning: Unexpected labels found")
```

#### 🧪 **Khuyến nghị kiểm tra:**
- [ ] Phân tích distribution của labels trong 500K samples
- [ ] Kiểm tra text quality: length, character encoding, duplicates
- [ ] So sánh sample quality giữa dataset nhỏ và lớn
- [ ] Thống kê missing values, empty texts

---

### 2. **Class Imbalance Problem (Mất Cân Bằng Class)**

#### 🔍 **Nghi ngờ chính:**
- Dataset 500K có thể có **severe class imbalance** 
- Tỷ lệ positive/negative samples không đều, gây bias cho model

#### 📊 **Evidence từ code:**
```python
# Model training không sử dụng class balancing
optimize_hyperparameters=False  # Không optimize class_weight
```

```python
# Logistic Regression có support cho class balancing
'class_weight': [None, 'balanced']  # Nhưng không được enable
```

#### 🧪 **Khuyến nghị kiểm tra:**
- [ ] Phân tích chi tiết distribution của 2 classes trong 500K dataset
- [ ] So sánh với distribution trong dataset nhỏ hơn
- [ ] Test với class_weight='balanced' parameter
- [ ] Sử dụng stratified sampling

---

### 3. **Feature Engineering Limitations (Hạn Chế Feature Engineering)**

#### 🔍 **Nghi ngờ chính:**
- **TF-IDF configuration** không phù hợp với dataset lớn
- **Feature dimensionality** có thể quá thấp hoặc quá cao
- **Text preprocessing** không đủ mạnh cho dataset phức tạp

#### 📊 **Evidence từ code:**
```python
# TF-IDF config trong main.py
CONFIG = {
    "tfidf_max_features": 5000,  # Có thể quá thấp cho 500K samples
    "tfidf_min_df": 2,
    "tfidf_max_df": 0.8,
    "ngram_range": (1, 2),
}
```

```python
# Trong gradient_boosting_classifier.py
max_features=30000,  # Cao hơn nhưng không được sử dụng
```

#### 🧪 **Khuyến nghị điều chỉnh:**
- [ ] Tăng `max_features` từ 5,000 lên 15,000-30,000
- [ ] Điều chỉnh `min_df` và `max_df` cho dataset lớn
- [ ] Test với trigrams: `ngram_range=(1, 3)`
- [ ] Thêm numerical features: text length, punctuation count, etc.

---

### 4. **Model Complexity vs Dataset Size (Độ Phức Tạp Mô Hình)**

#### 🔍 **Nghi ngờ chính:**
- Models quá **đơn giản** cho dataset 500K
- Không đủ **capacity** để học patterns phức tạp
- **Regularization** quá mạnh, gây underfitting

#### 📊 **Evidence từ code:**
```python
# Logistic Regression với default parameters
model = LogisticRegression(random_state=42, max_iter=1000)
# Không có hyperparameter tuning

# Random Forest với default config
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
# n_estimators có thể quá thấp

# Gradient Boosting training time rất cao
'training_time_seconds': 361.58  # Quá lâu, có thể có vấn đề
```

#### 🧪 **Khuyến nghị điều chỉnh:**
- [ ] Enable hyperparameter optimization: `optimize_hyperparameters=True`
- [ ] Tăng model complexity:
  - Random Forest: `n_estimators=300-500`
  - Gradient Boosting: `n_estimators=200`, `max_depth=8`
- [ ] Giảm regularization cho Logistic Regression

---

### 5. **Memory và Performance Constraints**

#### 🔍 **Nghi ngờ chính:**
- **Memory limitations** ảnh hưởng đến model training
- **Computational constraints** gây ra suboptimal training
- **Sparse matrix operations** không hiệu quả

#### 📊 **Evidence từ code:**
```python
# Memory usage được track
print(f"   Memory usage: ~{X_train_tfidf.data.nbytes / (1024**2):.2f} MB")

# Gradient Boosting training time quá lâu
'training_time_seconds': 361.58  # > 6 phút cho 1 model
```

#### 🧪 **Khuyến nghị:**
- [ ] Monitor memory usage during training
- [ ] Sử dụng batch training cho models lớn
- [ ] Optimize sparse matrix operations
- [ ] Consider model parallelization

---

### 6. **Train/Test Split và Cross-Validation Issues**

#### 🔍 **Nghi ngờ chính:**
- **Test set** quá nhỏ (10K) so với train set (500K)
- Không có **proper validation strategy**
- **Data leakage** có thể xảy ra

#### 📊 **Evidence từ code:**
```python
CONFIG = {
    "train_size": 500_000,
    "test_size": 10000,  # Ratio 50:1, không balanced
}

# Không có validation set
# Không có cross-validation trong main pipeline
```

#### 🧪 **Khuyến nghị:**
- [ ] Tăng test_size lên 50K-100K (ratio 10:1 or 5:1)
- [ ] Implement proper train/validation/test split
- [ ] Add k-fold cross-validation
- [ ] Ensure stratified sampling

---

### 7. **Evaluation Metrics và Bias**

#### 🔍 **Nghi ngờ chính:**
- **Accuracy** không phải metric tốt nhất cho binary classification
- **Class distribution** trong test set không representative
- **Threshold optimization** chưa được thực hiện

#### 📊 **Evidence từ code:**
```python
# Chỉ focus vào accuracy
'test_accuracy': 0.8905

# F1-scores cũng thấp
'test_f1_macro': 0.8904176753729849
'test_f1_weighted': 0.8904935833209457
```

#### 🧪 **Khuyến nghị:**
- [ ] Focus on F1-score, Precision, Recall
- [ ] Analyze per-class performance
- [ ] ROC-AUC analysis
- [ ] Threshold optimization for better balance

---

## Kế Hoạch Thử Nghiệm Cụ Thể

### Phase 1: Data Analysis
1. **Dataset Quality Check**
   ```python
   # Analyze 500K dataset
   - Label distribution
   - Text length distribution  
   - Duplicate analysis
   - Missing/empty text analysis
   ```

2. **Comparative Analysis**
   ```python
   # Compare với dataset nhỏ hơn
   - Sample 50K from 500K dataset
   - Train models on both
   - Compare results
   ```

### Phase 2: Feature Engineering Optimization
1. **TF-IDF Tuning**
   ```python
   CONFIG_NEW = {
       "tfidf_max_features": 20000,  # Tăng từ 5000
       "tfidf_min_df": 5,           # Tăng từ 2
       "tfidf_max_df": 0.9,         # Tăng từ 0.8
       "ngram_range": (1, 3),       # Thêm trigrams
   }
   ```

2. **Additional Features**
   ```python
   # Thêm numerical features
   - text_length
   - word_count  
   - punctuation_count
   - capitalization_ratio
   ```

### Phase 3: Model Optimization
1. **Hyperparameter Tuning**
   ```python
   # Enable optimization
   optimize_hyperparameters=True
   
   # Test specific configurations
   - Logistic: Higher C values, different solvers
   - Random Forest: More trees, deeper trees
   - Gradient Boosting: More conservative settings
   ```

2. **Advanced Models**
   ```python
   # Test additional models
   - XGBoost
   - LightGBM  
   - Neural Networks (if feasible)
   ```

### Phase 4: Validation Strategy
1. **Cross-Validation**
   ```python
   # Implement proper CV
   - 5-fold stratified CV
   - Time-based split if applicable
   ```

2. **Ensemble Methods**
   ```python
   # Combine best models
   - Voting classifier
   - Stacking ensemble
   ```

---

## Dự Đoán Nguyên Nhân Chính

Dựa trên phân tích, **3 nguyên nhân có khả năng cao nhất**:

1. **🥇 Feature Engineering Inadequate (40%)**
   - TF-IDF max_features=5000 quá thấp cho 500K samples
   - Cần tăng lên 15K-30K features

2. **🥈 Class Imbalance (30%)**  
   - Dataset lớn có thể có distribution không cân bằng
   - Cần class balancing strategies

3. **🥉 Model Complexity Insufficient (30%)**
   - Models với default parameters quá đơn giản
   - Cần hyperparameter tuning và model complexity tăng

---

## Kết Luận

Vấn đề performance drop khi scale up dataset là **common issue** trong ML. Các action items ưu tiên:

1. ✅ **Immediate**: Tăng TF-IDF max_features và enable hyperparameter tuning
2. ✅ **Short-term**: Phân tích data quality và class distribution  
3. ✅ **Medium-term**: Implement proper validation và ensemble methods

**Expected outcome**: Với các điều chỉnh trên, accuracy có thể cải thiện lên **0.92-0.95** cho dataset 500K.
