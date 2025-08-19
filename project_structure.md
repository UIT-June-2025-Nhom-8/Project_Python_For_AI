# Cấu Trúc và Chức Năng Project Amazon Reviews Analysis

## 1. Tổng quan dự án

Dự án "Amazon Reviews Analysis" là một hệ thống phân tích tự động các đánh giá sản phẩm Amazon sử dụng các kỹ thuật Machine Learning và Natural Language Processing. Mục tiêu chính là thực hiện phân tích cảm xúc (Sentiment Analysis) và mô hình hóa chủ đề (Topic Modeling) để hiểu rõ hơn về ý kiến khách hàng.

## 2. Cấu trúc thư mục chi tiết

```
Project_Python_For_AI/
├── .git/                          # Hệ thống quản lý phiên bản Git
├── .venv/                         # Môi trường ảo Python
├── .gitignore                     # Quy tắc bỏ qua file cho Git
├── README.MD                      # Tài liệu hướng dẫn dự án
├── requirements.txt               # Danh sách thư viện Python cần thiết
├── example.kaggle.json           # Template cấu hình Kaggle API
│
├── docs/                          # Thư mục tài liệu
│   ├── overview.md                # Tổng quan dự án chi tiết
│   ├── gensim_lda_report.md      # Báo cáo phân tích Gensim LDA
│   ├── lda_topic.md              # Phân tích Topic Modeling
│   └── preprocessing_analyse.md   # Phân tích tiền xử lý dữ liệu
│
├── notebook/                      # Thư mục Jupyter Notebooks
│   ├── main.ipynb                # Notebook phân tích chính
│   ├── 0.data_downloader.ipynb   # Notebook tải dữ liệu
│   └── amazon_reviews_lda_gensim_topic_modeling.ipynb
│
├── src/                           # Thư mục mã nguồn
│   ├── main.py                    # Script chính thực thi pipeline
│   ├── kaggle_data_loader.py     # Module tải dữ liệu từ Kaggle
│   ├── local_data_loader.py      # Module tải dữ liệu từ file local
│   ├── pre_processor.py          # Module tiền xử lý văn bản
│   ├── text_analyzer.py          # Module phân tích văn bản
│   ├── stopwords_config.py       # Cấu hình từ dừng
│   ├── tf_idf_vectorizer.py      # Implementation TF-IDF
│   ├── sklearn_lda.py            # LDA với Scikit-learn
│   ├── gensim_lda.py             # LDA với Gensim
│   ├── lda_utils.py              # Tiện ích cho LDA
│   ├── model_trainer.py          # Module huấn luyện mô hình
│   ├── logistic_regression_classifier.py
│   ├── random_forest_classifier.py
│   ├── gradient_boosting_classifier.py
│   │
│   ├── images/                    # Thư mục lưu trữ hình ảnh
│   │   ├── wordcloud_train.png
│   │   ├── wordcloud_test_*.png
│   │   ├── wordcloud_*_positive.png
│   │   ├── wordcloud_*_negative.png
│   │   └── gensim_lda/
│   │       ├── lda.png
│   │       └── Coherence_perpelexity_analyst.png
│   │
│   └── reports/                   # Báo cáo huấn luyện
│       └── model_training_results_*.json
│
├── output/                        # Thư mục kết quả đầu ra
│   └── models/
│       └── tfidf_vectorizer.pkl
│
└── reports/                       # Báo cáo cuối cùng
    └── model_training_results_*.json
```

## 3. Mô tả chức năng từng thành phần

### 3.1 Thư mục tài liệu (docs/)

**overview.md**

- Tài liệu tổng quan toàn diện về dự án
- Mô tả mục tiêu, phương pháp AI Thinking
- Phân tích kết quả và đánh giá hiệu suất
- Khuyến nghị phát triển tương lai

**gensim_lda_report.md**

- Báo cáo chi tiết về Topic Modeling sử dụng thư viện Gensim
- Phân tích coherence score và perplexity
- Visualization các chủ đề được tìm thấy

**lda_topic.md**

- Phân tích chuyên sâu các chủ đề được phát hiện
- Mô tả nội dung từng topic
- Đánh giá tính hợp lý của phân loại

**preprocessing_analyse.md**

- Phân tích quá trình tiền xử lý dữ liệu
- Đánh giá hiệu quả các bước cleaning
- Thống kê về dữ liệu sau xử lý

### 3.2 Thư mục Notebooks (notebook/)

**main.ipynb**

- Notebook chính chứa toàn bộ pipeline phân tích
- Kết hợp tất cả các bước từ loading data đến đánh giá mô hình
- Visualization và báo cáo kết quả

**0.data_downloader.ipynb**

- Notebook chuyên dụng để tải dữ liệu từ Kaggle
- Cấu hình API key và download dataset
- Kiểm tra tính toàn vẹn dữ liệu

**amazon_reviews_lda_gensim_topic_modeling.ipynb**

- Phân tích LDA chuyên sâu sử dụng Gensim
- Tối ưu hóa số lượng topics
- Visualization interactive với pyLDAvis

### 3.3 Thư mục mã nguồn (src/)

#### 3.3.1 Module xử lý dữ liệu

**kaggle_data_loader.py**

- Class KaggleDataLoader để tải dữ liệu từ Kaggle API
- Xử lý authentication và download tự động
- Chuẩn bị dataframe cho các bước tiếp theo

**local_data_loader.py**

- Class LocalDataLoader để tải dữ liệu từ file local
- Hỗ trợ nhiều định dạng file (CSV, JSON, etc.)
- Backup option khi không có kết nối internet

**pre_processor.py**

- Class PreProcessor cho tiền xử lý văn bản
- Text cleaning: loại bỏ URL, HTML tags, punctuation
- Tokenization, stopwords removal, stemming/lemmatization
- Normalization và standardization

**text_analyzer.py**

- Module phân tích sentiment sử dụng VADER
- Tính toán các metrics văn bản (word count, sentence length)
- Phân loại cảm xúc: positive, negative, neutral

**stopwords_config.py**

- Cấu hình danh sách từ dừng tiếng Anh
- Customizable stopwords cho domain cụ thể
- Hỗ trợ thêm/bớt từ dừng theo yêu cầu

#### 3.3.2 Module Feature Engineering

**tf_idf_vectorizer.py**

- Implementation TF-IDF vectorization
- Tùy chỉnh max_features, min_df, max_df
- Hỗ trợ n-gram features

#### 3.3.3 Module Topic Modeling

**sklearn_lda.py**

- Class SKLearnLDATopicModeler sử dụng Scikit-learn
- Latent Dirichlet Allocation implementation
- Auto-labeling topics dựa trên keywords
- Evaluation metrics: perplexity, log-likelihood

**gensim_lda.py**

- Class GensimLDATopicModeler sử dụng Gensim
- Advanced LDA với coherence optimization
- Interactive visualization với pyLDAvis
- Bigram/trigram detection

**lda_utils.py**

- Tiện ích hỗ trợ LDA analysis
- Function run_lda_experiments cho grid search
- Plotting coherence và perplexity curves
- Comparison utilities cho multiple models

#### 3.3.4 Module Machine Learning

**model_trainer.py**

- Class ModelTrainer để huấn luyện các mô hình ML
- Pipeline tự động: preprocessing -> training -> evaluation
- Support multiple algorithms
- Cross-validation và hyperparameter tuning

**logistic_regression_classifier.py**

- Implementation Logistic Regression cho sentiment classification
- Feature importance analysis
- Regularization options (L1, L2)

**random_forest_classifier.py**

- Random Forest classifier với ensemble learning
- Feature importance ranking
- Out-of-bag score evaluation

**gradient_boosting_classifier.py**

- Gradient Boosting implementation
- Đạt hiệu suất cao nhất: 93.13% accuracy
- Learning curve analysis

#### 3.3.5 Main Pipeline

**main.py**

- Script chính để chạy toàn bộ pipeline
- Orchestrate tất cả các module
- Configuration management
- Logging và error handling

### 3.4 Thư mục hình ảnh (src/images/)

**Word Clouds**

- Visualization từ phổ biến theo sentiment
- wordcloud_train.png: Word cloud từ training data
- wordcloud\_\*\_positive.png: Từ phổ biến trong reviews tích cực
- wordcloud\_\*\_negative.png: Từ phổ biến trong reviews tiêu cực

**LDA Visualizations**

- lda.png: Biểu đồ phân bố topics
- Coherence_perpelexity_analyst.png: Phân tích model performance

### 3.5 Thư mục đầu ra (output/ và reports/)

**output/models/**

- tfidf_vectorizer.pkl: TF-IDF vectorizer đã được huấn luyện
- Các mô hình khác sau khi save

**reports/**

- model*training_results*\*.json: Kết quả đánh giá hiệu suất model
- Performance metrics cho tất cả algorithms
- Timestamp để tracking experiments

## 4. Thư viện và Dependencies

Dựa trên file requirements.txt, dự án sử dụng các thư viện chính:

### 4.1 Data Processing

- **pandas (>=1.5.0)**: Xử lý và manipulate dữ liệu
- **numpy (>=1.21.0)**: Tính toán số học và ma trận

### 4.2 Machine Learning

- **scikit-learn (>=1.1.0)**: Thuật toán ML cơ bản
- **lightgbm (>=3.3.0)**: Gradient boosting framework

### 4.3 Natural Language Processing

- **nltk (>=3.7)**: Toolkit xử lý ngôn ngữ tự nhiên
- **gensim (>=4.2.0)**: Topic modeling và word embeddings
- **vaderSentiment (>=3.3.0)**: Sentiment analysis

### 4.4 Visualization

- **matplotlib (>=3.5.0)**: Plotting cơ bản
- **seaborn (>=0.11.0)**: Statistical visualization
- **plotly (>=5.0.0)**: Interactive plots
- **wordcloud (>=1.9.0)**: Word cloud generation

### 4.5 Data Source & Development

- **kaggle (>=1.5.0)**: Kaggle API client
- **kagglehub**: Kaggle dataset hub
- **jupyter (>=1.0.0)**: Jupyter notebook environment
- **ipykernel (>=6.0.0)**: IPython kernel for Jupyter

### 4.6 Utilities

- **tqdm (>=4.64.0)**: Progress bars cho long-running processes

## 5. Workflow và Pipeline

### 5.1 Data Pipeline

1. **Data Loading**: Kaggle API hoặc local files
2. **Data Cleaning**: Loại bỏ duplicates, handle missing values
3. **Text Preprocessing**: Cleaning, tokenization, normalization
4. **Feature Engineering**: TF-IDF vectorization

### 5.2 Analysis Pipeline

1. **Sentiment Analysis**: VADER sentiment classification
2. **Topic Modeling**: LDA với Gensim và Scikit-learn
3. **Model Training**: Multiple ML algorithms
4. **Evaluation**: Performance metrics và comparison

### 5.3 Output Pipeline

1. **Visualization**: Word clouds, topic distributions
2. **Model Saving**: Pickle serialization
3. **Reporting**: JSON format results
4. **Documentation**: Markdown reports

## 6. Kết quả chính

### 6.1 Sentiment Classification

- **Best Model**: Gradient Boosting Classifier
- **Accuracy**: 93.13% trên test set
- **Features**: TF-IDF với 5000 max features

### 6.2 Topic Modeling

- **Optimal Topics**: 16 topics (Gensim LDA)
- **Coherence Score**: 0.5141
- **Main Categories**: Thiết bị công nghệ Amazon, dịch vụ streaming, chất lượng âm thanh/hiển thị

### 6.3 Business Impact

- Tự động phân loại feedback khách hàng
- Cải thiện hệ thống recommendation
- Market research và business intelligence
- Reduced manual labeling effort

## 7. Ứng dụng thực tế

Dự án có thể được áp dụng cho:

- **E-commerce platforms**: Phân tích review tự động
- **Customer service**: Routing complaints theo priority
- **Product development**: Understanding customer needs
- **Marketing research**: Trend analysis và customer insights
- **Quality assurance**: Identifying product issues từ reviews

## 8. Hướng phát triển

### 8.1 Cải tiến ngắn hạn

- Hyperparameter tuning với Grid Search
- Cross-validation cho robust evaluation
- Advanced feature engineering
- Model ensemble techniques

### 8.2 Phát triển dài hạn

- Deep Learning models (BERT, Transformers)
- Real-time processing pipeline
- Multi-language support
- Interactive web dashboard
- Cloud deployment và scaling
