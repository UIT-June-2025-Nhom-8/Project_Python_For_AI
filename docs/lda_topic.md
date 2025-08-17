# 📊 LDA Topic Modeling Report

## 1. Giới thiệu
Latent Dirichlet Allocation (LDA) là một phương pháp phân tích chủ đề tiềm ẩn (topic modeling) thường được áp dụng trên tập dữ liệu văn bản lớn. Trong bối cảnh này, LDA được thử nghiệm trên tập Amazon Reviews nhằm rút trích các chủ đề chính và đánh giá khả năng của mô hình.

## 2. Dữ liệu & Tiền xử lý
- **Nguồn dữ liệu**: Amazon Reviews (tập con đã được tiền xử lý).  
- **Các bước xử lý**:  
  - Chuẩn hoá văn bản (lowercase, loại bỏ ký tự đặc biệt).  
  - Tokenize thành danh sách từ.  
  - Biến đổi `normalized_input` (list token) thành chuỗi để đưa vào Bag-of-Words.

## 3. Mô hình & Cài đặt
### 3.1 Class `LDATopicModeler`
Một lớp bao bọc `sklearn.decomposition.LatentDirichletAllocation` được xây dựng để:  
- Huấn luyện và suy diễn phân phối chủ đề (`fit`, `transform`, `fit_transform`).  
- Trích xuất top-words cho từng topic (`get_top_words_per_topic`).  
- Gán nhãn topic tự động theo từ khoá (`auto_label_topics`).  
- Lưu/khôi phục mô hình và từ vựng (`save`, `load`).  

### 3.2 Grid-search theo số lượng topics
Hàm `run_lda_experiments` cho phép chạy LDA với nhiều giá trị `n_topics` và ghi lại các chỉ số:  
- Train/Test Perplexity (càng thấp càng tốt).  
- Train/Test Log-likelihood (càng cao càng tốt).  
- Thời gian huấn luyện.  

## 4. Thiết kế Thực nghiệm
- **Số lượng topics thử nghiệm**: `[10, 15, 20, 30, 50]`.  
- **CountVectorizer**:  
  - `max_features=20000`  
  - `min_df=5`  
  - `max_df=0.7`  
- **Tham số LDA**:  
  - `max_iter=20`  
  - `learning_method="online"`  
  - `random_state=42`  

## 5. Kết quả
Bảng kết quả thực nghiệm:

| n_topics | Test Perplexity | Train Perplexity | Test Log-likelihood | Train Log-likelihood | Fit Seconds | Vocab Size | n_train_docs |
|---------:|----------------:|-----------------:|--------------------:|---------------------:|------------:|-----------:|-------------:|
| 10 | 3164.49 | 2159.89 | -3.14e+06 | -3.00e+07 | 371.56 | 20000 | 100000 |
| 15 | 3501.49 | 2308.35 | -3.18e+06 | -3.02e+07 | 434.95 | 20000 | 100000 |
| 20 | 3864.58 | 2502.98 | -3.22e+06 | -3.05e+07 | 498.45 | 20000 | 100000 |
| 30 | 4410.71 | 2808.21 | -3.27e+06 | -3.10e+07 | 613.50 | 20000 | 100000 |
| 50 | 40713.84 | 29798.52 | -4.13e+06 | -4.02e+07 | 951.65 | 20000 | 100000 |

**Phân tích**:  
- Khi tăng số lượng topics, test perplexity tăng mạnh, cho thấy mô hình không khái quát tốt trên dữ liệu chưa thấy.  
- Log-likelihood giảm dần trên test set, hiệu năng mô hình không cải thiện theo số topics.  
- Với `n_topics=10`, perplexity thấp nhất và thời gian huấn luyện ngắn nhất.  

## 6. Diễn giải Chủ đề

Top-words theo từng chủ đề (LDA với `n_topics=12`, từ đã được stem):

Topic 01: book, read, life, one, peopl, work, histori, world, interest, us  
Topic 02: work, game, use, product, one, get, would, buy, time, great  
Topic 03: like, dont, get, love, old, buy, money, go, one, time  
Topic 04: use, one, product, great, good, work, would, like, look, get  
Topic 05: album, song, cd, music, like, one, listen, great, sound, good  
Topic 06: love, famili, stori, life, beauti, great, live, fun, enjoy, wonder  
Topic 07: book, read, charact, stori, one, like, end, novel, bore, good  
Topic 08: book, use, inform, good, help, learn, need, would, author, look  
Topic 09: movi, film, watch, one, like, good, time, see, great, bad  
Topic 10: great, movi, good, best, one, fan, like, seri, action, star  
Topic 11: book, read, great, one, love, stori, time, like, good, would  
Topic 12: dvd, version, movi, video, qualiti, edit, pictur, buy, great, watch


Một số nhóm chủ đề có ý nghĩa rõ rệt như **sách, âm nhạc, phim ảnh, sản phẩm**. Tuy nhiên, nhiều topic chứa từ ngữ chung chung (như *one, like, good*), làm giảm khả năng diễn giải.

## 7. Hạn chế & Thảo luận
- Perplexity và log-likelihood đều ở mức cao, cho thấy LDA chưa phù hợp để nắm bắt ngữ nghĩa trong dữ liệu review.  
- Chủ đề trích xuất đôi khi khó đặt nhãn, nhiều từ nhiễu xuất hiện.  
- Nguyên nhân:  
  - Văn bản review ngắn, ngôn ngữ đa dạng, chứa nhiều từ phổ biến.  
  - LDA dựa trên Bag-of-Words, thiếu khả năng xử lý ngữ cảnh và semantics.  
