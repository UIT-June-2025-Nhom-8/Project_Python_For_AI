# LDA Topic Modeling Report (Amazon Reviews)

## 1. Giới thiệu

Latent Dirichlet Allocation (LDA) là mô hình chủ đề xác suất nhằm rút trích các chủ đề tiềm ẩn từ tập văn bản. Mục tiêu ở đây: huấn luyện LDA trên Amazon Reviews và đánh giá bằng các chỉ số xác suất (perplexity, log-likelihood, log-perplexity) cùng **coherence (c_v)** để đo mức độ “mạch lạc” của topic.

---

## 2. Dữ liệu & Tiền xử lý

- **Nguồn**: Amazon Reviews (train/test đã chuẩn bị sẵn).
- **Tiền xử lý**:
  - Làm sạch, chuẩn hóa chữ thường, loại ký tự nhiễu.
  - Tokenize, remove stopwords, stemming (SnowballStemmer).
  - `normalized_input`: danh sách token (list[str]).
  - Đưa vào LDA bằng **Bag-of-Words (CountVectorizer)** (không dùng TF-IDF cho LDA).

---

## 3. Mô hình & Cài đặt

- **Triển khai**: class `LDATopicModeler` (wrapper cho `sklearn.decomposition.LatentDirichletAllocation`), hỗ trợ:
  - `fit/transform/fit_transform` trên chuỗi token đã tiền xử lý.
  - `get_top_words_per_topic` để lấy top-words mỗi topic.
- **Đánh giá**:
  - **Perplexity** (sklearn): càng thấp càng tốt.
  - **Log-likelihood** (sklearn): càng cao càng tốt.
  - **Log-perplexity per-word**: tính gián tiếp từ `-log_likelihood / tổng_số_từ`.
  - **Coherence (c_v)**: tính bằng `gensim` với `CoherenceModel(topics=..., texts=..., dictionary=...)`.
- **Vectorizer (CountVectorizer)**: `max_features=20000`, `min_df=5`, `max_df=0.7`.
- **Tham số LDA** (online VB): `max_iter=20`, `learning_method="online"`, `random_state=42`.

---

## 4. Thiết kế Thực nghiệm

- So sánh theo số topic: `n_topics ∈ {10, 11, 12, …, 20}`.
- Tính trên **train** và **test**:
  - Perplexity, Log-likelihood, Log-perplexity (per-word).
  - Coherence c_v (gensim).
  - Thời gian huấn luyện (giây), kích thước vocab, #doc train.

---

## 5. Kết quả

| n_topics | Test Perplexity ↓ | Train Perplexity | Test Log-Perplx | Train Log-Perplx | Test Log-Lik ↑ | Train Log-Lik ↑ | Coherence c_v ↑ | Fit (s) | Vocab | #Train docs |
| -------: | ----------------: | ---------------: | --------------: | ---------------: | -------------: | --------------: | :-------------: | ------: | ----: | ----------: |
|       10 |           3191.39 |          2172.43 |          8.0682 |           7.6836 |    -3.1461e+06 |     -3.0038e+07 |   **0.5075**    |  570.78 | 20000 |      100000 |
|       11 |           3255.38 |          2216.27 |          8.0881 |           7.7036 |    -3.1538e+06 |     -3.0116e+07 |     0.5029      |  575.62 | 20000 |      100000 |
|   **12** |           3295.08 |          2210.72 |          8.1002 |           7.7011 |    -3.1586e+06 |     -3.0106e+07 |   **0.5513**    |  557.73 | 20000 |      100000 |
|       13 |           3397.89 |          2267.37 |          8.1309 |           7.7264 |    -3.1705e+06 |     -3.0205e+07 |     0.5327      |  583.37 | 20000 |      100000 |
|       14 |           3478.26 |          2297.54 |          8.1543 |           7.7396 |    -3.1797e+06 |     -3.0257e+07 |     0.5070      |  601.41 | 20000 |      100000 |
|       15 |           3506.42 |          2321.64 |          8.1624 |           7.7500 |    -3.1828e+06 |     -3.0297e+07 |     0.5452      |  585.10 | 20000 |      100000 |
|       16 |           3609.87 |          2374.62 |          8.1914 |           7.7726 |    -3.1941e+06 |     -3.0386e+07 |     0.4978      |  630.74 | 20000 |      100000 |
|       17 |           3643.50 |          2407.39 |          8.2007 |           7.7863 |    -3.1978e+06 |     -3.0439e+07 |     0.4934      |  599.72 | 20000 |      100000 |
|       18 |           3692.20 |          2424.39 |          8.2140 |           7.7933 |    -3.2029e+06 |     -3.0467e+07 |     0.4917      |  655.47 | 20000 |      100000 |
|       19 |           3725.68 |          2435.36 |          8.2230 |           7.7979 |    -3.2065e+06 |     -3.0484e+07 |     0.5015      |  664.37 | 20000 |      100000 |
|       20 |           3879.74 |          2518.36 |          8.2635 |           7.8314 |    -3.2223e+06 |     -3.0615e+07 |     0.5099      |  696.07 | 20000 |      100000 |

**Nhận xét nhanh:**

- **Perplexity (test)** thấp nhất ở **n_topics=10** → tốt nhất theo tiêu chí xác suất tổng quát hóa.
- **Coherence (c_v)** cao nhất ở **n_topics=12** (**0.5513**) → chủ đề dễ diễn giải nhất theo c_v.
- Khi tăng số topic:
  - Perplexity (test) **tăng dần** → rủi ro overfitting/khó tổng quát.
  - Log-perplexity per-word (test) **tăng** (xấu hơn).
  - Thời gian huấn luyện tăng (tuyến tính–siêu tuyến tính theo số topic).
- **Trade-off**: n=10 tối ưu perplexity; n=12 tối ưu coherence. Chọn theo mục tiêu:
  - Ưu tiên **fit xác suất / generalization** → **n=10**.
  - Ưu tiên **tính mạch lạc/chủ đề dễ hiểu** → **n=12**.

---

## 6. Diễn giải Chủ đề (n_topics = 12)

> Lưu ý: Top-words đã được stem (ví dụ: _movi_ ~ _movie_, _stori_ ~ _story_), nhằm gộp các biến thể từ vựng.

**Top-words theo từng chủ đề:**

- **Topic 01** — book, read, life, one, peopl, work, histori, world, interest, us
- **Topic 02** — work, game, use, product, one, get, would, buy, time, great
- **Topic 03** — like, dont, get, love, old, buy, money, go, one, time
- **Topic 04** — use, one, product, great, good, work, would, like, look, get
- **Topic 05** — album, song, cd, music, like, one, listen, great, sound, good
- **Topic 06** — love, famili, stori, life, beauti, great, live, fun, enjoy, wonder
- **Topic 07** — book, read, charact, stori, one, like, end, novel, bore, good
- **Topic 08** — book, use, inform, good, help, learn, need, would, author, look
- **Topic 09** — movi, film, watch, one, like, good, time, see, great, bad
- **Topic 10** — great, movi, good, best, one, fan, like, seri, action, star
- **Topic 11** — book, read, great, one, love, stori, time, like, good, would
- **Topic 12** — dvd, version, movi, video, qualiti, edit, pictur, buy, great, watch

**Gợi ý gán nhãn nhanh (tùy bối cảnh):**

- Books/Reading: 01, 07, 08, 11
- General Product/Usage: 02, 04
- Sentiment/Purchase (chung chung): 03
- Music/Album: 05
- Family/Life Stories: 06
- Movies/Video: 09, 10, 12

> Lưu ý thêm: Một số topic có nhiều từ phổ dụng (_one, like, good_), làm giảm độ sắc nét. Có thể giảm `max_df`, tăng `min_df` hoặc tăng `max_features` trong CountVectorizer để giảm nhiễu và làm rõ ranh giới chủ đề.

Gợi ý gán nhãn (tùy ngữ cảnh): **Books/Reading**, **General Product/Usage**, **Sentiment/Purchase**, **Music/Album**, **Movies/Video**… Một số topic vẫn chứa từ chung chung (_one, like, good_), làm giảm độ sắc nét.

## 7. Kết luận: Lựa chọn số lượng chủ đề tối ưu

Dựa trên bảng so sánh:

- **Perplexity**: tăng dần khi `n_topics` tăng ⇒ mô hình càng phức tạp, khó tổng quát hóa.
- **Coherence (c_v)**: cao nhất tại **n_topics = 12** (≈0.55), cho thấy mức gắn kết chủ đề tốt nhất trong các mô hình thử nghiệm.
- **Log-likelihood**: xu hướng giảm nhưng ổn định quanh n=12–15.

Với cả ba chỉ số, **n_topics = 12** là lựa chọn hợp lý và cân bằng nhất.

---

## 8. Hạn chế & Thảo luận

- Perplexity trên test tăng theo số topic → lựa chọn `n_topics` lớn **không** cải thiện tổng quát hóa.
- Coherence cao nhất ở n=12 nhưng không chênh lệch nhiều so với n=10/15 → độ mạch lạc cải thiện có giới hạn.
- Dữ liệu review ngắn, ngôn ngữ đa dạng, nhiều từ phổ dụng → LDA (BoW) khó nắm bắt semantics sâu.
- Nếu mục tiêu là **topic dễ diễn giải**: coherence là thước đo chính; nếu mục tiêu là **khả năng mô hình hóa xác suất**: ưu tiên perplexity/log-perplexity.

---
