import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

def _tokens_to_text_list(series_or_list):
    """
    Join list-of-tokens -> string cho CountVectorizer.
    """
    if isinstance(series_or_list, pd.Series):
        series_or_list = series_or_list.tolist()
    out = []
    for x in series_or_list:
        out.append(" ".join(x) if isinstance(x, list) else str(x))
    return out


def run_lda_experiments(
    n_topics_list,
    train_tokens,
    test_tokens,
    base_params=None,
    vectorizer_params=None,
):
    """
    Chạy LDA cho nhiều giá trị n_topics và trả về DataFrame so sánh.
    Tính train/test Perplexity, Log-likelihood, Log-perplexity (per-word)
    và Coherence (c_v nếu có gensim).
    """
    from sklearn_lda import SKLearnLDATopicModeler as LDATopicModeler
    train_texts = _tokens_to_text_list(train_tokens)
    test_texts  = _tokens_to_text_list(test_tokens)

    results = []
    for k in n_topics_list:
        print(f"\n>>> Running LDA with n_topics={k}")
        params = dict(
            n_topics=k,
            max_iter=20,
            learning_method="online",
            random_state=42,
            evaluate_every=-1,
        )
        if base_params:
            params.update(base_params)

        lda = LDATopicModeler(
            n_topics=params["n_topics"],
            max_features=None,
            min_df=5,
            max_df=0.8,
            random_state=params["random_state"],
            max_iter=params["max_iter"],
            learning_method=params["learning_method"],
            evaluate_every=params["evaluate_every"],
        )
        if vectorizer_params:
            lda.vectorizer.set_params(**vectorizer_params)

        t0 = time.time()
        doc_topic_train = lda.fit_transform(train_texts)
        doc_topic_test  = lda.transform(test_texts)
        fit_secs = time.time() - t0

        X_train_bow = lda.vectorizer.transform(train_texts)
        X_test_bow  = lda.vectorizer.transform(test_texts)

        train_perp = lda.model.perplexity(X_train_bow)
        test_perp  = lda.model.perplexity(X_test_bow)
        train_ll   = lda.model.score(X_train_bow)
        test_ll    = lda.model.score(X_test_bow)

        n_words_train = float(X_train_bow.sum())
        n_words_test  = float(X_test_bow.sum())
        train_log_perp = -train_ll / n_words_train if n_words_train > 0 else np.nan
        test_log_perp  = -test_ll / n_words_test  if n_words_test  > 0 else np.nan

        coherence_cv = np.nan
        try:
            from gensim.corpora import Dictionary
            from gensim.models.coherencemodel import CoherenceModel
            top_words = lda.get_top_words_per_topic(top_n=10)
            topics_words = [[w for (w, _) in topic] for topic in top_words]
            texts_tokens_for_coh = (
                train_tokens.tolist()
                if isinstance(train_tokens, pd.Series)
                else train_tokens
            )
            dictionary = Dictionary(texts_tokens_for_coh)
            cm = CoherenceModel(
                topics=topics_words,
                texts=texts_tokens_for_coh,
                dictionary=dictionary,
                coherence="c_v",
            )
            coherence_cv = cm.get_coherence()
        except Exception as e:
            print(f"(Skip coherence for n_topics={k}: {e})")

        results.append(
            {
                "n_topics": k,
                "test_perplexity": test_perp,
                "train_perplexity": train_perp,
                "test_log_perplexity": test_log_perp,
                "train_log_perplexity": train_log_perp,
                "test_log_likelihood": test_ll,
                "train_log_likelihood": train_ll,
                "coherence_c_v": coherence_cv,
                "fit_seconds": fit_secs,
                "vocab_size": len(lda.vectorizer.get_feature_names_out()),
                "n_train_docs": X_train_bow.shape[0],
            }
        )

    df = pd.DataFrame(results)
    df = df.sort_values(by=["test_perplexity"], ascending=True).reset_index(drop=True)
    return df

def plot_coherence_and_perplexity(
    lda_grid_df: pd.DataFrame,
    *,
    x_col: str = "n_topics",
    coherence_col: str = "coherence_c_v",
    log_perp_col: str = "test_log_perplexity",
    perp_fallback_col: str = "test_perplexity",
    title_left: str = "Coherence Score Analysis",
    title_right: str = "Perplexity Analysis (log-per-word)",
    save_path: str | None = None,
    show: bool = True,
):
    """
    Vẽ 2 biểu đồ:
      - Trái: Coherence (nếu có) + vạch đỏ tại n_topics tối ưu (max coherence).
      - Phải: Log-Perplexity per-word (nếu không có thì dùng log(perplexity)) + vạch đỏ tại min.

    Parameters
    ----------
    lda_grid_df : DataFrame có cột n_topics, coherence_c_v (tuỳ chọn), 
                  test_log_perplexity hoặc test_perplexity (fallback).
    save_path : nếu không None, lưu PNG vào đường dẫn này.
    show : có plt.show() hay không.
    """
    df = lda_grid_df.copy().sort_values(x_col).reset_index(drop=True)

    # --- X ---
    if x_col not in df.columns:
        raise ValueError(f"'{x_col}' not found.")
    x = df[x_col].to_numpy()

    # --- Coherence (có thể NaN/thiếu) ---
    have_coh = coherence_col in df.columns
    if have_coh:
        coh = df[coherence_col].to_numpy()
        valid = ~np.isnan(coh)
        if not valid.any():
            have_coh = False  # toàn NaN → bỏ chart trái
    else:
        coh = None

    # --- Log-Perplexity (hoặc fallback: log(perplexity)) ---
    if log_perp_col in df.columns:
        logppl = df[log_perp_col].to_numpy()
    else:
        if perp_fallback_col not in df.columns:
            raise ValueError(f"Need '{log_perp_col}' or '{perp_fallback_col}'.")
        logppl = np.log(df[perp_fallback_col].to_numpy())

    # --- Best points ---
    best_ppl_idx = np.nanargmin(logppl)
    best_ppl_x   = int(x[best_ppl_idx])
    best_ppl_val = float(logppl[best_ppl_idx])

    if have_coh:
        best_coh_idx = np.nanargmax(coh)
        best_coh_x   = int(x[best_coh_idx])
        best_coh_val = float(coh[best_coh_idx])

        # Figure 2 subplots
        plt.figure(figsize=(14, 5))

        # Left: Coherence
        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(x, coh, marker="o")
        ax1.axvline(best_coh_x, linestyle="--", color="red", label=f"Optimal: {best_coh_x}")
        ax1.set_title(title_left)
        ax1.set_xlabel("Number of Topics")
        ax1.set_ylabel("Coherence Score")
        ax1.set_xticks(x.astype(int))
        ax1.grid(True, linestyle="--", alpha=0.3)
        ax1.legend()

        # Right: Log-Perplexity
        ax2 = plt.subplot(1, 2, 2)
        ax2.plot(x, logppl, marker="s", color="green")
        ax2.axvline(best_ppl_x, linestyle="--", color="red", label=f"Optimal: {best_ppl_x}")
        ax2.set_title(title_right)
        ax2.set_xlabel("Number of Topics")
        ax2.set_ylabel("Log Perplexity")
        ax2.set_xticks(x.astype(int))
        ax2.grid(True, linestyle="--", alpha=0.3)
        ax2.legend()

        # Annotation nhẹ giá trị min ở đồ thị phải
        ax2.annotate(f"{best_ppl_val:.3f}", xy=(best_ppl_x, best_ppl_val),
                     xytext=(8, 8), textcoords="offset points")
        plt.tight_layout()

    else:
        # Chỉ có đồ thị phải
        plt.figure(figsize=(7,5))
        ax = plt.gca()
        ax.plot(x, logppl, marker="s", color="green")
        ax.axvline(best_ppl_x, linestyle="--", color="red", label=f"Optimal: {best_ppl_x}")
        ax.set_title(title_right)
        ax.set_xlabel("Number of Topics")
        ax.set_ylabel("Log Perplexity")
        ax.set_xticks(x.astype(int))
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
        ax.annotate(f"{best_ppl_val:.3f}", xy=(best_ppl_x, best_ppl_val),
                    xytext=(8, 8), textcoords="offset points")
        plt.tight_layout()

    if save_path:
        plt.gcf().savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()
