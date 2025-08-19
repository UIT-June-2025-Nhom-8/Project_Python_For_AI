from typing import List, Tuple, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import SnowballStemmer


class SKLearnLDATopicModeler:
    """
    A thin wrapper around scikit-learn's LatentDirichletAllocation for topic modeling
    with helper utilities to work with pre-tokenized texts.

    Parameters
    ----------
    n_topics : int, default=10
        Number of latent topics to learn.
    max_features : Optional[int], default=None
        Max vocabulary size for CountVectorizer. If None, use all tokens.
    min_df : int, default=2
        Ignore terms that appear in fewer than `min_df` documents.
    max_df : float, default=0.9
        Ignore terms that appear in more than `max_df` fraction of documents.
    random_state : int, default=42
        Random seed for reproducibility.
    max_iter : int, default=10
        Maximum number of iterations for the LDA optimizer.
    learning_method : str, default='batch'
        'batch' (full pass) or 'online' (incremental). See sklearn docs.
    learning_decay : float, default=0.7
        Learning rate decay (if using online learning).
    learning_offset : float, default=10.0
        A parameter that down-weights early iterations in online learning.
    evaluate_every : int, default=-1
        How often to evaluate perplexity; set to -1 to never evaluate on perplexity during training.

    Notes
    -----
    - This class expects *pre-tokenized* input (list of tokens per document) OR raw strings.
    If you pass list-of-tokens, we will join with spaces before vectorizing.
    - Uses CountVectorizer (bow) because LDA is a probabilistic count model (not TF-IDF).
    """

    def __init__(
        self,
        n_topics: int = 10,
        max_features: Optional[int] = None,
        min_df: int = 2,
        max_df: float = 0.9,
        random_state: int = 42,
        max_iter: int = 10,
        learning_method: str = "batch",
        learning_decay: float = 0.7,
        learning_offset: float = 10.0,
        evaluate_every: int = -1,
    ) -> None:
        """
        Initialize vectorizer and LDA model with provided hyperparameters.
        """
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
        )
        self.model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=random_state,
            max_iter=max_iter,
            learning_method=learning_method,
            learning_decay=learning_decay,
            learning_offset=learning_offset,
            evaluate_every=evaluate_every,
        )
        self._is_fitted = False

    @staticmethod
    def _ensure_texts(
        texts: Union[pd.Series, List[Union[str, List[str]]]],
    ) -> List[str]:
        """
        Ensure inputs are plain strings. If items are token lists, join them with spaces.

        Parameters
        ----------
        texts : list/Series of str or list of tokens

        Returns
        -------
        list of str
            Cleaned list of documents, each as a single string.
        """
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        out = []
        for item in texts:
            if isinstance(item, list):
                out.append(" ".join(map(str, item)))
            else:
                out.append(str(item))
        return out

    def fit(
        self, texts: Union[pd.Series, List[Union[str, List[str]]]]
    ) -> "LDATopicModeler":
        """
        Fit the LDA model on training texts.

        Parameters
        ----------
        texts : list/Series of str or list of tokens

        Returns
        -------
        self : LDATopicModeler
            Fitted instance for chaining.
        """
        docs = self._ensure_texts(texts)
        X = self.vectorizer.fit_transform(docs)
        self.model.fit(X)
        self._is_fitted = True
        return self

    def transform(
        self, texts: Union[pd.Series, List[Union[str, List[str]]]]
    ) -> np.ndarray:
        """
        Transform texts into topic-distribution (document-topic) matrix.

        Parameters
        ----------
        texts : list/Series of str or list of tokens

        Returns
        -------
        ndarray of shape (n_docs, n_topics)
            Topic proportions per document (rows sum ~ 1).
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() before transform().")
        docs = self._ensure_texts(texts)
        X = self.vectorizer.transform(docs)
        return self.model.transform(X)

    def fit_transform(
        self, texts: Union[pd.Series, List[Union[str, List[str]]]]
    ) -> np.ndarray:
        """
        Fit the model and return topic-distributions for the same texts.

        Parameters
        ----------
        texts : list/Series of str or list of tokens

        Returns
        -------
        ndarray of shape (n_docs, n_topics)
        """
        docs = self._ensure_texts(texts)
        X = self.vectorizer.fit_transform(docs)
        dtm = self.model.fit_transform(X)
        self._is_fitted = True
        return dtm

    def get_top_words_per_topic(self, top_n: int = 10) -> List[List[Tuple[str, float]]]:
        """
        Get top-N words (and their weights) for each topic.

        Parameters
        ----------
        top_n : int, default=10
            Number of top words per topic.

        Returns
        -------
        list of list of (word, weight)
            Outer list is topics, inner list are (word, weight) pairs in descending order.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        words = np.array(self.vectorizer.get_feature_names_out())
        topics = []
        for topic_idx, comp in enumerate(self.model.components_):
            # Larger weight in components_ => higher probability given the topic
            top_ids = np.argsort(comp)[::-1][:top_n]
            topics.append([(words[i], float(comp[i])) for i in top_ids])
        return topics

    from nltk.stem import SnowballStemmer

    def auto_label_topics(
        self,
        top_n: int = 10,
        keywords_bank=None,
        fallback_top_k: int = 3,
    ) -> list[dict]:
        """
        Auto-label topics by matching each topic's top words against a keyword bank.

        Parameters
        ----------
        top_n : int, default=10
            Number of top words per topic to consider when labeling.
        keywords_bank : dict or None
            Optional mapping label -> list of indicative keywords.
            If None, a default general-purpose bank for e-commerce reviews is used.
        fallback_top_k : int, default=3
            If no label matches, build a fallback label from the top-k words.

        Returns
        -------
        list of dict
            Each item:
            {
                'topic_index': int,
                'label': str,
                'score': float,          # matching score (0 if fallback)
                'top_words': list[str],  # the top words we used to decide
                'matched_keywords': list[str]  # which keywords matched (if any)
            }

        Notes
        -----
        - Uses rank-weighted overlap: earlier (higher-weight) words contribute more.
        - Normalizes both topic words and keyword bank with Snowball stemmer
          to handle morphology (battery -> batteri, etc.).
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # default keyword bank (e-commerce flavored; chỉnh theo domain của bạn nếu cần)
        if keywords_bank is None:
            keywords_bank = {
                "electronics": [
                    "phone",
                    "smartphone",
                    "battery",
                    "screen",
                    "display",
                    "charge",
                    "camera",
                    "laptop",
                    "computer",
                    "tablet",
                    "android",
                    "iphone",
                    "headphone",
                    "bluetooth",
                    "app",
                ],
                "books": [
                    "book",
                    "story",
                    "novel",
                    "author",
                    "read",
                    "chapter",
                    "plot",
                    "literature",
                ],
                "clothing": [
                    "shirt",
                    "jeans",
                    "dress",
                    "fit",
                    "size",
                    "fabric",
                    "material",
                    "wear",
                ],
                "beauty": [
                    "skin",
                    "makeup",
                    "cream",
                    "lotion",
                    "hair",
                    "fragrance",
                    "smell",
                    "shade",
                ],
                "home_kitchen": [
                    "kitchen",
                    "cook",
                    "knife",
                    "pan",
                    "pot",
                    "coffee",
                    "vacuum",
                    "clean",
                ],
                "sports_outdoors": [
                    "sport",
                    "run",
                    "hike",
                    "bike",
                    "gym",
                    "camp",
                    "outdoor",
                    "ball",
                ],
                "toys_games": [
                    "toy",
                    "game",
                    "play",
                    "kid",
                    "puzzle",
                    "lego",
                    "board",
                    "controller",
                ],
                "grocery_food": [
                    "food",
                    "snack",
                    "taste",
                    "flavor",
                    "chocolate",
                    "coffee",
                    "tea",
                ],
                "health": [
                    "vitamin",
                    "supplement",
                    "pill",
                    "allergy",
                    "medical",
                    "health",
                    "symptom",
                ],
                "pet": [
                    "dog",
                    "cat",
                    "pet",
                    "litter",
                    "treat",
                    "leash",
                    "kennel",
                    "aquarium",
                ],
                "music": [
                    "music",
                    "song",
                    "album",
                    "sound",
                    "speaker",
                    "headphone",
                    "audio",
                ],
                "movies_tv": [
                    "movie",
                    "film",
                    "tv",
                    "series",
                    "episode",
                    "blu-ray",
                    "dvd",
                ],
                "tools_auto": [
                    "tool",
                    "wrench",
                    "drill",
                    "screwdriver",
                    "car",
                    "auto",
                    "tire",
                    "oil",
                ],
                "baby": ["baby", "diaper", "bottle", "stroller", "infant", "toddler"],
                "office": [
                    "office",
                    "paper",
                    "pen",
                    "printer",
                    "stapler",
                    "desk",
                    "chair",
                ],
                "video_games": [
                    "game",
                    "console",
                    "controller",
                    "xbox",
                    "playstation",
                    "nintendo",
                    "pc",
                ],
            }

        # stemmer để chuẩn hoá
        stemmer = SnowballStemmer("english")

        def _stem_set(words: list[str]) -> set[str]:
            return {stemmer.stem(w.lower()) for w in words}

        # stem hoá keyword bank
        stemmed_bank = {
            label: _stem_set(words) for label, words in keywords_bank.items()
        }

        topics = self.get_top_words_per_topic(
            top_n=top_n
        )  # [(word, weight), ...] per topic
        results = []

        for t_idx, topic_words in enumerate(topics):
            # lấy danh sách top words (giữ thứ hạng)
            words_ranked = [w for (w, _) in topic_words]
            words_stemmed = [stemmer.stem(w.lower()) for w in words_ranked]

            # rank-weighted scoring: weight = (top_n - rank)/top_n
            best_label = None
            best_score = 0.0
            best_matches: list[str] = []

            for label, kw_stemmed in stemmed_bank.items():
                score = 0.0
                matches = []
                for rank, w in enumerate(words_stemmed):
                    if w in kw_stemmed:
                        weight = (top_n - rank) / top_n  # 1.0 for rank 0, decreasing
                        score += weight
                        matches.append(
                            words_ranked[rank]
                        )  # lưu từ gốc (chưa stem) để dễ đọc
                if score > best_score:
                    best_score = score
                    best_label = label
                    best_matches = matches

            if best_label is None or best_score <= 0.0:
                # fallback: ghép top-k words
                label = " / ".join(words_ranked[:fallback_top_k])
                results.append(
                    {
                        "topic_index": t_idx,
                        "label": label,
                        "score": 0.0,
                        "top_words": words_ranked,
                        "matched_keywords": [],
                    }
                )
            else:
                results.append(
                    {
                        "topic_index": t_idx,
                        "label": best_label,
                        "score": float(best_score),
                        "top_words": words_ranked,
                        "matched_keywords": best_matches,
                    }
                )

        return results

    def print_topic_labels(self, labels: list[dict]) -> None:
        """
        Pretty-print auto-labeled topics.

        Parameters
        ----------
        labels : list[dict]
            Output from auto_label_topics().
        """
        print("\n=== AUTO TOPIC LABELS ===")
        for item in labels:
            t = item["topic_index"]
            lbl = item["label"]
            sc = item["score"]
            matches = (
                ", ".join(item["matched_keywords"]) if item["matched_keywords"] else "-"
            )
            topw = ", ".join(item["top_words"][:10])
            print(f"Topic {t:02d}: {lbl:20s} | score={sc:.3f} | matches=[{matches}]")
            print(f"  top-words: {topw}")

    def save(self, model_path: str, vocab_path: Optional[str] = None) -> None:
        """
        Save the LDA model and (optionally) the vocabulary.

        Parameters
        ----------
        model_path : str
            Filepath to save the fitted LDA model via joblib.
        vocab_path : Optional[str]
            If provided, save CountVectorizer via joblib for future use.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Nothing to save.")
        joblib.dump(self.model, model_path)
        if vocab_path is not None:
            joblib.dump(self.vectorizer, vocab_path)

    def load(self, model_path: str, vocab_path: Optional[str] = None) -> None:
        """
        Load a previously saved LDA model (and vocabulary).

        Parameters
        ----------
        model_path : str
            Path to joblib-pickled LDA model.
        vocab_path : Optional[str]
            Path to joblib-pickled CountVectorizer. If omitted, you must fit a new vectorizer.
        """
        self.model = joblib.load(model_path)
        if vocab_path is not None:
            self.vectorizer = joblib.load(vocab_path)
        self._is_fitted = True
