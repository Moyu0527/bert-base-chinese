import os
import re
import json
import jieba
import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix
from gensim import corpora
from gensim.models import LdaModel
import config


class TextPreprocessor:
    def __init__(self, stopwords_path: Optional[str] = None):
        self.stopwords = self._load_stopwords(stopwords_path)
        self.user_dict_loaded = False

    def _load_stopwords(self, stopwords_path: Optional[str]) -> Set[str]:
        default_stopwords = {
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
            "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
            "自己", "这", "那", "他", "她", "它", "们", "这个", "那个", "什么", "怎么",
            "为什么", "哪", "哪里", "谁", "如何", "多少", "几", "多", "少", "大", "小",
            "可以", "没", "只是", "但是", "所以", "因为", "如果", "虽然", "还是", "或者",
            "而且", "然后", "并且", "如此", "这样", "那样", "怎么", "这样", "那样", "一样",
            "被", "把", "让", "给", "向", "从", "由", "对", "于", "为", "以", "及", "与",
            "等", "等等", "之", "而", "且", "或", "但", "只", "已", "已经", "正在", "将",
            "将要", "曾", "曾经", "再", "又是", "还是", "只", "仅", "仅仅", "不过", "除了",
        }
        if stopwords_path and os.path.exists(stopwords_path):
            with open(stopwords_path, "r", encoding="utf-8") as f:
                custom_stopwords = set(line.strip() for line in f)
            default_stopwords.update(custom_stopwords)
        return default_stopwords

    def add_user_dict(self, words: List[str]):
        for word in words:
            jieba.add_word(word)
        self.user_dict_loaded = True

    def tokenize(self, text: str) -> List[str]:
        text = self._clean_text(text)
        tokens = jieba.cut(text, cut_all=False)
        tokens = [t.strip() for t in tokens if t.strip() and len(t.strip()) > 1]
        tokens = [t for t in tokens if t not in self.stopwords and not self._is_numeric(t)]
        return tokens

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#\w+#", "", text)
        text = re.sub(r"\[.*?\]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _is_numeric(self, text: str) -> bool:
        try:
            float(text)
            return True
        except ValueError:
            return False

    def process_corpus(self, corpus: List[Dict]) -> List[List[str]]:
        processed = []
        for item in corpus:
            text = item.get("text", "")
            tokens = self.tokenize(text)
            if tokens:
                processed.append(tokens)
        return processed


class LDABiasGraphBuilder:
    def __init__(
        self,
        num_topics: int = 50,
        passes: int = 10,
        alpha: str = "auto",
        eta: str = "auto"
    ):
        self.num_topics = num_topics
        self.passes = passes
        self.alpha = alpha
        self.eta = eta
        self.preprocessor = TextPreprocessor()
        self.dictionary = None
        self.corpus = None
        self.lda_model = None
        self.topic_word_dist = None
        self.bias_graph = {}

    def build_from_corpus(self, corpus: List[Dict], bias_seed_words: Dict[str, List[str]]) -> Dict:
        processed_corpus = self.preprocessor.process_corpus(corpus)

        if not processed_corpus:
            raise ValueError("No valid tokens after preprocessing")

        self.dictionary = corpora.Dictionary(processed_corpus)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in processed_corpus]

        self.lda_model = LdaModel(
            self.corpus,
            num_topics=self.num_topics,
            id2word=self.dictionary,
            passes=self.passes,
            alpha=self.alpha,
            eta=self.eta,
            random_state=config.HYPERPARAM_CONFIG["random_seed"]
        )

        self.topic_word_dist = self.lda_model.get_topics()

        self._extract_attribute_pairs(bias_seed_words)

        self._build_bias_graph(bias_seed_words)

        return self.bias_graph

    def _extract_attribute_pairs(self, bias_seed_words: Dict[str, List[str]]):
        all_seed_words = []
        for dimension, pairs in bias_seed_words.items():
            for attr_list in pairs.values():
                all_seed_words.extend(attr_list)
        self.preprocessor.add_user_dict(all_seed_words)

    def _build_bias_graph(self, bias_seed_words: Dict[str, List[str]]) -> Dict:
        self.bias_graph = {
            "dimension": {},
            "attribute_pairs": {},
            "cooccurrence_matrix": {}
        }

        for dimension, pairs in bias_seed_words.items():
            self.bias_graph["dimension"][dimension] = {
                "positive_attributes": pairs.get("positive", []),
                "negative_attributes": pairs.get("negative", []),
            }
            self.bias_graph["attribute_pairs"][dimension] = pairs

        topic_attribute_scores = self._calculate_topic_attribute_scores(bias_seed_words)

        self._build_cooccurrence_matrix(bias_seed_words, topic_attribute_scores)

        return self.bias_graph

    def _calculate_topic_attribute_scores(self, bias_seed_words: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        topic_attribute_scores = {}
        vocab_size = len(self.dictionary)

        for dimension, pairs in bias_seed_words.items():
            positive_words = pairs.get("positive", [])
            negative_words = pairs.get("negative", [])

            positive_indices = self._get_word_indices(positive_words)
            negative_indices = self._get_word_indices(negative_words)

            positive_scores = np.zeros(self.num_topics)
            negative_scores = np.zeros(self.num_topics)

            for idx in positive_indices:
                if idx < vocab_size:
                    positive_scores += self.topic_word_dist[:, idx]
            for idx in negative_indices:
                if idx < vocab_size:
                    negative_scores += self.topic_word_dist[:, idx]

            positive_scores /= max(len(positive_indices), 1)
            negative_scores /= max(len(negative_indices), 1)

            topic_attribute_scores[dimension] = {
                "positive": positive_scores,
                "negative": negative_scores,
            }

        return topic_attribute_scores

    def _get_word_indices(self, words: List[str]) -> List[int]:
        indices = []
        for word in words:
            word_id = self.dictionary.token2id.get(word, -1)
            if word_id != -1:
                indices.append(word_id)
        return indices

    def _build_cooccurrence_matrix(
        self,
        bias_seed_words: Dict[str, List[str]],
        topic_attribute_scores: Dict[str, np.ndarray]
    ):
        for dimension, pairs in bias_seed_words.items():
            positive_words = pairs.get("positive", [])
            negative_words = pairs.get("negative", [])

            positive_idx = self._get_word_indices(positive_words)
            negative_idx = self._get_word_indices(negative_words)

            cooccurrence = defaultdict(float)

            for i, w1_idx in enumerate(positive_idx):
                for j, w2_idx in enumerate(negative_idx):
                    cooccurrence[(positive_words[i], negative_words[j])] = 1.0

            self.bias_graph["cooccurrence_matrix"][dimension] = dict(cooccurrence)

    def save_bias_graph(self, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.bias_graph, f, ensure_ascii=False, indent=2)

    def load_bias_graph(self, input_path: str) -> Dict:
        with open(input_path, "r", encoding="utf-8") as f:
            self.bias_graph = json.load(f)
        return self.bias_graph

    def get_top_attribute_words(
        self,
        dimension: str,
        attribute_type: str,
        top_n: int = 20
    ) -> List[Tuple[str, float]]:
        if self.lda_model is None:
            raise ValueError("LDA model not built yet. Call build_from_corpus first.")

        topic_idx = self.lda_model.id2word.token2id
        word_scores = []

        for word, idx in topic_idx.items():
            if idx < len(self.topic_word_dist):
                score = self.topic_word_dist[:, idx].sum()
                word_scores.append((word, score))

        word_scores.sort(key=lambda x: x[1], reverse=True)
        return word_scores[:top_n]
