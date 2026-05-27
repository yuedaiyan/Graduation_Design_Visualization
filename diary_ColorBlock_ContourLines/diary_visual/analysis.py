"""
analysis.py — 分词、虚词分析、情绪分类、实词分簇
"""

import math
import re
import hashlib
from collections import defaultdict

import jieba
import numpy as np

from .config import (
    CLUSTER_MAX_GROUPS,
    CLUSTER_MIN_GROUPS,
    CLUSTER_WORDS_PER_GROUP,
    FUNCTION_WORDS,
    FUNCTION_WORD_CATEGORIES,
)

_WORD_TO_CATEGORY = {}
for _cat, _words in FUNCTION_WORD_CATEGORIES.items():
    for _w in _words:
        _WORD_TO_CATEGORY[_w] = _cat

_FUNCTION_WORD_SET = set(FUNCTION_WORDS)
_STOP_CHARS = set("，。！？；：""''「」【】（）《》、\n\r\t —…·")


def _stable_hash_int(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def get_function_word_category(word: str) -> str:
    return _WORD_TO_CATEGORY.get(word, "particle")


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"[。！？\n]+", text)
    return [p.strip() for p in parts if len(p.strip()) >= 4]


def tokenize(text: str):
    total_len = len(text)
    if total_len == 0:
        return [], []

    function_tokens = []
    content_tokens = []
    for word, start, _end in jieba.tokenize(text, mode="search"):
        word = word.strip()
        if not word or word in _STOP_CHARS:
            continue
        norm_pos = start / total_len if total_len > 0 else 0.5
        if word in _FUNCTION_WORD_SET:
            function_tokens.append(
                {
                    "word": word,
                    "category": get_function_word_category(word),
                    "char_pos": norm_pos,
                }
            )
        elif len(word) >= 2 or "\u4e00" <= word <= "\u9fff":
            content_tokens.append(word)
    return function_tokens, content_tokens


def _cosine_similarity(A, B):
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return A_norm @ B_norm.T


def classify_emotion(entry_vec: np.ndarray, emotion_region_centers: np.ndarray, emotion_names: list[str]) -> str:
    entry_vec = entry_vec.reshape(1, -1)
    sims = _cosine_similarity(entry_vec, emotion_region_centers)[0]
    return emotion_names[int(np.argmax(sims))]


def analyze_function_words(entry_content: str, sent_vectors: np.ndarray | None, entry_vec: np.ndarray) -> list[dict]:
    function_tokens, _ = tokenize(entry_content)
    if not function_tokens:
        return []

    sentences = split_sentences(entry_content)
    total_len = len(entry_content)
    sentence_ranges = []
    search_start = 0
    for sent in sentences:
        idx = entry_content.find(sent, search_start)
        if idx >= 0:
            sentence_ranges.append((idx / total_len, (idx + len(sent)) / total_len))
            search_start = idx + 1
        else:
            sentence_ranges.append((0.0, 1.0))

    def find_sentence_idx(pos):
        for si, (s, e) in enumerate(sentence_ranges):
            if s <= pos <= e:
                return si
        if sentence_ranges:
            dists = [abs(pos - (s + e) / 2) for s, e in sentence_ranges]
            return int(np.argmin(dists))
        return 0

    token_intensities = []
    vec_len = len(entry_vec)
    for tok in function_tokens:
        sent_idx = find_sentence_idx(tok["char_pos"])
        if sent_vectors is not None and sent_idx < len(sent_vectors):
            sv = sent_vectors[sent_idx].reshape(1, -1)
            ev = entry_vec.reshape(1, -1)
            sim = float(_cosine_similarity(sv, ev)[0, 0])
        else:
            if vec_len == 0:
                sim = 0.5
            else:
                idx = (_stable_hash_int(tok["word"]) + sent_idx * 17) % vec_len
                sim = float(entry_vec[idx])
        token_intensities.append(sim)

    min_i, max_i = min(token_intensities), max(token_intensities)
    rng = max_i - min_i
    norm_intensities = [(v - min_i) / rng if rng > 0 else 0.5 for v in token_intensities]

    word_data = defaultdict(lambda: {"positions": [], "intensities": [], "sentence_indices": []})
    for tok, intensity in zip(function_tokens, norm_intensities):
        w = tok["word"]
        word_data[w]["positions"].append(tok["char_pos"])
        word_data[w]["intensities"].append(intensity)
        word_data[w]["sentence_indices"].append(find_sentence_idx(tok["char_pos"]))
        word_data[w]["category"] = tok["category"]

    all_positions = [tok["char_pos"] for tok in function_tokens]
    n_sentences = max(len(sentences), 1)

    def local_density_at(pos):
        return sum(1 for p in all_positions if abs(p - pos) <= 0.1)

    results = []
    for word, data in word_data.items():
        positions = data["positions"]
        intensities = data["intensities"]
        sent_indices = data["sentence_indices"]
        count = len(positions)
        spread = float(np.std(positions)) if len(positions) > 1 else 0.0
        context_diversity = len(set(sent_indices)) / n_sentences
        burst_score = 0.0
        if count >= 3:
            sorted_pos = sorted(positions)
            max_run = 1
            cur_run = 1
            for j in range(1, len(sorted_pos)):
                if sorted_pos[j] - sorted_pos[j - 1] < 0.05:
                    cur_run += 1
                    max_run = max(max_run, cur_run)
                else:
                    cur_run = 1
            if max_run >= 3:
                burst_score = min(1.0, (max_run - 2) / 5.0)

        for pos, intensity in zip(positions, intensities):
            results.append(
                {
                    "word": word,
                    "category": data["category"],
                    "count": count,
                    "intensity": intensity,
                    "position": pos,
                    "spread": spread,
                    "context_diversity": context_diversity,
                    "local_density": local_density_at(pos) / max(len(all_positions), 1),
                    "burst_score": burst_score,
                    "avg_intensity": float(np.mean(intensities)),
                }
            )
    return results


def cluster_content_words(words: list[str], n_clusters: int | None = None) -> dict:
    if not words:
        return {}
    unique_words = sorted(set(words))
    if len(unique_words) == 1:
        return {0: unique_words}

    if n_clusters is None:
        n_clusters = max(
            CLUSTER_MIN_GROUPS,
            min(len(unique_words) // CLUSTER_WORDS_PER_GROUP, CLUSTER_MAX_GROUPS),
        )
    n_clusters = max(1, min(n_clusters, len(unique_words)))

    clusters = defaultdict(list)
    for w in unique_words:
        clusters[_stable_hash_int(w) % n_clusters].append(w)
    return dict(clusters)


def compute_word_weight(word_count_in_entry: int, total_words_in_entry: int, doc_freq: int, total_docs: int) -> float:
    tf = word_count_in_entry / max(total_words_in_entry, 1)
    idf = math.log(total_docs / (doc_freq + 1)) + 1.0
    return float(tf * idf)


def get_entry_content_word_data(entry_content: str, doc_freqs: dict, total_docs: int) -> tuple:
    _, content_tokens = tokenize(entry_content)
    if not content_tokens:
        return [], {}, {}

    word_counts = defaultdict(int)
    for w in content_tokens:
        word_counts[w] += 1

    total_words = len(content_tokens)
    unique_words = list(word_counts.keys())
    clusters = cluster_content_words(unique_words)

    weights = {}
    for w in unique_words:
        weights[w] = compute_word_weight(word_counts[w], total_words, doc_freqs.get(w, 1), total_docs)

    max_w = max(weights.values())
    min_w = min(weights.values())
    rng = max_w - min_w
    if rng > 0:
        weights = {w: (v - min_w) / rng for w, v in weights.items()}
    else:
        weights = {w: 0.5 for w in weights}
    return content_tokens, clusters, weights
