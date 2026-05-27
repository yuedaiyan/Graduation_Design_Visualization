"""
读取输入
从 diary_entries.json (line 1) 读取日记，只保留同时有 content 和 date 的条目。

句子拆分
每篇日记先经过 atomic_sentence_split()，再经过 merge_short_atomic_sentences()。

生成句向量
用本地模型 Qwen3-Embedding-0.6B，通过 QwenEmbedder.embed_texts() 批量生成每个句子的 embedding。

构造语义段
用相邻句子的余弦相似度决定是否把几个句子合成一个语义窗口。

保存结果
每篇日记会生成一个目录，里面保存：

sentence_vectors.npy：每句话的向量
window_vectors.npy：每个语义段的向量
diary_vector.npy：整篇日记的向量
sentence_similarity.npy：句子之间的相似度矩阵
meta.json：句子文本、语义段边界、参数等元数据
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from diary_core.embedding import QwenEmbedder

LOCAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), "Qwen3-Embedding-0.6B")
INPUT_FILE = os.path.join(os.path.dirname(__file__), "diary_entries.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "diary_sentence_vectors")
MAX_LENGTH = 512
EMBED_BATCH_SIZE = 32

# 句级过滤与分段参数
MIN_SENT_CHAR = 6
SEMANTIC_BREAK_THRESHOLD = 0.55
WINDOW_MIN_SENT = 2
WINDOW_MAX_SENT = 4


@dataclass
class SentenceItem:
    idx: int
    text: str


def _normalize_text(text: str) -> str:
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


def atomic_sentence_split(text: str) -> list[str]:
    """规则切分：按中文终止符+换行，保留语义最小单元。"""
    text = _normalize_text(text)
    if not text:
        return []

    parts: list[str] = []
    buf: list[str] = []

    def flush():
        s = "".join(buf).strip()
        if s:
            parts.append(s)
        buf.clear()

    hard_breaks = {"。", "！", "？", "；", "\n"}

    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        buf.append(ch)

        if ch in hard_breaks:
            # 连续标点归到同一句末尾
            j = i + 1
            while j < n and text[j] in "。！？；!?…~～":
                buf.append(text[j])
                j += 1
            flush()
            i = j
            continue

        i += 1

    flush()

    cleaned = [re.sub(r"\s+", " ", p).strip() for p in parts]
    return [c for c in cleaned if c]


def merge_short_atomic_sentences(
    sentences: list[str], min_len: int = MIN_SENT_CHAR
) -> list[str]:
    """先做长度层面的稳健化：把太短句子并到前后。"""
    if not sentences:
        return []
    merged: list[str] = []

    for s in sentences:
        if len(s) < min_len:
            if merged:
                merged[-1] = f"{merged[-1]} {s}".strip()
            else:
                merged.append(s)
        else:
            merged.append(s)

    # 处理最后一句过短
    if len(merged) >= 2 and len(merged[-1]) < min_len:
        merged[-2] = f"{merged[-2]} {merged[-1]}".strip()
        merged.pop()

    return merged


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def build_semantic_windows(
    sentences: list[str], sentence_vecs: np.ndarray
) -> list[dict[str, Any]]:
    """按相邻句相似度 + 窗口大小限制做语义分段。"""
    n = len(sentences)
    if n == 0:
        return []

    windows: list[dict[str, Any]] = []
    start = 0

    for i in range(1, n):
        sim = cosine(sentence_vecs[i - 1], sentence_vecs[i])
        cur_size = i - start

        # 满足最低窗口大小后，出现明显语义跳变就切断
        should_break = cur_size >= WINDOW_MIN_SENT and sim < SEMANTIC_BREAK_THRESHOLD
        # 达到最大窗口大小强制切断
        should_force_break = cur_size >= WINDOW_MAX_SENT

        if should_break or should_force_break:
            seg = sentence_vecs[start:i]
            seg_vec = seg.mean(axis=0)
            seg_vec = seg_vec / (np.linalg.norm(seg_vec) + 1e-12)
            windows.append(
                {
                    "start_sentence_idx": start,
                    "end_sentence_idx": i - 1,
                    "text": " ".join(sentences[start:i]),
                    "size": i - start,
                    "vector": seg_vec.astype(np.float32),
                }
            )
            start = i

    # 最后一段
    seg = sentence_vecs[start:n]
    seg_vec = seg.mean(axis=0)
    seg_vec = seg_vec / (np.linalg.norm(seg_vec) + 1e-12)
    windows.append(
        {
            "start_sentence_idx": start,
            "end_sentence_idx": n - 1,
            "text": " ".join(sentences[start:n]),
            "size": n - start,
            "vector": seg_vec.astype(np.float32),
        }
    )

    return windows


def safe_stem(date: str, count: int) -> str:
    base = re.sub(r"[^0-9A-Za-z_-]", "_", date.strip()) or "unknown_date"
    return base if count == 0 else f"{base}_{count + 1}"


def main():
    embedder = QwenEmbedder(LOCAL_MODEL_DIR)
    print(f"使用设备: {embedder.device}")

    with open(INPUT_FILE, encoding="utf-8") as f:
        entries = json.load(f)

    entries = [e for e in entries if e.get("content") and e.get("date")]
    if not entries:
        print("没有有效的日记条目，退出。")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    date_counter: dict[str, int] = {}
    total_sentences = 0
    total_windows = 0

    for idx, entry in enumerate(entries, start=1):
        date = entry["date"]
        content = entry["content"]

        atomic = atomic_sentence_split(content)
        sentences = merge_short_atomic_sentences(atomic, min_len=MIN_SENT_CHAR)

        if not sentences:
            continue

        sent_vecs = embedder.embed_texts(
            sentences,
            max_tokens=MAX_LENGTH,
            batch_size=EMBED_BATCH_SIZE,
        )

        windows = build_semantic_windows(sentences, sent_vecs)
        win_vecs = np.stack([w["vector"] for w in windows], axis=0)

        # 篇级向量：句向量均值（再归一化）
        diary_vec = sent_vecs.mean(axis=0)
        diary_vec = diary_vec / (np.linalg.norm(diary_vec) + 1e-12)
        diary_vec = diary_vec.astype(np.float32)

        # 句间相似度矩阵
        sim_matrix = sent_vecs @ sent_vecs.T

        count = date_counter.get(date, 0)
        date_counter[date] = count + 1
        stem = safe_stem(date, count)

        out_dir = os.path.join(OUTPUT_DIR, stem)
        os.makedirs(out_dir, exist_ok=True)

        np.save(
            os.path.join(out_dir, "sentence_vectors.npy"), sent_vecs.astype(np.float32)
        )
        np.save(
            os.path.join(out_dir, "window_vectors.npy"), win_vecs.astype(np.float32)
        )
        np.save(os.path.join(out_dir, "diary_vector.npy"), diary_vec)
        np.save(
            os.path.join(out_dir, "sentence_similarity.npy"),
            sim_matrix.astype(np.float32),
        )

        meta = {
            "date": date,
            "entry_index": idx - 1,
            "num_atomic_sentences": len(atomic),
            "num_sentences": len(sentences),
            "num_windows": len(windows),
            "semantic_break_threshold": SEMANTIC_BREAK_THRESHOLD,
            "window_min_sent": WINDOW_MIN_SENT,
            "window_max_sent": WINDOW_MAX_SENT,
            "sentences": [{"idx": i, "text": s} for i, s in enumerate(sentences)],
            "windows": [
                {
                    "idx": wi,
                    "start_sentence_idx": w["start_sentence_idx"],
                    "end_sentence_idx": w["end_sentence_idx"],
                    "size": w["size"],
                    "text": w["text"],
                }
                for wi, w in enumerate(windows)
            ],
        }

        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as mf:
            json.dump(meta, mf, ensure_ascii=False, indent=2)

        total_sentences += len(sentences)
        total_windows += len(windows)

        print(
            f"[{idx}/{len(entries)}] {stem}: 句子 {len(sentences)}，语义段 {len(windows)}"
        )

    summary = {
        "entries": len(entries),
        "total_sentences": total_sentences,
        "total_windows": total_windows,
        "avg_sentences_per_entry": round(total_sentences / len(entries), 2),
        "avg_windows_per_entry": round(total_windows / len(entries), 2),
        "embedding_dim": 1024,
        "output_dir": OUTPUT_DIR,
    }

    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as sf:
        json.dump(summary, sf, ensure_ascii=False, indent=2)

    print("\n完成。")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
