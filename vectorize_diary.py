"""
vectorize_diary.py
这是“生产向量”的程序。

它读取 diary_entries.json，把每篇日记的 content 用本地模型 Qwen3-Embedding-0.6B 转成 embedding 向量，然后保存到：
diary_vectors/

每篇日记会生成一个 .npy 文件，比如：
diary_vectors/2024-01-01.npy
diary_vectors/2024-01-02.npy

如果同一天有多篇，会变成：
2024-01-01.npy
2024-01-01_2.npy

简单说：
vectorize_diary.py 是把“文字日记”变成“数学向量”的脚本。

"""

import json
import os
import numpy as np

from diary_core.embedding import QwenEmbedder

LOCAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), "Qwen3-Embedding-0.6B")
INPUT_FILE = os.path.join(os.path.dirname(__file__), "diary_entries.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "diary_vectors")
MAX_LENGTH = 4096  # 16GB Mac 上单次 MPS 推理更稳
CHUNK_OVERLAP = 128  # 长日记分块时保留少量上下文重叠


def estimate_token_length(text: str) -> int:
    """中文大致 1 char ≈ 1 token,粗估即可,只用来排序。"""
    return len(text)


def get_batch_size(max_char_len: int) -> int:
    """根据 batch 内最长文本动态决定 batch size。
    16GB 统一内存,留 4GB 给系统,大约 10GB 可用。
    """
    if max_char_len <= 256:
        return 32
    elif max_char_len <= 1024:
        return 16
    elif max_char_len <= 4096:
        return 8
    elif max_char_len <= 8192:
        return 2
    else:
        return 1


def main():
    embedder = QwenEmbedder(LOCAL_MODEL_DIR)
    print(f"使用设备: {embedder.device}")

    with open(INPUT_FILE, encoding="utf-8") as f:
        entries = json.load(f)

    entries = [e for e in entries if e.get("content") and e.get("date")]
    if not entries:
        print("没有有效的日记条目,退出。")
        return

    # 按长度排序,保留原始索引以便最后写回
    indexed = list(enumerate(entries))
    indexed.sort(key=lambda x: estimate_token_length(x[1]["content"]))

    lengths = [estimate_token_length(e["content"]) for _, e in indexed]
    print(f"日记总数: {len(entries)}")
    print(
        f"长度分布: 最短 {lengths[0]} 字, 最长 {lengths[-1]} 字, 中位数 {lengths[len(lengths) // 2]} 字"
    )
    print(f"向量化中...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_vecs = [None] * len(entries)

    i = 0
    processed = 0
    while i < len(indexed):
        tentative_bs = get_batch_size(estimate_token_length(indexed[i][1]["content"]))
        batch = indexed[i : i + tentative_bs]
        max_len_in_batch = max(estimate_token_length(e["content"]) for _, e in batch)
        actual_bs = get_batch_size(max_len_in_batch)
        if actual_bs < len(batch):
            batch = batch[:actual_bs]

        texts = [e["content"] for _, e in batch]
        vecs = embedder.embed_long_texts(
            texts,
            max_tokens=MAX_LENGTH,
            chunk_overlap=CHUNK_OVERLAP,
        )

        for (orig_idx, _), vec in zip(batch, vecs):
            all_vecs[orig_idx] = vec

        i += len(batch)
        processed += len(batch)
        print(
            f"  进度: {processed}/{len(entries)}  (batch={len(batch)}, max_len={max_len_in_batch})"
        )

    # 保存每篇日记的向量为独立文件，日期相同时加序号
    date_counter: dict[str, int] = {}
    for entry, vec in zip(entries, all_vecs):
        date = entry["date"]
        count = date_counter.get(date, 0)
        date_counter[date] = count + 1
        filename = f"{date}.npy" if count == 0 else f"{date}_{count + 1}.npy"
        np.save(os.path.join(OUTPUT_DIR, filename), np.asarray(vec, dtype=np.float32))

    total = len(entries)
    total_size = sum(
        os.path.getsize(os.path.join(OUTPUT_DIR, f))
        for f in os.listdir(OUTPUT_DIR)
        if f.endswith(".npy")
    )
    print(f"\n完成。")
    print(f"  输出目录: {OUTPUT_DIR}/  共 {total} 个文件")
    print(f"  总大小: {total_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
