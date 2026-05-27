# diary_core

仓库内稳定复用的 Python 工具模块。目前主要内容是 `embedding.py`，用于统一封装本地 Qwen embedding 推理。

这个文件夹不是独立可视化程序，没有直接的批量启动命令；它被其他脚本 import 使用。

## 典型使用方式

```python
from diary_core.embedding import QwenEmbedder

embedder = QwenEmbedder("Qwen3-Embedding-0.6B")
vectors = embedder.embed_texts(
    texts,
    max_tokens=512,
    batch_size=32,
    sort_by_length=True,
)
```

长文本：

```python
vectors = embedder.embed_long_texts(
    texts,
    max_tokens=4096,
    chunk_overlap=128,
)
```

## 被哪些程序使用

常见调用方：

- `vectorize_diary.py`
- `vectorize_diary_sentences.py`
- `diary_ColorfulLines/main.py`
- `diary_MultiBackground/pipeline.py`

## 主要函数和类

- `default_device(torch_module)`：自动选择推理设备，优先 `mps`，其次 `cuda`，最后 `cpu`。
- `default_dtype(torch_module, device)`：在 `mps/cuda` 上默认用 `float16`，在 CPU 上用 `float32`。
- `last_token_pool(last_hidden_states, attention_mask)`：从最后一个有效 token 取 embedding，适配 Qwen embedding 模型。
- `split_token_ids(token_ids, max_tokens, chunk_overlap=0)`：把长 token 序列切成带可选 overlap 的块。
- `QwenEmbedder.__init__()`：加载 tokenizer 和 model，并切到合适设备。
- `QwenEmbedder.clear_cache()`：每个 batch 后清理 MPS/CUDA 缓存。
- `QwenEmbedder.embed_texts()`：批量把短文本/句子/片段转成 L2 归一化向量。
- `QwenEmbedder.embed_token_chunks()`：对已经切好的 token chunk 直接推理。
- `QwenEmbedder.embed_long_texts()`：长文本切块推理，再按 token 数加权平均并归一化。

## 验证命令

这个文件夹本身没有 CLI。修改后建议在仓库根目录运行：

```bash
uv run python -m py_compile diary_core/embedding.py
uv run python -m unittest tests/test_embedding.py
```
