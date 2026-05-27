# AI 协作和项目开发习惯

这份文档的主要功能是给未来继续写代码时定习惯。新增脚本、重构旧脚本、接入新数据源、增加缓存或依赖时，优先按这里的约定做；README 只保留项目入口和运行说明。

## 基本原则

这个项目本质上是“共享数据资产 + 多个视觉实验”。每个实验可以有自己的视觉逻辑、参数、颜色、布局和输出方式，但不要重复实现那些稳定的基础设施。

适合复用的内容：

- Qwen embedding 模型加载与推理
- 文本转向量的 batch 逻辑
- MPS/16GB Mac 内存保护策略
- 长文本切块与 chunk overlap
- `last_token_pool`
- 全局统计缓存，例如词频、全库向量均值、SVD/PCA 上下文
- 数据不变时可以复用的中间结果

不一定需要复用的内容：

- 每个视觉实验自己的布局算法
- 颜色、线条、粒子、形状映射
- 实验自己的 `next_try_dir`
- 某个实验专用的 `safe_stem` 或日期筛选规则
- 输出图片命名、summary 字段、局部参数

原因：视觉实验应该保持自由；但 embedding 和缓存属于基础设施，重复实现会导致结果不可比较，也更容易浪费内存和时间。

## uv 环境管理

项目根目录已经有 `pyproject.toml` 和 `uv.lock`。以后新增依赖时，优先改 `pyproject.toml`，不要再给每个子目录单独新增零散的 `requirements.txt`。

已有子目录里的 `requirements.txt` 只当作历史记录或兼容说明；真正的环境入口以根目录 `pyproject.toml` 为准。

基础环境：

```bash
uv sync
```

需要运行 embedding、HuggingFace 数据集、UMAP 等重依赖时：

```bash
uv sync --extra embed
```

需要运行 `diary_ColorBlock_ContourLines` 且本机具备 cairo 系统库时：

```bash
uv sync --extra cairo
```

运行脚本时优先用：

```bash
uv run python path/to/script.py
```

不要依赖系统 `python3`，因为系统环境可能没有 `numpy`、`torch` 等包。

## Qwen Embedding 复用

所有直接调用 Qwen3-Embedding 的新代码，应优先使用：

```python
from diary_core.embedding import QwenEmbedder
```

典型用法：

```python
embedder = QwenEmbedder(model_dir)
vectors = embedder.embed_texts(
    texts,
    max_tokens=256,
    batch_size=32,
    sort_by_length=True,
)
```

长文本用：

```python
vectors = embedder.embed_long_texts(
    texts,
    max_tokens=4096,
    chunk_overlap=128,
)
```

MacBook Pro 16GB 注意：

- 背景文本建议 `max_tokens=256` 起步。
- 句子/短片段建议 `max_tokens=512` 起步。
- 整篇日记可以 `max_tokens=4096`，但长文本切块默认一次只推理一个 chunk。
- 不要随意把 batch size 调很大。MPS 内存不够时，优先降低 `batch_size` 和 `max_tokens`。
- 多背景脚本里应尽量复用同一个 `QwenEmbedder` 实例，不要每个背景重新加载模型。

如果只是使用已有向量文件，不要加载 Qwen 模型。

## 向量数据的区别

项目里有两套天级相关向量，它们不是同一份数据：

- `diary_vectors/<stem>.npy`：由 `vectorize_diary.py` 对整篇日记文本直接 embedding 得到。
- `diary_sentence_vectors/<stem>/diary_vector.npy`：由 `vectorize_diary_sentences.py` 对句向量取均值再归一化得到。

新增脚本前要先想清楚：

- 如果需要“整篇文本作为一个整体”的语义位置，用 `diary_vectors/<stem>.npy`。
- 如果需要和句子、窗口、事件内部结构一致，用 `diary_sentence_vectors/<stem>/diary_vector.npy`。
- 不要假设这两份向量相同，也不要随便互相替代。

句级目录中常见文件：

- `sentence_vectors.npy`
- `window_vectors.npy`
- `diary_vector.npy`
- `sentence_similarity.npy`
- `meta.json`

## 缓存策略

理论上只要输入数据没有变，就应该缓存的内容：

- 全库词频、文档频率、IDF
- 所有日记向量拼成的大矩阵
- 全库 centroid / distinctiveness
- SVD/PCA/UMAP 的 reducer 结果
- 背景文本采样结果
- 背景 embedding
- 字符级或片段级 embedding

缓存位置建议：

- 某个实验自己的缓存：`diary_xxx/.cache/`
- 根目录共享派生缓存：`.cache/derived/<experiment_name>/`
- 大型输出结果：`output_All/<experiment_name>/try_N/`

缓存文件名应包含影响结果的关键参数，例如：

- 输入文件大小或修改时间
- 样本数量
- `max_tokens`
- `random_seed`
- 背景名称
- reducer 类型

如果参数变了，应该自然生成新的缓存，而不是覆盖旧缓存。

缓存文件或 payload 里建议带 `version` / `algorithm_version`。只要算法含义变了，即使输入文件没变，也应该换版本，避免误读旧缓存。

## 输出目录习惯

视觉实验输出统一放在：

```text
output_All/<实验名>/try_N/
```

每次运行新建下一个 `try_N`，不要覆盖上一次结果。

背景类实验保持：

```text
output_All/diary_InTheWorld/<实验名>/try_N/<background_key>/
```

每次输出建议包含：

- 图片或 SVG
- `summary.json` 或 `summary.txt`
- 关键参数快照，例如 `params.json`
- 如果是二维空间图，保存 `points.csv`

## 新增实验脚本建议结构

简单实验可以继续单文件：

```text
diary_NewExperiment/
  main.py
  README.md
```

复杂实验建议拆成：

```text
diary_NewExperiment/
  main.py
  README.md
  diary_visual/
    config.py
    pipeline.py
    analysis.py
    rendering.py
```

参数优先放在文件顶部或 `config.py`，方便之后调节。

不要把生成结果、缓存、大模型、虚拟环境提交到 git。

## 新增代码前的检查清单

写新脚本前先确认：

- 是否真的需要重新 embedding？如果已有向量够用，就直接读向量。
- 需要的是整篇向量，还是句级目录里的派生篇级向量？
- 是否有全局统计会在每次运行重复计算？如果有，考虑缓存。
- 是否会读取外部数据集？如果会，先做采样缓存。
- 是否会生成大量图片？输出到 `output_All/.../try_N/`。
- 是否需要新增依赖？如果需要，改根目录 `pyproject.toml`。
- 是否会在 16GB Mac 上跑？先用小 batch、小样本、小 token 测试。

## 什么时候不要抽共享模块

如果一个函数表达的是某个视觉实验自己的审美判断或布局规则，不要为了“减少重复”强行抽共享模块。

例如这些通常可以留在各自脚本里：

- 某种颜色如何映射情绪
- 某种线条如何弯曲
- 某种点如何排布
- 某个实验的 summary 字段
- 某个实验的 try 目录命名细节

共享模块应该只放稳定、低争议、未来不太会按实验分叉的基础能力。
