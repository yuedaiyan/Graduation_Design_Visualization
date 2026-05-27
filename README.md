# Biue Diary Visualization

这个仓库是“共享日记数据资产 + 多个视觉实验”的项目。

核心数据：

- `diary_entries.json`：原始日记条目。
- `diary_vectors/`：按整篇日记生成的天级 embedding。
- `diary_sentence_vectors/`：按句子和语义窗口生成的事件级 embedding。
- `Qwen3-Embedding-0.6B/`：本地 embedding 模型目录，按需使用。

主要脚本：

- `vectorize_diary.py`：生成 `diary_vectors/`。
- `vectorize_diary_sentences.py`：生成 `diary_sentence_vectors/`。
- `diary_Ring/main.py`：圆环形式的单篇或全量日记视觉。
- `diary_ColorfulLines/main.py`：基于字符/句级语义的彩色线条视觉。
- `diary_PerspectiveSpace/main.py`：基于全库语义上下文的透视空间图。
- `diary_MultiBackground/main.py`：把日记放进一个或多个背景语料中比较位置。
- `diary_ColorBlock_ContourLines/`：色块和等高线视觉实验。
- `diary_SingleCharacterDensityMap/main.py`：保留原文版面坐标，只显示指定字符/词的密度地图。

AI 协作和未来开发约定见 `AGENTS.md`。那份文档的用途是给以后新增或修改代码时定规矩，不是用户手册。

## 数据依赖核查

这些 `diary_*` 视觉实验大多使用同一批日记日期 stem，但不是所有逻辑都读取完全相同的原始文件，也不是所有“篇目向量”都是同一种数值来源。

当前工作区核查结果：

- `diary_entries.json` 有 1121 条日记。
- `diary_vectors/` 有 1121 个 `.npy` 文件。
- `diary_sentence_vectors/` 有 1121 个日期目录。
- `diary_entries.json` 推导出的日期 stem、`diary_vectors/*.npy` 的 stem、`diary_sentence_vectors/<stem>/` 的目录名完全对齐。
- 每个 `diary_sentence_vectors/<stem>/` 都包含 `sentence_vectors.npy`、`window_vectors.npy`、`diary_vector.npy`、`sentence_similarity.npy` 和 `meta.json`。

主要逻辑的默认依赖：

| 逻辑 | 默认依赖 |
| --- | --- |
| `diary_ColorBlock_ContourLines` | `diary_entries.json`、`diary_vectors/`、`diary_sentence_vectors/` |
| `diary_ColorfulLines` | `diary_entries.json`、`diary_vectors/`、`diary_sentence_vectors/`，可选 `cache_char_vectors/` 或 `Qwen3-Embedding-0.6B/` |
| `diary_Glitch` | `diary_entries.json`、`diary_vectors/`、`diary_sentence_vectors/<stem>/sentence_vectors.npy` |
| `diary_GridSpace` | 用 `diary_vectors/` 发现日期，但实际布局读取 `diary_sentence_vectors/<stem>/diary_vector.npy`、`sentence_vectors.npy` 和 `meta.json` |
| `diary_InTheMe` | `diary_vectors/`，并辅助读取 `diary_entries.json` 的日期、地点、时间和字数信息 |
| `diary_PerspectiveSpace` | `diary_vectors/`、`diary_sentence_vectors/<stem>/window_vectors.npy`，并在 `.cache/derived/diary_PerspectiveSpace/` 生成派生缓存 |
| `diary_Ring` | `diary_entries.json`、`diary_analysis_result.json`、`diary_vectors/`、`diary_sentence_vectors/`，并在 `.cache/derived/diary_Ring/` 生成派生缓存 |
| `diary_MultiBackground` | `diary_vectors/`、`diary_entries.merged.json`，以及背景语料文本和 `diary_MultiBackground/.cache/` 背景向量缓存 |
| `diary_BackgroundRelations` | 不读取日记原文或日记向量，只读取 `diary_MultiBackground/.cache/` 中已有的背景向量缓存 |
| `diary_ReferentialNetwork` | 默认读取 `diary_entries.merged.json`，不读取向量库 |
| `diary_SingleCharacterDensityMap` | 默认读取 `diary_entries.json` 和 `diary_entries.merged.sentence_counts.json` |

需要特别注意的差异：

- `diary_entries.merged.json` 在当前工作区是软链接，指向 `/Users/yuedaiyan/code_school/biue_code_text/diary_entries.merged.json`。它和根目录 `diary_entries.json` 的 `date`、`content`、`time_of_day` 一致，但有 702 条 `location` 不同，并且额外包含 `id`、`people_tags`、`tags`、`weather` 等字段。
- `diary_analysis_result.json` 有 1122 条，而 `diary_entries.json` 有 1121 条。`diary_Ring/io_data.py` 已处理这个一条偏移：当前分析文件的有效映射从 `analysis[1]` 对应 `entries[0]` 开始。
- `diary_vectors/<stem>.npy` 和 `diary_sentence_vectors/<stem>/diary_vector.npy` 不是重复文件。前者由 `vectorize_diary.py` 对整篇日记直接 embedding 生成；后者由 `vectorize_diary_sentences.py` 对句向量求均值再归一化生成。当前 1121 对文件中没有完全相同的向量，1102 对差异大于 `1e-5`。

## 环境管理

本项目可以用 `uv` 管理 Python 环境。第一次准备基础环境：

```bash
uv sync
```

如果要运行需要本地 embedding 模型、HuggingFace 数据集、UMAP 的脚本：

```bash
uv sync --extra embed
```

如果要运行 `diary_ColorBlock_ContourLines` 且本机已有 cairo 相关系统库：

```bash
uv sync --extra cairo
```

常用运行方式：

```bash
uv run python vectorize_diary.py
uv run python vectorize_diary_sentences.py
uv run python diary_Ring/main.py --date 2026-03-10
uv run python diary_ColorfulLines/main.py --date 2026-03-10 --char-vector-source semantic
uv run python diary_MultiBackground/main.py
```

运行轻量测试：

```bash
uv run python -m unittest tests/test_embedding.py
```

## 输出和缓存

视觉实验默认输出到：

```text
output_All/<实验名>/try_N/
```

派生缓存默认放在：

```text
.cache/derived/
```

已有向量足够时，脚本应优先复用 `.npy` 和缓存，避免重新加载 Qwen 模型。
