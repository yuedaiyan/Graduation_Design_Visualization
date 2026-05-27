# diary_ColorfulLines

根据篇级、句级/窗口级、字符级语义向量生成彩色铅笔排线风格图像。每篇日记输出一张 PNG 和一张 SVG。

## 启动命令

在仓库根目录运行：

```bash
uv run python diary_ColorfulLines/main.py
```

常用命令：

```bash
# 只生成某一天或某个 stem
uv run python diary_ColorfulLines/main.py --date 2021-09-05

# 只生成 PNG，不生成 SVG
uv run python diary_ColorfulLines/main.py --png-only

# 限制每篇参与渲染的字符数
uv run python diary_ColorfulLines/main.py --max-chars 400

# 强制用 Qwen 模型重新生成字符级上下文向量
uv run python diary_ColorfulLines/main.py --char-vector-source model --date 2021-09-05

# 只用缓存，不允许重新派生
uv run python diary_ColorfulLines/main.py --char-vector-source cache
```

## 输入和输出

输入：

- `diary_entries.json`
- `diary_vectors/*.npy`
- `diary_sentence_vectors/<stem>/diary_vector.npy`
- `diary_sentence_vectors/<stem>/sentence_vectors.npy`
- `diary_sentence_vectors/<stem>/window_vectors.npy`
- 可选 `diary_ColorfulLines/cache_char_vectors/*.npz`
- 当 `--char-vector-source model` 时会加载 `Qwen3-Embedding-0.6B`

输出：

```text
output_All/diary_ColorfulLines/try_N/<stem>.png
output_All/diary_ColorfulLines/svg_try_N/<stem>.svg
output_All/diary_ColorfulLines/try_N/summary.json
```

字符级向量缓存默认写在：

```text
diary_ColorfulLines/cache_char_vectors/
```

## 主函数和关键函数

- `main()`：解析参数、建立输出目录、读取日记、按条目加载向量、渲染 PNG/SVG、写 `summary.json`。
- `parse_args()`：定义 `--root`、`--entries`、`--model-dir`、`--out-dir`、`--cache-dir`、`--char-vector-source`、`--date`、`--max-chars` 等参数。
- `load_entries()`：从日记 JSON 读取有效条目，并把重复日期转成 stem。
- `load_event_vectors()`：读取篇级、句级、窗口级向量和 `meta.json`。
- `char_context_units()`：把日记正文拆成字符上下文片段，记录字符位置和字形。
- `load_or_build_char_vectors()`：按 `auto/semantic/cache/model` 策略读取、派生或生成字符级向量。
- `dynamic_embed()`：用 `QwenEmbedder.embed_texts()` 批量生成字符上下文 embedding。
- `semantic_char_vectors()`：从句向量、窗口向量和篇级向量派生字符级向量，避免每次加载模型。
- `render_one()`：把语义向量映射成彩色排线，保存 PNG/SVG。
- `write_svg()`、`svg_line_elements()`：输出透明前景 SVG。

## 生成逻辑

- 篇级向量控制整体色调和构图。
- 句级/窗口级向量控制局部语义区域。
- 字符级向量控制每个字附近排线的颜色、方向、长度和密度。
- 默认 `--char-vector-source auto` 会优先复用缓存，缓存不合适时从现有句级/窗口级向量派生；只有显式传 `model` 才会重新加载 Qwen。
- 文件顶部的 `DEFAULT_CONTENT_SIZE` 控制内容坐标和排线生成画布，`DEFAULT_RESOLUTION` 控制最终 PNG 尺寸和 SVG `width`/`height`；默认仍是 `1000 x 1000`，只改输出尺寸不会改变排线内容。
