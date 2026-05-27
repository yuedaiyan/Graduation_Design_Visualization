# diary_Glitch

把每篇日记渲染成 glitch/发光网格风格图像。它融合篇级向量、句级向量和原文内容，生成语义背景、句子光晕和文本网格层；默认每篇输出 PNG 和 SVG。

## 启动命令

在仓库根目录运行：

```bash
uv run python diary_Glitch/main.py
```

常用命令：

```bash
# 只生成某一天或某个 stem
uv run python diary_Glitch/main.py --date 2022-04-18

# 只生成 PNG，不生成 SVG
uv run python diary_Glitch/main.py --png-only

# 只生成 SVG，不生成 PNG
uv run python diary_Glitch/main.py --svg-only

# 为指定日期生成 5000 x 5000 高分辨率 PNG
uv run python diary_Glitch/main.py --date 2022-04-18 --high
```

`--high` 必须和 `--date` 一起使用。

## 输入和输出

输入：

- `diary_entries.json`
- `diary_vectors/<stem>.npy`
- `diary_sentence_vectors/<stem>/sentence_vectors.npy`

输出：

```text
output_All/diary_Glitch/try_N/<stem>.png
output_All/diary_Glitch/svg_try_N/<stem>.svg
output_All/diary_Glitch/high_try_N/<stem>.png
```

## 主函数和关键函数

- `main()`：解析参数，检查输入目录，决定批量/单日/高分辨率模式，逐个调用 `render_one_date()`。
- `parse_args()`：定义 `--date`、`--high`、`--png-only`、`--svg-only`。
- `load_entries_by_date()`：读取日记正文并生成和向量文件一致的 stem。
- `ensure_day_alignment()`：批量生成前检查日记、篇级向量、句向量是否能对齐。
- `load_render_inputs()`：读取单篇日记所需的篇级向量、句向量和正文。
- `render_one_date()`：调度单篇 PNG/SVG 输出。
- `compose_diary_image()`：合成 PNG 图像。
- `write_diary_svg()`：输出 SVG 版本。
- `vector_to_semantic_palette()`：把篇级向量映射成基础色盘。
- `build_semantic_background()`：由篇级向量生成底层语义背景。
- `blend_sentence_glows()`：用句向量和句子文本生成局部光晕。
- `collect_text_grid_lines()` / `draw_text_grid_layer()`：根据正文统计生成 glitch 文本网格层。

## 生成逻辑

- 篇级向量决定整体色盘、背景波形和构图方向。
- 句级向量决定局部光晕位置、颜色和强度。
- 原文的标点、短语、冷暖情绪词和长度统计决定网格线密度与偏移。
- SVG 输出主要保留可编辑前景线条，PNG 则包含完整背景合成。
- 文件顶部的 `DEFAULT_CONTENT_SIZE` 控制内容生成画布，`DEFAULT_RESOLUTION` 控制普通模式最终 PNG 尺寸和 SVG `width`/`height`；只改 `DEFAULT_RESOLUTION` 不会改变 glitch 内容本身。`--high` 仍是单日 PNG 的 `5000 x 5000` 放大模式。
