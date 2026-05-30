# diary_ColorBlock_ContourLines

把每篇日记渲染成色块和等高线图。图像由三层组成：情绪背景色、实词驱动的 metaball 等高线、虚词驱动的前景矩形色块。

## 启动命令

在仓库根目录运行：

```bash
uv run python diary_ColorBlock_ContourLines/main.py
```

常用命令：

```bash
# 只渲染某个日期或 stem
uv run python diary_ColorBlock_ContourLines/main.py --date 2026-03-10

# 只渲染前 N 个匹配条目
uv run python diary_ColorBlock_ContourLines/main.py --limit 20

# 只生成 PNG，不生成 SVG
uv run python diary_ColorBlock_ContourLines/main.py --png-only

# 只生成 SVG
uv run python diary_ColorBlock_ContourLines/main.py --svg-only

# 指定输出根目录
uv run python diary_ColorBlock_ContourLines/main.py --out-dir output_All/diary_ColorBlock_ContourLines_test
```

## 输入和输出

输入来自仓库根目录：

- `diary_entries.json`
- `diary_vectors/*.npy`
- `diary_sentence_vectors/<stem>/sentence_vectors.npy`

默认输出：

```text
output_All/diary_ColorBlock_ContourLines/try_N/<stem>.png
output_All/diary_ColorBlock_ContourLines/svg_try_N/<stem>.svg
```

默认同时写 PNG 和 SVG；传入 `--png-only` 时只写 PNG，传入 `--svg-only` 时只写 SVG。

等高线密度可以通过 `--contour-density` 调整，默认 `1.0` 保持原样：

```bash
uv run python diary_ColorBlock_ContourLines/main.py --date 2025-06-27 --svg-only --contour-density 2.0
```

`2.0` 会在原有等值线范围内插入更多阈值，约等于把背景等高线数量翻倍；小于 `1.0` 则会减少线条。

## 主函数和关键函数

- `main.py`：很薄的命令入口，只导入并执行 `diary_visual.pipeline.run()`。
- `diary_visual.pipeline._parse_args()`：定义 `--date`、`--limit`、`--out-dir`、`--png-only`、`--svg`、`--svg-only`、`--contour-density`。
- `diary_visual.pipeline.run()`：总调度函数，读取日记、篇级向量和句向量，匹配 stem，建立输出目录，逐篇渲染。
- `_load_entries_by_stem()`：把 `diary_entries.json` 中重复日期转成和向量文件一致的 stem，如 `2024-03-08_2`。
- `_load_or_build_doc_freq()`：统计或读取词的文档频率缓存，用于实词权重。
- `_build_emotion_centers()`：按本次参与生成的篇级向量建立情绪中心。
- `analysis.classify_emotion()`：把单篇日记向量分配到当前最接近的情绪类别。
- `analysis.analyze_function_words()`：识别虚词 token，并结合句向量估计强度。
- `analysis.get_entry_content_word_data()`：提取实词、聚类并计算类似 TF-IDF 的权重。
- `rendering.map_function_word_to_block()`：把虚词 token 映射成色块的位置、尺寸、透明度。
- `rendering.map_content_words_to_metaballs()`：把实词组映射成 metaball 球。
- `rendering.compose_and_save()`：合成背景、等高线、色块并保存 PNG。
- `rendering.compose_and_save_svg()`：保存对应 SVG。

## 图像生成逻辑

- 背景色来自当前日记被分到的情绪类别。
- 前景色块来自虚词，例如助词、连词、副词、时间词、认知词、存在/否定词等。
- 等高线来自实词。实词越高频且越少见，权重越高，对 metaball 场的影响越强。
- 如果缺少句向量，虚词强度会退回到篇级向量和稳定哈希生成的兜底值，不会跳过整张图。

参数主要在 `diary_visual/config.py`。其中 `DEFAULT_CONTENT_SIZE` 控制内容坐标和构图计算，`DEFAULT_RESOLUTION` 控制最终导出的 PNG/SVG 尺寸；只改 `DEFAULT_RESOLUTION` 不会重新改变内容布局。
