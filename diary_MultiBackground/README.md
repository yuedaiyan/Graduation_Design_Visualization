# diary_MultiBackground

把 `diary_vectors/*.npy` 放进多个背景文本语境中，观察日记相对于百科、新闻、问答、评论、诗词、歌词、法律文本、心理话语等背景的位置。日记保持“一篇/一天一个点”；背景文本会被切分或合并到接近日记长度的 chunk 后再向量化。

## 启动命令

在仓库根目录运行：

```bash
uv run python diary_MultiBackground/main.py
```

常用命令：

```bash
# 小规模 smoke test
uv run python diary_MultiBackground/main.py --test

# 只根据已有二维缓存重新绘图，不下载、不向量化、不降维
uv run python diary_MultiBackground/main.py --replot-cache

# 只生成 PNG，不生成 SVG
uv run python diary_MultiBackground/main.py --png-only

# 降低 replot 所需缓存样本数
uv run python diary_MultiBackground/main.py --replot-cache --replot-min-items 20

# 预先下载/整理背景文本缓存，不做 embedding
uv run python diary_MultiBackground/prefetch_texts.py
```

常用环境变量：

```bash
MULTIBG_ENABLED_BACKGROUNDS=zh_wikipedia,zhihu_kol \
MULTIBG_TEXT_CACHE_TARGET_ITEMS=200 \
MULTIBG_TEXT_CACHE_MIN_ITEMS=20 \
MULTIBG_VECTORIZE_TEXT_LIMIT=100 \
uv run python diary_MultiBackground/main.py --test
```

## 输入和输出

输入：

- `diary_vectors/*.npy`
- `Qwen3-Embedding-0.6B`
- `diary_MultiBackground/corpora/<background_key>/` 下的本地补充语料
- `config.py` 中配置的 HuggingFace、本地文件或 URL 数据源

缓存：

```text
diary_MultiBackground/.cache/<background_key>/
```

输出：

```text
output_All/diary_MultiBackground/try_N/
```

每个背景会有单独子目录，例如：

```text
output_All/diary_MultiBackground/try_N/zh_wikipedia/
output_All/diary_MultiBackground/try_N/zhihu_kol/
```

主要输出：

- `01_world_umap_or_pca.svg/.png`：背景点和日记点的全局图。
- `02_diary_region_zoom.svg/.png`：日记区域局部放大图。
- `03_world_diary_area50.svg/.png`：全局投影中把日记区域视觉压缩到固定比例的图。
- `points.csv`：背景点、日记点和二维坐标。
- `summary.txt`：本背景的面积、采样、缓存、耗时摘要。
- `summary_all.txt`：本次所有背景的总摘要。
- `params.json`：本次参数快照。
- `png/`：所有 PNG 的集中镜像。
- `viewer.html` 和 `viewer_data/`：可交互查看点位文本。
- `background_relations/`：背景之间的关系图和矩阵。

## 主函数和关键函数

- `main.py`：命令入口，只负责把仓库根目录加入 import path，然后调用 `pipeline.main()`。
- `pipeline.main()`：总调度函数，解析 `--test`、`--replot-cache` 等参数，建立 `try_N`，逐个背景生成或重绘。
- `pipeline.enabled_background_keys()`：决定本次启用哪些背景。
- `pipeline.load_diary_vectors()`：读取日记篇级向量。
- `pipeline.collect_texts()`：从本地语料、数据集或 URL 收集背景文本，并写 text cache。
- `pipeline.split_text_chunks()` / `pipeline.row_to_records()`：把背景文本切成接近日记长度的 chunk。
- `pipeline.load_or_create_background_vectors()`：读取或生成背景 embedding 缓存。
- `pipeline.embed_texts()`：通过 `diary_core.embedding.QwenEmbedder` 批量向量化背景文本。
- `pipeline.reduce_points()`：把背景点和日记点投影到同一二维空间。
- `pipeline.build_one_background()`：完整生成一个背景的图、CSV、summary 和 viewer 数据。
- `pipeline.replot_cached_background()`：只读取已有二维缓存重新输出图。
- `pipeline.write_background_relation_chart()`：调用 `diary_BackgroundRelations` 生成背景关系图。
- `output.save_figure_all_formats()`：按 `OUTPUT_IMAGE_FORMATS` 保存 SVG/PNG，并同步到 `png/`。
- `output.plot_world()`、`plot_zoom()`、`plot_world_diary_area_scaled()`：绘制三类主要图。
- `output.save_points_csv()`、`summarize_metrics()`：写坐标和统计。
- `viewer.write_viewer_background_data()`、`write_viewer_shell()`：生成交互式查看器。
- `prefetch_texts.main()`：只预抓取文本缓存，适合正式跑图前先准备语料。

## 参数位置

主要参数在 `config.py`，也支持环境变量覆盖：

- `MULTIBG_ENABLED_BACKGROUNDS`
- `MULTIBG_TEXT_CACHE_TARGET_ITEMS`
- `MULTIBG_TEXT_CACHE_MIN_ITEMS`
- `MULTIBG_VECTORIZE_TEXT_LIMIT`
- `MULTIBG_TEXT_CACHE_MAX_BYTES`
- `MULTIBG_OUTPUT_IMAGE_FORMATS`
- `HF_ENDPOINT`

尺寸相关参数在 `config.py` 顶部：`DEFAULT_CONTENT_SIZE` 控制图表内容画布，`DEFAULT_RESOLUTION` 控制最终 PNG 尺寸；默认仍是 `1000 x 1000`，只改 `DEFAULT_RESOLUTION` 不会重新改变图表内容布局。
