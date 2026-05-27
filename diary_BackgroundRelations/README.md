# diary_BackgroundRelations

比较 `diary_MultiBackground` 已生成的各类背景语境向量，判断哪些背景在 embedding 空间里更接近、哪些更分立。这个文件夹只读已有缓存，不下载文本，也不重新向量化。

## 启动命令

在仓库根目录运行：

```bash
uv run python diary_BackgroundRelations/main.py
```

只生成 PNG、不生成 SVG：

```bash
uv run python diary_BackgroundRelations/main.py --png-only
```

常用环境变量覆盖：

```bash
BGREL_BACKGROUNDS=zh_wikipedia,zhihu_kol,douban_reviews,weibo_senti \
BGREL_METRIC_SAMPLE_PER_BACKGROUND=500 \
BGREL_PLOT_SAMPLE_PER_BACKGROUND=260 \
uv run python diary_BackgroundRelations/main.py
```

## 输入和输出

输入来自：

- `diary_MultiBackground/.cache/<background>/vectors_*.npy`
- 背景列表默认读取 `diary_MultiBackground/config.py` 的 `ENABLED_BACKGROUNDS`

输出写入：

```text
output_All/diary_BackgroundRelations/try_N/
```

主要输出：

- `01_background_scatter_map.svg/.png`：所有背景的二维散点关系图。
- `02_background_overlap_report.svg/.png`：最近邻重合矩阵和最重合背景对榜单。
- `nearest_neighbor_overlap_matrix.csv`：局部最近邻混合程度。
- `centroid_cosine_similarity_matrix.csv`：背景中心向量余弦相似度。
- `relation_pairs.csv`：背景对排序表。
- `background_summary.csv`：每个背景样本量、中心点、自聚集程度。
- `summary.txt`、`params.json`、可选 `skipped.txt`。

## 主函数和关键函数

- `main()`：解析 `config.py` 中的路径和背景列表，创建新的 `try_N` 输出目录，然后调用 `build_background_relation_outputs()`。
- `build_background_relation_outputs()`：总调度函数，加载背景向量、降维、计算关系矩阵、保存 CSV/图片/摘要。
- `load_background()`：从 `diary_MultiBackground/.cache` 中选择最新的 `vectors_*.npy`，抽取 metric 样本和 plot 样本。
- `reduce_backgrounds()`：用 PCA 把所有背景向量投到同一二维空间。
- `centroid_similarity()`：计算每个背景中心向量之间的余弦相似度。
- `neighbor_overlap()`：统计每个背景点的最近邻来自哪些背景，用来衡量局部混合。
- `plot_scatter_map()`：输出散点关系图和背景椭圆。
- `plot_overlap_report()`：输出最近邻重合热力图和文字榜单。
- `save_matrix_csv()`、`save_pair_csv()`、`save_summary_csv()`、`write_summary()`：写出结构化结果。

## 参数位置

主要参数在 `config.py`，都支持环境变量覆盖：

- `BGREL_BACKGROUNDS`
- `BGREL_METRIC_SAMPLE_PER_BACKGROUND`
- `BGREL_PLOT_SAMPLE_PER_BACKGROUND`
- `BGREL_NEIGHBOR_K`
- `BGREL_OUTPUT_IMAGE_FORMATS`
- `BGREL_SCATTER_CANVAS_WIDTH` / `BGREL_SCATTER_CANVAS_HEIGHT`
- `BGREL_SCATTER_OUTPUT_WIDTH` / `BGREL_SCATTER_OUTPUT_HEIGHT`
- `BGREL_REPORT_CANVAS_WIDTH` / `BGREL_REPORT_CANVAS_HEIGHT`
- `BGREL_REPORT_OUTPUT_WIDTH` / `BGREL_REPORT_OUTPUT_HEIGHT`

`*_CANVAS_*` 控制图表内容画布，`*_OUTPUT_*` 控制最终 PNG 尺寸；默认输出尺寸等于当前内容画布尺寸，所以现有默认结果不变。
