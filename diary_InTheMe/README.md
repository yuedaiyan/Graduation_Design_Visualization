# diary_InTheMe

把全部日记篇级向量放进一个全局二维语义地图，用来观察所有日记之间的聚类、孤立点和年份/篇幅分布。这个程序不是逐日输出，而是一次输出整套日记的总览图。

## 启动命令

在仓库根目录运行：

```bash
uv run python diary_InTheMe/main.py
```

常用命令：

```bash
# 调整聚类数量和近邻数
uv run python diary_InTheMe/main.py --clusters 14 --neighbors 18

# 指定输出目录
uv run python diary_InTheMe/main.py --out-dir output_All/diary_InTheMe_test

# 只生成 PNG，不生成 SVG
uv run python diary_InTheMe/main.py --png-only

# 指定项目根目录
uv run python diary_InTheMe/main.py --root /Users/yuedaiyan/code_school/biue_code_all
```

## 输入和输出

输入：

- `diary_vectors/*.npy`
- 可选 `diary_entries.json`，用于补充日期、字数、地点、时间等摘要信息。

输出：

```text
output_All/diary_InTheMe/try_N/diary_semantic_space.png
output_All/diary_InTheMe/try_N/diary_semantic_space.svg
output_All/diary_InTheMe/try_N/semantic_layout_cache.npz
output_All/diary_InTheMe/try_N/cluster_summary.json
```

## 主函数和关键函数

- `main()`：读取全部篇级向量，降维、聚类、绘图并写摘要。
- `parse_args()`：定义 `--root`、`--out-dir`、`--dpi`、`--seed`、`--clusters`、`--neighbors`、`--png-only`。
- `load_diary_entries()`：读取日记元数据，并处理重复日期 stem。
- `load_vectors()`：读取所有 `diary_vectors/*.npy`。
- `semantic_layout()`：优先使用 UMAP 做二维布局；如果 UMAP 不可用，回退到 PCA/SVD。
- `cluster_points()`：先尝试 DBSCAN，聚类不足时回退到 KMeans。
- `cluster_summary()`：统计每个聚类代表日期、年份范围、字数中位数和孤立日记。
- `draw_map()`：绘制全局语义地图。

## 生成逻辑

- 每个点是一篇日记。
- 点的位置来自篇级向量的全局降维结果。
- 聚类标签来自 DBSCAN 或 KMeans。
- `isolation` 表示该日记在语义空间里相对孤立的程度。
- 文件顶部的 `CANVAS_W` / `CANVAS_H` 控制 Matplotlib 内容画布比例；`OUTPUT_W` / `OUTPUT_H` 默认为 `0`，表示保留当前由 `CANVAS_W`、`CANVAS_H` 和 `--dpi` 得到的 PNG 尺寸。需要只改最终 PNG 尺寸时，把 `OUTPUT_W` / `OUTPUT_H` 改成目标像素即可。
