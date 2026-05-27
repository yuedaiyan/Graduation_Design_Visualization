# diary_SingleCharacterDensityMap

把所有日记放进一个模拟日记本版面，并为高频关键词生成字符密度地图。每个关键词输出一张全库叠加图，显示这个关键词相关字符在日记本坐标中的出现密度。

## 启动命令

在仓库根目录运行：

```bash
uv run python diary_SingleCharacterDensityMap/main.py
```

常用命令：

```bash
# 只生成前 20 个关键词
uv run python diary_SingleCharacterDensityMap/main.py --top-keywords 20

# 只生成 SVG
uv run python diary_SingleCharacterDensityMap/main.py --top-keywords 200 --svg-only

# 只用某一天测试；--date 可重复
uv run python diary_SingleCharacterDensityMap/main.py --date 2026-03-10 --top-keywords 20

# 修改日记本网格
uv run python diary_SingleCharacterDensityMap/main.py --chars-per-line 56 --lines-per-page 28

# 额外跳过某个关键词原始排名
uv run python diary_SingleCharacterDensityMap/main.py --skip-rank 12
```

## 输入和输出

默认输入：

- `diary_entries.json`
- `diary_entries.merged.sentence_counts.json`
- 字体：`/Users/yuedaiyan/Library/Fonts/SourceHanSansCN-VF-2.otf`

输出：

```text
output_All/diary_SingleCharacterDensityMap/try_N/<rank>_<keyword>.png
output_All/diary_SingleCharacterDensityMap/svg_try_N/<rank>_<keyword>.svg
output_All/diary_SingleCharacterDensityMap/try_N/summary.json
```

默认前 100 个关键词，默认同时输出 PNG 和 SVG。脚本内默认跳过关键词原始排名 `{43, 92}`。

## 主函数和关键函数

- `main()`：读取参数、日记和关键词，分配日记本页面，逐关键词渲染 PNG/SVG 和 summary。
- `parse_args()`：定义 `--root`、`--entries`、`--keyword-counts`、`--out-dir`、`--top-keywords`、`--date`、版面参数、字体参数、密度参数、输出格式参数。
- `load_entries()`：读取日记正文并生成 stem。
- `select_entries()`：按 `--date` 过滤日记。
- `skip_keyword_ranks()`：合并默认跳过排名和命令行 `--skip-rank`。
- `load_top_keywords()`：从关键词统计 JSON 读取要生成的关键词。
- `notebook_positions()`：把正文字符映射到页、行、列。
- `assign_notebook_pages()`：给每篇日记分配连续日记本页面。
- `global_xy()`：把页/行/列转换成画布坐标。
- `render_items_for_keyword()`：找出某关键词在所有日记中的字符位置。
- `aggregate_items()`：把同一位置、同一字符的多次出现合并成密度 glyph。
- `density_for_count()`、`blur_for_density()`、`stroke_for_density()`：把叠加次数映射成透明度、模糊和描边。
- `save_png()`：保存带纸色背景的 PNG。
- `save_svg()`：保存透明前景 SVG。

## 生成逻辑

- 默认日记本版面是 56 字/行、28 行/页。
- 所有页面叠加到同一张方形画布。
- 同一关键词在同一格反复出现时，会通过加粗、多次偏移绘制和模糊显示密度。
- PNG 包含背景色；SVG 只保留字符密度前景。
