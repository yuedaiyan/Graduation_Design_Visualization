# diary_ReferentialNetwork

从日记正文生成“指称网络”图。每篇日记会抽取人物、地点、物、抽象概念等实体，并把同一句中共现的实体连接成有方向的曲线网络。

## 启动命令

在仓库根目录运行：

```bash
uv run python diary_ReferentialNetwork/main.py
```

常用命令：

```bash
# 随机抽样 100 篇
uv run python diary_ReferentialNetwork/main.py --sample-size 100 --seed 20260516

# 只生成某一天或某个 stem/id；--date 可重复
uv run python diary_ReferentialNetwork/main.py --date 2020-12-08

# 明确生成全部条目
uv run python diary_ReferentialNetwork/main.py --sample-size 0

# 调整最大节点数
uv run python diary_ReferentialNetwork/main.py --max-nodes 48

# 只生成 SVG
uv run python diary_ReferentialNetwork/main.py --svg-only

# 只生成 PNG，不生成 SVG
uv run python diary_ReferentialNetwork/main.py --png-only

# 关闭某种输出格式
uv run python diary_ReferentialNetwork/main.py --no-png --svg
```

## 输入和输出

默认输入：

```text
diary_entries.merged.json
```

输出：

```text
output_All/diary_ReferentialNetwork/try_N/<stem>.png
output_All/diary_ReferentialNetwork/svg_try_N/<stem>.svg
output_All/diary_ReferentialNetwork/try_N/summary.json
```

默认 `SAMPLE_SIZE = 0`，也就是生成全部日记；默认同时生成 PNG 和 SVG。

## 主函数和关键函数

- `main()`：解析参数、读取日记、筛选条目、逐篇调用 `render_entry()`，最后写 `summary.json`。
- `parse_args()`：定义 `--root`、`--entries`、`--out-dir`、`--sample-size`、`--seed`、`--max-nodes`、`--date`、`--png/--no-png`、`--svg/--no-svg`、`--png-only`、`--svg-only`。
- `load_entries()`：从 JSON 读取正文，并生成稳定的 entry id 和 stem。
- `select_entries()`：按日期/stem/id 过滤，或按 seed 抽样。
- `split_sentences()`：切句。
- `normalize_entity()`、`is_noise()`：清理实体文本并过滤噪声词。
- `classify_token()`、`regex_entities()`、`extract_sentence_entities()`：结合词性和正则抽取实体类别。
- `build_graph()`：把同一句共现实体构造成 `networkx.Graph`，并记录节点/边信息。
- `graph_layout()`、`normalize_positions()`：生成力导向布局并归一化位置。
- `make_node_codes()`：把真实实体名映射成 A/B/C/D 编码。
- `draw_curved_arrow()`、`draw_png_foreground()`：绘制 PNG 前景网络。
- `write_svg_foreground()`：写透明前景 SVG。
- `render_entry()`：单篇日记的完整渲染入口。
- `write_summary()`：写出节点数、边数、输出文件等摘要。

## 图像规则

- 节点类别：人物 `A`、地点 `B`、物 `C`、抽象概念 `D`。
- 图中不显示真实实体名，只显示编码。
- 边表示同一句共现；方向按实体在句子里的出现顺序。
- 重复共现或相邻共现使用实线；松散共现一次使用虚线。
- PNG 有纸张背景；SVG 只保留前景网络。
- 文件顶部的 `SVG_CANVAS_SIZE` 控制 SVG 内容坐标，`SVG_OUTPUT_SIZE` 控制 SVG 显示尺寸。PNG 默认保留当前 Matplotlib 输出尺寸；需要只改最终 PNG 尺寸时，把 `PNG_OUTPUT_SIZE` 从 `0` 改成目标像素。
