# diary_Ring

把每篇日记渲染成一个完整圆环。语义窗口按原文顺序沿圆环顺时针排列：更聚焦的窗口更接近线段，更松散/发散的窗口更容易散成粒子。

## 启动命令

在仓库根目录运行：

```bash
uv run python diary_Ring/main.py
```

常用命令：

```bash
# 只生成某一天或某个 stem
uv run python diary_Ring/main.py --date 2025-12-17

# 调整画布尺寸
uv run python diary_Ring/main.py --size 2000

# 调整粒子密度、发散影响、窗口平滑和背景变化
uv run python diary_Ring/main.py --particles 1.35
uv run python diary_Ring/main.py --impact 0.7
uv run python diary_Ring/main.py --smoothing 0.4
uv run python diary_Ring/main.py --background-variation 1.4

# 控制并发渲染数量
uv run python diary_Ring/main.py --workers 5
```

背景/潮汐波预览：

```bash
uv run python diary_Ring/tide_wave.py
uv run python diary_Ring/tide_wave.py --all
uv run python diary_Ring/tide_wave.py --date 2025-12-17
uv run python diary_Ring/tide_wave.py --date 2025-12-17 --raw-wave
```

## 输入和输出

输入：

- `diary_entries.json`
- `diary_analysis_result.json`
- `diary_vectors/<stem>.npy`
- `diary_sentence_vectors/<stem>/sentence_vectors.npy`
- `diary_sentence_vectors/<stem>/window_vectors.npy`
- `diary_sentence_vectors/<stem>/meta.json`

输出：

```text
output_All/diary_Ring/try_N/<stem>.png
output_All/diary_Ring/SVG_try_N/<stem>.svg
output_All/diary_Ring/try_N/summary.json
output_All/diary_Ring/wave_try_N/
```

SVG 只包含透明背景上的中心圆环笔触，不包含情绪背景。

## 主函数和关键函数

- `main.py.parse_args()`：定义 `--root`、`--out-dir`、`--date`、`--seed`、`--size`、`--particles`、`--impact`、`--smoothing`、`--background-variation`、`--workers`。
- `main.py.main()`：读取日记、情绪分析和全局 distinctiveness，建立 `try_N` 和 `SVG_try_N`，并发渲染每篇日记。
- `main.py.render_entry()`：单篇日记完整渲染入口，生成 PNG、SVG 和 summary item。
- `io_data.load_entries()`：读取日记并生成 stem。
- `io_data.load_emotion_profiles()`：读取 `emotion_primary` 和 `emotion_arc`。
- `io_data.next_try_dir()`：创建递增输出目录。
- `metrics.load_diary_distinctiveness()`：根据全部篇级向量计算每篇日记的全局独特性。
- `metrics.build_window_metrics()`：读取句/窗口向量，计算窗口 focus、looseness、novelty 等指标。
- `ring.merge_metrics()`：把窗口指标转成圆环 segment。
- `ring.draw_ring_content()`：实际绘制线段和粒子。
- `renderer.render_ring()`：保存带情绪背景的 PNG。
- `renderer.render_ring_svg()`：保存透明 SVG 前景。
- `background.create_emotion_background()`：根据情绪生成背景场。
- `tide_wave.main()`：独立输出背景潮汐波预览。

## 生成逻辑

- 第一句从 12 点钟方向开始，按原文顺序顺时针走完整个圆。
- segment 长度按语义窗口字符数分配。
- 高窗口内相似度和高篇级对齐度会增加线段覆盖。
- 低连贯性、高新颖性和更多主题窗口会增加粒子密度和径向扩散。
- `emotion_primary` 控制背景调色板；`emotion_arc` 控制背景运动/潮汐感。
