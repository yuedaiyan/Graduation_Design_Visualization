# diary_PerspectiveSpace

把每篇日记渲染成透视空间图。篇级向量决定整体风格和地平线，窗口级向量决定多个消失点和透视线结构。

## 启动命令

在仓库根目录运行：

```bash
uv run python diary_PerspectiveSpace/main.py
```

常用命令：

```bash
# 只生成某一天或某个 stem
uv run python diary_PerspectiveSpace/main.py --date 2026-03-10

# 批量模式下只生成前 100 个有效日期
uv run python diary_PerspectiveSpace/main.py --count 100

# 只生成 SVG
uv run python diary_PerspectiveSpace/main.py --count 100 --svg-only

# 只生成 PNG，不生成 SVG
uv run python diary_PerspectiveSpace/main.py --count 100 --png-only

# 指定输出目录
uv run python diary_PerspectiveSpace/main.py --out-dir output_All/diary_PerspectiveSpace_test
```

## 输入和输出

输入：

- `diary_vectors/<stem>.npy`
- `diary_sentence_vectors/<stem>/window_vectors.npy`

输出：

```text
output_All/diary_PerspectiveSpace/try_N/<stem>.png
output_All/diary_PerspectiveSpace/svg_try_N/<stem>.svg
```

样式上下文缓存位于：

```text
.cache/derived/diary_PerspectiveSpace/
```

## 主函数和关键函数

- `main()`：解析参数、筛选日期、建立全局样式上下文、逐篇输出 PNG/SVG。
- `parse_args()`：定义 `--root`、`--out-dir`、`--date`、`--count`、`--png-only`、`--svg-only`、`--dpi`、`--seed`。
- `discover_dates()`：筛选同时有篇级向量和窗口向量的日期。
- `build_style_context()`：根据全部日记篇级向量建立全局风格基准和缓存。
- `vec_style()`：把单篇篇级向量映射为地平线、色彩、线条密度等风格参数。
- `style_colors()`：把风格值转成背景和线条颜色。
- `event_vanishing_points()`：从窗口向量生成透视消失点。
- `draw_background()`：绘制 PNG 背景渐变和空间底色。
- `collect_perspective_lines()`：生成可复用的透视线几何。
- `draw_perspective_grid()`：绘制 PNG 透视线。
- `draw_noise()`：加入轻微噪点。
- `render_one()`：输出单篇 PNG。
- `write_svg_foreground()`：输出透明前景 SVG。

## 生成逻辑

- 篇级向量负责全局画面性格。
- 窗口级向量负责一篇日记内部事件/语义段的透视方向。
- PNG 包含背景、边框、日期和噪点；SVG 主要保留可编辑前景线条。
- 文件顶部的 `DEFAULT_CONTENT_SIZE` 控制内容坐标，`DEFAULT_RESOLUTION` 控制最终 PNG 尺寸和 SVG `width`/`height`；默认仍是 `1000 x 1000`，只改输出尺寸不会改变透视线内容。
