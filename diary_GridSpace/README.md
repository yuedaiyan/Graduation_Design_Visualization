# diary_GridSpace

从篇级向量和句级向量生成带“语义洞口”的网格空间图。每篇日记会形成一个被语义场扭曲的规则网格，中心区域被切出不规则空洞。

## 启动命令

在仓库根目录运行：

```bash
uv run python diary_GridSpace/main.py
```

常用命令：

```bash
# 只生成某一天或某个 stem
uv run python diary_GridSpace/main.py --date 2026-03-10

# 批量模式下只生成前 100 个有效日期
uv run python diary_GridSpace/main.py --count 100

# 只生成 PNG，不生成 SVG
uv run python diary_GridSpace/main.py --count 100 --png-only

# 只生成 SVG
uv run python diary_GridSpace/main.py --count 100 --svg-only

# 内容按 1000x1000 的坐标生成，最终导出 2000x2000 像素
uv run python diary_GridSpace/main.py --date 2026-03-10 --content-size 1000 --resolution 2000

# 兼容旧写法：--size 等同于 --resolution
uv run python diary_GridSpace/main.py --date 2026-03-10 --size 2000

# 指定项目根目录或输出目录
uv run python diary_GridSpace/main.py --root /Users/yuedaiyan/code_school/biue_code_all
uv run python diary_GridSpace/main.py --out-dir output_All/diary_GridSpace_test
```

## 输入和输出

输入：

- `diary_vectors/<stem>.npy`
- `diary_sentence_vectors/<stem>/diary_vector.npy`
- `diary_sentence_vectors/<stem>/sentence_vectors.npy`
- `diary_sentence_vectors/<stem>/meta.json`

输出：

```text
output_All/diary_GridSpace/try_N/<stem>.png
output_All/diary_GridSpace/svg_try_N/<stem>.svg
```

## 主函数和关键函数

- `main()`：解析参数、发现有效日期、建立共享网格场、逐篇生成 PNG/SVG。
- `parse_args()`：定义 `--date`、`--count`、`--png-only`、`--with-svg`、`--svg-only`、`--root`、`--out-dir`、`--seed`、`--dpi`、`--content-size`、`--resolution`、`--size`。
- `discover_dates()`：从 `diary_vectors` 和 `diary_sentence_vectors` 中筛选输入完整的日期。
- `load_vectors()`：读取篇级向量、句向量和 `meta.json`。
- `build_semantic_layout()`：把句向量投到二维语义布局。
- `semantic_fields()`：根据语义点生成洞口场和网格扭曲场。
- `pick_hole_polygon()`：从语义场中选出不规则洞口边界。
- `warp_points()`：把规则网格点按语义场扭曲。
- `collect_foreground_polylines()`：收集 SVG 前景网格线。
- `plot_grid_with_hole()`：保存完整 PNG。
- `write_svg_foreground()`：保存透明前景 SVG。

## 生成逻辑

- 句级向量形成语义点云。
- 语义点云决定画面中空洞的位置、大小和边缘形态。
- 篇级向量参与整体布局和扭曲方向。
- `--content-size` 控制内容自身的坐标尺寸，边距、网格步长、线宽和文字会随它等比缩放。
- `--resolution` 控制最终 PNG 像素尺寸和 SVG 的 `width` / `height`，不改变内容构图。
- 默认同时生成 PNG 和 SVG；需要只生成 PNG 时传 `--png-only`。
- PNG 包含背景、边框和网格；SVG 主要用于后续编辑前景线条。
