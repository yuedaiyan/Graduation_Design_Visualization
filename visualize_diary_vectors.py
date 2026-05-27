"""
visualize_diary_vectors.py
这是“基础可视化”的程序。

它不再读取原始日记文本，而是读取：
diary_vectors/*.npy

然后生成一些静态图和统计摘要，输出到：
diary_vectors_viz/

它主要看的是向量本身的数值结构，比如：
每篇日记向量的热力图
每个维度的均值、波动
相邻维度之间的变化
哪些维度数值更集中、更活跃
summary.txt 统计摘要

简单说：
visualize_diary_vectors.py 是看“这些向量内部长什么样”。
"""

import glob
import os
from datetime import datetime

# 保证在受限环境也可写缓存
CACHE_ROOT = os.path.abspath(".plot_cache")
os.makedirs(CACHE_ROOT, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(CACHE_ROOT, "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(CACHE_ROOT, "xdg_cache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INPUT_DIR = "diary_vectors"
OUTPUT_DIR = "diary_vectors_viz"


def load_vectors(input_dir: str):
    files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {input_dir}")

    dates = []
    vectors = []
    for fp in files:
        name = os.path.splitext(os.path.basename(fp))[0]
        try:
            dt = datetime.strptime(name, "%Y-%m-%d")
        except ValueError:
            dt = None
        x = np.load(fp).reshape(-1)
        dates.append((name, dt))
        vectors.append(x)

    X = np.stack(vectors)
    return files, dates, X


def moving_avg(x: np.ndarray, window: int = 25):
    if window <= 1:
        return x
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(x, kernel, mode="same")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _, dates, X = load_vectors(INPUT_DIR)

    n_samples, n_dims = X.shape
    dim_idx = np.arange(n_dims)

    mean_by_dim = X.mean(axis=0)
    std_by_dim = X.std(axis=0)
    mean_abs_by_dim = np.abs(X).mean(axis=0)

    adj_gap = np.abs(np.diff(X, axis=1))
    mean_adj_gap = adj_gap.mean(axis=0)
    smooth_adj_gap = moving_avg(mean_adj_gap, window=21)

    bin_size = 32
    n_bins = n_dims // bin_size
    trimmed = X[:, : n_bins * bin_size]
    bins = np.abs(trimmed).reshape(n_samples, n_bins, bin_size).mean(axis=2)
    bin_mean = bins.mean(axis=0)

    plt.figure(figsize=(14, 7))
    im = plt.imshow(X, aspect="auto", cmap="coolwarm", interpolation="nearest")
    plt.colorbar(im, label="value")
    plt.title("Diary Embedding Values Heatmap (samples x dimensions)")
    plt.xlabel("dimension index")
    plt.ylabel("date sample")
    plt.yticks(np.arange(n_samples), [d[0] for d in dates], fontsize=8)
    plt.tight_layout()
    p1 = os.path.join(OUTPUT_DIR, "01_heatmap_values.png")
    plt.savefig(p1, dpi=180)
    plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(dim_idx, mean_by_dim, label="mean", linewidth=1.2)
    axes[0].plot(dim_idx, mean_abs_by_dim, label="mean(|value|)", linewidth=1.0)
    axes[0].fill_between(
        dim_idx,
        mean_by_dim - std_by_dim,
        mean_by_dim + std_by_dim,
        alpha=0.18,
        label="mean ± std",
    )
    axes[0].set_title("Per-Dimension Mean / Volatility")
    axes[0].set_ylabel("value")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.2)

    axes[1].plot(
        np.arange(n_dims - 1),
        mean_adj_gap,
        alpha=0.45,
        linewidth=0.9,
        label="mean adjacent gap",
    )
    axes[1].plot(
        np.arange(n_dims - 1),
        smooth_adj_gap,
        linewidth=2.0,
        label="smoothed adjacent gap",
    )
    axes[1].set_title("Adjacent-Dimension Gap (|x[i+1]-x[i]|)")
    axes[1].set_xlabel("dimension index i")
    axes[1].set_ylabel("gap")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.2)

    plt.tight_layout()
    p2 = os.path.join(OUTPUT_DIR, "02_dim_stats_and_gaps.png")
    plt.savefig(p2, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    x_bins = np.arange(n_bins)
    axes[0].bar(x_bins, bin_mean, width=0.85)
    axes[0].set_title(f"Concentration by Dimension Blocks ({bin_size} dims per block)")
    axes[0].set_xlabel("block index")
    axes[0].set_ylabel("mean(|value|)")
    axes[0].grid(axis="y", alpha=0.2)

    for i in range(n_samples):
        axes[1].plot(dim_idx, X[i], alpha=0.25, linewidth=0.8)
    axes[1].plot(
        dim_idx, mean_by_dim, color="black", linewidth=2.0, label="mean profile"
    )
    axes[1].set_title("All Vector Profiles (overlay)")
    axes[1].set_xlabel("dimension index")
    axes[1].set_ylabel("value")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.2)

    plt.tight_layout()
    p3 = os.path.join(OUTPUT_DIR, "03_concentration_and_profiles.png")
    plt.savefig(p3, dpi=180)
    plt.close(fig)

    top_k = 30
    top_abs_idx = np.argsort(-mean_abs_by_dim)[:top_k]
    top_std_idx = np.argsort(-std_by_dim)[:top_k]

    summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"samples: {n_samples}\n")
        f.write(f"dimensions: {n_dims}\n")
        f.write(f"global min/max: {X.min():.6f} / {X.max():.6f}\n")
        f.write(f"global mean/std: {X.mean():.6f} / {X.std():.6f}\n")
        f.write("\nTop dimensions by mean(|value|):\n")
        for idx in top_abs_idx:
            f.write(
                f"dim {idx:4d}: mean={mean_by_dim[idx]: .6f}, mean_abs={mean_abs_by_dim[idx]: .6f}, std={std_by_dim[idx]: .6f}\n"
            )
        f.write("\nTop dimensions by std:\n")
        for idx in top_std_idx:
            f.write(
                f"dim {idx:4d}: mean={mean_by_dim[idx]: .6f}, mean_abs={mean_abs_by_dim[idx]: .6f}, std={std_by_dim[idx]: .6f}\n"
            )

    print("Generated:")
    print(p1)
    print(p2)
    print(p3)
    print(summary_path)


if __name__ == "__main__":
    main()
