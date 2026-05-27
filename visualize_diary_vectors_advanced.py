"""
visualize_diary_vectors_advanced.py
这是“高级可视化”的程序。

它也读取：
diary_vectors/*.npy
但它关注的是日记之间的关系和时间变化，而不是单纯看每个维度。

它会生成：
04_pca_trajectory.png
用 PCA 把高维向量压缩到二维，看日记在语义空间里的轨迹。

05_umap_trajectory.png 或 05_t-sne_trajectory.png
用 UMAP 或 t-SNE 做更强的二维降维，看日记之间的聚类和距离。

06_vector_profiles_over_time.gif
动画，展示每一天的向量轮廓如何随时间变化。

advanced_summary.txt
保存 PCA 坐标、解释方差等信息。

简单说：
visualize_diary_vectors_advanced.py 是看“不同日记之间的语义距离和时间轨迹”。
"""

import glob
import os
from datetime import datetime

import numpy as np

# headless plot cache for sandbox env
CACHE_ROOT = os.path.abspath(".plot_cache")
os.makedirs(CACHE_ROOT, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(CACHE_ROOT, "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(CACHE_ROOT, "xdg_cache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

INPUT_DIR = "diary_vectors"
OUT_DIR = "diary_vectors_viz"


def load_data():
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.npy")))
    if not files:
        raise FileNotFoundError("No vector files found")

    dates = []
    vecs = []
    for fp in files:
        date_str = os.path.splitext(os.path.basename(fp))[0]
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            dt = datetime.min
        dates.append((date_str, dt))
        vecs.append(np.load(fp).reshape(-1))

    order = np.argsort([d[1] for d in dates])
    dates_sorted = [dates[i][0] for i in order]
    X = np.stack([vecs[i] for i in order])
    return dates_sorted, X


def make_pca_trajectory(dates, X):
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X)

    plt.figure(figsize=(10, 8))
    plt.plot(Z[:, 0], Z[:, 1], "-o", linewidth=1.8, markersize=5)

    for i, d in enumerate(dates):
        if i == 0 or i == len(dates) - 1 or i % 4 == 0:
            plt.annotate(d, (Z[i, 0], Z[i, 1]), fontsize=8, alpha=0.9)

    plt.title(
        f"Diary Vector Trajectory in PCA Space "
        f"(PC1 {pca.explained_variance_ratio_[0]*100:.1f}%, "
        f"PC2 {pca.explained_variance_ratio_[1]*100:.1f}%)"
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    out = os.path.join(OUT_DIR, "04_pca_trajectory.png")
    plt.savefig(out, dpi=180)
    plt.close()
    return out, Z, pca


def make_umap_or_tsne(dates, X):
    # UMAP optional; fallback to t-SNE when UMAP is unavailable.
    try:
        import umap

        reducer = umap.UMAP(
            n_components=2, random_state=42, n_neighbors=min(10, len(X) - 1)
        )
        Z = reducer.fit_transform(X)
        method = "UMAP"
    except Exception:
        from sklearn.manifold import TSNE

        Z = TSNE(
            n_components=2,
            random_state=42,
            init="pca",
            perplexity=min(8, max(2, len(X) // 3)),
            learning_rate="auto",
        ).fit_transform(X)
        method = "t-SNE"

    plt.figure(figsize=(10, 8))
    plt.plot(Z[:, 0], Z[:, 1], "-o", linewidth=1.6, markersize=4.5)
    for i, d in enumerate(dates):
        if i == 0 or i == len(dates) - 1 or i % 4 == 0:
            plt.annotate(d, (Z[i, 0], Z[i, 1]), fontsize=8, alpha=0.9)
    plt.title(f"Diary Vector Trajectory in {method} Space")
    plt.xlabel(f"{method}-1")
    plt.ylabel(f"{method}-2")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    out = os.path.join(OUT_DIR, f"05_{method.lower()}_trajectory.png")
    plt.savefig(out, dpi=180)
    plt.close()
    return out, method


def make_profile_animation_gif(dates, X):
    import imageio.v2 as imageio

    n_samples, n_dims = X.shape
    x = np.arange(n_dims)
    ymin, ymax = float(X.min()), float(X.max())

    tmp_dir = os.path.join(OUT_DIR, "_frames")
    os.makedirs(tmp_dir, exist_ok=True)
    frame_paths = []

    for i in range(n_samples):
        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(x, X[i], color="#1f77b4", linewidth=1.3, label="current vector")
        ax.plot(
            x,
            X[: i + 1].mean(axis=0),
            color="black",
            linewidth=2.0,
            alpha=0.85,
            label="running mean",
        )

        ax.set_title(f"Vector Profile Over Dimensions - {dates[i]} ({i+1}/{n_samples})")
        ax.set_xlabel("dimension index")
        ax.set_ylabel("value")
        ax.set_ylim(ymin * 1.05, ymax * 1.05)
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right")

        fp = os.path.join(tmp_dir, f"frame_{i:03d}.png")
        plt.tight_layout()
        plt.savefig(fp, dpi=120)
        plt.close(fig)
        frame_paths.append(fp)

    gif_path = os.path.join(OUT_DIR, "06_vector_profiles_over_time.gif")
    images = [imageio.imread(fp) for fp in frame_paths]
    imageio.mimsave(gif_path, images, duration=0.8, loop=0)

    # keep frames for optional debugging if needed; but clean by default
    for fp in frame_paths:
        try:
            os.remove(fp)
        except OSError:
            pass
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    return gif_path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    dates, X = load_data()

    pca_out, Z, pca = make_pca_trajectory(dates, X)
    manifold_out, method = make_umap_or_tsne(dates, X)
    gif_out = make_profile_animation_gif(dates, X)

    out_txt = os.path.join(OUT_DIR, "advanced_summary.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"samples: {X.shape[0]}\n")
        f.write(f"dimensions: {X.shape[1]}\n")
        f.write(
            f"PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.6f}, PC2={pca.explained_variance_ratio_[1]:.6f}\n"
        )
        f.write(f"manifold method: {method}\n")
        f.write("\nPCA coordinates by date:\n")
        for d, (a, b) in zip(dates, Z):
            f.write(f"{d}: ({a:.6f}, {b:.6f})\n")

    print("Generated:")
    print(pca_out)
    print(manifold_out)
    print(gif_out)
    print(out_txt)


if __name__ == "__main__":
    main()
