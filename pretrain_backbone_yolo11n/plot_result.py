# plot_results_like_ultralytics.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 1) Đường dẫn results.csv của Ultralytics
# Ví dụ: runs/detect/train/results.csv (đổi lại cho đúng)
RESULTS_CSV = r"runs/detect/train2/results.csv"

# 2) Thông số smooth (giống Ultralytics hay dùng rolling mean)
SMOOTH_WIN = 7

TOP = [
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
]
BOT = [
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
]

df = pd.read_csv(RESULTS_CSV).sort_values("epoch").reset_index(drop=True)

# rolling smooth
smooth = df.copy()
for c in df.columns:
    if c != "epoch" and pd.api.types.is_numeric_dtype(df[c]):
        smooth[c] = df[c].rolling(SMOOTH_WIN, min_periods=1).mean()

fig, axes = plt.subplots(2, 5, figsize=(14, 6), sharex=True)

def plot_one(ax, col, show_legend=False):
    if col not in df.columns:
        ax.set_title(f"{col}\n(missing)")
        ax.axis("off")
        return
    ax.plot(df["epoch"], df[col], marker="o", markersize=2, linewidth=1, label="results")
    ax.plot(df["epoch"], smooth[col], linestyle="--", linewidth=1.5, label="smooth")
    ax.set_title(col)
    if show_legend:
        ax.legend(fontsize=8)

# hàng trên
for i, col in enumerate(TOP):
    plot_one(axes[0, i], col, show_legend=(i == 1))  # legend giống ảnh (ở ô train/cls_loss)

# hàng dưới
for i, col in enumerate(BOT):
    plot_one(axes[1, i], col, show_legend=False)

for ax in axes[1, :]:
    ax.set_xlabel("epoch")
for ax in axes[:, 0]:
    ax.set_ylabel("value")
for ax in axes.ravel():
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)  # ẩn số epoch
    ax.set_xlabel("")  # bỏ chữ epoch (nếu có)

plt.tight_layout()
out = Path("results_like_ultralytics.png")
plt.savefig(out, dpi=200)
plt.close()
print("Saved:", out.resolve())
