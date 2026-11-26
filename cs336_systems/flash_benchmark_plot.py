"""Block 5: figures for the FlashAttention benchmark report.

    uv run python -m cs336_systems.flash_benchmark_plot

Reads the CSVs written by flash_benchmark.py and writes PNGs to reports/.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

R = "reports"
# Categorical slots 1-3 of the dataviz reference palette, in fixed order.
# Identity is never color-alone: each impl also gets its own marker + direct label.
STYLE = {
    "naive": dict(color="#257de0", marker="o", label="naive PyTorch"),
    "triton": dict(color="#f47230", marker="s", label="Triton FlashAttention-2"),
    "tiled": dict(color="#00b894", marker="^", label="eager tiled PyTorch"),
}
TEXT, GRID = "#333333", "#e5e5e5"
plt.rcParams.update({
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "lines.linewidth": 2, "lines.markersize": 6,
})
SUBTITLE = "RTX 3070 · fp32 (no TF32) · causal · batch 1 · tile 16 · do_bench median GPU time"


def load(path):
    df = pd.read_csv(path, comment="#")
    df["fwd"] = pd.to_numeric(df["fwd_ms"], errors="coerce")  # OOM / skipped -> NaN
    return df


def log2_x(ax, seqs):
    ax.set_xscale("log", base=2)
    ax.set_xticks(seqs)
    ax.set_xticklabels([f"{s // 1024}K" if s >= 1024 else str(s) for s in seqs])
    ax.set_xlabel("sequence length N")


def finish(fig, ax, title, out, subtitle=SUBTITLE):
    ax.set_title(title, loc="left", fontsize=12, color=TEXT, pad=22)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=8.5, color="#777777")
    fig.tight_layout()
    fig.savefig(f"{R}/{out}", dpi=160)
    plt.close(fig)
    print("wrote", f"{R}/{out}")


def fig_latency():
    df = load(f"{R}/flash_fp32_coarse_skip.csv")
    df = df[df.d_head == 64]
    seqs = sorted(df.seq_len.unique())
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for impl in ("tiled", "naive", "triton"):
        g = df[df.impl == impl].dropna(subset=["fwd"]).sort_values("seq_len")
        s = STYLE[impl]
        ax.plot(g.seq_len, g.fwd, color=s["color"], marker=s["marker"], label=s["label"])
        last = g.iloc[-1]
        ax.annotate(s["label"], (last.seq_len, last.fwd), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=TEXT)
    # Mark where naive stops: out of memory, not "not measured".
    oom = df[(df.impl == "naive") & (df.fwd_ms == "OOM")].seq_len.min()
    ax.axvline(oom, color="#999999", linestyle=":", linewidth=1)
    ax.text(oom, ax.get_ylim()[1], " naive OOM (8 GB) →", fontsize=8.5, color=TEXT, va="top")
    log2_x(ax, seqs)
    ax.set_yscale("log")
    ax.set_ylabel("forward latency (ms, log)")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    finish(fig, ax, "Forward latency vs sequence length (d_head = 64)", "flash_fwd_latency.png")


def speedup(df, d=64):
    p = df[df.d_head == d].pivot_table(index="seq_len", columns="impl", values="fwd")
    return (p["naive"] / p["triton"]).dropna()


def fig_speedup_skip():
    before = speedup(load(f"{R}/flash_fp32_coarse.csv"))
    after = speedup(load(f"{R}/flash_fp32_coarse_skip.csv"))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.axhline(1, color="#999999", linestyle="--", linewidth=1)
    ax.text(after.index[-1], 0.97, "parity (below = naive faster)", fontsize=8.5, color="#777777",
            ha="right", va="top")
    for series, label, color, marker in ((before, "no tile skip", "#257de0", "o"),
                                         (after, "causal tile skip", "#f47230", "s")):
        ax.plot(series.index, series.values, color=color, marker=marker, label=label)
        ax.annotate(f"{label}: {series.values[-1]:.2f}×", (series.index[-1], series.values[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color=TEXT)
    log2_x(ax, list(before.index))
    ax.set_ylabel("speedup = naive / Triton (forward)")
    ax.set_ylim(0, max(after.max(), before.max()) * 1.15)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    finish(fig, ax, "Triton speedup over naive grows with N only after skipping future tiles",
           "flash_speedup_skip.png")


def fig_speedup_dims():
    df = pd.concat([load(f"{R}/flash_fp32_coarse_skip.csv"), load(f"{R}/flash_fp32_coarse_skip_dims.csv")])
    dims = sorted(df.d_head.unique())
    colors = ["#257de0", "#f47230", "#00b894", "#f2a600"]  # slots 1-4, fixed order by d
    markers = ["o", "s", "^", "D"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.axhline(1, color="#999999", linestyle="--", linewidth=1)
    for d, c, m in zip(dims, colors, markers):
        s = speedup(df, d)
        ax.plot(s.index, s.values, color=c, marker=m, label=f"d_head = {d}")
        ax.annotate(f"d={d}", (s.index[-1], s.values[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=TEXT)
    log2_x(ax, sorted(df.seq_len.unique()))
    ax.set_ylabel("speedup = naive / Triton (forward)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    finish(fig, ax, "Speedup by head dimension (with causal tile skip)", "flash_speedup_dims.png")


def fig_memory():
    df = pd.read_csv(f"{R}/flash_fp32_memory.csv", comment="#")
    df["mib"] = pd.to_numeric(df["peak_mib"], errors="coerce")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for impl in ("naive", "triton"):
        g = df[df.impl == impl].dropna(subset=["mib"])
        s = STYLE[impl]
        ax.plot(g.seq_len, g.mib, color=s["color"], marker=s["marker"], label=s["label"])
        last = g.iloc[-1]
        ax.annotate(f"{last.mib:,.1f} MiB", (last.seq_len, last.mib), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=TEXT)
    oom = df[(df.impl == "naive") & (df.peak_mib == "OOM")].seq_len.min()
    ax.axvline(oom, color="#999999", linestyle=":", linewidth=1)
    ax.text(oom, 2000, " naive OOM →", fontsize=8.5, color=TEXT)
    log2_x(ax, sorted(df.seq_len.unique()))
    ax.set_yscale("log")
    ax.set_ylabel("peak extra memory in forward (MiB, log)")
    ax.legend(frameon=False, loc="center left", fontsize=8.5)
    finish(fig, ax, "Memory: naive grows O(N²), FlashAttention O(N·d)", "flash_memory.png",
           subtitle="RTX 3070 · fp32 · causal · batch 1 · d_head 64 · torch.cuda.max_memory_allocated")


if __name__ == "__main__":
    fig_latency()
    fig_speedup_skip()
    fig_memory()
    import os
    if os.path.exists(f"{R}/flash_fp32_coarse_skip_dims.csv"):
        fig_speedup_dims()
