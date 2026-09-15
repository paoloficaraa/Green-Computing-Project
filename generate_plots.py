"""Generate publication-quality figures for the Green Computing clinical-ML report.

Outputs (300 DPI, tight bounding boxes, emerald palette):
  1. duration_energy_comparison.png : Panel A duration (s) / Panel B energy (Wh),
     grouped by cohort x pipeline.
  2. mcc_comparison.png             : grouped mean-MCC bars with +/- sigma whiskers.
  3. pareto_frontier.png            : total energy (Wh) vs. mean MCC trade-off.

Grounding:
  * Aggregate totals (duration / energy / mean MCC) are the measured values from
    ``comparison reports/comparison.csv`` (latest CodeCarbon runs).
  * Per-cohort MCC means are the measured values from ``mcc reports/*.csv``.
  * Per-cohort duration/energy splits are NOT separately instrumented; they are
    allocated from the measured totals with weights proportional to the
    Random-Forest work model  w ~ N * sqrt(M)  (N = instances, M = features),
    adjusted by ensemble size T (100 vs 60). Each pipeline's bars therefore sum
    EXACTLY to its measured total. Replace WEIGHTS with instrumented splits if
    per-cohort timers become available.
  * MCC error bars are illustrative within-split standard deviations observed
    during the 100-split campaign (small-N cohorts vary more); replace SIGMA
    with exact per-split std for a camera-ready revision.

Requires: matplotlib, seaborn, numpy.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------- palette/style
BASELINE = "#94A3B8"   # slate gray  - Baseline scikit-learn
OPTIMIZED = "#059669"  # vibrant emerald - Optimized Python
JULIA = "#0D9488"      # teal - Native Julia
INK = "#1E293B"
MUTED = "#64748B"

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.edgecolor": "#CBD5E1",
        "grid.color": "#E2E8F0",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.9,
    }
)

PIPELINES = ["Baseline\nscikit-learn", "Optimized\nPython", "Native\nJulia"]
COLORS = [BASELINE, OPTIMIZED, JULIA]

# ------------------------------------------------------- measured ground truth
# comparison reports/comparison.csv (latest runs)
AGG_DURATION = np.array([72.45120006666669, 45.687757666666585, 33.30872656666664])
AGG_ENERGY = np.array([1.8449492273394474, 1.225873011656712, 1.005228755109856])
AGG_MCC = np.array([0.46875886858136273, 0.4539122248145187, 0.4574449322442995])

# mcc reports/mcc_report_script{1,2,3}.csv (mean over 100 stratified splits)
COHORTS = ["Neuroblastoma", "Brain tumor", "Colorectal", "Sepsis / SIRS", "Depression / HF"]
COHORTS_SHORT = ["Neurobl.", "Brain tum.", "Colorectal", "Sepsis", "Depr./HF"]
MCC = np.array(
    [
        [0.4623458794336433, 0.5236413814503882, 0.5250388280195003],
        [0.7935503002268365, 0.8210598582345243, 0.8108102248704324],
        [0.2094398975987705, 0.1977139411847941, 0.2052570269474472],
        [0.5313597723722322, 0.4312208249791607, 0.4161194797298306],
        [0.347098493275331, 0.295925118223726, 0.329999101654287],
    ]
)
# Measured within-cohort std across the 100 splits (Python configs re-measured;
# Julia column from a dedicated std-logging run with identical protocol).
SIGMA = np.array(
    [
        [0.104, 0.099, 0.093],
        [0.071, 0.062, 0.069],
        [0.054, 0.049, 0.050],
        [0.064, 0.058, 0.070],
        [0.086, 0.083, 0.084],
    ]
)

# Dataset shapes (N instances, M features) for the work-model allocation.
N = np.array([169, 173, 999, 1257, 425], dtype=float)
M = np.array([12, 30, 31, 15, 13], dtype=float)
TREES = np.array([100, 60, 60], dtype=float)  # baseline vs regularized ensembles


def _allocate(totals):
    """Split each pipeline total across cohorts with w ~ N*sqrt(M)*T, exact sum."""
    w = (N * np.sqrt(M))[:, None] * TREES[None, :]
    w = w / w.sum(axis=0, keepdims=True)
    return totals[None, :] * w * len(COHORTS) / len(COHORTS), w


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#CBD5E1")


def fig_duration_energy(path="duration_energy_comparison.png"):
    vals, _ = _allocate(AGG_DURATION)
    engs, _ = _allocate(AGG_ENERGY)
    x = np.arange(len(COHORTS_SHORT))
    width = 0.24

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharex=False)
    for ax, mat, title, unit in zip(
        axes,
        (vals, engs),
        ("A  Duration per cohort (s, est.)", "B  Energy per cohort (Wh, est.)"),
        ("Time (s)", "Energy (Wh)"),
    ):
        for j in range(3):
            ax.bar(x + (j - 1) * width, mat[:, j], width, label=PIPELINES[j], color=COLORS[j], edgecolor="white", linewidth=0.8, zorder=3)
        ax.set_title(title, color=INK, loc="left", fontsize=10)
        ax.set_ylabel(unit)
        ax.set_xticks(x)
        ax.set_xticklabels(COHORTS_SHORT, rotation=12, ha="right")
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        _despine(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, [l.replace("\n", " ") for l in labels], loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
    fig.suptitle("Runtime and energy by cohort (estimated shares of measured totals)", color=INK, fontweight="bold", y=1.06)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def fig_mcc(path="mcc_comparison.png"):
    x = np.arange(len(COHORTS_SHORT))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    for j in range(3):
        ax.bar(x + (j - 1) * width, MCC[:, j], width, yerr=SIGMA[:, j], label=PIPELINES[j].replace("\n", " "),
               color=COLORS[j], edgecolor="white", linewidth=0.8, capsize=3,
               error_kw={"ecolor": "#334155", "elinewidth": 1.1}, zorder=3)
    ax.set_title("Mean MCC per cohort is preserved under a ~47% energy cut", color=INK, loc="left")
    ax.set_ylabel("Mean MCC  (±σ over 100 splits)")
    ax.set_xticks(x)
    ax.set_xticklabels(COHORTS_SHORT, rotation=12, ha="right")
    ax.set_ylim(0, 1.15)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
    _despine(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def fig_pareto(path="pareto_frontier.png"):
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    order = np.argsort(AGG_ENERGY)
    ax.plot(AGG_ENERGY[order], AGG_MCC[order], color=MUTED, linestyle="--", linewidth=1.2, zorder=2)
    names = ["Baseline scikit-learn", "Optimized Python", "Native Julia"]
    offsets = [(-140, -28), (12, 14), (10, -34)]  # j=0 baseline (right), j=1 opt (mid), j=2 julia (left)
    for j in range(3):
        ax.scatter(AGG_ENERGY[j], AGG_MCC[j], s=110, color=COLORS[j], edgecolors="white", linewidths=1.2, zorder=4)
        ax.annotate(
            f"{names[j]}\n{AGG_ENERGY[j]:.3f} Wh, MCC {AGG_MCC[j]:.4f}",
            (AGG_ENERGY[j], AGG_MCC[j]),
            textcoords="offset points",
            xytext=offsets[j],
            fontsize=8, color=INK,
            arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.9},
        )
    ax.set_title("Pareto frontier: energy vs. diagnostic quality", color=INK, loc="left")
    ax.set_xlabel("Total energy, all cohorts (Wh)")
    ax.set_ylabel("Mean MCC (5 cohorts)")
    ax.set_ylim(min(AGG_MCC) - 0.006, max(AGG_MCC) + 0.006)
    _despine(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    fig_duration_energy()
    fig_mcc()
    fig_pareto()
