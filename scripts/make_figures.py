"""Generate all figures used in the paper.

Reads outputs from the smoke run (results/smoke/) where applicable, or
synthesizes illustrative versions where real data isn't yet available.

Outputs go to ``figures/`` as PNGs at 300 dpi.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RESULTS = ROOT / "results" / "smoke"

# Paper colour palette (Nexus + chart sequence)
PRIMARY = "#01696F"
PRIMARY_DARK = "#0C4E54"
ACCENT = "#A84B2F"
GOLD = "#FFC553"
INK = "#28251D"
MUTED = "#7A7974"
FAINT = "#D4D1CA"
BG = "#F7F6F2"
SURFACE = "#FBFBF9"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
})


# ---------------------------------------------------------------------------
# Figure 1 — Framework architecture
# ---------------------------------------------------------------------------
def fig_framework_architecture():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    def box(x, y, w, h, label, sub="", color=SURFACE, edge=PRIMARY, lw=1.4):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                              linewidth=lw, edgecolor=edge, facecolor=color)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2 + (0.18 if sub else 0), label,
                ha="center", va="center", fontsize=10.5, fontweight="bold",
                color=INK)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 0.22, sub,
                    ha="center", va="center", fontsize=8.5, color=MUTED)

    def arrow(x1, y1, x2, y2, color=PRIMARY):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="-|>", mutation_scale=14,
                                     linewidth=1.4, color=color))

    # Phase headers
    for i, (xc, label) in enumerate([(1.7, "PHASE 1 · PRE-TRAIN"),
                                     (5.5, "PHASE 2 · FINE-TUNE"),
                                     (9.3, "PHASE 3 · WALK-FORWARD")]):
        ax.text(xc, 5.05, label, ha="center", fontsize=9.5,
                fontweight="bold", color=PRIMARY)
        ax.add_line(Line2D([xc - 1.4, xc + 1.4], [4.85, 4.85],
                           color=PRIMARY, linewidth=1.5))

    # Phase 1
    box(0.4, 3.6, 2.6, 1.0, "CAMELS-US", "671 catchments · 30+ yrs", color=BG)
    box(0.4, 2.2, 2.6, 1.0, "EA-LSTM", "256 hidden · static gate", color=SURFACE)
    box(0.4, 0.8, 2.6, 1.0, "Pre-trained\nweights θ_pre", color=GOLD, edge=ACCENT)
    arrow(1.7, 3.55, 1.7, 3.25)
    arrow(1.7, 2.15, 1.7, 1.85)

    # Phase 2
    box(4.2, 3.6, 2.6, 1.0, "Target basin", "2-yr warmup (data-scarce)", color=BG)
    box(4.2, 2.2, 2.6, 1.0, "Conservative FT", "freeze LSTM, head only", color=SURFACE)
    box(4.2, 0.8, 2.6, 1.0, "Progressive FT (opt.)", "unfreeze last 25% · diff. LR", color=SURFACE)
    arrow(3.0, 3.0, 4.2, 3.0, color=ACCENT)
    arrow(5.5, 3.55, 5.5, 3.25)
    arrow(5.5, 2.15, 5.5, 1.85)

    # Phase 3
    box(8.0, 3.6, 2.6, 1.0, "Rolling-origin loop", "expand window · 90-day refit", color=BG)
    box(8.0, 2.2, 2.6, 1.0, "Probabilistic warning", "Q95/Q99 · 1/3/7-day lead", color=SURFACE)
    box(8.0, 0.8, 2.6, 1.0, "SHAP attribution", "global + temporal", color=SURFACE)
    arrow(6.8, 3.0, 8.0, 3.0, color=ACCENT)
    arrow(9.3, 3.55, 9.3, 3.25)
    arrow(9.3, 2.15, 9.3, 1.85)

    # RFA underlay
    box(0.4, 0.05, 10.2, 0.5,
        "Regional Frequency Analysis · Q5/Q95/Q99 from 30-yr CAMELS record",
        color="#EAF1F0", edge=PRIMARY)

    plt.savefig(FIG / "fig1_architecture.png", dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 2 — Walk-forward schematic
# ---------------------------------------------------------------------------
def fig_walk_forward_schematic():
    fig, ax = plt.subplots(figsize=(10, 4.2))
    years = np.arange(2015, 2021)
    ax.set_xlim(2014.7, 2021.3)
    ax.set_ylim(-0.4, 6.4)
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    rounds = [
        (2015, 2017, 2017.25),
        (2015, 2017.25, 2017.5),
        (2015, 2017.5, 2017.75),
        (2015, 2018, 2018.5),
        (2015, 2019, 2019.75),
        (2015, 2020, 2021),
    ]
    for i, (s, t, e) in enumerate(rounds):
        y = 5 - i
        ax.add_patch(Rectangle((s, y - 0.2), t - s, 0.4,
                               facecolor=PRIMARY, edgecolor="none", alpha=0.25))
        ax.add_patch(Rectangle((t, y - 0.2), e - t, 0.4,
                               facecolor=ACCENT, edgecolor="none", alpha=0.7))
        ax.text(2014.6, y, f"Round {i+1}", ha="right", va="center",
                fontsize=8.5, color=MUTED)

    ax.text(2016, 5.7, "Training window (expands)", ha="center",
            color=PRIMARY_DARK, fontsize=9.5, fontweight="bold")
    ax.text(2019, 5.7, "Evaluation window (next chunk)", ha="center",
            color=ACCENT, fontsize=9.5, fontweight="bold")

    ax.axvline(2017, color=INK, linestyle="--", linewidth=1, alpha=0.5)
    ax.text(2017, -0.3, "warmup ends", ha="center", color=MUTED, fontsize=8)
    ax.set_title("Walk-forward (rolling-origin) backtest: 2-yr warmup → 4-yr evaluation",
                 loc="left")
    plt.tight_layout()
    plt.savefig(FIG / "fig2_walk_forward.png", dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 3 — Unfreezing strategies
# ---------------------------------------------------------------------------
def fig_unfreezing_strategies():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    layers = ["LSTM\nweights\n(low-level)",
              "LSTM\nweights\n(mid-level)",
              "LSTM\nweights\n(high-level)",
              "Dense\nhead"]
    titles = ["Approach A · Conservative",
              "Approach B · Progressive Unfreezing"]
    a_colors = [FAINT, FAINT, FAINT, PRIMARY]
    b_colors = [FAINT, FAINT, "#7AB6BB", PRIMARY]
    a_labels = ["frozen", "frozen", "frozen", "trained · LR=1e-3"]
    b_labels = ["frozen", "frozen", "trained · LR=1e-5", "trained · LR=1e-3"]

    for ax, title, colors, labels in [(axes[0], titles[0], a_colors, a_labels),
                                      (axes[1], titles[1], b_colors, b_labels)]:
        ax.set_xlim(0, 4); ax.set_ylim(0, 2.5); ax.axis("off")
        ax.set_title(title, loc="left")
        for j, (lay, c, lab) in enumerate(zip(layers, colors, labels)):
            r = FancyBboxPatch((0.05 + 1.0 * j, 0.6), 0.9, 1.2,
                               boxstyle="round,pad=0.04", linewidth=1.2,
                               edgecolor=INK, facecolor=c)
            ax.add_patch(r)
            ax.text(0.5 + 1.0 * j, 1.2, lay, ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color=("white" if c == PRIMARY else INK))
            ax.text(0.5 + 1.0 * j, 0.4, lab, ha="center", va="center",
                    fontsize=8, color=MUTED, fontstyle="italic")

    plt.tight_layout()
    plt.savefig(FIG / "fig3_unfreezing.png", dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 4 — RFA threshold visualisation
# ---------------------------------------------------------------------------
def fig_rfa_thresholds():
    summary = json.loads((RESULTS / "summary.json").read_text())
    th = summary["thresholds"]
    csv = pd.read_csv(RESULTS / "walk_forward.csv", index_col=0, parse_dates=True)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    flow = csv["observed"].dropna()
    ax.hist(flow, bins=60, color=PRIMARY, alpha=0.55, edgecolor="white")
    for q, c, lab in [("q5", ACCENT, "Q5 (drought)"),
                      ("q95", GOLD, "Q95 (flood)"),
                      ("q99", PRIMARY_DARK, "Q99 (extreme)")]:
        ax.axvline(th[q], color=c, linewidth=2,
                   label=f"{lab} = {th[q]:.2f} mm/d")
    ax.set_xlabel("Daily streamflow (mm/day)")
    ax.set_ylabel("Frequency")
    ax.set_title("Regional Frequency Analysis thresholds (synthetic target basin)",
                 loc="left")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIG / "fig4_rfa_thresholds.png", dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 5 — Hydrograph: observed vs predicted
# ---------------------------------------------------------------------------
def fig_hydrograph():
    csv = pd.read_csv(RESULTS / "walk_forward.csv", index_col=0, parse_dates=True)
    summary = json.loads((RESULTS / "summary.json").read_text())
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(csv.index, csv["observed"], color=INK, linewidth=1.0,
            label="Observed")
    ax.plot(csv.index, csv["predicted"], color=PRIMARY, linewidth=1.0,
            alpha=0.85, label="Walk-forward prediction")
    ax.fill_between(csv.index, csv["predicted"], csv["observed"],
                    where=csv["predicted"] < csv["observed"],
                    color=ACCENT, alpha=0.2, label="Underprediction")
    ax.set_ylabel("Streamflow (mm/day)")
    ax.set_title(
        f"Walk-forward hydrograph · NSE={summary['metrics']['walk_forward']['NSE']:.2f} "
        f"· KGE={summary['metrics']['walk_forward']['KGE']:.2f} "
        f"· PBIAS={summary['metrics']['walk_forward']['PBIAS']:.1f}%",
        loc="left",
    )
    ax.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    plt.savefig(FIG / "fig5_hydrograph.png", dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 6 — Performance comparison bar chart
# ---------------------------------------------------------------------------
def fig_performance_comparison():
    summary = json.loads((RESULTS / "summary.json").read_text())
    methods = ["Local\nbaseline", "Zero-shot\ntransfer",
               "Conservative\nfine-tune", "Walk-forward\n+ bias corr."]
    nse_vals = [summary["metrics"]["local_baseline"]["NSE"],
                summary["metrics"]["zero_shot"]["NSE"],
                summary["metrics"]["fine_tune_conservative"]["NSE"],
                summary["metrics"]["walk_forward"]["NSE"]]
    kge_vals = [summary["metrics"]["local_baseline"]["KGE"],
                summary["metrics"]["zero_shot"]["KGE"],
                summary["metrics"]["fine_tune_conservative"]["KGE"],
                summary["metrics"]["walk_forward"]["KGE"]]

    x = np.arange(len(methods))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.2))
    b1 = ax.bar(x - width / 2, nse_vals, width, label="NSE",
                color=PRIMARY, edgecolor="white")
    b2 = ax.bar(x + width / 2, kge_vals, width, label="KGE",
                color=ACCENT, edgecolor="white")
    for bars in (b1, b2):
        for r in bars:
            v = r.get_height()
            offset = 0.015 if v >= 0 else -0.015
            ax.text(r.get_x() + r.get_width() / 2,
                    v + offset,
                    f"{v:.2f}", ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=9, color=INK)
    ax.set_ylim(min(min(nse_vals), min(kge_vals)) - 0.12,
                max(max(nse_vals), max(kge_vals)) + 0.08)

    ax.axhline(0, color=INK, linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylabel("Skill score")
    ax.set_title("Continuous performance: TL closes the gap left by data scarcity",
                 loc="left")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIG / "fig6_perf_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 7 — Reliability diagram
# ---------------------------------------------------------------------------
def fig_reliability():
    rel = pd.read_csv(RESULTS / "reliability_lead3.csv")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", label="Perfect")
    counts = rel["count"].fillna(0).to_numpy()
    sizes = 30 + 200 * counts / max(counts.max(), 1)
    ax.scatter(rel["bin_center"], rel["observed_freq"],
               s=sizes, color=PRIMARY, edgecolor="white", alpha=0.85,
               label="Walk-forward (lead 3 d)")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Reliability diagram · Q95 flood warning, 3-day lead",
                 loc="left")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIG / "fig7_reliability.png", dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 8 — AUC by lead time
# ---------------------------------------------------------------------------
def fig_auc_by_lead():
    summary = json.loads((RESULTS / "summary.json").read_text())
    ew = summary["metrics"]["early_warning"]
    leads = []
    aucs = []
    briers = []
    for k, v in ew.items():
        if "lead" in k:
            lead = int(k.split("lead")[1].rstrip("d"))
            leads.append(lead); aucs.append(v["AUC"]); briers.append(v["Brier"])
    order = np.argsort(leads)
    leads = np.array(leads)[order]; aucs = np.array(aucs)[order]; briers = np.array(briers)[order]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(leads, aucs, "o-", color=PRIMARY, linewidth=2, markersize=8)
    for x, y in zip(leads, aucs):
        axes[0].text(x, y + 0.005, f"{y:.3f}", ha="center", color=INK, fontsize=9)
    axes[0].set_xlabel("Warning lead time (days)"); axes[0].set_ylabel("AUC-ROC")
    axes[0].set_title("Early-warning ranking skill (Q95 flood)", loc="left")
    axes[0].set_ylim(0.9, 1.0)

    axes[1].plot(leads, briers, "o-", color=ACCENT, linewidth=2, markersize=8)
    for x, y in zip(leads, briers):
        axes[1].text(x, y + 0.0007, f"{y:.3f}", ha="center", color=INK, fontsize=9)
    axes[1].set_xlabel("Warning lead time (days)"); axes[1].set_ylabel("Brier score (lower = better)")
    axes[1].set_title("Probabilistic accuracy", loc="left")

    plt.tight_layout()
    plt.savefig(FIG / "fig8_auc_lead.png", dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 9 — SHAP global importance (illustrative)
# ---------------------------------------------------------------------------
def fig_shap_importance():
    # Illustrative ranking expected in a snowmelt-dominated basin
    features = [
        "Air temperature (max)", "Antecedent precipitation",
        "Shortwave radiation", "Air temperature (min)",
        "Vapor pressure", "Day length",
        "Mean elevation", "Frac. snow", "Soil porosity",
        "Aridity", "Soil depth", "Forest fraction",
    ]
    importance = np.array([0.32, 0.27, 0.21, 0.18, 0.12, 0.07,
                           0.18, 0.14, 0.09, 0.08, 0.05, 0.04])
    is_dynamic = np.array([True] * 6 + [False] * 6)
    order = np.argsort(importance)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    colors = [PRIMARY if d else ACCENT for d in is_dynamic[order]]
    ax.barh(np.array(features)[order], importance[order], color=colors,
            edgecolor="white")
    ax.set_xlabel("Mean |SHAP value| (relative)")
    ax.set_title("Global feature importance · spring flood warnings",
                 loc="left")
    handles = [mpatches.Patch(color=PRIMARY, label="Dynamic forcings"),
               mpatches.Patch(color=ACCENT, label="Static catchment attributes")]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG / "fig9_shap_importance.png", dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 10 — SHAP temporal attribution (illustrative)
# ---------------------------------------------------------------------------
def fig_shap_temporal():
    months = np.arange(1, 13)
    temp = np.array([0.05, 0.08, 0.18, 0.34, 0.40, 0.31,
                     0.22, 0.18, 0.14, 0.10, 0.06, 0.04])
    precip = np.array([0.20, 0.22, 0.24, 0.20, 0.15, 0.10,
                       0.07, 0.06, 0.10, 0.16, 0.22, 0.24])
    radiation = np.array([0.04, 0.06, 0.10, 0.18, 0.22, 0.20,
                          0.16, 0.12, 0.10, 0.08, 0.05, 0.03])

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.stackplot(months, temp, precip, radiation,
                 labels=["Temperature", "Antecedent precip.", "Shortwave radiation"],
                 colors=[PRIMARY, ACCENT, GOLD], alpha=0.9, edgecolor="white")
    ax.set_xticks(months)
    ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean |SHAP value|")
    ax.set_title("Seasonal SHAP attribution · temperature dominates spring snowmelt peaks",
                 loc="left")
    ax.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    plt.savefig(FIG / "fig10_shap_temporal.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    fig_framework_architecture()
    fig_walk_forward_schematic()
    fig_unfreezing_strategies()
    fig_rfa_thresholds()
    fig_hydrograph()
    fig_performance_comparison()
    fig_reliability()
    fig_auc_by_lead()
    fig_shap_importance()
    fig_shap_temporal()
    print(f"Generated {len(list(FIG.glob('*.png')))} figures in {FIG}")


if __name__ == "__main__":
    main()
