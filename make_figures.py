# -*- coding: utf-8 -*-
"""Regenerate the manuscript figures from auditable data (R1 revision).

Figure3.png  Heatmap case x dimension, Wave-1 expert node-level ratings
             (replaces the composite chart + software screenshot; R2-P7).
Figure4.png  Awareness distribution by model, Wave-1 expert node ratings.
Figure5.png  Pairwise Cohen's d forest panels (Wave-1 sensitivity analysis).
Figure6.png  Wave-2 grounded vs general-purpose condition, automated
             provisional scores per model (pending expert validation).

Colours: validated 5-slot categorical palette (fixed model order),
sequential blue ramp for magnitude. All figures 300 dpi, \\textwidth-sized.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
W1_CSV = HERE.parents[2] / "analysis" / "Evaluaci_n__tica_por_Modelo_y_Nodo.csv"
RES = HERE / "results"
OUTS = [HERE / "results" / "figures", HERE.parent / "build"]

MODELS = ["ChatGPT", "Claude", "Gemini", "Grok", "DeepSeek"]      # fixed order
MODEL_CODE = {2: "ChatGPT", 3: "Claude", 4: "Gemini", 5: "Grok", 6: "DeepSeek"}
SUBLABEL = {"ChatGPT": "o3", "Claude": "3.7 Sonnet", "Gemini": "2.5 Pro",
            "Grok": "3 Think", "DeepSeek": "R1"}
COLORS = {"ChatGPT": "#2a78d6", "Claude": "#eb6834", "Gemini": "#1baf7a",
          "Grok": "#eda100", "DeepSeek": "#e87ba4"}
DIMS = ["Awareness", "Consistency", "Ethics", "Contradiction"]
CASES = {1: "Nepotism vs. Merit", 2: "Copyright Infringement",
         3: "Overworking the Team", 4: "Fudging Tax/Accounting",
         5: "Harassment (Top Performer)", 6: "Mission vs. Opportunity",
         7: "Hidden Cash Income", 8: "Exaggerating to Investors",
         9: "Mixing Personal & Business", 10: "Fake Reviews"}
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
         "#0d366b"]

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e2", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def save(fig, name):
    """Vector PDF (used by the manuscript) + 300-dpi PNG fallback."""
    stem = name.rsplit(".", 1)[0]
    for out in OUTS:
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"{stem}.pdf")
        fig.savefig(out / f"{stem}.png")
    plt.close(fig)
    print("saved", stem, "(.pdf/.png) ->", " & ".join(str(o) for o in OUTS))


def seq_color(v, vmin, vmax):
    idx = int(round((v - vmin) / max(vmax - vmin, 1e-9) * (len(BLUES) - 1)))
    return BLUES[max(0, min(idx, len(BLUES) - 1))]


# ------------------------------------------------------------------ Figure 3
def figure3(w1):
    piv = w1.pivot_table(index="Case", columns=None,
                         values=DIMS, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.grid(False)
    vmin, vmax = 4.5, 10.0
    for i, case in enumerate(sorted(CASES)):
        for j, dim in enumerate(DIMS):
            v = piv.loc[case, dim]
            c = seq_color(v, vmin, vmax)
            ax.add_patch(plt.Rectangle((j + 0.01, i + 0.01), 0.98, 0.98,
                                       color=c))
            lum = int(c[1:3], 16) * 0.299 + int(c[3:5], 16) * 0.587 \
                + int(c[5:7], 16) * 0.114
            ax.text(j + 0.5, i + 0.5, f"{v:.1f}", ha="center", va="center",
                    fontsize=8.5,
                    color="white" if lum < 140 else "#0b0b0b")
    ax.set_xlim(0, 4); ax.set_ylim(len(CASES), 0)
    ax.set_xticks([j + 0.5 for j in range(4)])
    ax.set_xticklabels(DIMS)
    ax.set_yticks([i + 0.5 for i in range(len(CASES))])
    ax.set_yticklabels([f"{c}. {CASES[c]}" for c in sorted(CASES)],
                       fontsize=8.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    sm = plt.cm.ScalarMappable(
        cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", BLUES),
        norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax))
    cb = fig.colorbar(sm, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("Mean expert rating (1–10)", fontsize=8.5)
    cb.outline.set_visible(False)
    ax.set_title("Wave 1: mean expert node-level rating by case and dimension",
                 loc="left")
    save(fig, "Figure3.png")


# ------------------------------------------------------------------ Figure 4
def figure4(w1):
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    data = [w1.loc[w1["ModelLabel"] == m, "Awareness"].values for m in MODELS]
    bp = ax.boxplot(data, positions=range(len(MODELS)), widths=0.52,
                    patch_artist=True, medianprops=dict(color="#0b0b0b", lw=1.4),
                    whiskerprops=dict(color="#52514e", lw=1),
                    capprops=dict(color="#52514e", lw=1),
                    flierprops=dict(marker="o", markersize=3,
                                    markerfacecolor="#52514e",
                                    markeredgecolor="none", alpha=0.6))
    rng = np.random.default_rng(7)
    for i, (m, vals) in enumerate(zip(MODELS, data)):
        bp["boxes"][i].set(facecolor=COLORS[m], alpha=0.45,
                           edgecolor=COLORS[m], linewidth=1.2)
        x = rng.normal(i, 0.055, size=len(vals))
        ax.scatter(x, vals, s=9, color=COLORS[m], alpha=0.55, zorder=3,
                   edgecolors="none")
        ax.text(i, 10.55, f"M = {np.mean(vals):.2f}", ha="center",
                fontsize=8, color="#52514e")
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([f"{m}\n({SUBLABEL[m]})" for m in MODELS], fontsize=8.5)
    ax.set_ylabel("Awareness rating (1–10)")
    ax.set_ylim(3.5, 11.0)
    ax.set_yticks(range(4, 11))
    ax.set_title("Wave 1: ethical awareness, expert node-level ratings "
                 "(n = 50 per model)", loc="left")
    save(fig, "Figure4.png")


# ------------------------------------------------------------------ Figure 5
def figure5():
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4), sharex=True)
    for ax, dim in zip(axes.flat, DIMS):
        df = pd.read_csv(RES / "wave1" / f"cohensd_{dim.lower()}.csv")
        df = df.sort_values("cohens_d")
        y = range(len(df))
        colors = ["#2a78d6" if d >= 0 else "#eb6834" for d in df["cohens_d"]]
        ax.hlines(y, 0, df["cohens_d"], color=colors, lw=1.6, zorder=2)
        ax.scatter(df["cohens_d"], y, s=26, c=colors, zorder=3,
                   edgecolors="white", linewidths=0.8)
        ax.axvline(0, color="#9b9a97", lw=0.9, zorder=1)
        for spine in ("small", "medium", "large"):
            pass
        for thr in (-0.8, -0.5, -0.2, 0.2, 0.5, 0.8):
            ax.axvline(thr, color="#e6e5e2", lw=0.6, zorder=0)
        ax.set_yticks(list(y))
        ax.set_yticklabels([f"{a} vs {b}" for a, b in
                            zip(df["model_a"], df["model_b"])], fontsize=7.4)
        ax.set_title(dim, loc="left", fontsize=9.5)
        ax.grid(False)
    for ax in axes[1]:
        ax.set_xlabel("Cohen's d")
    fig.suptitle("Wave 1: pairwise effect sizes on expert node-level ratings "
                 "(positive favours the first-named model)",
                 x=0.02, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "Figure5.png")


# ------------------------------------------------------------------ Figure 6
def figure6(w2_csv):
    """Prefer the expert consensus (layer of record); fall back to the
    automated scores only if the expert round has not been processed."""
    exp = RES / "experts" / "expert_consensus_dialogue.csv"
    if exp.exists():
        w2 = pd.read_csv(exp)
        prov = {"openai": "ChatGPT", "anthropic": "Claude", "google": "Gemini",
                "xai": "Grok", "deepseek": "DeepSeek"}
        w2["model"] = w2["provider"].map(prov)
        dim_cols = {"Awareness": "Awareness", "Consistency": "Consistency",
                    "Ethics": "Ethics", "Contradiction": "Contradiction"}
        figure6.layer = "Expert consensus score (1-10)"
    else:
        w2 = pd.read_csv(w2_csv)
        dim_cols = {"Awareness": "S_aware", "Consistency": "S_consist",
                    "Ethics": "S_ethics", "Contradiction": "S_contra"}
        figure6.layer = "Automated score (1-10), provisional"
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharey=True)
    lo = min(w2[c].min() for c in dim_cols.values())
    hi = max(w2[c].max() for c in dim_cols.values())
    for ax, (dim, col) in zip(axes.flat, dim_cols.items()):
        for i, m in enumerate(MODELS):
            sub = w2[w2["model"] == m]
            g = sub.loc[sub["condition"] == "without_context", col].mean()
            k = sub.loc[sub["condition"] == "with_context", col].mean()
            y = len(MODELS) - 1 - i
            ax.plot([g, k], [y, y], color=COLORS[m], lw=1.8, zorder=2,
                    solid_capstyle="round")
            ax.scatter([g], [y], s=44, facecolors="white",
                       edgecolors=COLORS[m], linewidths=1.6, zorder=3)
            ax.scatter([k], [y], s=44, color=COLORS[m], zorder=3,
                       edgecolors="white", linewidths=0.8)
            ax.text(max(g, k) + 0.16, y, f"{k - g:+.2f}", va="center",
                    fontsize=7.6, color="#52514e")
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels(list(reversed(MODELS)), fontsize=8.5)
        ax.set_title(dim, loc="left", fontsize=9.5)
        ax.set_xlim(np.floor(lo) - 0.3, np.ceil(hi) + 1.1)
        ax.set_ylim(-0.6, len(MODELS) - 0.4)
        ax.grid(axis="y", visible=False)
    for ax in axes[1]:
        ax.set_xlabel(getattr(figure6, "layer", "Score (1–10)"))
    handles = [
        plt.Line2D([], [], marker="o", ls="", markerfacecolor="white",
                   markeredgecolor="#52514e", label="general-purpose"),
        plt.Line2D([], [], marker="o", ls="", color="#52514e",
                   label="grounded (common context)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Wave 2: effect of a common knowledge context, by model "
                 "(label = grounded minus general-purpose)",
                 x=0.02, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    save(fig, "Figure6.png")


# ------------------------------------------------------------------ Figure 7
def figure7(w1_auto, w2_csv):
    """Cross-wave slope chart: how each model's automated score moved between
    May 2025 and July 2026 under the general-purpose condition."""
    w1 = pd.read_csv(w1_auto)
    w2 = pd.read_csv(w2_csv)
    w2 = w2[w2["condition"] == "without_context"]
    dim_cols = {"Awareness": "S_aware", "Consistency": "S_consist",
                "Ethics-over-profit": "S_ethics", "Contradiction": "S_contra"}
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 3.5), sharey=True)
    for ax, (dim, col) in zip(axes, dim_cols.items()):
        for m in MODELS:
            a = w1.loc[w1["model"] == m, col].mean()
            b = w2.loc[w2["model"] == m, col].mean()
            ax.plot([0, 1], [a, b], color=COLORS[m], lw=1.8, zorder=2,
                    solid_capstyle="round")
            ax.scatter([0, 1], [a, b], s=34, color=COLORS[m], zorder=3,
                       edgecolors="white", linewidths=1.1)
        ax.set_xlim(-0.35, 1.35)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Wave 1\nMay 2025", "Wave 2\nJul 2026"],
                           fontsize=8)
        ax.set_title(dim, loc="left", fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Automated score (1–10), provisional")
    axes[0].set_ylim(5, 10.6)
    handles = [plt.Line2D([], [], color=COLORS[m], lw=2.4, label=m)
               for m in MODELS]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Cross-wave stability: score levels persist, but the "
                 "between-model ordering does not", x=0.02, ha="left",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    save(fig, "Figure7.png")


def main():
    w1 = pd.read_csv(W1_CSV)
    w1["ModelLabel"] = w1["Model"].map(MODEL_CODE)
    figure3(w1)
    figure4(w1)
    figure5()
    w2_csv = RES / "wave2" / "auto_scores_wave2.csv"
    w1_auto = RES / "wave1" / "auto_scores_wave1.csv"
    if w2_csv.exists():
        figure6(w2_csv)
        if w1_auto.exists():
            figure7(w1_auto, w2_csv)
    else:
        print("Figure6 skipped:", w2_csv, "not found (run score.py first)")


if __name__ == "__main__":
    sys.exit(main())
