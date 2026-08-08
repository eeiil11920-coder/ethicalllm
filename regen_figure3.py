#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate Figure 2 (Figure3.pdf) at the manuscript's real text width.

The manuscript's \textwidth is 372pt = 5.15in. The figure was drawn 7.0in wide,
so \includegraphics[width=\textwidth] shrank it to 74% and the 8.5pt labels
rendered at ~6.2pt. Drawing it at 5.15in keeps the type at its nominal size.
Same data, same palette, same layout as make_figures.py:figure3.
"""
import shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors
import pandas as pd

BASE = Path(r"C:\Users\Usuario\OneDrive - Universidad Pablo de Olavide de Sevilla\Escritorio\1_ARTÍCULO\ENVIADOS\5_ETHICS_XXX_David")
W1_CSV = BASE / "analysis" / "Evaluaci_n__tica_por_Modelo_y_Nodo.csv"
OVERLEAF = BASE / "mayor revissions" / "ENTREGA" / "overleaf"
REPOFIG = BASE / "mayor revissions" / "ENTREGA" / "github_ethicalllm" / "results" / "figures"

DIMS = ["Awareness", "Consistency", "Ethics", "Contradiction"]
CASES = {1: "Nepotism vs. Merit", 2: "Copyright Infringement",
         3: "Overworking the Team", 4: "Fudging Tax/Accounting",
         5: "Harassment (Top Performer)", 6: "Mission vs. Opportunity",
         7: "Hidden Cash Income", 8: "Exaggerating to Investors",
         9: "Mixing Personal & Business", 10: "Fake Reviews"}
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
         "#0d366b"]
CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("b", BLUES)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "axes.axisbelow": True,
    "figure.dpi": 300, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def seq_color(v, vmin, vmax):
    t = 0.0 if vmax == vmin else (v - vmin) / (vmax - vmin)
    return matplotlib.colors.to_hex(CMAP(max(0.0, min(1.0, t))))


w1 = pd.read_csv(W1_CSV)
piv = w1.pivot_table(index="Case", values=DIMS, aggfunc="mean")

# 5.15in = the manuscript's \textwidth (372pt); no downscaling at include time
fig, ax = plt.subplots(figsize=(5.15, 4.9))
vmin, vmax = 4.5, 10.0
for i, case in enumerate(sorted(CASES)):
    for j, dim in enumerate(DIMS):
        v = piv.loc[case, dim]
        c = seq_color(v, vmin, vmax)
        ax.add_patch(plt.Rectangle((j + 0.01, i + 0.01), 0.98, 0.98, color=c))
        lum = int(c[1:3], 16) * .299 + int(c[3:5], 16) * .587 + int(c[5:7], 16) * .114
        ax.text(j + .5, i + .5, f"{v:.1f}", ha="center", va="center",
                fontsize=9, color="white" if lum < 140 else "#0b0b0b")
ax.set_xlim(0, 4); ax.set_ylim(len(CASES), 0)
ax.set_xticks([j + .5 for j in range(4)])
ax.set_xticklabels(DIMS, fontsize=9)
ax.set_yticks([i + .5 for i in range(len(CASES))])
ax.set_yticklabels([f"{c}. {CASES[c]}" for c in sorted(CASES)], fontsize=9)
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
sm = plt.cm.ScalarMappable(cmap=CMAP,
                           norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax))
cb = fig.colorbar(sm, ax=ax, shrink=0.82, pad=0.02)
cb.set_label("Mean expert rating (1–10)", fontsize=9)
cb.ax.tick_params(labelsize=8.5)
cb.outline.set_visible(False)
ax.set_title("Wave 1: mean expert node-level rating by case and dimension",
             loc="left", fontsize=9.5)

for d in (OVERLEAF, REPOFIG):
    d.mkdir(parents=True, exist_ok=True)
    bak = d / "Figure3.pdf.bak"
    if (d / "Figure3.pdf").exists() and not bak.exists():
        shutil.copy2(d / "Figure3.pdf", bak)
    fig.savefig(d / "Figure3.pdf")
    fig.savefig(d / "Figure3.png")
    print("written ->", d)
plt.close(fig)

print("\ncase-1 row (sanity):", [round(piv.loc[1, d], 1) for d in DIMS])
