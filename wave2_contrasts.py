# -*- coding: utf-8 -*-
"""Wave-2 pre-registered contrasts (R1 revision).

(1) Condition contrast: model x condition interaction (reviewer 2's
    falsification test) and the between-model spread under each condition.
(2) Wave contrast: rank stability wave 1 -> wave 2 by dimension.
(3) Pressure-response contrast: stance-flip rates by model, wave, condition.

Writes results/wave2/contrasts.md and the per-condition means table.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
W2 = pd.read_csv(RES / "wave2" / "auto_scores_wave2.csv")
W1 = pd.read_csv(RES / "wave1" / "auto_scores_wave1.csv")
DIMS = {"S_aware": "Awareness", "S_consist": "Consistency",
        "S_ethics": "Ethics-over-profit", "S_contra": "Contradiction"}
MODELS = ["ChatGPT", "Claude", "Gemini", "Grok", "DeepSeek"]
out = []


def fit(formula, data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return smf.mixedlm(formula, data=data,
                               groups=data["case"]).fit(reml=False), "mixed"
        except Exception:
            return (smf.ols(formula, data=data)
                    .fit(cov_type="cluster",
                         cov_kwds={"groups": data["case"]}), "OLS-cluster")


out.append("# Wave-2 pre-registered contrasts\n")

# ---------------------------------------------------------------- contrast 1
out.append("## 1. Condition contrast (model x condition interaction)\n")
rows = []
for col, name in DIMS.items():
    f, kind = fit(f"{col} ~ C(model)*C(condition)", W2)
    inter = [t for t in f.params.index if ":" in t]
    w = f.wald_test(", ".join(f"({t} = 0)" for t in inter), scalar=True)
    fmain, _ = fit(f"{col} ~ C(model) + C(condition)", W2)
    cterm = [t for t in fmain.params.index if "condition" in t][0]
    # between-model spread per condition
    spread = {c: (W2[W2.condition == c].groupby("model")[col].mean().max()
                  - W2[W2.condition == c].groupby("model")[col].mean().min())
              for c in ("without_context", "with_context")}
    rows.append({
        "dimension": name,
        "context_effect": round(f.params.get(cterm, fmain.params[cterm]), 3)
        if cterm in fmain.params else np.nan,
        "context_p": round(fmain.pvalues[cterm], 4),
        "interaction_chi2": round(float(w.statistic), 2),
        "interaction_p": round(float(w.pvalue), 4),
        "spread_general": round(spread["without_context"], 2),
        "spread_grounded": round(spread["with_context"], 2),
        "fit": kind,
    })
c1 = pd.DataFrame(rows)
out.append(c1.to_markdown(index=False))
out.append("\nBetween-model spread = max-min of model means within the "
           "condition. If grounding removed the differences, spread_grounded "
           "would collapse towards zero.\n")

# ---------------------------------------------------------------- contrast 2
out.append("\n## 2. Wave contrast (rank stability, general-purpose condition)\n")
w1g = W1.copy()
w2g = W2[W2.condition == "without_context"]
rows = []
for col, name in DIMS.items():
    m1 = w1g.groupby("model")[col].mean().reindex(MODELS)
    m2 = w2g.groupby("model")[col].mean().reindex(MODELS)
    rho, p = sps.spearmanr(m1.values, m2.values)
    rows.append({"dimension": name,
                 "wave1_mean": round(m1.mean(), 2),
                 "wave2_mean": round(m2.mean(), 2),
                 "delta": round(m2.mean() - m1.mean(), 2),
                 "rank_rho": round(rho, 3), "rho_p": round(p, 4)})
c2 = pd.DataFrame(rows)
out.append(c2.to_markdown(index=False))
per_model = pd.concat(
    [w1g.groupby("model")[list(DIMS)].mean().add_suffix("_w1"),
     w2g.groupby("model")[list(DIMS)].mean().add_suffix("_w2")],
    axis=1).round(2).reindex(MODELS)
out.append("\nPer-model means (general-purpose condition only):\n")
out.append(per_model.to_markdown())

# ---------------------------------------------------------------- contrast 3
out.append("\n## 3. Pressure-response contrast (stance flips)\n")
rows = []
for wave, df in (("1 (May 2025)", W1), ("2 (Jul 2026)", W2)):
    for cond in sorted(df["condition"].unique()):
        sub = df[df.condition == cond]
        rows.append({"wave": wave, "condition": cond,
                     "mean_flips": round(sub["n_flips"].mean(), 2),
                     "dialogues_with_flip":
                         f"{int((sub['n_flips'] > 0).sum())}/{len(sub)}",
                     "mean_modal_rate": round(sub["modal_rate"].mean(), 3)})
c3 = pd.DataFrame(rows)
out.append(c3.to_markdown(index=False))
flips = (W2.groupby(["model", "condition"])["n_flips"].mean()
         .unstack().round(2).reindex(MODELS))
out.append("\nWave-2 mean stance flips per dialogue by model and condition:\n")
out.append(flips.to_markdown())

# per-model x condition means table (for the manuscript table)
tab = (W2.groupby(["model", "condition"])[list(DIMS)].mean()
       .round(2).reindex(MODELS, level=0))
tab.to_csv(RES / "wave2" / "means_by_model_condition.csv")
out.append("\n\n## Table source: means by model and condition\n")
out.append(tab.to_markdown())

(RES / "wave2" / "contrasts.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
