#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Section 4.4 contrasts computed on the EXPERT layer, from the dialogue-level data.
Nothing is written back to the manuscript; this only reports what comes out."""
import itertools, warnings
import numpy as np, pandas as pd
from scipy import stats as sps
import statsmodels.formula.api as smf

REPO = r"C:\Users\Usuario\OneDrive - Universidad Pablo de Olavide de Sevilla\Escritorio\1_ARTÍCULO\ENVIADOS\5_ETHICS_XXX_David\mayor revissions\ENTREGA\github_ethicalllm"
W1N = r"C:\Users\Usuario\OneDrive - Universidad Pablo de Olavide de Sevilla\Escritorio\1_ARTÍCULO\ENVIADOS\3_SPECIAL ETHICS_solo\ULTIMO Comuters in human Behaviour Reports\ENTREGA_github_ethicalgorithm\data\expert_scores_by_node.csv"

DIMS = ["Awareness", "Consistency", "Ethics", "Contradiction"]
PROV = {"openai": "ChatGPT", "anthropic": "Claude", "google": "Gemini",
        "xai": "Grok", "deepseek": "DeepSeek"}
ORDER = ["ChatGPT", "Claude", "Gemini", "Grok", "DeepSeek"]

w2 = pd.read_csv(REPO + r"\results\experts\expert_consensus_dialogue.csv")
w2["model"] = w2["provider"].map(PROV)
w1 = pd.read_csv(W1N)
w1d = w1.groupby(["Model", "Case"], as_index=False)[DIMS].mean().rename(columns={"Model": "model", "Case": "case"})

print("=" * 78)
print("0.  DOES THE WAVE-1 COLUMN OF TABLE 6 COME FROM THIS DATA?")
print("=" * 78)
tab6_w1 = {"ChatGPT": [7.40, 5.30, 6.36, 9.64], "Claude": [8.00, 5.12, 7.28, 9.72],
           "Gemini": [7.98, 5.28, 6.76, 9.78], "Grok": [7.86, 5.04, 6.40, 9.84],
           "DeepSeek": [6.82, 5.00, 5.66, 10.00]}
g = w1.groupby("Model")[DIMS].mean()
print("%-10s %-28s %s" % ("model", "computed from node data", "Table 6"))
allok = True
for m in ORDER:
    c = [round(g.loc[m, d], 2) for d in DIMS]
    ok = all(abs(a - b) < .01 for a, b in zip(c, tab6_w1[m]))
    allok &= ok
    print("%-10s %-28s %-24s %s" % (m, c, tab6_w1[m], "match" if ok else "MISMATCH"))
print("\n-> wave-1 column of Table 6 reproduced exactly:", allok)

print()
print("=" * 78)
print("1.  CONDITION CONTRAST  (wave 2, n=100 dialogues, expert layer)")
print("    mixed model  score ~ model * condition,  random intercept for case")
print("=" * 78)
rows = []
for d in DIMS:
    dat = w2[["model", "condition", "case", d]].rename(columns={d: "y"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = smf.mixedlm("y ~ C(model)*C(condition)", dat, groups=dat["case"]).fit(reml=False)
        add = smf.mixedlm("y ~ C(model)+C(condition)", dat, groups=dat["case"]).fit(reml=False)
        mod0 = smf.mixedlm("y ~ C(condition)", dat, groups=dat["case"]).fit(reml=False)
    # LR test: interaction
    lr_i = 2 * (full.llf - add.llf); df_i = 4
    p_i = sps.chi2.sf(lr_i, df_i)
    # LR test: model main effect
    lr_m = 2 * (add.llf - mod0.llf); df_m = 4
    p_m = sps.chi2.sf(lr_m, df_m)
    cterm = [t for t in add.params.index if "condition" in t][0]
    eff = add.params[cterm]; se = add.bse[cterm]
    p_c = 2 * sps.norm.sf(abs(eff / se))
    sp = {c: (w2[w2.condition == c].groupby("model")[d].mean().max()
              - w2[w2.condition == c].groupby("model")[d].mean().min())
          for c in ("without_context", "with_context")}
    rows.append([d, lr_i, p_i, lr_m, p_m, eff, se, p_c,
                 sp["without_context"], sp["with_context"]])

print("%-14s %-22s %-22s %-24s %s" % ("dimension", "model x condition", "model main effect",
                                      "grounding effect", "spread gen/grnd"))
for r in rows:
    print("%-14s chi2(4)=%5.2f p=%.3f  chi2(4)=%6.2f p=%.4f  %+.2f (SE %.2f) p=%.3f   %.2f / %.2f"
          % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9]))

print()
print("=" * 78)
print("2.  WAVE CONTRAST  (wave 1 vs wave 2 general-purpose, expert layer)")
print("    Spearman on 5 instance means, EXACT permutation p (120 perms)")
print("=" * 78)
w2gp = w2[w2.condition == "without_context"].groupby("model")[DIMS].mean()
w1m = w1.groupby("Model")[DIMS].mean()
for d in DIMS:
    a = np.array([w1m.loc[m, d] for m in ORDER])
    b = np.array([w2gp.loc[m, d] for m in ORDER])
    rho = sps.spearmanr(a, b).statistic
    perms = [sps.spearmanr(a, [b[i] for i in p]).statistic
             for p in itertools.permutations(range(5))]
    p_exact = np.mean([abs(x) >= abs(rho) - 1e-12 for x in perms])
    print("%-14s rho=%+.2f   exact two-sided p=%.3f   (w1 %s -> w2 %s)"
          % (d, rho, p_exact,
             [ORDER[i] for i in np.argsort(-a)][:2], [ORDER[i] for i in np.argsort(-b)][:2]))

print()
print("=" * 78)
print("3.  IS THE MODEL EFFECT PRESENT *WITHIN* EACH CONDITION? (falsification test)")
print("=" * 78)
for d in DIMS:
    line = "%-14s" % d
    for c, lab in (("without_context", "general"), ("with_context", "grounded")):
        dat = w2[w2.condition == c][["model", "case", d]].rename(columns={d: "y"})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f1 = smf.mixedlm("y ~ C(model)", dat, groups=dat["case"]).fit(reml=False)
            f0 = smf.mixedlm("y ~ 1", dat, groups=dat["case"]).fit(reml=False)
        lr = 2 * (f1.llf - f0.llf); p = sps.chi2.sf(lr, 4)
        line += "   %s: chi2(4)=%6.2f p=%.4f" % (lab, lr, p)
    print(line)
