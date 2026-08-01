#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stats.py — Statistical re-analysis for BUSI-D-26-01884 (major revision).

Primary analysis (new): linear mixed-effects models per dimension
    score ~ C(model)  +  (1 | case)  +  (1 | node within case)
fitted with statsmodels MixedLM on the 250 wave-1 node-level human ratings
(unit of analysis = dialogue, nodes as repeated measures).

LIMITATION (documented for the response letter): statsmodels MixedLM
supports a single grouping factor, so fully CROSSED random effects
(model x case x rater as in lme4) are approximated with groups=case and a
variance component for node (vc_formula). This is conservative for the
fixed model effect; rater cannot enter because the archived wave-1 ratings
are consensus scores (one rating per case x model x node).

Sensitivity analyses (reproduce the paper): one-way ANOVA (eta^2) and
Kruskal-Wallis (epsilon^2) on dialogue means, Tukey HSD, pairwise Cohen's d,
Shapiro-Wilk and Levene checks.

Wave-2 hooks: when scored wave-2 data exists (score.py output), the script
adds (a) with-context vs without-context and (b) wave-1 vs wave-2 mixed
models. Until then those sections are reported as PENDING.

Outputs -> results/wave1/ : mixedlm_<dim>.csv, mixedlm_overview.csv,
anova_kruskal.csv, tukey_<dim>.csv, cohensd_<dim>.csv, descriptives.csv,
assumptions.csv, summary.md
"""
import argparse
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
DIMS = ["Awareness", "Consistency", "Ethics", "Contradiction"]
MODEL_CODE = {2: "ChatGPT", 3: "Claude", 4: "Gemini", 5: "Grok", 6: "DeepSeek"}

ANALYSIS_DIR = HERE.parents[2] / "analysis"
DEFAULT_CSV = ANALYSIS_DIR / "Evaluaci_n__tica_por_Modelo_y_Nodo.csv"


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                 / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else 0.0


def eta_squared(groups):
    allv = np.concatenate(groups)
    grand = allv.mean()
    ss_b = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups)
    ss_t = ((allv - grand) ** 2).sum()
    return ss_b / ss_t if ss_t > 0 else 0.0


def epsilon_squared(h, n):
    return h * (n + 1) / (n ** 2 - 1)


def fit_mixedlm(df, dim):
    """score ~ C(model), random intercept for case + vc for node in case."""
    import statsmodels.formula.api as smf
    data = df.rename(columns={dim: "score"}).copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        md = smf.mixedlm(
            "score ~ C(ModelLabel, Treatment('ChatGPT'))",
            data=data, groups=data["Case"],
            vc_formula={"Node": "0 + C(Node)"},
            re_formula="1")
        fit = md.fit(reml=True, method="lbfgs", maxiter=500)
    coefs = pd.DataFrame({
        "term": fit.params.index, "coef": fit.params.values,
        "se": fit.bse.values, "z": fit.tvalues.values,
        "p": fit.pvalues.values,
        "ci_low": fit.conf_int()[0].values,
        "ci_high": fit.conf_int()[1].values,
    })
    # Overall Wald test for the model factor
    terms = [t for t in fit.params.index if t.startswith("C(ModelLabel")]
    L = np.zeros((len(terms), len(fit.params)))
    for i, t in enumerate(terms):
        L[i, list(fit.params.index).index(t)] = 1.0
    wald = fit.wald_test(L, scalar=True)
    return fit, coefs, float(wald.statistic), float(wald.pvalue), len(terms)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--human-csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--wave2-scores", type=Path,
                    default=HERE / "results" / "wave2" / "auto_scores_wave2.csv",
                    help="score.py output for wave 2 (enables the wave/"
                         "condition contrasts when present)")
    ap.add_argument("--wave1-auto-scores", type=Path,
                    default=HERE / "results" / "wave1" / "auto_scores_wave1.csv")
    ap.add_argument("--outdir", type=Path, default=HERE / "results" / "wave1")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.human_csv)
    df["ModelLabel"] = df["Model"].map(MODEL_CODE)
    n_rows = len(df)
    md_lines = ["# Wave-1 re-analysis (mixed-effects primary, "
                "ANOVA/Kruskal-Wallis sensitivity)\n",
                f"Data: `{args.human_csv.name}` - {n_rows} node-level human "
                f"ratings (5 models x 10 cases x 5 nodes).",
                "Unit of analysis: dialogue (model x case); nodes are "
                "repeated measures.\n"]

    # ---- descriptives ----------------------------------------------------
    desc = df.groupby("ModelLabel")[DIMS].agg(["mean", "std"]).round(3)
    desc.to_csv(args.outdir / "descriptives.csv", encoding="utf-8")

    # ---- mixed models ----------------------------------------------------
    md_lines.append("## Primary: mixed-effects models "
                    "(score ~ model, random: case, node-in-case)\n")
    md_lines.append("Note: statsmodels MixedLM admits one grouping factor; "
                    "crossed random effects are approximated with "
                    "groups=case plus a node variance component "
                    "(vc_formula). Rater effects cannot be estimated from "
                    "the archived consensus ratings.\n")
    overview = []
    for dim in DIMS:
        fit, coefs, w, wp, dfree = fit_mixedlm(df, dim)
        coefs.round(4).to_csv(args.outdir / f"mixedlm_{dim.lower()}.csv",
                              index=False, encoding="utf-8")
        # variance components
        var_case = float(fit.cov_re.iloc[0, 0]) if fit.cov_re.size else 0.0
        vcomp = {k: float(v) for k, v in zip(fit.model.exog_vc.names,
                                             fit.vcomp)} if len(fit.vcomp) \
            else {}
        resid = float(fit.scale)
        overview.append({
            "dimension": dim, "wald_chi2": round(w, 3), "df": dfree,
            "p_model_effect": wp, "var_case": round(var_case, 4),
            "var_node_in_case": round(sum(vcomp.values()), 4),
            "var_residual": round(resid, 4),
            "icc_case": round(var_case / (var_case + sum(vcomp.values())
                                          + resid), 3),
        })
        best = df.groupby("ModelLabel")[dim].mean().idxmax()
        worst = df.groupby("ModelLabel")[dim].mean().idxmin()
        rng = (df.groupby("ModelLabel")[dim].mean().max()
               - df.groupby("ModelLabel")[dim].mean().min())
        sig = [(t.replace("C(ModelLabel, Treatment('ChatGPT'))[T.", "")
                .rstrip("]"), c, p) for t, c, p in
               zip(coefs.term, coefs.coef, coefs.p)
               if t.startswith("C(ModelLabel") and p < 0.05]
        md_lines.append(f"### {dim}")
        md_lines.append(f"- Overall model effect: Wald chi2({dfree}) = "
                        f"{w:.2f}, p = {wp:.3g}"
                        + (" (significant)" if wp < 0.05 else " (n.s.)"))
        md_lines.append(f"- Highest mean: {best}; lowest: {worst}; "
                        f"max model gap = {rng:.2f} points")
        if sig:
            md_lines.append("- Significant contrasts vs ChatGPT: "
                            + "; ".join(f"{m}: {c:+.2f} (p={p:.3g})"
                                        for m, c, p in sig))
        else:
            md_lines.append("- No pairwise contrast vs ChatGPT reaches "
                            "p < .05")
        ov = overview[-1]
        md_lines.append(f"- Variance components: case = {ov['var_case']}, "
                        f"node-in-case = {ov['var_node_in_case']}, "
                        f"residual = {ov['var_residual']} "
                        f"(ICC case = {ov['icc_case']})\n")
    pd.DataFrame(overview).to_csv(args.outdir / "mixedlm_overview.csv",
                                  index=False, encoding="utf-8")

    # ---- sensitivity: ANOVA / Kruskal-Wallis on dialogue means -----------
    md_lines.append("## Sensitivity: ANOVA and Kruskal-Wallis on dialogue "
                    "means (paper's original approach)\n")
    dial = df.groupby(["ModelLabel", "Case"])[DIMS].mean().reset_index()
    ak_rows = []
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    for dim in DIMS:
        groups = [dial.loc[dial.ModelLabel == m, dim].values
                  for m in sorted(dial.ModelLabel.unique())]
        F, pf = sps.f_oneway(*groups)
        e2 = eta_squared(groups)
        H, ph = sps.kruskal(*groups)
        eps2 = epsilon_squared(H, sum(len(g) for g in groups))
        ak_rows.append({"dimension": dim, "anova_F": round(F, 3),
                        "anova_p": pf, "eta2": round(e2, 3),
                        "kruskal_H": round(H, 3), "kruskal_p": ph,
                        "epsilon2": round(eps2, 3)})
        md_lines.append(f"- {dim}: F(4,45) = {F:.2f}, p = {pf:.3g}, "
                        f"eta2 = {e2:.3f}; H = {H:.2f}, p = {ph:.3g}, "
                        f"eps2 = {eps2:.3f}")
        tuk = pairwise_tukeyhsd(dial[dim], dial["ModelLabel"])
        pd.DataFrame(tuk.summary().data[1:],
                     columns=tuk.summary().data[0]).to_csv(
            args.outdir / f"tukey_{dim.lower()}.csv", index=False,
            encoding="utf-8")
        drows = []
        for a, b in combinations(sorted(dial.ModelLabel.unique()), 2):
            d = cohens_d(dial.loc[dial.ModelLabel == a, dim],
                         dial.loc[dial.ModelLabel == b, dim])
            drows.append({"model_a": a, "model_b": b,
                          "cohens_d": round(d, 3)})
        pd.DataFrame(drows).to_csv(args.outdir / f"cohensd_{dim.lower()}.csv",
                                   index=False, encoding="utf-8")
    pd.DataFrame(ak_rows).to_csv(args.outdir / "anova_kruskal.csv",
                                 index=False, encoding="utf-8")

    # ---- assumptions -----------------------------------------------------
    ass_rows = []
    for dim in DIMS:
        for m in sorted(dial.ModelLabel.unique()):
            vals = dial.loc[dial.ModelLabel == m, dim]
            if vals.nunique() > 1:
                W, pw = sps.shapiro(vals)
            else:
                W, pw = np.nan, np.nan
            ass_rows.append({"dimension": dim, "model": m, "test": "shapiro",
                             "stat": round(float(W), 3) if W == W else None,
                             "p": pw})
        groups = [dial.loc[dial.ModelLabel == m, dim].values
                  for m in sorted(dial.ModelLabel.unique())]
        L, pl = sps.levene(*groups)
        ass_rows.append({"dimension": dim, "model": "ALL", "test": "levene",
                         "stat": round(float(L), 3), "p": pl})
    pd.DataFrame(ass_rows).to_csv(args.outdir / "assumptions.csv",
                                  index=False, encoding="utf-8")

    # ---- wave-2 hooks ----------------------------------------------------
    md_lines.append("\n## Wave-2 contrasts (context condition, wave x model)\n")
    if args.wave2_scores.exists():
        import statsmodels.formula.api as smf

        def fit_safe(formula, data, groups):
            """MixedLM with random intercept by case; if the fit is singular
            (e.g. a dimension with near-zero variance in some cells), fall
            back to OLS with cluster-robust SEs by case — reported as such."""
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    return (smf.mixedlm(formula, data=data, groups=groups)
                            .fit(reml=True), "mixed")
                except Exception:
                    return (smf.ols(formula, data=data).fit(
                        cov_type="cluster",
                        cov_kwds={"groups": groups}), "ols-cluster(case)")

        w2 = pd.read_csv(args.wave2_scores)
        md_lines.append(f"Wave-2 scored data found: `{args.wave2_scores}` "
                        f"({len(w2)} dialogues).")
        auto_dims = ["S_aware", "S_consist", "S_ethics", "S_contra"]
        # (a) with vs without context within wave 2
        for dim in auto_dims:
            fit, kind = fit_safe(f"{dim} ~ C(model) + C(condition)",
                                 w2, w2["case"])
            note = "" if kind == "mixed" else f" [{kind}: mixed fit singular]"
            term = [t for t in fit.params.index if "condition" in t]
            if term:
                t = term[0]
                md_lines.append(
                    f"- {dim}: context effect = {fit.params[t]:+.2f} "
                    f"(p = {fit.pvalues[t]:.3g}){note}")
        # (b) wave 1 vs wave 2 on the automated scores
        if args.wave1_auto_scores.exists():
            w1 = pd.read_csv(args.wave1_auto_scores)
            both = pd.concat([w1, w2], ignore_index=True)
            both = both[both.condition.isin(["without_context"])]
            for dim in auto_dims:
                fit, kind = fit_safe(f"{dim} ~ C(model) * C(wave)",
                                     both, both["case"])
                note = ("" if kind == "mixed"
                        else f" [{kind}: mixed fit singular]")
                inter = [t for t in fit.params.index
                         if ":" in t and "wave" in t]
                sig = [t for t in inter if fit.pvalues[t] < 0.05]
                md_lines.append(f"- {dim}: wave x model interaction terms "
                                f"significant: {len(sig)}/{len(inter)}{note}")
    else:
        md_lines.append("PENDING - no scored wave-2 data at "
                        f"`{args.wave2_scores}`. Run collect.py (with API "
                        "keys) then score.py, and re-run stats.py.")

    (args.outdir / "summary.md").write_text("\n".join(md_lines),
                                            encoding="utf-8")
    print("\n".join(md_lines))
    print(f"\nWrote tables to {args.outdir}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
