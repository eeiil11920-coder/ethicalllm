# -*- coding: utf-8 -*-
"""Process the returned expert rating sheets.

Reads every *_results.xlsx in EXPERT_PACK/Rater_*/, de-anonymises the dialogue
codes with the organiser keys, computes inter-rater reliability (Krippendorff's
ordinal alpha and pairwise Spearman), aggregates to the expert consensus, and
compares the consensus against the automated pipeline scores.

Outputs -> results/experts/
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
PACK = HERE.parent / "EXPERT_PACK"
OUT = HERE / "results" / "experts"
OUT.mkdir(parents=True, exist_ok=True)
DIMS = ["Awareness", "Consistency", "Ethics", "Contradiction"]


def krippendorff_ordinal(matrix):
    """Krippendorff's alpha, ordinal metric.
    matrix: rows = raters, cols = units; NaN allowed."""
    m = np.asarray(matrix, dtype=float)
    vals = m[~np.isnan(m)]
    if vals.size == 0:
        return np.nan
    levels = np.unique(vals)
    idx = {v: i for i, v in enumerate(levels)}
    n_c = np.zeros(len(levels))
    for v in vals:
        n_c[idx[v]] += 1

    def delta2(a, b):
        i, j = idx[a], idx[b]
        if i > j:
            i, j = j, i
        s = n_c[i:j + 1].sum() - (n_c[i] + n_c[j]) / 2.0
        return s ** 2

    Do_num, n_pairs = 0.0, 0
    for u in range(m.shape[1]):
        col = m[:, u]
        col = col[~np.isnan(col)]
        mu = len(col)
        if mu < 2:
            continue
        for a, b in itertools.permutations(col, 2):
            Do_num += delta2(a, b) / (mu - 1)
        n_pairs += mu
    if n_pairs == 0:
        return np.nan
    Do = Do_num / n_pairs
    De_num = 0.0
    for a in levels:
        for b in levels:
            De_num += n_c[idx[a]] * n_c[idx[b]] * delta2(a, b)
    De = De_num / (n_pairs * (n_pairs - 1))
    return 1.0 - Do / De if De else np.nan


# --------------------------------------------------------------- load sheets
frames, keys = [], []
# raters returned their sheets in different formats and with slightly different
# file names (*_result.csv / *_results.xlsx), so accept both.
returned = sorted(p for p in PACK.rglob("*")
                  if p.is_file() and "result" in p.stem.lower()
                  and p.suffix.lower() in (".xlsx", ".csv"))
for f in returned:
    rater = f.parent.name.replace("Rater_", "")
    if f.suffix.lower() == ".xlsx":
        xl = pd.ExcelFile(f)
        sheet = "Scores" if "Scores" in xl.sheet_names else xl.sheet_names[0]
        d = xl.parse(sheet)
    else:
        d = pd.read_csv(f)
    d = d.loc[:, ~d.columns.astype(str).str.startswith("Unnamed")]
    d["RaterID"] = rater
    print(f"  loaded {f.name} -> rater {rater} ({len(d)} rows)")
    frames.append(d)
    k = PACK / "_ORGANISER_ONLY" / f"provider_key_{rater}.csv"
    if k.exists():
        kk = pd.read_csv(k)
        kk["RaterID"] = rater
        keys.append(kk)
if not frames:
    raise SystemExit("no returned sheets found")

r = pd.concat(frames, ignore_index=True)
key = pd.concat(keys, ignore_index=True)
print("raters returned:", sorted(r.RaterID.unique()))
print("rows per rater:", r.groupby("RaterID").size().to_dict())

# CRITICAL: each rater received the dialogues in a DIFFERENT random order, so
# `DialogueCode` is rater-specific (only ~4% of codes coincide across raters).
# Reliability must therefore be computed on the TRUE dialogue identity, which
# is the `file` column of the organiser key, never on DialogueCode.
merge_cols = [c for c in ("DialogueCode", "RaterID", "file", "provider",
                          "model_code", "case_id", "condition")
              if c in key.columns or c == "RaterID"]
r = r.merge(key[merge_cols], on=["DialogueCode", "RaterID"], how="left")
r["dialogue_uid"] = r["file"].str.replace(".jsonl", "", regex=False)
missing = r["dialogue_uid"].isna().sum()
if missing:
    print(f"WARNING: {missing} ratings could not be de-anonymised")
r.to_csv(OUT / "expert_ratings_long.csv", index=False)

# ------------------------------------------------------------- reliability
rel = []
for dim in DIMS:
    piv = r.pivot_table(index="RaterID",
                        columns=["dialogue_uid", "Node"], values=dim)
    a = krippendorff_ordinal(piv.values)
    row = {"dimension": dim, "krippendorff_alpha_ordinal": round(a, 3),
           "n_raters": piv.shape[0], "n_units": piv.shape[1]}
    rhos = []
    for x, y in itertools.combinations(piv.index, 2):
        sub = piv.loc[[x, y]].T.dropna()
        if len(sub) > 5:
            rho, _ = sps.spearmanr(sub[x], sub[y])
            rhos.append(rho)
            row[f"rho_{x}_{y}"] = round(rho, 3)
    row["mean_pairwise_rho"] = round(float(np.mean(rhos)), 3) if rhos else np.nan
    rel.append(row)
rel = pd.DataFrame(rel)
print("\n== Inter-rater reliability ==")
print(rel.to_string(index=False))
rel.to_csv(OUT / "reliability.csv", index=False)

# ------------------------------------------------------- consensus scores
grp = ["dialogue_uid"]
cons_node = (r.groupby(grp + ["Node"])[DIMS].median().reset_index())
cons_node.to_csv(OUT / "expert_consensus_node.csv", index=False)
cons_dlg = r.groupby(grp)[DIMS].median().reset_index()

meta_cols = [c for c in ("provider", "case_id", "condition") if c in r.columns]
if meta_cols:
    meta = r.groupby(grp)[meta_cols].first().reset_index()
    cons_dlg = cons_dlg.merge(meta, on=grp)
cons_dlg = cons_dlg.rename(columns={"case_id": "case"})
cons_dlg.to_csv(OUT / "expert_consensus_dialogue.csv", index=False)
print(f"\nconsensus dialogues: {len(cons_dlg)}")

mcol = next((c for c in ("model", "model_label", "provider")
             if c in cons_dlg.columns), None)
if mcol:
    by_model = cons_dlg.groupby(mcol)[DIMS].mean().round(2)
    print("\n== Expert consensus by model ==")
    print(by_model.to_string())
    by_model.to_csv(OUT / "expert_by_model.csv")
    ccol = "condition" if "condition" in cons_dlg.columns else None
    if ccol:
        by_mc = cons_dlg.groupby([mcol, ccol])[DIMS].mean().round(2)
        print("\n== Expert consensus by model and condition ==")
        print(by_mc.to_string())
        by_mc.to_csv(OUT / "expert_by_model_condition.csv")

# --------------------------------------- agreement with automated pipeline
auto_p = HERE / "results" / "wave2" / "auto_scores_wave2.csv"
if auto_p.exists() and mcol:
    auto = pd.read_csv(auto_p)
    ren = {"S_aware": "Awareness", "S_consist": "Consistency",
           "S_ethics": "Ethics", "S_contra": "Contradiction"}
    auto = auto.rename(columns=ren)
    # Join on the DIALOGUE IDENTITY, never on (case, condition): the expert
    # consensus carries `provider` while the automated table carries `model`,
    # so a (case, condition) join silently becomes many-to-many and pairs each
    # dialogue with all five models' scores, destroying the correlation.
    mg = cons_dlg.merge(auto, left_on="dialogue_uid", right_on="dialogue_id",
                        suffixes=("_exp", "_auto"))
    assert len(mg) == len(cons_dlg), (
        f"join produced {len(mg)} rows for {len(cons_dlg)} dialogues")
    if True:
        rows = []
        for dim in DIMS:
            ce, ca = f"{dim}_exp", f"{dim}_auto"
            if ce in mg and ca in mg:
                rho, p = sps.spearmanr(mg[ce], mg[ca])
                rows.append({"dimension": dim, "n": len(mg),
                             "rho_expert_vs_auto": round(rho, 3),
                             "p": round(p, 4),
                             "MAE": round((mg[ce] - mg[ca]).abs().mean(), 2),
                             "pct_gap_gt_2": round(
                                 100 * ((mg[ce] - mg[ca]).abs() > 2).mean(), 1)})
        if rows:
            adj = pd.DataFrame(rows)
            print("\n== Expert consensus vs automated pipeline ==")
            print(adj.to_string(index=False))
            adj.to_csv(OUT / "expert_vs_auto.csv", index=False)

print("\nWROTE ->", OUT)
