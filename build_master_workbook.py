# -*- coding: utf-8 -*-
"""Consolidate every dataset of the study into one Excel workbook for
consultation: transcripts, scores, statistics and validation results.

Output: results/MASTER_DATA.xlsx
"""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
ANALYSIS = HERE.parents[2] / "analysis"
OUT = RES / "MASTER_DATA.xlsx"

MODEL_CODE = {2: "ChatGPT", 3: "Claude", 4: "Gemini", 5: "Grok", 6: "DeepSeek"}
NODES = ["Node 1", "Node 2", "Node 3", "Conclusion", "Confirmation"]
sheets = {}


def add(name, df, note=""):
    if df is None or len(df) == 0:
        return
    sheets[name[:31]] = (df, note)


# ------------------------------------------------------------- wave 1 data
w1x = ANALYSIS / "Cases_LLM_Ethics_2.xlsx"
if w1x.exists():
    raw = pd.read_excel(w1x)
    long = []
    for _, r in raw.iterrows():
        for n in NODES:
            long.append({"wave": 1, "collected": str(r["Id"])[:10],
                         "case": r["Case"], "model": r["Model"],
                         "version": r["Submodel"], "node": n,
                         "response": str(r[n]).strip()})
    add("W1_transcripts", pd.DataFrame(long),
        "Wave-1 dialogues, one row per node response (50 dialogues x 5 nodes).")

w1csv = ANALYSIS / "Evaluaci_n__tica_por_Modelo_y_Nodo.csv"
if w1csv.exists():
    df = pd.read_csv(w1csv)
    df.insert(0, "wave", 1)
    df["ModelLabel"] = df["Model"].map(MODEL_CODE)
    add("W1_expert_scores", df,
        "Wave-1 expert node-level ratings (scores of record), 1-10 scale.")

for f, note in [
    ("wave1/auto_scores_wave1.csv",
     "Wave-1 automated dialogue-level scores (comparable with wave 2)."),
    ("wave1/auto_scores_wave1_nodes.csv",
     "Wave-1 automated node-level detail."),
    ("wave1/descriptives.csv", "Wave-1 descriptive statistics by model."),
]:
    p = RES / f
    if p.exists():
        add("W1_" + Path(f).stem.replace("auto_scores_wave1", "auto")
            .replace("descriptives", "descriptives"), pd.read_csv(p), note)

# ------------------------------------------------------------- wave 2 data
w2dir = HERE / "data" / "wave2"
if w2dir.exists():
    rows, meta = [], []
    for jf in sorted(w2dir.glob("*.jsonl")):
        if "manifest" in jf.name:
            continue
        m = {}
        for line in jf.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("type") == "meta":
                m = rec
            elif rec.get("type") == "turn" and rec.get("role") == "assistant":
                rows.append({
                    "wave": 2, "collected": rec.get("timestamp", "")[:10],
                    "provider": m.get("provider"),
                    "model": m.get("model_label"),
                    "api_model_id": rec.get("api_model_id"),
                    "case": m.get("case_id"), "case_title": m.get("case_title"),
                    "condition": m.get("condition"), "node": rec.get("node"),
                    "response": (rec.get("content") or "").strip()})
            elif rec.get("type") == "result":
                meta.append({
                    "provider": m.get("provider"), "model": m.get("model_label"),
                    "case": m.get("case_id"), "condition": m.get("condition"),
                    "branch": rec.get("branch_taken"),
                    "api_model_ids": ", ".join(rec.get("api_model_ids", []))
                    if isinstance(rec.get("api_model_ids"), list)
                    else rec.get("api_model_ids"),
                    "param_notes": " | ".join(rec.get("param_notes") or [])})
    add("W2_transcripts", pd.DataFrame(rows),
        "Wave-2 dialogues, one row per node response (100 dialogues x 5 nodes).")
    add("W2_collection_log", pd.DataFrame(meta),
        "Wave-2 run log: branch taken, exact API model id, parameter notes.")

for f, name, note in [
    ("wave2/auto_scores_wave2.csv", "W2_auto_scores",
     "Wave-2 automated dialogue-level scores (provisional, expert round pending)."),
    ("wave2/auto_scores_wave2_nodes.csv", "W2_auto_nodes",
     "Wave-2 automated node-level detail."),
    ("wave2/means_by_model_condition.csv", "W2_means_by_condition",
     "Wave-2 means by model and deployment condition (source of the article table)."),
]:
    p = RES / f
    if p.exists():
        add(name, pd.read_csv(p), note)

# -------------------------------------------------------------- statistics
for f, name, note in [
    ("wave1/mixedlm_awareness.csv", "Stat_mixed_awareness", "Mixed-effects model, Awareness."),
    ("wave1/mixedlm_consistency.csv", "Stat_mixed_consistency", "Mixed-effects model, Consistency."),
    ("wave1/mixedlm_ethics.csv", "Stat_mixed_ethics", "Mixed-effects model, Ethics."),
    ("wave1/mixedlm_contradiction.csv", "Stat_mixed_contradiction", "Mixed-effects model, Contradiction."),
    ("wave1/mixedlm_overview.csv", "Stat_mixed_overview", "Mixed-effects omnibus overview."),
    ("wave1/anova_kruskal.csv", "Stat_anova_kruskal", "ANOVA / Kruskal-Wallis sensitivity analysis."),
    ("wave1/cohensd_awareness.csv", "Stat_d_awareness", "Pairwise Cohen's d, Awareness."),
    ("wave1/cohensd_consistency.csv", "Stat_d_consistency", "Pairwise Cohen's d, Consistency."),
    ("wave1/cohensd_ethics.csv", "Stat_d_ethics", "Pairwise Cohen's d, Ethics."),
    ("wave1/cohensd_contradiction.csv", "Stat_d_contradiction", "Pairwise Cohen's d, Contradiction."),
    ("nli_validation/predictions.csv", "NLI_validation", "NLI classifier vs human labels."),
]:
    p = RES / f
    if p.exists():
        add(name, pd.read_csv(p), note)

# ------------------------------------------------------------------- index
index = pd.DataFrame(
    [{"sheet": k, "rows": len(v[0]), "columns": len(v[0].columns),
      "content": v[1]} for k, v in sheets.items()])
index = pd.concat([
    pd.DataFrame([{"sheet": "README", "rows": "", "columns": "",
                   "content": "Master data workbook. Wave 1 = 50 dialogues, "
                              "12-14 May 2025. Wave 2 = 100 dialogues "
                              "(5 models x 10 cases x 2 conditions), "
                              "31 July 2026. Both waves via official APIs. "
                              "Expert scores are the scores of record for "
                              "wave 1; wave-2 expert round pending."}]),
    index], ignore_index=True)

with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
    index.to_excel(xw, sheet_name="README", index=False)
    for name, (df, _) in sheets.items():
        df.to_excel(xw, sheet_name=name, index=False)

print("WROTE:", OUT)
for _, r in index.iterrows():
    if r["sheet"] != "README":
        print(f"  {r['sheet']:26} {r['rows']:>6} rows")
