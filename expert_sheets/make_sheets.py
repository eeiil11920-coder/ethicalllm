#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_sheets.py — Builds blinded expert-rating materials from collected
wave-2 dialogues (see anonymization.md for the protocol).

For each rater it produces, under expert_sheets/output/:
  * rater_<ID>.csv               blank scoring sheet (randomised order)
  * transcripts_<ID>.md          anonymised transcripts in that order
  * provider_key_<ID>.csv        PRIVATE blinding key (do NOT distribute)

Usage:
  python make_sheets.py --wave2-dir ..\\data\\wave2 --raters R1,R2,R3 --seed 2026
"""
import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NODES = ["node1", "node2", "node3", "conclusion", "confirmation"]
NODE_LABEL = {"node1": "Node 1", "node2": "Node 2", "node3": "Node 3",
              "conclusion": "Conclusion", "confirmation": "Confirmation"}

SELF_ID = re.compile(
    r"\b(as|i am|i'm)\s+(chatgpt|claude|gemini|grok|deepseek|gpt-[\w.-]+|"
    r"an ai (?:developed|created|made) by (?:openai|anthropic|google|xai|"
    r"deepseek))\b", re.I)
BRAND = re.compile(r"\b(chatgpt|claude|gemini|grok|deepseek|openai|"
                   r"anthropic|google ai|xai)\b", re.I)


def load_dialogues(d: Path):
    out = []
    for f in sorted(d.glob("*.jsonl")):
        if f.name == "manifest.jsonl":
            continue
        meta, turns = {}, []
        for line in f.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec["type"] == "meta":
                meta = rec
            elif rec["type"] == "turn":
                turns.append(rec)
        out.append({"file": f.name, "meta": meta, "turns": turns})
    return out


def anonymise(text: str) -> str:
    text = re.sub(r"\[dry-run:[^\]]*\]\s*", "", text)
    text = SELF_ID.sub("As an AI assistant", text)
    return BRAND.sub("the assistant", text)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wave2-dir", type=Path,
                    default=HERE.parent / "data" / "wave2")
    ap.add_argument("--raters", default="R1,R2,R3")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--outdir", type=Path, default=HERE / "output")
    args = ap.parse_args()

    dialogues = load_dialogues(args.wave2_dir)
    if not dialogues:
        raise SystemExit(f"no dialogues found in {args.wave2_dir}")
    args.outdir.mkdir(parents=True, exist_ok=True)
    providers = sorted({d["meta"]["provider"] for d in dialogues})

    for rater in [r.strip() for r in args.raters.split(",")]:
        rng = random.Random(f"{args.seed}|{rater}")
        codes = [f"M{i+1}" for i in range(len(providers))]
        rng.shuffle(codes)
        pmap = dict(zip(providers, codes))
        order = list(range(len(dialogues)))
        rng.shuffle(order)

        key_rows, sheet_rows, md = [], [], []
        md.append(f"# Anonymised transcripts — rater {rater}\n")
        for pos, idx in enumerate(order, 1):
            d = dialogues[idx]
            meta = d["meta"]
            dcode = f"D{pos:03d}"
            key_rows.append({
                "DialogueCode": dcode, "file": d["file"],
                "provider": meta["provider"],
                "model_code": pmap[meta["provider"]],
                "case_id": meta["case_id"], "condition": meta["condition"],
            })
            md.append(f"\n## {dcode}  (model {pmap[meta['provider']]})\n")
            for t in d["turns"]:
                who = "USER" if t["role"] == "user" else "ASSISTANT"
                md.append(f"**{NODE_LABEL.get(t['node'], t['node'])} — "
                          f"{who}:** {anonymise(t['content'])}\n")
            for n in NODES:
                sheet_rows.append({
                    "RaterID": rater, "DialogueCode": dcode,
                    "Node": NODE_LABEL[n], "Awareness": "",
                    "Consistency": "", "Ethics": "", "Contradiction": "",
                    "Comments": ""})

        with open(args.outdir / f"rater_{rater}.csv", "w", newline="",
                  encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(sheet_rows[0].keys()))
            w.writeheader()
            w.writerows(sheet_rows)
        (args.outdir / f"transcripts_{rater}.md").write_text(
            "\n".join(md), encoding="utf-8")
        with open(args.outdir / f"provider_key_{rater}.csv", "w",
                  newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(key_rows[0].keys()))
            w.writeheader()
            w.writerows(key_rows)
        print(f"[{rater}] {len(dialogues)} dialogues -> rater_{rater}.csv, "
              f"transcripts_{rater}.md, provider_key_{rater}.csv (PRIVATE)")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
