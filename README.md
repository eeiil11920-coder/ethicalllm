# Ethical stress-testing of conversational systems

Protocol, corpus, code and results accompanying a manuscript under review.

**Browse the materials:** https://eeiil11920-coder.github.io/ethicalllm/

## Contents
- `index.html` — the materials site: protocol, applied case (appendices A1–A5), data and code.
- `prompts/cases.json` — the ten dilemmas with their full prompt trees.
- `collect.py` — administers the protocol; `--dry-run` needs no credentials.
- `score.py`, `validate_nli.py`, `process_expert_ratings.py`, `stats.py`, `wave2_contrasts.py`, `make_figures.py` — scoring, validation, analysis and figures.
- `data/wave2/` — one JSONL per dialogue with the full message sequence and the effective parameters.
- `results/` — scores, reliability, models, contrasts and figures.
- `results/MASTER_DATA.xlsx` — everything in one workbook.

## Reproducing without API access

```bash
pip install -r requirements.txt
python collect.py --dry-run
python score.py
python stats.py
python make_figures.py
```

Credentials are read from a local `.env` that is not part of this repository. Random seeds are fixed and dependencies are version-locked.

Released for peer review. Contains no identifying information about the authors or the raters.
