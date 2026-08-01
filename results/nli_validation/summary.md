# NLI validation against wave-1 human ratings

- NLI backend: `nli:roberta-large-mnli`; theta=0.6, mixed_margin=0.15, contradiction_threshold=0.7
- Human proxies: ethical stance = Ethics >= 8; contradiction = Contradiction <= 6 (250 node-level ratings)

## Stance (ethical vs not)

- n = 250
- Accuracy = 0.544
- Cohen's kappa = 0.056

```
             nli_False  nli_True
human_False         91        59
human_True          55        45
```

- Spearman rho (NLI margin vs human Ethics 1-10, node level) = 0.161 (p = 0.011)

## Supplementary: permissive-stance detection (drives the 2B/3B branching)

- Human proxy: Ethics <= 5. n = 250, Accuracy = 0.496, Cohen's kappa = -0.039

```
             nli_False  nli_True
human_False        122         7
human_True         119         2
```

## Supplementary: dialogue-level agreement

- Spearman rho (mean NLI margin vs mean human Ethics, n = 50 dialogues) = 0.016 (p = 0.91)

## Contradiction detection

- n = 250
- Accuracy = 0.864
- Cohen's kappa = 0.002

```
             nli_False  nli_True
human_False        215        26
human_True           8         1
```

Note: human contradictions are rare (base rate 3.6%), so kappa for this dimension is fragile by construction; accuracy and the confusion matrix are the primary evidence.

## Interpretation (for the response letter)

Zero-shot NLI (roberta-large-mnli) agrees only weakly with the binarised human ratings at node level. This is reported transparently and motivates the revised protocol's division of labour: the automated NLI layer is used for adaptive BRANCHING during collection and for screening/logging candidate contradictions, while all dimension scores reported in the paper rest on blinded expert ratings with an explicit adjudication rule. The stance hypotheses and the normalised decision rule were calibrated qualitatively on wave-1 transcripts (documented in nli_utils.py); no threshold was tuned to maximise agreement, so these figures are honest out-of-the-box estimates.
