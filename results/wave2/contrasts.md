# Wave-2 pre-registered contrasts

## 1. Condition contrast (model x condition interaction)

| dimension          |   context_effect |   context_p |   interaction_chi2 |   interaction_p |   spread_general |   spread_grounded | fit   |
|:-------------------|-----------------:|------------:|-------------------:|----------------:|-----------------:|------------------:|:------|
| Awareness          |            0.2   |      0.0204 |               4.49 |          0.3434 |             0.88 |              1.52 | mixed |
| Consistency        |            0.2   |      0.2233 |              10.8  |          0.0289 |             1.18 |              1.56 | mixed |
| Ethics-over-profit |            0.295 |      0.0186 |               2.64 |          0.6204 |             0.62 |              0.63 | mixed |
| Contradiction      |            1.4   |      0.0062 |               2.58 |          0.6304 |             3.2  |              3    | mixed |

Between-model spread = max-min of model means within the condition. If grounding removed the differences, spread_grounded would collapse towards zero.


## 2. Wave contrast (rank stability, general-purpose condition)

| dimension          |   wave1_mean |   wave2_mean |   delta |   rank_rho |   rho_p |
|:-------------------|-------------:|-------------:|--------:|-----------:|--------:|
| Awareness          |         8.26 |         8.22 |   -0.04 |      0.205 |  0.7406 |
| Consistency        |         9.54 |         9.16 |   -0.38 |      0.3   |  0.6238 |
| Ethics-over-profit |         6.26 |         6.07 |   -0.19 |      0.5   |  0.391  |
| Contradiction      |         8.26 |         8.32 |    0.06 |     -0.4   |  0.5046 |

Per-model means (general-purpose condition only):

| model    |   S_aware_w1 |   S_consist_w1 |   S_ethics_w1 |   S_contra_w1 |   S_aware_w2 |   S_consist_w2 |   S_ethics_w2 |   S_contra_w2 |
|:---------|-------------:|---------------:|--------------:|--------------:|-------------:|---------------:|--------------:|--------------:|
| ChatGPT  |         8.12 |           8.86 |          6.4  |           6   |         8.4  |           9.38 |          5.98 |           9   |
| Claude   |         8.16 |           9.68 |          6.16 |           8.7 |         8.36 |           9.74 |          6.33 |           8.4 |
| Gemini   |         8.28 |           9.5  |          6.03 |          10   |         7.56 |           8.56 |          5.84 |           6.2 |
| Grok     |         8.12 |           9.74 |          6.2  |           8.8 |         8.36 |           8.62 |          5.8  |           9.4 |
| DeepSeek |         8.64 |           9.94 |          6.52 |           7.8 |         8.44 |           9.52 |          6.41 |           8.6 |

## 3. Pressure-response contrast (stance flips)

| wave         | condition       |   mean_flips | dialogues_with_flip   |   mean_modal_rate |
|:-------------|:----------------|-------------:|:----------------------|------------------:|
| 1 (May 2025) | without_context |         0.12 | 5/50                  |             0.928 |
| 2 (Jul 2026) | with_context    |         0.1  | 3/50                  |             0.72  |
| 2 (Jul 2026) | without_context |         0.04 | 2/50                  |             0.748 |

Wave-2 mean stance flips per dialogue by model and condition:

| model    |   with_context |   without_context |
|:---------|---------------:|------------------:|
| ChatGPT  |            0.2 |               0.1 |
| Claude   |            0.1 |               0.1 |
| Gemini   |            0   |               0   |
| Grok     |            0   |               0   |
| DeepSeek |            0.2 |               0   |


## Table source: means by model and condition

|                                 |   S_aware |   S_consist |   S_ethics |   S_contra |
|:--------------------------------|----------:|------------:|-----------:|-----------:|
| ('ChatGPT', 'with_context')     |      8.2  |        9.18 |       5.69 |        7.6 |
| ('ChatGPT', 'without_context')  |      8.4  |        9.38 |       5.98 |        9   |
| ('Claude', 'with_context')      |      8.44 |        9.2  |       6.32 |        6.8 |
| ('Claude', 'without_context')   |      8.36 |        9.74 |       6.33 |        8.4 |
| ('Gemini', 'with_context')      |      6.92 |        7.78 |       5.7  |        6   |
| ('Gemini', 'without_context')   |      7.56 |        8.56 |       5.84 |        6.2 |
| ('Grok', 'with_context')        |      8.16 |        9.34 |       5.68 |        9   |
| ('Grok', 'without_context')     |      8.36 |        8.62 |       5.8  |        9.4 |
| ('DeepSeek', 'with_context')    |      8.08 |        9.3  |       6.05 |        7.2 |
| ('DeepSeek', 'without_context') |      8.44 |        9.52 |       6.41 |        8.6 |