# Wave-1 re-analysis (mixed-effects primary, ANOVA/Kruskal-Wallis sensitivity)

Data: `Evaluaci_n__tica_por_Modelo_y_Nodo.csv` - 250 node-level human ratings (5 models x 10 cases x 5 nodes).
Unit of analysis: dialogue (model x case); nodes are repeated measures.

## Primary: mixed-effects models (score ~ model, random: case, node-in-case)

Note: statsmodels MixedLM admits one grouping factor; crossed random effects are approximated with groups=case plus a node variance component (vc_formula). Rater effects cannot be estimated from the archived consensus ratings.

### Awareness
- Overall model effect: Wald chi2(4) = 33.96, p = 7.58e-07 (significant)
- Highest mean: Claude; lowest: DeepSeek; max model gap = 1.18 points
- Significant contrasts vs ChatGPT: Claude: +0.60 (p=0.0143); DeepSeek: -0.58 (p=0.0179); Gemini: +0.58 (p=0.0179)
- Variance components: case = 0.439, node-in-case = 0.2034, residual = 1.5011 (ICC case = 0.205)

### Consistency
- Overall model effect: Wald chi2(4) = 15.48, p = 0.0038 (significant)
- Highest mean: ChatGPT; lowest: DeepSeek; max model gap = 0.30 points
- Significant contrasts vs ChatGPT: DeepSeek: -0.30 (p=0.00229); Grok: -0.26 (p=0.0082)
- Variance components: case = 0.0005, node-in-case = 0.0955, residual = 0.2418 (ICC case = 0.001)

### Ethics
- Overall model effect: Wald chi2(4) = 35.43, p = 3.79e-07 (significant)
- Highest mean: Claude; lowest: DeepSeek; max model gap = 1.62 points
- Significant contrasts vs ChatGPT: Claude: +0.92 (p=0.00111); DeepSeek: -0.70 (p=0.0131)
- Variance components: case = 0.5065, node-in-case = 0.4578, residual = 1.9911 (ICC case = 0.171)

### Contradiction
- Overall model effect: Wald chi2(4) = 3.28, p = 0.512 (n.s.)
- Highest mean: DeepSeek; lowest: ChatGPT; max model gap = 0.36 points
- No pairwise contrast vs ChatGPT reaches p < .05
- Variance components: case = 0.0706, node-in-case = 0.0101, residual = 1.1274 (ICC case = 0.058)

## Sensitivity: ANOVA and Kruskal-Wallis on dialogue means (paper's original approach)

- Awareness: F(4,45) = 2.67, p = 0.0444, eta2 = 0.192; H = 9.21, p = 0.056, eps2 = 0.188
- Consistency: F(4,45) = 4.14, p = 0.00614, eta2 = 0.269; H = 13.01, p = 0.0112, eps2 = 0.265
- Ethics: F(4,45) = 2.55, p = 0.0522, eta2 = 0.185; H = 9.93, p = 0.0416, eps2 = 0.203
- Contradiction: F(4,45) = 0.62, p = 0.652, eta2 = 0.052; H = 2.35, p = 0.672, eps2 = 0.048

## Wave-2 contrasts (context condition, wave x model)

Wave-2 scored data found: `<repository> - Universidad Pablo de  de Sevilla\Escritorio\1_ARTÍCULO\ENVIADOS\5_ETHICS_XXX_David\mayor revissions\revision\repository\results\wave2\auto_scores_wave2.csv` (100 dialogues).
- S_aware: context effect = +0.26 (p = 0.0242)
- S_consist: context effect = +0.20 (p = 0.231)
- S_ethics: context effect = +0.19 (p = 0.0232)
- S_contra: context effect = +1.00 (p = 0.00906)
- S_aware: wave x model interaction terms significant: 1/4
- S_consist: wave x model interaction terms significant: 3/4
- S_ethics: wave x model interaction terms significant: 1/4
- S_contra: wave x model interaction terms significant: 2/4