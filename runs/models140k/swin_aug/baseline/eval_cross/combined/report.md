# Cross-generator (imdb reals + sd_wiki fakes) — AUC view

Rows: 20676 | threshold: 0.5

## Overall detection performance

| metric | value |
| --- | --- |
| n | 20676 |
| accuracy | 0.4513 |
| balanced_accuracy | 0.4963 |
| precision | 0.6814 |
| recall | 0.3751 |
| f1 | 0.4839 |
| auc | 0.4936 |
| fpr | 0.3825 |
| fnr | 0.6249 |
| positive_rate | 0.3774 |

## Fairness audit — grouped by `skin_tone`

| group   | small_n   |     n |   accuracy |   balanced_accuracy |   precision |   recall |     f1 |    auc |    fpr |    fnr |   positive_rate |
|:--------|:----------|------:|-----------:|--------------------:|------------:|---------:|-------:|-------:|-------:|-------:|----------------:|
| Black   | False     |  3166 |     0.4049 |              0.4174 |      0.4178 |   0.2604 | 0.3208 | 0.3768 | 0.4255 | 0.7396 |          0.3364 |
| Indian  | False     |  1088 |     0.4476 |              0.4473 |      0.7854 |   0.4478 | 0.5704 | 0.4233 | 0.5533 | 0.5522 |          0.4669 |
| White   | False     | 16422 |     0.4605 |              0.5119 |      0.7180 |   0.3865 | 0.5025 | 0.5164 | 0.3626 | 0.6135 |          0.3794 |

Gaps (groups with n >= 30; grey bars in plots are small-n groups):

| metric | value |
| --- | --- |
| fpr_gap | 0.1907 |
| tpr_gap | 0.1874 |
| auc_gap | 0.1396 |
| equalized_odds_gap | 0.1907 |
| demographic_parity_gap | 0.1305 |
| worst_group_auc | 0.3768 |
| n_groups | 3 |
| fpr_gap_ci | [0.12211370618831643, 0.2619986451292539] |
| auc_gap_ci | [0.11828297044618319, 0.1624560814771823] |
