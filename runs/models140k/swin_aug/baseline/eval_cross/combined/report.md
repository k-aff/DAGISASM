# Cross-generator (imdb reals + sd_wiki fakes) — AUC view

Rows: 20676 | threshold: 0.5

## Overall detection performance

| metric | value |
| --- | --- |
| n | 20676 |
| accuracy | 0.3831 |
| balanced_accuracy | 0.5206 |
| precision | 0.7504 |
| recall | 0.1503 |
| f1 | 0.2505 |
| auc | 0.5529 |
| fpr | 0.1091 |
| fnr | 0.8497 |
| positive_rate | 0.1374 |

## Fairness audit — grouped by `skin_tone`

| group   | small_n   |     n |   accuracy |   balanced_accuracy |   precision |   recall |     f1 |    auc |    fpr |    fnr |   positive_rate |
|:--------|:----------|------:|-----------:|--------------------:|------------:|---------:|-------:|-------:|-------:|-------:|----------------:|
| Black   | False     |  3166 |     0.4659 |              0.4991 |      0.5346 |   0.0813 | 0.1412 | 0.4515 | 0.0830 | 0.9187 |          0.0821 |
| Indian  | False     |  1088 |     0.3134 |              0.5057 |      0.8273 |   0.2043 | 0.3276 | 0.5384 | 0.1929 | 0.7957 |          0.2022 |
| White   | False     | 16422 |     0.3718 |              0.5214 |      0.7669 |   0.1563 | 0.2597 | 0.5696 | 0.1135 | 0.8437 |          0.1437 |

Gaps (groups with n >= 30; grey bars in plots are small-n groups):

| metric | value |
| --- | --- |
| fpr_gap | 0.1098 |
| tpr_gap | 0.1229 |
| auc_gap | 0.1181 |
| equalized_odds_gap | 0.1229 |
| demographic_parity_gap | 0.1201 |
| worst_group_auc | 0.4515 |
| n_groups | 3 |
| fpr_gap_ci | [0.05552141284425611, 0.17174359752664062] |
| auc_gap_ci | [0.09592826122569563, 0.144081056897297] |
