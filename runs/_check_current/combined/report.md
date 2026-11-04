# Cross-generator (imdb reals + sd_wiki fakes) — AUC view

Rows: 600 | threshold: 0.5

## Overall detection performance

| metric | value |
| --- | --- |
| n | 600 |
| accuracy | 0.4883 |
| balanced_accuracy | 0.4883 |
| precision | 0.4848 |
| recall | 0.3733 |
| f1 | 0.4218 |
| auc | 0.4901 |
| fpr | 0.3967 |
| fnr | 0.6267 |
| positive_rate | 0.3850 |

## Fairness audit — grouped by `skin_tone`

| group   | small_n   |   n |   accuracy |   balanced_accuracy |   precision |   recall |     f1 |    auc |    fpr |    fnr |   positive_rate |
|:--------|:----------|----:|-----------:|--------------------:|------------:|---------:|-------:|-------:|-------:|-------:|----------------:|
| Black   | False     |  97 |     0.4742 |              0.4254 |      0.2500 |   0.2727 | 0.2609 | 0.4697 | 0.4219 | 0.7273 |          0.3711 |
| Indian  | True      |  27 |     0.4444 |              0.4765 |      0.6000 |   0.3529 | 0.4444 | 0.4059 | 0.4000 | 0.6471 |          0.3704 |
| White   | False     | 476 |     0.4937 |              0.4993 |      0.5243 |   0.3880 | 0.4460 | 0.4980 | 0.3894 | 0.6120 |          0.3887 |

Gaps (groups with n >= 30; grey bars in plots are small-n groups):

| metric | value |
| --- | --- |
| fpr_gap | 0.0325 |
| tpr_gap | 0.1153 |
| auc_gap | 0.0283 |
| equalized_odds_gap | 0.1153 |
| demographic_parity_gap | 0.0175 |
| worst_group_auc | 0.4697 |
| n_groups | 2 |
| fpr_gap_ci | [0.006194789448358958, 0.22973862942612935] |
| auc_gap_ci | [0.005202670251029388, 0.24091529663556857] |
