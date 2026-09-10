# Swin-Tiny (fair) — test

Rows: 2400 | threshold: 0.5

## Overall detection performance

| metric | value |
| --- | --- |
| n | 2400 |
| accuracy | 0.9971 |
| balanced_accuracy | 0.9971 |
| precision | 0.9967 |
| recall | 0.9975 |
| f1 | 0.9971 |
| auc | 0.9999 |
| fpr | 0.0033 |
| fnr | 0.0025 |
| positive_rate | 0.5004 |

## Fairness audit — grouped by `skin_tone`

| group   | small_n   |   n |   accuracy |   balanced_accuracy |   precision |   recall |     f1 |    auc |    fpr |    fnr |   positive_rate |
|:--------|:----------|----:|-----------:|--------------------:|------------:|---------:|-------:|-------:|-------:|-------:|----------------:|
| Black   | False     | 800 |     1.0000 |              1.0000 |      1.0000 |   1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |          0.5000 |
| Indian  | False     | 800 |     0.9975 |              0.9975 |      0.9975 |   0.9975 | 0.9975 | 0.9999 | 0.0025 | 0.0025 |          0.5000 |
| White   | False     | 800 |     0.9938 |              0.9938 |      0.9925 |   0.9950 | 0.9938 | 0.9999 | 0.0075 | 0.0050 |          0.5012 |

Gaps (groups with n >= 30; grey bars in plots are small-n groups):

| metric | value |
| --- | --- |
| fpr_gap | 0.0075 |
| tpr_gap | 0.0050 |
| auc_gap | 0.0001 |
| equalized_odds_gap | 0.0075 |
| demographic_parity_gap | 0.0012 |
| worst_group_auc | 0.9999 |
| n_groups | 3 |
| fpr_gap_ci | [0.0023416474236146374, 0.017114914425427872] |
| auc_gap_ci | [1.6053950492997498e-05, 0.0003533409706478859] |
