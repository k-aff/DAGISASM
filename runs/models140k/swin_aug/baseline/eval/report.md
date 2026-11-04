# Swin-Tiny (baseline) — test

Rows: 2400 | threshold: 0.5

## Overall detection performance

| metric | value |
| --- | --- |
| n | 2400 |
| accuracy | 0.9967 |
| balanced_accuracy | 0.9967 |
| precision | 0.9983 |
| recall | 0.9950 |
| f1 | 0.9967 |
| auc | 0.9998 |
| fpr | 0.0017 |
| fnr | 0.0050 |
| positive_rate | 0.4983 |

## Fairness audit — grouped by `skin_tone`

| group   | small_n   |   n |   accuracy |   balanced_accuracy |   precision |   recall |     f1 |    auc |    fpr |    fnr |   positive_rate |
|:--------|:----------|----:|-----------:|--------------------:|------------:|---------:|-------:|-------:|-------:|-------:|----------------:|
| Black   | False     | 800 |     0.9975 |              0.9975 |      1.0000 |   0.9950 | 0.9975 | 1.0000 | 0.0000 | 0.0050 |          0.4975 |
| Indian  | False     | 800 |     0.9988 |              0.9988 |      1.0000 |   0.9975 | 0.9987 | 0.9994 | 0.0000 | 0.0025 |          0.4988 |
| White   | False     | 800 |     0.9938 |              0.9938 |      0.9950 |   0.9925 | 0.9937 | 0.9999 | 0.0050 | 0.0075 |          0.4988 |

Gaps (groups with n >= 30; grey bars in plots are small-n groups):

| metric | value |
| --- | --- |
| fpr_gap | 0.0050 |
| tpr_gap | 0.0050 |
| auc_gap | 0.0006 |
| equalized_odds_gap | 0.0050 |
| demographic_parity_gap | 0.0013 |
| worst_group_auc | 0.9994 |
| n_groups | 3 |
| fpr_gap_ci | [0.0, 0.01282051282051282] |
| auc_gap_ci | [1.888402867221696e-05, 0.002055309037570386] |
