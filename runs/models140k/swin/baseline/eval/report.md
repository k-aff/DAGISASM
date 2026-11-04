# Swin-Tiny (baseline) — test

Rows: 1848 | threshold: 0.5

## Overall detection performance

| metric | value |
| --- | --- |
| n | 1848 |
| accuracy | 0.9968 |
| balanced_accuracy | 0.9968 |
| precision | 0.9989 |
| recall | 0.9946 |
| f1 | 0.9967 |
| auc | 0.9999 |
| fpr | 0.0011 |
| fnr | 0.0054 |
| positive_rate | 0.4978 |

## Fairness audit — grouped by `skin_tone`

| group   | small_n   |   n |   accuracy |   balanced_accuracy |   precision |   recall |     f1 |    auc |    fpr |    fnr |   positive_rate |
|:--------|:----------|----:|-----------:|--------------------:|------------:|---------:|-------:|-------:|-------:|-------:|----------------:|
| Black   | False     | 616 |     1.0000 |              1.0000 |      1.0000 |   1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |          0.5000 |
| Indian  | False     | 616 |     0.9951 |              0.9951 |      0.9967 |   0.9935 | 0.9951 | 0.9999 | 0.0032 | 0.0065 |          0.4984 |
| White   | False     | 616 |     0.9951 |              0.9951 |      1.0000 |   0.9903 | 0.9951 | 0.9999 | 0.0000 | 0.0097 |          0.4951 |

Gaps (groups with n >= 30; grey bars in plots are small-n groups):

| metric | value |
| --- | --- |
| fpr_gap | 0.0032 |
| tpr_gap | 0.0097 |
| auc_gap | 0.0001 |
| equalized_odds_gap | 0.0097 |
| demographic_parity_gap | 0.0049 |
| worst_group_auc | 0.9999 |
| n_groups | 3 |
| fpr_gap_ci | [0.0, 0.01145253497818563] |
| auc_gap_ci | [4.909053327864533e-06, 0.00035218516144705487] |
