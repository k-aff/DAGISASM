# ViT-Small (baseline) — test

Rows: 1848 | threshold: 0.5

## Overall detection performance

| metric | value |
| --- | --- |
| n | 1848 |
| accuracy | 0.9957 |
| balanced_accuracy | 0.9957 |
| precision | 0.9946 |
| recall | 0.9968 |
| f1 | 0.9957 |
| auc | 1.0000 |
| fpr | 0.0054 |
| fnr | 0.0032 |
| positive_rate | 0.5011 |

## Fairness audit — grouped by `skin_tone`

| group   | small_n   |   n |   accuracy |   balanced_accuracy |   precision |   recall |     f1 |    auc |    fpr |    fnr |   positive_rate |
|:--------|:----------|----:|-----------:|--------------------:|------------:|---------:|-------:|-------:|-------:|-------:|----------------:|
| Black   | False     | 616 |     0.9951 |              0.9951 |      0.9935 |   0.9968 | 0.9951 | 1.0000 | 0.0065 | 0.0032 |          0.5016 |
| Indian  | False     | 616 |     0.9951 |              0.9951 |      0.9904 |   1.0000 | 0.9952 | 1.0000 | 0.0097 | 0.0000 |          0.5049 |
| White   | False     | 616 |     0.9968 |              0.9968 |      1.0000 |   0.9935 | 0.9967 | 1.0000 | 0.0000 | 0.0065 |          0.4968 |

Gaps (groups with n >= 30; grey bars in plots are small-n groups):

| metric | value |
| --- | --- |
| fpr_gap | 0.0097 |
| tpr_gap | 0.0065 |
| auc_gap | 0.0000 |
| equalized_odds_gap | 0.0097 |
| demographic_parity_gap | 0.0081 |
| worst_group_auc | 1.0000 |
| n_groups | 3 |
| fpr_gap_ci | [0.0033536138809081182, 0.022104933110367885] |
| auc_gap_ci | [1.0180405702420181e-05, 0.00016595460941300474] |
