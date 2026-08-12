# Xception (baseline) — test

Rows: 1848 | threshold: 0.5

## Overall detection performance

| metric | value |
| --- | --- |
| n | 1848 |
| accuracy | 0.9567 |
| balanced_accuracy | 0.9567 |
| precision | 0.9597 |
| recall | 0.9535 |
| f1 | 0.9566 |
| auc | 0.9932 |
| fpr | 0.0400 |
| fnr | 0.0465 |
| positive_rate | 0.4968 |

## Fairness audit — grouped by `skin_tone`

| group   | small_n   |   n |   accuracy |   balanced_accuracy |   precision |   recall |     f1 |    auc |    fpr |    fnr |   positive_rate |
|:--------|:----------|----:|-----------:|--------------------:|------------:|---------:|-------:|-------:|-------:|-------:|----------------:|
| Black   | False     | 616 |     0.9724 |              0.9724 |      0.9739 |   0.9708 | 0.9724 | 0.9978 | 0.0260 | 0.0292 |          0.4984 |
| Indian  | False     | 616 |     0.9562 |              0.9562 |      0.9607 |   0.9513 | 0.9560 | 0.9924 | 0.0390 | 0.0487 |          0.4951 |
| White   | False     | 616 |     0.9416 |              0.9416 |      0.9444 |   0.9383 | 0.9414 | 0.9888 | 0.0552 | 0.0617 |          0.4968 |

Gaps (groups with n >= 30; grey bars in plots are small-n groups):

| metric | value |
| --- | --- |
| fpr_gap | 0.0292 |
| tpr_gap | 0.0325 |
| auc_gap | 0.0090 |
| equalized_odds_gap | 0.0325 |
| demographic_parity_gap | 0.0032 |
| worst_group_auc | 0.9888 |
| n_groups | 3 |
| fpr_gap_ci | [0.008436009506334729, 0.05982161674985018] |
| auc_gap_ci | [0.0043517635769621905, 0.016320606449091567] |
