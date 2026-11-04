# Swin-Tiny skewed (baseline) - test

Rows: 2400 | threshold: 0.5

## Overall detection performance

| metric | value |
| --- | --- |
| n | 2400 |
| accuracy | 0.9958 |
| balanced_accuracy | 0.9958 |
| precision | 0.9950 |
| recall | 0.9967 |
| f1 | 0.9958 |
| auc | 0.9998 |
| fpr | 0.0050 |
| fnr | 0.0033 |
| positive_rate | 0.5008 |

## Fairness audit — grouped by `skin_tone`

| group   | small_n   |   n |   accuracy |   balanced_accuracy |   precision |   recall |     f1 |    auc |    fpr |    fnr |   positive_rate |
|:--------|:----------|----:|-----------:|--------------------:|------------:|---------:|-------:|-------:|-------:|-------:|----------------:|
| Black   | False     | 800 |     0.9962 |              0.9963 |      0.9950 |   0.9975 | 0.9963 | 0.9999 | 0.0050 | 0.0025 |          0.5012 |
| Indian  | False     | 800 |     0.9975 |              0.9975 |      0.9975 |   0.9975 | 0.9975 | 0.9998 | 0.0025 | 0.0025 |          0.5000 |
| White   | False     | 800 |     0.9938 |              0.9938 |      0.9925 |   0.9950 | 0.9938 | 0.9995 | 0.0075 | 0.0050 |          0.5012 |

Gaps (groups with n >= 30; grey bars in plots are small-n groups):

| metric | value |
| --- | --- |
| fpr_gap | 0.0050 |
| tpr_gap | 0.0025 |
| auc_gap | 0.0004 |
| equalized_odds_gap | 0.0050 |
| demographic_parity_gap | 0.0012 |
| worst_group_auc | 0.9995 |
| n_groups | 3 |
| fpr_gap_ci | [0.000582898565884157, 0.015975256330368774] |
| auc_gap_ci | [4.9624567810799286e-05, 0.0013659928543873193] |
