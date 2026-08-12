# Pilot baseline — test split

Rows: 301 | threshold: 0.5

## Overall detection performance

| metric | value |
| --- | --- |
| n | 301 |
| accuracy | 0.9402 |
| balanced_accuracy | 0.9402 |
| precision | 0.9290 |
| recall | 0.9536 |
| f1 | 0.9412 |
| auc | 0.9799 |
| fpr | 0.0733 |
| fnr | 0.0464 |
| positive_rate | 0.5150 |

## Fairness audit — grouped by `skin_tone`

| group   | small_n   |   n |   accuracy |   balanced_accuracy |   precision |   recall |     f1 |    auc |    fpr |    fnr |   positive_rate |
|:--------|:----------|----:|-----------:|--------------------:|------------:|---------:|-------:|-------:|-------:|-------:|----------------:|
| dark    | False     | 101 |     0.9208 |              0.9210 |      0.9388 |   0.9020 | 0.9200 | 0.9718 | 0.0600 | 0.0980 |          0.4851 |
| light   | False     | 100 |     0.9600 |              0.9600 |      0.9259 |   1.0000 | 0.9615 | 0.9844 | 0.0800 | 0.0000 |          0.5400 |
| medium  | False     | 100 |     0.9400 |              0.9400 |      0.9231 |   0.9600 | 0.9412 | 0.9856 | 0.0800 | 0.0400 |          0.5200 |

Gaps (groups with n >= 25; grey bars in plots are small-n groups):

| metric | value |
| --- | --- |
| fpr_gap | 0.0200 |
| tpr_gap | 0.0980 |
| auc_gap | 0.0138 |
| equalized_odds_gap | 0.0980 |
| demographic_parity_gap | 0.0549 |
| worst_group_auc | 0.9718 |
| n_groups | 3 |
| fpr_gap_ci | [0.01394644290136094, 0.15493872549019602] |
| auc_gap_ci | [0.006176486530620524, 0.052447071889903815] |

## Fairness audit — grouped by `dataset`

| group          | small_n   |   n |   accuracy |   balanced_accuracy |   precision |   recall |     f1 |    auc |    fpr |    fnr |   positive_rate |
|:---------------|:----------|----:|-----------:|--------------------:|------------:|---------:|-------:|-------:|-------:|-------:|----------------:|
| ai_vs_real     | True      |   2 |     0.5000 |              0.5000 |    nan      |   0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |          0.0000 |
| deepdetect2025 | False     | 299 |     0.9431 |              0.9431 |      0.9290 |   0.9600 | 0.9443 | 0.9824 | 0.0738 | 0.0400 |          0.5184 |

Gaps (groups with n >= 25; grey bars in plots are small-n groups):

| metric | value |
| --- | --- |
| fpr_gap | nan |
| tpr_gap | nan |
| auc_gap | nan |
| equalized_odds_gap | nan |
| demographic_parity_gap | nan |
| worst_group_auc | 0.9824 |
| n_groups | 1 |
| fpr_gap_ci | [nan, nan] |
| auc_gap_ci | [nan, nan] |
