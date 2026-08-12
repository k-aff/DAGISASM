# Pilot fair — test split

Rows: 301 | threshold: 0.5

## Overall detection performance

| metric | value |
| --- | --- |
| n | 301 |
| accuracy | 0.9203 |
| balanced_accuracy | 0.9203 |
| precision | 0.9320 |
| recall | 0.9073 |
| f1 | 0.9195 |
| auc | 0.9781 |
| fpr | 0.0667 |
| fnr | 0.0927 |
| positive_rate | 0.4884 |

## Fairness audit — grouped by `skin_tone`

| group   | small_n   |   n |   accuracy |   balanced_accuracy |   precision |   recall |     f1 |    auc |    fpr |    fnr |   positive_rate |
|:--------|:----------|----:|-----------:|--------------------:|------------:|---------:|-------:|-------:|-------:|-------:|----------------:|
| dark    | False     | 101 |     0.9307 |              0.9310 |      0.9583 |   0.9020 | 0.9293 | 0.9702 | 0.0400 | 0.0980 |          0.4752 |
| light   | False     | 100 |     0.9000 |              0.9000 |      0.9000 |   0.9000 | 0.9000 | 0.9804 | 0.1000 | 0.1000 |          0.5000 |
| medium  | False     | 100 |     0.9300 |              0.9300 |      0.9388 |   0.9200 | 0.9293 | 0.9900 | 0.0600 | 0.0800 |          0.4900 |

Gaps (groups with n >= 25; grey bars in plots are small-n groups):

| metric | value |
| --- | --- |
| fpr_gap | 0.0600 |
| tpr_gap | 0.0200 |
| auc_gap | 0.0198 |
| equalized_odds_gap | 0.0600 |
| demographic_parity_gap | 0.0248 |
| worst_group_auc | 0.9702 |
| n_groups | 3 |
| fpr_gap_ci | [0.016166125541125585, 0.15460919110085228] |
| auc_gap_ci | [0.0043884803504497085, 0.0559506422704914] |

## Fairness audit — grouped by `dataset`

| group          | small_n   |   n |   accuracy |   balanced_accuracy |   precision |   recall |     f1 |    auc |    fpr |    fnr |   positive_rate |
|:---------------|:----------|----:|-----------:|--------------------:|------------:|---------:|-------:|-------:|-------:|-------:|----------------:|
| ai_vs_real     | True      |   2 |     0.5000 |              0.5000 |    nan      |   0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |          0.0000 |
| deepdetect2025 | False     | 299 |     0.9231 |              0.9231 |      0.9320 |   0.9133 | 0.9226 | 0.9804 | 0.0671 | 0.0867 |          0.4916 |

Gaps (groups with n >= 25; grey bars in plots are small-n groups):

| metric | value |
| --- | --- |
| fpr_gap | nan |
| tpr_gap | nan |
| auc_gap | nan |
| equalized_odds_gap | nan |
| demographic_parity_gap | nan |
| worst_group_auc | 0.9804 |
| n_groups | 1 |
| fpr_gap_ci | [nan, nan] |
| auc_gap_ci | [nan, nan] |
