# Cross-generator (imdb reals + sd_wiki fakes) — AUC view

Rows: 20676 | threshold: 0.5

## Overall detection performance

| metric | value |
| --- | --- |
| n | 20676 |
| accuracy | 0.5428 |
| balanced_accuracy | 0.5869 |
| precision | 0.7763 |
| recall | 0.4682 |
| f1 | 0.5841 |
| auc | 0.6189 |
| fpr | 0.2944 |
| fnr | 0.5318 |
| positive_rate | 0.4135 |

## Fairness audit — grouped by `skin_tone`

| group   | small_n   |     n |   accuracy |   balanced_accuracy |   precision |   recall |     f1 |    auc |    fpr |    fnr |   positive_rate |
|:--------|:----------|------:|-----------:|--------------------:|------------:|---------:|-------:|-------:|-------:|-------:|----------------:|
| Black   | False     |  3166 |     0.5038 |              0.5192 |      0.5708 |   0.3253 | 0.4145 | 0.5549 | 0.2869 | 0.6747 |          0.3076 |
| Indian  | False     |  1088 |     0.5947 |              0.6122 |      0.8801 |   0.5847 | 0.7026 | 0.6448 | 0.3604 | 0.4153 |          0.5441 |
| White   | False     | 16422 |     0.5469 |              0.5932 |      0.7961 |   0.4803 | 0.5991 | 0.6238 | 0.2939 | 0.5197 |          0.4253 |

Gaps (groups with n >= 30; grey bars in plots are small-n groups):

| metric | value |
| --- | --- |
| fpr_gap | 0.0735 |
| tpr_gap | 0.2594 |
| auc_gap | 0.0899 |
| equalized_odds_gap | 0.2594 |
| demographic_parity_gap | 0.2365 |
| worst_group_auc | 0.5549 |
| n_groups | 3 |
| fpr_gap_ci | [0.016162310851529586, 0.14869680547707415] |
| auc_gap_ci | [0.054292008813532994, 0.136721958364728] |
