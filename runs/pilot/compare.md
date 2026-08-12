# Comparison: baseline vs fair

## Overall detection

| metric | baseline | fair | delta |
| --- | --- | --- | --- |
| accuracy | 0.9402 | 0.9203 | -0.0199 |
| precision | 0.9290 | 0.9320 | +0.0029 |
| recall | 0.9536 | 0.9073 | -0.0464 |
| auc | 0.9799 | 0.9781 | -0.0018 |
| fpr | 0.0733 | 0.0667 | -0.0067 |

## Fairness gaps (grouped by `skin_tone`)

| gap | baseline | fair | delta (neg = fairer) |
| --- | --- | --- | --- |
| fpr_gap | 0.0200 | 0.0600 | +0.0400 |
| auc_gap | 0.0138 | 0.0198 | +0.0060 |
| tpr_gap | 0.0980 | 0.0200 | -0.0780 |
| equalized_odds_gap | 0.0980 | 0.0600 | -0.0380 |
| worst_group_auc | 0.9718 | 0.9702 | -0.0016 |

baseline FPR-gap 95% CI: [0.0139, 0.1549]

fair FPR-gap 95% CI: [0.0162, 0.1546]