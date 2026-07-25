# TriageSense Evaluation Report

Model: `llama-3.1-8b-instant`, `temperature=0` (reproducible). Metrics computed with pandas (no scikit-learn).

## Headline numbers

| Set | Accuracy | What it means |
|---|---|---|
| Tuning set (71 examples) | **98.59% (70/71)** | Full labeled set, including hard and ambiguous cases. The prompt was refined by error analysis on this set, so it is fitted to it. |
| Held-out set (16 fresh examples) | **93.8% (15/16)** | Examples the prompt never saw. This is the honest generalization estimate. |
| Baseline (before prompt tuning) | 90.14% (64/71) | Same 71-example set, before the disambiguation rules were added. |

Method: the baseline was run first. Error analysis on the confusion matrix showed the classifier was over-predicting enquiry on multi-intent billing and on vague messages, so precedence and disambiguation rules were added to the prompt. Re-scoring gave 98.6% on the tuning set, up from 90.1%, and 93.8% on the held-out set, which shows the rules generalize rather than memorize. The labeled data is self-generated, so the next step is a larger held-out set of real, human-labeled requests.

## Tuning-set detail (71 examples)

**Accuracy:** 98.59% (70/71)

## Per-class Precision / Recall / F1

| class | precision | recall | f1 |
|---|---|---|---|
| Billing/Claim Dispute | 0.941 | 1.0 | 0.97 |
| Claim Status/Coverage Enquiry | 1.0 | 0.938 | 0.968 |
| Prior-Auth/Appointment Service Request | 1.0 | 1.0 | 1.0 |
| Complaint/Urgent Escalation | 1.0 | 1.0 | 1.0 |
| Unknown/Needs Human Review | 1.0 | 1.0 | 1.0 |

## Confusion Matrix (rows=true, columns=predicted)

| true \\ pred | Billing/Claim Dispute | Claim Status/Coverage Enquiry | Prior-Auth/Appointment Service Request | Complaint/Urgent Escalation | Unknown/Needs Human Review |
|---|---|---|---|---|---|
| Billing/Claim Dispute | 16 | 0 | 0 | 0 | 0 |
| Claim Status/Coverage Enquiry | 1 | 15 | 0 | 0 | 0 |
| Prior-Auth/Appointment Service Request | 0 | 0 | 16 | 0 | 0 |
| Complaint/Urgent Escalation | 0 | 0 | 0 | 14 | 0 |
| Unknown/Needs Human Review | 0 | 0 | 0 | 0 | 9 |
