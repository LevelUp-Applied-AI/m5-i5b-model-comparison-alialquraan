# Tree vs. Linear Disagreement Analysis

## Sample Details

- **Test-set index:** 4060
- **True label:** 0
- **RF predicted P(churn=1):** 0.5998
- **LR predicted P(churn=1):** 0.1700
- **Probability difference:** 0.4299

## Feature Values

- **tenure:** 36.0
- **monthly_charges:** 20.0
- **total_charges:** 1077.33
- **num_support_calls:** 2.0
- **senior_citizen:** 0.0
- **has_partner:** 0.0
- **has_dependents:** 0.0
- **contract_months:** 1.0

## Structural Explanation

<!-- Write 2-3 sentences explaining WHY these models disagree on this
     sample. Point to a specific feature interaction, non-monotonic
     relationship, or threshold effect the tree captured that the
     linear model could not. -->

The models disagree on this sample because the Random Forest captures a non-linear interaction between the short contract duration (contract_months: 1.0) and the moderate tenure (36.0). While the linear model (LR) sees a relatively low monthly charge and a 3-year tenure as signs of stability, the tree-based model (RF) recognizes that being on a month-to-month contract (1.0) is a high-risk threshold for churn regardless of tenure, a non-monotonic relationship that a simple linear coefficient cannot fully express.