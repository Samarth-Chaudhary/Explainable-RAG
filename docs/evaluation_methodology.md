# Evaluation Methodology & Metrics

## 1. Pillar A: Predictive & Mathematical Fidelity
- **Prediction Invariance**: Given fixed model parameters and applicant input, output probabilities must be identical across invocations ($diff < 10^{-7}$).
- **Additivity Residual**:
  $$\text{Residual} = \left| base\_value + \sum_{i=1}^M \phi_i - raw\_margin \right|$$
  - Requirement: $\text{Residual} \le 10^{-4}$ (Status: `valid`).
  - If residual exceeds tolerance, status transitions to `degraded` or `invalid`, blocking narrative synthesis.
- **Cross-Space Integrity**: Validates that raw margin log-odds contributions $\phi_i$ are never directly compared against probability thresholds without appropriate link transformation (e.g., sigmoid).

## 2. Pillar B: Counterfactual Feasibility & Metrics
All metrics report explicit denominators:
- **Validity**: Proportion of returned candidates whose re-scored probability falls below the target threshold:
  $$\text{Validity} = \frac{\sum \mathbb{I}(\text{re\_scored\_prob} < \text{threshold})}{\text{total candidates evaluated}}$$
- **Actionability**: Proportion of candidates that satisfy the target threshold **AND** all business, immutability, range, and derivation constraints:
  $$\text{Actionability} = \frac{\sum \mathbb{I}(\text{valid} \land \text{all constraints passed})}{\text{total candidates evaluated}}$$
- **Generation Failure Rate**:
  $$\text{Failure Rate} = 1.0 - \text{Validity}$$
- **Proximity**: Mean continuous L1 distance normalized by feature range:
  $$\text{Proximity} = \frac{1}{|A|} \sum_{j \in A} \frac{|x'_j - x_j|}{\max(X_j) - \min(X_j)}$$
- **Diversity**: Mean pairwise L1 distance among generated candidates.
- **Out-of-Distribution (OOD) Rate**: Percentage of counterfactual values falling outside the empirical domain bounds observed in training data.
- **Target Miss Magnitude**: Mean shortfall for non-satisfying candidates:
  $$\text{Miss Magnitude} = \max(0.0, \text{re\_scored\_prob} - \text{threshold})$$
