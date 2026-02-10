# 35) Finance Analysis via Bottom-Up Estimation + Reconciliation

## Pitch
Produce a **bottom-up** estimate by summing task-level costs (labor hours × rates, materials, vendors) and reconcile it against the top-down estimate so both align within a tolerance.

## Why
Bottom-up is needed for bids and execution, but it can be wrong if tasks are missing or rates are unrealistic. Reconciliation catches gaps.

## Proposal
### 1) Bottom-up build
- For each WBS task:
  - assign roles + hour estimates
  - apply rate card
  - add materials/vendor line items
- Sum to phase and total.

### 2) Reconciliation step
- Compare top-down P50 vs bottom-up total.
- If delta > threshold (e.g., 15–25%), generate:
  - missing scope hypotheses
  - rate corrections
  - contingency adjustments
  - iteration plan to converge

### 3) Outputs
- Bottom-up table by WBS
- Variance report: top-down vs bottom-up
- “Convergence actions” checklist

## Success metrics
- % plans where top-down and bottom-up converge within tolerance
- Reduced budget surprises during execution
