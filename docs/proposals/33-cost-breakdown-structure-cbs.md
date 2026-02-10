# 33) Cost Breakdown Structure (CBS) Generation

## Pitch
Generate a **Cost Breakdown Structure** alongside the WBS so users can see where money goes (labor, materials, vendors, permits, contingency) and tie costs back to tasks.

## Why
A plan without a CBS is hard to fund, bid, or govern. CBS improves transparency and enables tracking actuals vs plan.

## Proposal
### 1) CBS tree aligned to WBS
- Map each WBS node to cost categories:
  - labor (role-based)
  - materials
  - software/services
  - subcontractors
  - travel
  - legal/compliance
  - contingency

### 2) Cost assumptions registry
- Every cost line references assumptions (rates, quantities, quotes, inflation)

### 3) Outputs
- CBS table (hierarchical)
- CBS ↔ WBS mapping table
- “Top cost drivers” section

## Data model additions
- `cost_items` (wbs_id, cbs_path, amount_cents, currency, assumption_id)
- `cost_assumptions` (id, description, source, confidence)

## Success metrics
- % WBS nodes mapped to CBS categories
- User ability to export CBS to spreadsheet/accounting tools
