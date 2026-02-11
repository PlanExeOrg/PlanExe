---
title: Execution Readiness Scoring: Technical Documentation
date: 2026-02-11
status: proposal
author: PlanExe Team
---

# Execution Readiness Scoring

**Author:** PlanExe Team  
**Date:** 2026-02-11  
**Status:** Proposal  
**Audience:** Program Managers, VCs  

---

## Overview
The **Execution Readiness Scoring** system provides a quantitative "Credit Score" for a plan's executability. It prevents the common failure mode where a plan *looks* good on paper but lacks the critical prerequisites (resources, permits, contracts) to actually start.

It acts as a "Gatekeeper" service: a plan cannot move to the "Execution Phase" until its readiness score exceeds a threshold (e.g., 80/100).

## Core Problem
Plans are often approved based on *optimism* rather than *evidence*. Teams commit to dates without having the underlying resources or regulatory approvals secured, leading to immediate delays.

## System Architecture

### 1. Scoring Engine
The core service. It aggregates data from multiple "Validator Agents":
-   **Evidence Validator:** Checks if all critical claims are backed by Level 3 evidence.
-   **Resource Validator:** Cross-checks "Roles Needed" vs "Staff Available".
-   **Dependency Validator:** Ensures upstream constraints (e.g., "Seed funding secured") are met.
-   **Risk Validator:** Verifies that all "Critical" risks have mitigation plans.

### 2. The Scorecard (0-100)
A weighted sum of the validator outputs.

$$Score = (0.3 \times Evidence) + (0.3 \times Capacity) + (0.2 \times Risk) + (0.1 \times Dependencies) + (0.1 \times Financials)$$

### 3. Gap Analysis
The system doesn't just say "No". It generates a `GapReport` listing exactly *what* is missing (e.g., "Missing permit from EPA").

---

## Scoring Dimensions (The 100 Points)

| Dimension | Weight | Criteria |
| :--- | :--- | :--- |
| **Evidence Coverage** | 30 pts | % of Level 3 verified claims. (100% = 30 pts) |
| **Resource Capacity** | 30 pts | (Available / Required) FTEs. (1.0 = 30 pts) |
| **Risk Mitigation** | 20 pts | % of High/Critical risks with active mitigation plans. |
| **Dependency Maturity** | 10 pts | % of long-lead items (e.g., chips) ordered/secured. |
| **Financial Viability** | 10 pts | Cashway runway > 12 months. (True = 10 pts) |

---

## Output Schema (JSON)

The API response for a readiness check:

```json
{
  "plan_id": "plan_123",
  "overall_score": 67, 
  "status": "conditional", # "ready" (>80), "conditional" (60-79), "not_ready" (<60)
  "breakdown": {
    "evidence": 22, # out of 30
    "capacity": 15, # out of 30 (Major Gap)
    "risk": 18,     # out of 20
    "deps": 8,      # out of 10
    "finance": 4    # out of 10
  },
  "gaps": [
    {
      "severity": "blocker",
      "category": "capacity",
      "description": "Missing Lead Engineer (Role ID: role_55)",
      "action": "Open requisition or contract agency"
    },
    {
      "severity": "warning",
      "category": "evidence",
      "description": "Market sizing is based on 2023 report (stale)",
      "action": "Update source to 2025 data"
    }
  ]
}
```

---

## User Interface

### "The Launch Button"
A prominent button on the plan dashboard.
-   **Disabled (Gray):** Score < 60. Tooltip lists major blockers.
-   **Warning (Yellow):** Score 60-79. Pop-up warns: "Are you sure? Financials are weak."
-   **Enabled (Green):** Score > 80. "All systems go."

## Integration Logic
The Readiness Score is connected to other PlanExe modules:
-   **Evidence Ledger:** Feeds the "Evidence Coverage" score.
-   **Audit Pack:** The final score is printed on the cover page of the Investor Pack.
-   **Elo Ranking:** Uses readiness as a "Feasibility" signal for ranking plans.

## Future Enhancements
1.  **AI Gap Filling:** "You are missing a GDPR policy. Here is a draft based on similar plans."
2.  **Sector Benchmarks:** Compare readiness against industry averages (e.g., "Your hiring plan is 20% slower than peer startups").
