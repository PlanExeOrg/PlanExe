---
title: Assumption Drift Monitor: Technical Documentation
date: 2026-02-11
status: proposal
author: PlanExe Team
---

# Assumption Drift Monitor

**Author:** PlanExe Team  
**Date:** 2026-02-11  
**Status:** Proposal  
**Audience:** Engineers, Project Managers  

---

## Overview
The **Assumption Drift Monitor** is a real-time surveillance system for project plans. It continuously compares the foundational assumptions of a plan (e.g., "Steel costs $800/ton") against live data streams (commodity APIs, labor market reports, competitor pricing).

When reality deviates from the plan beyond a specific tolerance threshold, the system triggers alerts and suggests re-planning actions.

## Core Problem
Plans are static snapshots of a dynamic world. A plan created in January is often obsolete by March because key variables have shifted. Humans rarely manually check these variables until a crisis hits.

## System Architecture

### 1. Assumption Registry
A structured database of every variable the plan depends on.
-   **Static Assumptions:** "We need 5 engineers." (Verified internally)
-   **Dynamic Assumptions:** "EUR/USD exchange rate is 1.10." (Verified externally)

### 2. Data Ingestion Service
Connectors to external APIs:
-   **Financial:** Bloomberg, Yahoo Finance (FX, Rates)
-   **Commodities:** Metal/Energy spot prices.
-   **Macro:** Inflation rates, GDP growth.
-   **Custom:** Internal BI tools, Jira velocity.

### 3. Drift Detection Engine
Runs hourly/daily jobs to compare `Lesson learned` vs `Current Reality`.
-   **Thresholds:** defined per assumption (e.g., +/- 5%).
-   **Composite Drift:** aggregated impact of multiple small drifts.

### 4. Alerting & Governance
-   **Green:** Within tolerance.
-   **Yellow:** Warning (Approaching limit).
-   **Red:** Breach (Requires mandatory re-plan or waiver).

---

## Database Schema

### `assumptions`
The registry of monitored variables.

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | UUID | Primary Key |
| `plan_id` | UUID | FK to Plans |
| `variable_name` | TEXT | e.g., "Steel Price" |
| `baseline_value` | DECIMAL | Value at plan approval (e.g., 800.00) |
| `unit` | TEXT | e.g., "USD/Ton" |
| `data_source_id` | UUID | FK to Data Sources |
| `tolerance_pct` | DECIMAL | +/- % allowed before alert (e.g., 0.05) |

### `data_sources`
Configuration for external feeds.

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | UUID | Primary Key |
| `name` | TEXT | e.g., "London Metal Exchange API" |
| `adapter_type` | ENUM | `json_api`, `sql_query`, `manual_input` |
| `config` | JSONB | Auth tokens, endpoints, query paths |

### `observations`
Time-series log of actual values.

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | UUID | Primary Key |
| `assumption_id` | UUID | FK |
| `observed_at` | TIMESTAMPTZ | When the data was captured |
| `value` | DECIMAL | The live value (e.g., 850.00) |
| `drift_pct` | DECIMAL | `(value - baseline) / baseline` |

---

## Alerting Logic (DSL)

We use a simple domain-specific language for defining complex alerts.

```yaml
# rules/steel_price_breach.yml
rule_name: "Steel Price Surge"
condition:
  assumption: "steel_price"
  operator: ">"
  threshold: 10%
  duration: "3 days" # Must persist for 3 days to avoid noise
action:
  level: "critical"
  notify: ["project_manager", "procurement_lead"]
  trigger_workflow: "recalculate_budget"
```

**Composite Drift Example:**
If `steel_price` is up 4% (Green) AND `labor_rate` is up 4% (Green), the combined effect might be > 5%.
```python
def check_composite_drift(plan_id):
    total_impact = 0
    for assumption in get_assumptions(plan_id):
        impact = assumption.drift_pct * assumption.sensitivity_factor
        total_impact += impact
    
    if total_impact > 0.05:
        trigger_alert("Combined budget impact exceeds 5%")
```

---

## API Reference

### `GET /api/drift/status/{plan_id}`
Dashboard view of current health.

**Response:**
```json
{
  "plan_id": "uuid...",
  "status": "warning",
  "drift_score": 0.12, # 12% drift aggregated
  "breaches": [
    {
      "variable": "fuel_cost",
      "baseline": 3.50,
      "current": 4.10,
      "drift": "+17%",
      "status": "critical"
    }
  ]
}
```

### `POST /api/drift/simulate`
"What-if" analysis for manual scenario testing.

**Request:**
```json
{
  "assumptions": [
    {"variable": "exchange_rate", "value": 1.20}
  ]
}
```

---

## User Interface

### "The Watchtower"
A dashboard widget showing a "Health Bar" for the plan.
-   **Drift Chart:** Sparklines showing the trend of key variables over time.
-   **Impact Analysis:** "If this trend continues, you will be over budget by Nov 15th."

### Re-Plan Trigger
When a `Red` alert fires, a "Re-Plan" button appears.
1.  Clones the current plan.
2.  Updates the baselines to current actuals.
3.  Re-runs the critical path and budget estimation.
4.  Presents the "Delta" for approval.

## Future Enhancements
1.  **Predictive Drift:** Use ML time-series forecasting (Prophet/Arima) to alert *before* the breach happens.
2.  **News Sentiment:** Ingest news articles to detect "Qualitative Drift" (e.g., "Political instability in supplier region").
