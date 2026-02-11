---
title: Multi-Angle Topic Verification Engine: Technical Documentation
date: 2026-02-11
status: proposal
author: Larry the Laptop Lobster
---

# Multi-Angle Topic Verification Engine

**Author:** PlanExe Team  
**Date:** 2026-02-11  
**Status:** Proposal  
**Audience:** System Architects, Risk Managers  

---

## Overview
The **Multi-Angle Topic Verification Engine** ensures that critical plan topics are vetted from every relevant perspective (Technical, Legal, Financial, etc.) before a bid is approved. It prevents the common failure mode where a plan is technically sound but legally impossible (or vice versa).

It decomposes a plan into "Topics" and routes each topic to specialized "Lens Agents" for independent verification.

## Core Problem
Verification is often single-threaded. A technical reviewer focuses on the engineering, missing the regulatory blocker. A financial reviewer checks the spreadsheet, missing the technical impossibility.

## System Architecture

### 1. Topic Extractor
Uses NLP (LLM) to parse the plan into discrete assertions or "Topics".
*   *Example:* "We will use drone swarms for delivery." (Topic: Drone Operations)

### 2. Lens Routing
Determines which "Lenses" apply to a given topic.
*   **Legal Lens:** FDA regulations on drones? (Yes)
*   **Technical Lens:** Battery life sufficient? (Yes)
*   **Financial Lens:** Cost per mile vs truck? (Yes)
*   **Ethical Lens:** Privacy concerns? (Yes)

### 3. Lens Agents
Independent LLM instances (promoted with specific personas and knowledge bases) that evaluate the topic.
*   **Input:** The Topic + Evidence.
*   **Context:** Lens-specific regulations/standards.
*   **Output:** `ConfidenceScore` (0-1) + `ConcernList`.

### 4. Conflict Resolution (The Adjudicator)
If Lens A says "Go" and Lens B says "Stop", the Adjudicator (a meta-agent or human) reviews the conflict.

---

## Data Schema

### `verification_matrix`
Storage for the multi-angle results.

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | UUID | Primary Key |
| `topic_id` | UUID | FK to Topics |
| `lens_id` | ENUM | `legal`, `tech`, `finance`, `ops`, `market` |
| `status` | ENUM | `verified`, `flagged`, `rejected` |
| `confidence` | DECIMAL | 0.0 to 1.0 |
| `reasoning` | TEXT | Argument for the score |

---

## Conflict Resolution Logic

How we handle disagreement between lenses.

**Scenario:** "Crypto Payments"
*   **Tech Lens:** 0.95 (Easy to implement)
*   **Legal Lens:** 0.10 (Banned in target jurisdiction)

**Algorithm:**
```python
def adjudicate(topic, results):
    # Weighted average doesn't work for "Stop" signals.
    # Any "Critical" lens with < 0.3 score triggers a hard block.
    
    technical_score = results['tech'].score
    legal_score = results['legal'].score
    
    if legal_score < 0.3 and results['legal'].is_blocker:
        return {
            "verdict": "REJECTED",
            "reason": f"Legal blocker: {results['legal'].reason}"
        }
    
    # If standard disagreement, escalate to human
    if abs(technical_score - legal_score) > 0.5:
        return {
            "verdict": "ESCALATE",
            "reason": "High variance between lenses"
        }
        
    return {"verdict": "APPROVED"}
```

---

## API Reference

### `POST /api/verify/topic`
Submit a specific topic for multi-angle review.

**Request:**
```json
{
  "plan_id": "plan_123",
  "topic_content": "Use of autonomous heavy machinery",
  "lenses": ["legal", "safety", "union_labor"]
}
```

**Response:**
```json
{
  "verification_id": "ver_999",
  "results": {
    "legal": {"status": "pass", "confidence": 0.8},
    "safety": {"status": "cond_pass", "confidence": 0.6, "warning": "Requires Geo-fencing"},
    "union_labor": {"status": "fail", "confidence": 0.2, "error": "Violates CBA"}
  },
  "overall_status": "rejected"
}
```

---

## User Interface

### "The Prism View"
A radar chart showing the confidence score for a topic across all axes.
*   **Full Shape:** A large polygon means high confidence across the board.
*   **Spiked Shape:** Indicates imbalance (e.g., strong Tech, weak Legal).

## Future Enhancements
1.  **Lens Marketplace:** Allow third-party experts to plug in as a "Verifier Lens" (e.g., a "Cybersecurity Lens" provided by a security firm).
2.  **Historical Calibration:** "The Legal Lens is too pessimistic; adjust its weight down by 10%."
