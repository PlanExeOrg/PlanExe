---
title: Investor-Grade Audit Pack Generator: Technical Documentation
date: 2026-02-11
status: proposal
author: PlanExe Team
---

# Investor-Grade Audit Pack Generator

**Author:** PlanExe Team  
**Date:** 2026-02-11  
**Status:** Proposal  
**Audience:** Developers, Investment Analysts  

---

## Overview
The **Investor-Grade Audit Pack Generator** automates the final mile of deal preparation. It compiles verified evidence, financial models, risk registers, and governance logs into a standardized, cryptographically-signed artifact (PDF/HTML) suitable for institutional due diligence.

Instead of a scattered collection of Google Docs and Excel sheets, the Audit Pack is a single source of truth.

## Key Features
1.  **Redaction Engine:** Automatically hides sensitive data (e.g., specific salaries, trade secrets) based on the recipient's clearance level.
2.  **Versioning:** "Snapshot" a plan at a specific point in time (e.g., "Series A Pack - v1.0").
3.  **Digital Signature:** Signs the pack to prove it hasn't been tampered with since generation.

## System Architecture

### 1. Data Aggregation
The generator pulls data from multiple internal systems:
-   **Plan Content:** The narrative text.
-   **Evidence Ledger:** The verification status of claims.
-   **Financial Engine:** The 5-year pro-forma model.
-   **Risk Register:** The Monte Carlo simulation results.

### 2. Redaction Layer
Before generation, the `RedactionEngine` filters the data.
-   **Tags:** Fields are tagged as `public`, `investor_only`, or `internal_confidential`.
-   **Roles:** Recipients are assigned roles (e.g., `analyst`, `partner`).
-   **Logic:** If `role.clearance < field.sensitivity`, the field is replaced with `[REDACTED]`.

### 3. Artifact Generation
-   **HTML:** Renders a responsive, interactive report using Jinja2 templates.
-   **PDF:** Converts the HTML to a high-fidelity PDF using `WeasyPrint` or similar.
-   **ZIP:** Bundles the PDF with raw data files (CSV, JSON) for analysts who want to run their own models.

---

## Output Schema (JSON)

The core data structure passed to the renderer:

```json
{
  "pack_id": "pack_2026_02_11_abc",
  "plan_id": "plan_123",
  "generated_at": "2026-02-11T16:00:00Z",
  "recipient": "Sequoia Capital",
  "clearance_level": "investor_only",
  "sections": [
    {
      "title": "Executive Summary",
      "content": "..."
    },
    {
      "title": "Financial Model",
      "data": {
        "revenue_y1": 1000000,
        "revenue_y2": 5000000,
        "ebitda_margin": "[REDACTED]" 
      }
    }
  ],
  "signature": "sha256:..."
}
```

---

## Redaction Logic

We use a simplistic but robust tagging system.

| Tag | Visibility | Example |
| :--- | :--- | :--- |
| `public` | Everyone | Product description, Market size |
| `investor_only` | NDA Signed | High-level financials, Roadmap |
| `internal_confidential` | Founders | Cap table details, Employee salaries |

**Algorithm:**
```python
def redact(data: dict, clearance: str) -> dict:
    if clearance == 'internal_confidential':
        return data  # Show everything
    
    sanitized = {}
    for key, value in data.items():
        tag = get_tag(key)
        if can_view(clearance, tag):
            sanitized[key] = value
        else:
            sanitized[key] = "[REDACTED]"
    return sanitized
```

---

## API Reference

### `POST /api/audit/generate`
Create a new audit pack.

**Request:**
```json
{
  "plan_id": "plan_123",
  "recipient_name": "Acme VC",
  "clearance_level": "investor_only",
  "format": "pdf" 
}
```

**Response:**
```json
{
  "pack_id": "pack_456",
  "download_url": "https://planexe.org/api/audit/download/pack_456.pdf",
  "expires_at": "2026-02-18T16:00:00Z"
}
```

### `GET /api/audit/verify/{signature}`
Verify the integrity of a pack.

**Response:**
```json
{
  "valid": true,
  "generated_at": "2026-02-11",
  "plan_hash": "sha256:..."
}
```

---

## Future Enhancements
1.  **Watermarking:** Dynamic watermarks with the recipient's email on every page of the PDF.
2.  **Data Room Integration:** Auto-upload to Carta, DocSend, or specialized VDRs.
3.  **Interactive Models:** Embed "Live Excel" components in the HTML version, allowing investors to tweak assumptions (within bounds).
