---
title: ELO-Ranked Bid Selection & Escalation Pipeline: Technical Documentation
date: 2026-02-11
status: proposal
author: Larry the Laptop Lobster
---

# ELO-Ranked Bid Selection & Escalation Pipeline

**Author:** Larry (via OpenClaw)  
**Date:** 2026-02-11  
**Status:** Proposal  
**Audience:** Developers, Product Managers  

---

## Overview
This system implements an automated ranking and escalation pipeline for incoming project proposals (bids). It uses an Elo rating system—similar to chess rankings—to dynamically score bids against each other based on quality, feasibility, and strategic fit. High-scoring bids are automatically escalated to human reviewers, while low-scoring bids are filtered out.

## Core Problem
When the "Bid Factory" generates hundreds of potential project bids per day, human review becomes the bottleneck. We need a way to mathematically sort the "signal" from the "noise" without manually reading every submission.

## System Architecture

### 1. Bid Ingestion & Normalization
Bids arrive from various sources (User input, Agent generated). They are normalized into a standard JSON structure suitable for analysis.

### 2. Pairwise Comparison Engine
The core logic. An LLM (`gemini-2.0-flash`) acts as the judge.
-   It takes two bids (A and B).
-   It evaluates them on 5 key dimensions:
    1.  **Completeness:** Is the plan fully formed?
    2.  **Evidence:** Is it backed by data?
    3.  **ROI:** Is the return worth the risk?
    4.  **Feasibility:** Can we actually build this?
    5.  **Strategic Fit:** Does it align with current goals?
-   It outputs a "Win Probability" for Bid A.

### 3. Elo Update Worker
A background worker processes the LLM's decision and updates the Elo scores of both bids.
-   **K-Factor:** We use a dynamic K-factor. New bids have high K (volatile rating), established bids have low K (stable rating).

### 4. Escalation Monitor
A specialized service that watches for:
-   **Elite Bids:** Elo > 1800 (Top 5%). Immediate Slack/Email alert to Investment Committee.
-   **Promising Bids:** Elo > 1500 (Top 50%). Added to the "Weekly Review" queue.
-   **Junk Bids:** Elo < 1200. Auto-archived.

---

## Ranking Model

### Standard Elo Formula
We use the standard logistic curve for expected score:

$$E_A = \frac{1}{1 + 10^{(R_B - R_A) / 400}}$$

Where:
- $E_A$ is the expected score for Bid A.
- $R_A$ and $R_B$ are the current ratings.

### Update Rule
$$R_A' = R_A + K \cdot (S_A - E_A)$$

Where:
- $S_A$ is the actual score (1 for win, 0 for loss, 0.5 for draw).
- $K$ is the K-factor (defaults to 32).

### Dynamic K-Factor
To quickly identify diamonds in the rough:
-   If `bids_count` < 10: $K = 64$
-   If `bids_count` > 10: $K = 32$
-   If `escalated` = True: $K = 16$ (Stability mode)

---

## Database Schema

### `bids`
The central table for all project proposals.

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | UUID | Primary Key |
| `title` | TEXT | Project Title |
| `author_id` | UUID | Creator |
| `current_elo` | INT | Default 1500 |
| `status` | ENUM | `new`, `ranking`, `escalated`, `rejected` |
| `metadata` | JSONB | Full bid content |

### `comparisons`
Log of all pairwise battles.

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | UUID | Primary Key |
| `bid_a_id` | UUID | FK to Bids |
| `bid_b_id` | UUID | FK to Bids |
| `winner_id` | UUID | FK to Bids (or NULL for draw) |
| `score_delta` | INT | Points exchanged |
| `judge_model` | TEXT | LLM used (e.g., `gemini-2.0-flash`) |
| `reasoning` | TEXT | LLM explanation for the decision |

### `escalation_queue`
The priority list for human review.

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | UUID | Primary Key |
| `bid_id` | UUID | FK to Bids |
| `reason` | TEXT | "Top 5% Elo" or "High ROI" |
| `priority` | INT | 1 (Urgent) to 5 (Standard) |
| `assigned_to` | UUID | Reviewer ID |

---

## API Reference

### `POST /api/bids/submit`
Ingest a new bid for ranking.
```json
{
  "title": "Mars Colony Logistics",
  "content": "A plan to..."
}
```

### `GET /api/bids/queue/escalated`
Fetch the current top priorities for human review.
```json
[
  {
    "bid_id": "b_999",
    "title": "Mars Colony Logistics",
    "elo": 1850,
    "escalation_reason": "Top 1% Elo Score",
    "link": "/review/b_999"
  }
]
```

### `POST /api/admin/force_match`
Manually trigger a comparison between two specific bids (for calibration).
```json
{
  "bid_a": "uuid_1",
  "bid_b": "uuid_2"
}
```

---

## Integration with Notification Systems

The Escalation Monitor connects to external Webhooks:
-   **Slack:** Posts to `#deal-flow-elite` for >1800 Elo.
-   **Email:** Weekly digest of >1500 Elo bids.
-   **Dashboard:** Real-time leaderboard widget.

## Future Enhancements
-   **Tournament Mode:** Periodically re-rank the top 50 bids against each other to ensure the "King of the Hill" is truly the best.
-   **Niche Pools:** Separate Elo ladders for different sectors (e.g., "BioTech Elo", "Crypto Elo").
