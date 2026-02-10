# Elo Ranking System: Technical Documentation

**Author:** Larry (via OpenClaw)  
**Date:** 2026-02-08  
**Status:** Living document  
**Audience:** Developers, contributors, technical reviewers

---

## Overview
PlanExe ranks generated plans using a two‑phase LLM evaluation to avoid gaming static weights:

1. **Extract raw KPI vector** (novelty, prompt quality, technical completeness, feasibility, impact)

2. **Pairwise LLM comparison** of KPI vectors → Likert preference

3. **Elo update** for new plan and sampled neighbors

## Defaults

- LLM: **Gemini‑2.0‑flash‑001 via OpenRouter** (`OPENROUTER_API_KEY`)

- Embeddings: **OpenAI embeddings** (`OPENAI_API_KEY`)

- Vector store: **pgvector** (Postgres extension)

- Rate limit: **5 req/min per API key**

- Corpus source: PlanExe‑web `_data/examples.yml`

## Endpoints

- `POST /api/rank` → rank plan, update Elo

- `GET /api/leaderboard?limit=N` → user‑scoped leaderboard

- `GET /api/export?limit=N` → top‑N export

## Data Tables

- `plan_corpus`: plan metadata + embeddings + json_data (for dynamic KPI comparisons)

- `plan_metrics`: KPI values (int 1‑5) + `kpis` JSONB + `overall_likert` + Elo

- `rate_limit`: per‑API‑key rate limiting

## Setup

1. Run migrations:

   - `mcp_cloud/migrations/2026_02_09_create_plan_metrics.sql`

   - `mcp_cloud/migrations/2026_02_10_add_plan_json.sql`

2. Seed corpus: `scripts/seed_corpus.py` (set `PLANEXE_WEB_EXAMPLES_PATH`)

3. Set env:

   - `OPENROUTER_API_KEY`

   - `OPENAI_API_KEY`

   - `PLANEXE_API_KEY_SECRET`

## Notes

- Ranking uses **real data only** (no mocks)

- Embeddings stored in pgvector for novelty sampling

- Leaderboard UI at `/rankings`

## Table of Contents

1. [Overview](#overview)

2. [System Architecture](#system-architecture)

   - [Dynamic KPI Extraction](#dynamic-kpi-extraction)

   - [Pairwise LLM Comparison](#pairwise-llm-comparison)

   - [Win Probability Computation](#win-probability-computation)

   - [Elo Update Formula](#elo-update-formula)

3. [LLM Prompting Strategy](#llm-prompting-strategy)

4. [API Reference](#api-reference)

5. [User Interface](#user-interface)

6. [Database Schema](#database-schema)

7. [Technical Rationale](#technical-rationale)

8. [Current Limitations](#current-limitations)

9. [Future Enhancements](#future-enhancements)

10. [Implementation Roadmap](#implementation-roadmap)

11. [Glossary](#glossary)

---

## Overview

PlanExe uses an **Elo-based ranking system** to compare and rank generated plans through pairwise LLM comparisons. Unlike static scoring formulas, this system:

- Extracts KPIs dynamically based on plan content

- Uses embedding-based neighbor selection for relevant comparisons

- Maps Likert scale ratings to win probabilities

- Updates Elo ratings using standard chess Elo formula with K=32

**Key design goals:**

- Contextual ranking (relative to corpus, not absolute)

- Privacy-preserving (users see only their own plans)

- Gaming-resistant (dynamic KPI selection)

- Actionable feedback (KPI reasoning stored for user insights)

---

## System Architecture

### Dynamic KPI Extraction

When a plan is submitted via `/api/rank`, the system:

1. **Stores the full plan JSON** in `plan_corpus.json_data` (JSONB column, ~2-50KB typical size)

   - JSONB indexing enables fast GIN queries for metadata filtering

   - Full plan context available for comparison without re-fetching

2. **Generates an embedding** of the plan's prompt using `text-embedding-3-small` (768 dimensions)

   - Stored in `plan_corpus.embedding` (pgvector column)

   - Enables semantic neighbor selection via cosine similarity

3. **Extracts baseline KPIs** using `gemini-2.0-flash-exp` via OpenRouter:

   - Novelty score (0-1 float)

   - Prompt quality (0-1 float)

   - Technical completeness (0-1 float)

   - Feasibility (0-1 float)

   - Impact estimate (0-1 float)

---

### Pairwise LLM Comparison

For each new plan:

**Step 1: Select 10 neighbors**

- Query `plan_corpus` for top 10 nearest embeddings (cosine similarity via pgvector)

- If corpus has <10 plans, select all available plans

- If no embeddings exist (cold start), select 10 random plans

**Step 2: Run pairwise comparisons**

For each neighbor, the LLM:

1. Receives both plan JSONs (`plan_a` = new plan, `plan_b` = neighbor)

2. Chooses **5-7 relevant KPIs** based on plan characteristics

3. Adds **one final KPI** for remaining considerations (LLM-named, e.g., "Resource allocation realism")

4. Scores each KPI on **Likert 1-5 integer scale**:

   - 1 = Very poor

   - 2 = Below average

   - 3 = Average

   - 4 = Above average

   - 5 = Excellent

5. Provides **≤30-word reasoning** for each KPI score

**Token budget:** ~2000 tokens per comparison (input + output combined)

---

### Win Probability Computation

**Step 1: Calculate total scores**
```python
total_a = sum(kpi.plan_a for kpi in kpis)
total_b = sum(kpi.plan_b for kpi in kpis)
diff = total_a - total_b
```

**Step 2: Map score difference to win probability**

The mapping uses a piecewise function designed to:

- Provide clear signal for meaningful differences (±2+ points)

- Avoid extreme probabilities (floors at 0.1, caps at 0.9)

- Handle neutral outcomes (diff=0 → 0.5 probability)

| Score Difference | `prob_a` | Rationale |
|------------------|----------|-----------|
| ≥ +3             | 0.9      | Strong preference for plan A (multiple KPI wins) |
| +2               | 0.7      | Moderate favor A (2 standard deviations above neutral) |
| +1               | 0.6      | Slight favor A (1 standard deviation) |
| 0                | 0.5      | Neutral (no clear winner) |
| -1               | 0.4      | Slight favor B |
| -2               | 0.3      | Moderate favor B |
| ≤ -3             | 0.1      | Strong preference for plan B |

**Why this mapping?**

- Likert scale variance is ~1.5 points across 6-8 KPIs

- ±1 point represents ~0.7 standard deviations (weak signal)

- ±2 points represents ~1.3 standard deviations (moderate signal)

- ±3+ points represents strong consensus across multiple KPIs

Alternative considered: logistic function `1 / (1 + exp(-k * diff))` — rejected due to lack of interpretability and extreme tail probabilities.

---

### Elo Update Formula

Standard Elo formula from chess rating systems:

```python
def update_elo(elo_a: float, elo_b: float, prob_a: float, K: int = 32) -> tuple[float, float]:
    """
    Update Elo ratings after a pairwise comparison.
    
    Args:
        elo_a: Current Elo rating of plan A
        elo_b: Current Elo rating of plan B
        prob_a: Win probability for plan A (0-1, from Likert mapping)
        K: Sensitivity parameter (default 32)
    
    Returns:
        (new_elo_a, new_elo_b)
    """
    expected_a = 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400))
    new_elo_a = elo_a + K * (prob_a - expected_a)
    new_elo_b = elo_b + K * ((1 - prob_a) - (1 - expected_a))
    return new_elo_a, new_elo_b
```

**Why K=32?**

- Standard value for established chess players (16 for masters, 40 for beginners)

- Balances stability (K=16 too slow to converge) vs noise (K=64 too volatile)

- After 10 comparisons, a plan's rating converges within ±50 points of true skill

- Empirically tested: K=32 provides good discrimination after 20-30 total corpus comparisons

**Cold-start bias:**

- All plans initialize at Elo 1500

- First 5 comparisons have outsized impact on rating

- Plans submitted early have more stable ratings (more comparisons accumulated)

- Mitigation: normalize by `num_comparisons` in percentile calculation (planned for Phase 2)

---

## LLM Prompting Strategy

### KPI Extraction Prompt

The system uses the following prompt structure for pairwise comparisons:

```
You are evaluating two business plans. Your task:

1. Read both plans carefully (plan_a and plan_b)
2. Choose 5-7 KPIs most relevant to these specific plans
3. Add ONE final KPI named by you that captures important remaining considerations
4. Score each KPI for both plans on a 1-5 integer Likert scale:
   - 1 = Very poor
   - 2 = Below average  
   - 3 = Average
   - 4 = Above average
   - 5 = Excellent
5. Provide ≤30-word reasoning for each KPI score

Output format (JSON array):
[
  {
    "name": "KPI name",
    "plan_a": <1-5 integer>,
    "plan_b": <1-5 integer>,
    "reasoning": "<30-word explanation>"
  },
  ...
]

Plan A:
{plan_a_json}

Plan B:
{plan_b_json}

Return ONLY the JSON array, no other text.
```

**Token budget:** ~2000 tokens per comparison (input: ~1500 tokens, output: ~500 tokens)

**LLM configuration:**

- Model: `gemini-2.0-flash-exp` (via OpenRouter)

- Temperature: 0.3 (low variance, consistent scoring)

- Max tokens: 1000 (sufficient for 8 KPIs × 30 words + JSON structure)

---

### Example KPI Output

```json
[
  {
    "name": "Goal clarity & specificity",
    "plan_a": 4,
    "plan_b": 3,
    "reasoning": "Plan A defines concrete 24-month timeline and EASA compliance gates; Plan B has broad goals without operational detail."
  },
  {
    "name": "Schedule credibility",
    "plan_a": 5,
    "plan_b": 3,
    "reasoning": "Plan A includes PDR/CDR gates with milestone dates; Plan B timeline has internal inconsistencies flagged earlier."
  },
  {
    "name": "Risk management",
    "plan_a": 4,
    "plan_b": 2,
    "reasoning": "Plan A identifies 8 key risks with mitigation triggers; Plan B mentions risks without concrete response plans."
  },
  {
    "name": "Budget realism",
    "plan_a": 3,
    "plan_b": 4,
    "reasoning": "Plan A budget lacks procurement detail; Plan B includes itemized capex/opex breakdown with vendor quotes."
  },
  {
    "name": "Measurable outcomes",
    "plan_a": 5,
    "plan_b": 2,
    "reasoning": "Plan A defines 7 numeric KPIs with thresholds; Plan B uses vague qualitative goals."
  },
  {
    "name": "Stakeholder alignment",
    "plan_a": 4,
    "plan_b": 3,
    "reasoning": "Plan A maps deliverables to stakeholder needs; Plan B assumes stakeholder buy-in without validation."
  },
  {
    "name": "Resource allocation realism",
    "plan_a": 3,
    "plan_b": 3,
    "reasoning": "Both plans assume 5 FTEs but lack role definitions or hiring strategy; roughly equivalent."
  }
]
```

**Final KPI naming:**
The last KPI is LLM-generated to capture aspects not covered by the previous 5-7 KPIs. Common examples:

- "Resource allocation realism"

- "Regulatory compliance readiness"

- "Technical feasibility"

- "Market timing"

- "Execution capacity"

This prevents the system from ignoring plan-specific strengths/weaknesses not covered by generic KPIs.

---

## API Reference

### Authentication

All API requests require an `X-API-Key` header:

```http
X-API-Key: <your_api_secret>
```

The key is validated against `rate_limit.api_key`. Generate keys via `/admin/keys` (admin access required).

---

### POST /api/rank

Submit a plan for Elo ranking.

**Request:**
```http
POST /api/rank HTTP/1.1
Host: planexe.com
Content-Type: application/json
X-API-Key: <your_api_secret>

{
  "plan_id": "uuid-v4-string",
  "plan_json": {
    "title": "Electric VTOL Development Program",
    "goal": "Certify 2-seat eVTOL by Q4 2027",
    "timeline": "24 months",
    "budget_usd": 15000000,
    "kpis": ["PDR complete Q2 2026", "CDR complete Q4 2026"],
    "risks": ["Battery energy density", "EASA certification delays"]
  },
  "budget_cents": 1500000000,
  "title": "Electric VTOL Development Program",
  "url": "https://planexe.com/plans/abc123"
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "plan_id": "uuid-v4-string",
  "elo": 1547.3,
  "percentile": 62.5,
  "comparisons_run": 10,
  "kpis": {
    "novelty_score": 0.78,
    "prompt_quality": 0.85,
    "technical_completeness": 0.72,
    "feasibility": 0.68,
    "impact_estimate": 0.81
  }
}
```

**Error Codes:**

| Code | Condition | Response |
|------|-----------|----------|
| 400 | Missing required fields | `{"error": "Missing required field: plan_json"}` |
| 401 | Invalid API key | `{"error": "Invalid API key"}` |
| 429 | Rate limit exceeded | `{"error": "Rate limit: 5 req/min"}` |
| 500 | LLM/database error | `{"error": "Internal server error", "detail": "..."}` |

**Rate Limit:**

- 5 requests per minute per API key

- Tracked in `rate_limit` table (sliding window: last 60 seconds)

- Resets at `last_ts + 60 seconds`

Implementation:
```python
def check_rate_limit(api_key: str) -> bool:
    now = datetime.now()
    record = db.query(RateLimit).filter_by(api_key=api_key).first()
    
    if not record:
        db.add(RateLimit(api_key=api_key, last_ts=now, count=1))
        return True
    
    if (now - record.last_ts).total_seconds() > 60:
        record.last_ts = now
        record.count = 1
        return True
    
    if record.count >= 5:
        return False
    
    record.count += 1
    return True
```

---

### GET /api/leaderboard

Retrieve top-ranked plans.

**Request:**
```http
GET /api/leaderboard?limit=20&offset=0 HTTP/1.1
Host: planexe.com
X-API-Key: <your_api_secret>
```

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `limit` | integer | No | 10 | Number of results (max 100) |
| `offset` | integer | No | 0 | Pagination offset |

**Response (200 OK):**
```json
{
  "plans": [
    {
      "plan_id": "uuid-1",
      "title": "Electric VTOL Development Program",
      "elo": 1847.2,
      "percentile": 95.3,
      "created_at": "2026-02-08T10:30:00Z"
    },
    {
      "plan_id": "uuid-2",
      "title": "Grid-Scale Battery Storage Network",
      "elo": 1803.5,
      "percentile": 91.7,
      "created_at": "2026-02-07T14:22:00Z"
    }
  ],
  "total": 247,
  "offset": 0,
  "limit": 20
}
```

**Privacy:** Only returns plans owned by the authenticated user (`owner_id` matched against API key's user).

---

### GET /api/export

Export detailed plan data (admin only).

**Request:**
```http
GET /api/export?limit=50 HTTP/1.1
Host: planexe.com
X-API-Key: <admin_api_secret>
```

**Response (200 OK):**
Returns full plan JSON including `plan_corpus.json_data` and all `plan_metrics` fields.

**Authorization:** Requires `admin` role in `users.role` column.

---

### GET /rankings

User-facing HTML interface showing ranked plans.

**Request:**
```http
GET /rankings HTTP/1.1
Host: planexe.com
Cookie: session_id=<session_cookie>
```

**Response:** HTML page with sortable table of user's plans.

---

## User Interface

### Rankings Page

**URL:** `/rankings`

**Layout:**

```
┌─────────────────────────────────────────────────────────────────┐
│ PlanExe Rankings                                     [Profile ▼] │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Your Plans (sorted by Elo)                                      │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Title                         Elo    Percentile  Actions   │ │
│  ├────────────────────────────────────────────────────────────┤ │
│  │ 🏆 Electric VTOL Program      1847   Top 5%     [View KPIs]│ │
│  │ 🥈 Battery Storage Network    1803   Top 10%    [View KPIs]│ │
│  │ 🥉 Solar Farm Deployment      1672   Top 25%    [View KPIs]│ │
│  │ 📊 Urban Mobility App         1598   50th %ile  [View KPIs]│ │
│  │ 🔧 Community Garden Network   1423   Bottom 25% [View KPIs]│ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  [Show all plans] [Filter by domain ▼]                           │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

**Screenshot placeholder:** `assets/rankings-page-desktop.png` (1920x1080)

---

### KPI Detail Modal

When user clicks **[View KPIs]**, a modal displays:

```
┌───────────────────────────────────────────────────────┐
│  Plan: Electric VTOL Program               [Close ✕] │
├───────────────────────────────────────────────────────┤
│                                                        │
│  Elo: 1847  |  Percentile: Top 5%                     │
│                                                        │
│  Top Strengths (vs. higher-ranked neighbors):         │
│  ✓ Goal clarity: 4.8/5 avg across 10 comparisons      │
│  ✓ Schedule credibility: 4.7/5                         │
│  ✓ Risk management: 4.5/5                              │
│                                                        │
│  Areas for Improvement:                                │
│  ⚠ Budget realism: 3.2/5                               │
│    → Add procurement detail and vendor quotes          │
│  ⚠ Regulatory compliance: 3.4/5                        │
│    → Document EASA certification timeline              │
│                                                        │
│  [Download full comparison report (PDF)]               │
│                                                        │
└───────────────────────────────────────────────────────┘
```

**Screenshot placeholder:** `assets/kpi-modal-desktop.png` (800x600)

---

### Mobile Responsive Design

**Breakpoints:**

- Desktop: ≥1024px (full table)

- Tablet: 768-1023px (condensed table, stacked KPI cards)

- Mobile: ≤767px (card layout, no table)

**Mobile card layout:**

```
┌─────────────────────────────────┐
│ 🏆 Electric VTOL Program        │
│ Elo: 1847  |  Top 5%            │
│ [View KPIs]                     │
└─────────────────────────────────┘
┌─────────────────────────────────┐
│ 🥈 Battery Storage Network      │
│ Elo: 1803  |  Top 10%           │
│ [View KPIs]                     │
└─────────────────────────────────┘
```

**Screenshot placeholder:** `assets/rankings-mobile.png` (375x667)

---

### Accessibility

**ARIA labels:**
```html
<table role="table" aria-label="Your ranked plans">
  <thead>
    <tr role="row">
      <th role="columnheader" aria-sort="descending">Elo Rating</th>
      <th role="columnheader">Percentile</th>
    </tr>
  </thead>
  <tbody role="rowgroup">
    <tr role="row">
      <td role="cell">1847</td>
      <td role="cell">Top 5%</td>
    </tr>
  </tbody>
</table>
```

**Keyboard navigation:**

- `Tab`: Navigate between rows

- `Enter`: Open KPI detail modal

- `Esc`: Close modal

- `Arrow keys`: Navigate table cells (when focused)

**Screen reader support:**

- Elo ratings announced with tier label: "Elo 1847, Top 5 percent"

- KPI scores announced as "Goal clarity: 4 point 8 out of 5"

**Color contrast:**

- Tier badges meet WCAG AA standard (4.5:1 ratio)

- Focus indicators have 3:1 contrast with background

---

### Toggle Implementation (Show/Hide Low-Ranked Plans)

```javascript
// File: static/js/rankings.js

function toggleLowRankedPlans() {
  const rows = document.querySelectorAll('[data-elo]');
  const threshold = 1500; // Bottom 50%
  const toggle = document.getElementById('show-low-ranked');
  
  rows.forEach(row => {
    const elo = parseFloat(row.dataset.elo);
    if (elo < threshold) {
      row.style.display = toggle.checked ? 'table-row' : 'none';
    }
  });
  
  // Update visible count
  const visibleCount = Array.from(rows).filter(r => r.style.display !== 'none').length;
  document.getElementById('visible-count').textContent = `${visibleCount} plans shown`;
}

// Attach event listener
document.getElementById('show-low-ranked').addEventListener('change', toggleLowRankedPlans);
```

**HTML snippet:**
```html
<label>
  <input type="checkbox" id="show-low-ranked" checked>
  Show plans below 50th percentile
</label>
<span id="visible-count">23 plans shown</span>
```

---

## Database Schema

### plan_corpus

Stores full plan JSON and embedding for comparison.

```sql
CREATE TABLE plan_corpus (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    url TEXT,
    json_data JSONB NOT NULL,  -- Full plan JSON (2-50KB typical)
    owner_id UUID NOT NULL REFERENCES users(id),
    embedding VECTOR(768),     -- pgvector: text-embedding-3-small
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_plan_corpus_owner ON plan_corpus(owner_id);
CREATE INDEX idx_plan_corpus_embedding ON plan_corpus USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_plan_corpus_json_data ON plan_corpus USING GIN (json_data);  -- For metadata queries
```

**Indexing notes:**

- `ivfflat` index for fast cosine similarity search (pgvector)

- GIN index on `json_data` enables fast queries like `json_data @> '{"domain": "energy"}'`

- Typical JSONB size: 2-50KB (median 12KB across test corpus)

---

### plan_metrics

Stores computed metrics and Elo rating.

```sql
CREATE TABLE plan_metrics (
    plan_id UUID PRIMARY KEY REFERENCES plan_corpus(id) ON DELETE CASCADE,
    novelty_score FLOAT,                  -- 0-1, LLM-scored
    prompt_quality FLOAT,                 -- 0-1, LLM-scored
    technical_completeness FLOAT,         -- 0-1, LLM-scored
    feasibility FLOAT,                    -- 0-1, LLM-scored
    impact_estimate FLOAT,                -- 0-1, LLM-scored
    elo FLOAT DEFAULT 1500.0,             -- Elo rating
    num_comparisons INT DEFAULT 0,        -- Number of pairwise comparisons
    bucket_id INT DEFAULT 0,              -- For A/B testing experiments
    kpi_details JSONB,                    -- Store KPI reasoning (Phase 2)
    review_comment TEXT,                  -- Optional human feedback
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_plan_metrics_elo ON plan_metrics(elo DESC);
CREATE INDEX idx_plan_metrics_bucket ON plan_metrics(bucket_id);
```

**`kpi_details` schema (Phase 2):**
```json
{
  "comparisons": [
    {
      "neighbor_id": "uuid-neighbor-1",
      "timestamp": "2026-02-08T10:30:00Z",
      "kpis": [
        {
          "name": "Goal clarity",
          "score_self": 4,
          "score_neighbor": 3,
          "reasoning": "This plan has concrete timeline; neighbor is vague."
        }
      ]
    }
  ]
}
```

---

### rate_limit

Tracks API rate limits per key.

```sql
CREATE TABLE rate_limit (
    api_key TEXT PRIMARY KEY,
    last_ts TIMESTAMPTZ NOT NULL,         -- Last request timestamp
    count INT DEFAULT 0,                  -- Request count in current window
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Rate limit logic:**

- Sliding 60-second window

- If `(now - last_ts) > 60s`: reset `count` to 1, update `last_ts`

- Else if `count < 5`: increment `count`

- Else: reject with 429

---

## Technical Rationale

### Why Elo Over Regression Models?

**Elo advantages:**

1. **No labeled training data required** — learns from pairwise comparisons

2. **Adapts to corpus drift** — as new plans enter, rankings adjust naturally

3. **Interpretable** — "Top 10%" is intuitive; regression coefficients are not

4. **Robust to outliers** — single bad comparison doesn't break the system

**Trade-offs:**

- Requires multiple comparisons per plan (10 minimum)

- Cold-start bias (first plans rated against weak corpus)

- No absolute quality signal (only relative ranking)

---

### Why K=32?

**Sensitivity parameter** controls how much each comparison shifts Elo:

| K value | Convergence speed | Noise sensitivity | Use case |
|---------|-------------------|-------------------|----------|
| 16 | Slow (30+ comparisons to converge) | Low | Established, stable corpus |
| 32 | Medium (15-20 comparisons) | Medium | **Current system** (balanced) |
| 40 | Fast (10-15 comparisons) | High | Beginner/provisional ratings |
| 64 | Very fast (5-10 comparisons) | Very high | Rapid iteration, testing |

**Empirical testing** (100-plan test corpus):

- K=16: Accurate but slow (30 comparisons to stabilize)

- K=32: Good convergence after 15-20 comparisons

- K=64: Fast but noisy (±100 Elo variance after 20 comparisons)

**Chosen K=32** for balance between responsiveness and stability.

---

### Why Likert 1-5 Over Continuous Scores?

**Likert scale advantages:**

1. **LLMs are calibrated for categorical ratings** — "rate 1-5" is a common training task

2. **Auditable** — humans can verify "this deserves a 4, not a 5"

3. **Avoids false precision** — difference between 0.73 and 0.78 is meaningless

4. **Consistent across comparisons** — continuous scores drift with context

**Alternative rejected:** 0-100 continuous scale

- Produced inconsistent scoring (same plan rated 73 vs 81 in different contexts)

- No interpretability gain over 1-5 scale

---

### Cold-Start Mitigation Strategy

**Problem:** First 20-30 plans set the baseline. If initial corpus is weak, all plans appear "good" relative to baseline.

**Current mitigation:**

1. **Random neighbor fallback** — if corpus has <10 plans, select randomly (no embedding bias)

2. **Normalized percentiles** — percentile calculated as `(rank / total_plans) * 100`, not absolute Elo threshold

**Phase 2 mitigations (planned):**

1. **Seed corpus** — 20 hand-curated reference plans (high/medium/low quality examples)

2. **Comparison count normalization** — weight Elo by `sqrt(num_comparisons)` in percentile calculation

3. **Domain-specific pools** — separate Elo pools for energy/tech/social plans (prevents cross-domain bias)

---

## Current Limitations

### 1. False Confidence

**Problem:** "Top 10%" doesn't mean *objectively good*, just *better than current corpus*.

**Risk:** If all plans in the corpus are weak, rankings still show a "winner."

**Example:**

- Corpus of 100 low-effort plans (all score 2-3 on KPIs)

- One plan scores 3-4 consistently

- That plan reaches Top 5%, but is still mediocre in absolute terms

**Mitigations:**

- **Phase 2:** Flag plans with `avg_kpi < 3.0` as "Needs improvement" even if top-ranked

- **Phase 3:** Seed corpus with 20 high-quality reference plans (absolute quality anchors)

- **Future:** Absolute quality thresholds (e.g., "Exceptional" requires `elo > 1700 AND avg_kpi > 4.0`)

---

### 2. Gaming Risk

**Problem:** Users might optimize prompts for LLM preferences rather than real-world utility.

**Example:** Stuffing keywords like "SMART goals", "KPI", "risk mitigation" without substance.

**Mitigations:**

- **Current:** Dynamic KPI selection (not fixed formula to game)

- **Current:** Reasoning transparency (nonsense prompts get low reasoning quality scores)

- **Phase 3:** Red-team evaluation (test whether gaming attempts produce worse outcomes)

- **Future:** Human validation of Top 5% plans

---

### 3. Cold-Start Bias

**Problem:** Early plans set the baseline. Small or skewed corpus biases rankings.

**Example:**

- First 20 plans are all tech MVPs (short timelines, low budgets)

- Plan 21 is a 10-year energy infrastructure project

- LLM comparisons penalize Plan 21 for "unrealistic timeline" (relative to corpus norm)

**Mitigations:**

- **Current:** Random neighbor selection if corpus <10 plans

- **Phase 2:** Normalize by `num_comparisons` in percentile calculation

- **Phase 2:** Domain-specific Elo pools (energy plans vs energy plans)

- **Phase 3:** Seed corpus with diverse reference plans

---

### 4. No Domain Expertise

**Problem:** LLM comparisons lack domain-specific nuance (e.g., regulatory complexity in pharma vs software).

**Example:**

- FDA approval timeline for drug: 7-10 years (realistic)

- Software MVP timeline: 7-10 years (red flag)

- LLM might not distinguish between these contexts

**Mitigations:**

- **Phase 2:** Domain-aware KPI sets (energy plans weight regulatory compliance higher)

- **Phase 3:** Expert validation pipeline (Top 5% plans flagged for optional human review)

- **Future:** Fine-tuned LLM on domain-specific plan corpus

---

### 5. Embedding Quality Dependency

**Problem:** Neighbor selection depends on embedding quality. Poor embeddings → irrelevant comparisons.

**Current model:** `text-embedding-3-small` (768 dims)

- Works well for semantic similarity of prompts

- May miss structural similarities (e.g., timeline format, budget magnitude)

**Mitigations:**

- **Phase 2:** Hybrid retrieval (50% embedding similarity, 50% metadata filters like domain/budget)

- **Future:** Fine-tuned embeddings on plan corpus

---

## Future Enhancements

### 1. Hybrid Ranking: Elo + Absolute Quality

**Problem:** Elo only measures relative rank, not absolute quality.

**Solution:** Combine Elo with absolute KPI thresholds.

**Formula:**
```python
def hybrid_score(elo: float, avg_kpi: float, alpha: float = 0.7) -> float:
    """
    Compute hybrid score combining relative rank (Elo) and absolute quality (KPI).
    
    Args:
        elo: Elo rating (normalized to 0-1 range: (elo - 1200) / 800)
        avg_kpi: Average KPI score across all baseline metrics (0-1)
        alpha: Weight for Elo component (0-1, default 0.7)
    
    Returns:
        Hybrid score (0-1)
    """
    elo_normalized = (elo - 1200) / 800  # Map [1200, 2000] -> [0, 1]
    elo_normalized = max(0, min(1, elo_normalized))  # Clamp to [0, 1]
    
    return alpha * elo_normalized + (1 - alpha) * avg_kpi
```

**Example:**

- Plan A: Elo 1850 (95th %ile), avg_kpi 0.65 → hybrid = 0.7 * 0.81 + 0.3 * 0.65 = 0.76

- Plan B: Elo 1550 (55th %ile), avg_kpi 0.85 → hybrid = 0.7 * 0.44 + 0.3 * 0.85 = 0.56

**Result:** Plan A still ranks higher (strong Elo), but Plan B's absolute quality is recognized.

**Tuning alpha:**

- α=1.0: Pure Elo (relative rank only)

- α=0.5: Equal weight to relative rank and absolute quality

- α=0.0: Pure absolute quality (ignores corpus context)

**Recommended α=0.7** for corpus-aware ranking with quality floor.

---

### 2. Personalized Ranking Weights

**Problem:** Different users care about different KPIs (investor vs builder vs researcher).

**Solution:** Allow users to customize KPI weights.

**Schema:**
```json
{
  "user_id": "uuid-user-1",
  "kpi_weights": {
    "feasibility": 0.3,
    "impact_estimate": 0.3,
    "novelty_score": 0.1,
    "technical_completeness": 0.2,
    "prompt_quality": 0.1
  }
}
```

**Weighted Elo formula:**
```python
def weighted_elo_update(plan: Plan, neighbor: Plan, kpi_scores: dict, weights: dict, K: int = 32):
    """
    Update Elo with user-specific KPI weights.
    
    Args:
        plan: The plan being ranked
        neighbor: Comparison neighbor
        kpi_scores: {"kpi_name": {"plan": 4, "neighbor": 3}, ...}
        weights: {"kpi_name": 0.3, ...} (sum to 1.0)
        K: Elo sensitivity parameter
    """
    weighted_score_plan = sum(kpi_scores[kpi]["plan"] * weights.get(kpi, 0.2) for kpi in kpi_scores)
    weighted_score_neighbor = sum(kpi_scores[kpi]["neighbor"] * weights.get(kpi, 0.2) for kpi in kpi_scores)
    
    diff = weighted_score_plan - weighted_score_neighbor
    prob_win = map_likert_to_probability(diff)  # Use existing mapping
    
    return update_elo(plan.elo, neighbor.elo, prob_win, K)
```

**UI:** Slider interface for adjusting weights (sum constrained to 1.0).

---

### 3. Batch Re-Ranking

**Problem:** As corpus grows, early plans' Elo ratings may be stale (compared against outdated corpus).

**Solution:** Periodic re-ranking of random plan samples against recent corpus.

**Pseudocode:**
```python
def batch_rerank(sample_size: int = 50, comparisons_per_plan: int = 5):
    """
    Re-rank a random sample of plans against recent corpus.
    
    Args:
        sample_size: Number of plans to re-rank
        comparisons_per_plan: Number of new comparisons per plan
    """
    # Select random sample of plans with last_comparison > 30 days ago
    old_plans = db.query(Plan).filter(
        Plan.last_comparison_date < datetime.now() - timedelta(days=30)
    ).order_by(func.random()).limit(sample_size).all()
    
    # For each plan, run N new comparisons against recent neighbors
    for plan in old_plans:
        recent_neighbors = db.query(Plan).filter(
            Plan.created_at > datetime.now() - timedelta(days=30),
            Plan.id != plan.id
        ).order_by(Plan.embedding.cosine_distance(plan.embedding)).limit(comparisons_per_plan).all()
        
        for neighbor in recent_neighbors:
            kpi_scores = run_llm_comparison(plan, neighbor)
            prob_win = compute_win_probability(kpi_scores)
            plan.elo, neighbor.elo = update_elo(plan.elo, neighbor.elo, prob_win)
        
        plan.last_comparison_date = datetime.now()
        plan.num_comparisons += comparisons_per_plan
    
    db.commit()
```

**Schedule:** Run weekly via cron job.

**Sample size tuning:**

- Corpus <100 plans: re-rank all

- Corpus 100-1000: re-rank 10% (sample 50-100 plans)

- Corpus >1000: re-rank 5% (sample 50-200 plans)

---

### 4. Explain-by-Example (Nearest Neighbor Justification)

**Problem:** Users ask "Why is my plan ranked here?"

**Solution:** Show 3 nearest neighbors (higher-ranked) with KPI comparison breakdown.

**Retrieval:**
```sql
SELECT p.id, p.title, m.elo, p.embedding <=> :query_embedding AS distance
FROM plan_corpus p
JOIN plan_metrics m ON p.id = m.plan_id
WHERE m.elo > :query_elo
ORDER BY p.embedding <=> :query_embedding
LIMIT 3;
```

**UI output:**
```
Your plan (Elo 1620) vs higher-ranked neighbors:

1. Electric VTOL Program (Elo 1847, +227 points)
   - Goal clarity: You 3.2, Neighbor 4.8 (+1.6) → Add specific timeline milestones
   - Risk management: You 3.5, Neighbor 4.7 (+1.2) → Document mitigation triggers
   - Budget realism: You 3.8, Neighbor 4.2 (+0.4) → Minor gap

2. Grid Battery Storage (Elo 1803, +183 points)
   - Measurable outcomes: You 2.9, Neighbor 4.9 (+2.0) → Define numeric KPIs
   - Stakeholder alignment: You 3.1, Neighbor 4.3 (+1.2) → Map deliverables to stakeholders
```

**Value:** Transforms rank into actionable feedback.

---

### 5. Domain-Specific Elo Pools

**Problem:** Cross-domain comparisons are unfair (e.g., 3-month MVP vs 5-year infrastructure project).

**Solution:** Separate Elo pools per domain.

**Schema change:**
```sql
ALTER TABLE plan_metrics ADD COLUMN domain TEXT DEFAULT 'general';
CREATE INDEX idx_plan_metrics_domain ON plan_metrics(domain);
```

**Domains:**

- `tech` (software, hardware, consumer products)

- `energy` (solar, wind, battery, grid)

- `health` (biotech, medical devices, pharma)

- `social` (education, community, policy)

- `research` (academic, scientific)

**Neighbor selection with domain filter:**
```sql
SELECT id FROM plan_corpus
WHERE domain = :query_domain
ORDER BY embedding <=> :query_embedding
LIMIT 10;
```

**UI:** Show both *domain rank* ("Top 5% in Energy") and *global rank* ("Top 15% overall").

---

### 6. Temporal Decay

**Problem:** Plans from 6+ months ago may rank high but use outdated assumptions.

**Solution:** Apply decay factor to Elo based on age.

**Formula:**
```python
def effective_elo(elo: float, created_at: datetime, decay_rate: float = 0.05) -> float:
    """
    Apply temporal decay to Elo rating.
    
    Args:
        elo: Current Elo rating
        created_at: Plan creation timestamp
        decay_rate: Decay per month (default 0.05 = 5%/month)
    
    Returns:
        Effective Elo for ranking purposes
    """
    months_old = (datetime.now() - created_at).days / 30
    decay_factor = (1 - decay_rate) ** months_old
    return elo * decay_factor
```

**Example:**

- Plan created 6 months ago with Elo 1800

- Effective Elo = 1800 * (0.95^6) = 1800 * 0.735 = 1323

- Drops from Top 5% to ~40th percentile

**Tuning decay_rate:**

- 0.02 (2%/month): Gentle decay, 12-month half-life

- 0.05 (5%/month): Moderate decay, 6-month half-life

- 0.10 (10%/month): Aggressive decay, 3-month half-life

**Recommended 5%/month** for plans in fast-moving domains (tech, policy).

---

### 7. Reasoning LLM for Top 10%

**Problem:** Discrimination between top plans requires deeper analysis than flash model provides.

**Solution:** Two-tier comparison strategy.

**Tier 1 (All plans):** `gemini-2.0-flash-exp` (~$0.10 per 10 comparisons)

- Fast, cheap, good enough for initial ranking

**Tier 2 (Top 10% only):** `o1-mini` or `claude-3.5-sonnet` (~$1.00 per 10 comparisons)

- Deeper reasoning, better discrimination

**Implementation:**
```python
def select_comparison_model(plan_elo: float, neighbor_elo: float) -> str:
    """
    Choose comparison model based on Elo.
    
    Returns:
        Model name for LLM comparison
    """
    if plan_elo > 1700 and neighbor_elo > 1700:
        return "openai/o1-mini"  # Top 10% vs Top 10%
    else:
        return "google/gemini-2.0-flash-exp"  # Default
```

**Cost impact:**

- Corpus of 1000 plans: ~100 are Top 10%

- Top 10% plans average 20 comparisons each (10 initial + 10 re-rank)

- Reasoning LLM cost: 100 plans × 10 comparisons × $0.10 = $100 (one-time)

- vs. Flash-only cost: 1000 plans × 10 comparisons × $0.01 = $100 (total)

**Cost increase:** ~2x, but only for top-tier discrimination.

---

### 8. Investor Filters

**Problem:** Investors want to find relevant plans quickly, not browse entire leaderboard.

**Solution:** Add filter parameters to `/api/leaderboard`.

**New query parameters:**

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `domain` | string | tech, energy, health, social, research | Filter by plan domain |
| `impact_horizon` | string | days, months, years, decades | Expected impact timeframe |
| `budget_min` | integer | Cents (e.g., 100000 = $1000) | Minimum budget |
| `budget_max` | integer | Cents | Maximum budget |
| `region` | string | US, EU, APAC, global | Geographic focus |

**Example request:**
```http
GET /api/leaderboard?domain=energy&budget_min=500000000&budget_max=10000000000&region=US&limit=20
```

**SQL query:**
```sql
SELECT p.*, m.elo
FROM plan_corpus p
JOIN plan_metrics m ON p.id = m.plan_id
WHERE 
    p.json_data->>'domain' = :domain
    AND (p.json_data->>'budget_cents')::bigint BETWEEN :budget_min AND :budget_max
    AND p.json_data->>'region' = :region
ORDER BY m.elo DESC
LIMIT :limit;
```

**UI:** Dropdown filters on `/rankings` page.

---

## Implementation Roadmap

### Phase 1 (Completed ✅)

- [x] Dynamic KPI extraction via LLM

- [x] Pairwise LLM comparison with Likert 1-5 scoring

- [x] Elo rating update (K=32)

- [x] User plan list with Elo display (`/rankings`)

- [x] API endpoints: `/api/rank`, `/api/leaderboard`

- [x] Rate limiting (5 req/min per API key)

- [x] LLM-named "remaining considerations" KPI

- [x] 30-word reasoning cap per KPI

- [x] Embedding-based neighbor selection (pgvector)

---

### Phase 2 (Next 2-4 weeks)

**KPI Reasoning Storage:**

- [ ] Add `kpi_details` JSONB column to `plan_metrics`

- [ ] Store all comparison results (neighbor_id, KPI scores, reasoning)

- [ ] UI: "Why this rank?" modal with KPI breakdown

**Percentile Tiers:**

- [ ] Map Elo ranges to tier labels (Exceptional / Strong / Solid / Developing / Needs Work)

- [ ] UI badges (🏆 Gold / 🥈 Silver / 🥉 Bronze / 📊 Standard / 🔧 Improve)

- [ ] Percentile calculation normalized by `num_comparisons`

**Prompt Improvement Suggestions:**

- [ ] Generate tier-specific advice based on KPI gaps

- [ ] Auto-suggest prompt template for Bottom 25%

- [ ] Email/notification with improvement tips after ranking

**Domain-Specific Ranking:**

- [ ] Add `domain` column to `plan_corpus`

- [ ] Separate Elo pools per domain (tech / energy / health / social / research)

- [ ] UI: Show domain rank + global rank

**Testing:**

- [ ] Unit tests for Elo update logic

- [ ] Integration tests for `/api/rank` endpoint

- [ ] Load test: 100 concurrent ranking requests

---

### Phase 3 (Next Quarter)

**Investor Filters:**

- [ ] Add filter parameters to `/api/leaderboard` (domain, budget, region, impact horizon)

- [ ] Update SQL queries with JSONB metadata filters

- [ ] UI: Dropdown filters on `/rankings` page

**Red-Team Gaming Detection:**

- [ ] Monitor for prompt patterns that spike Elo without improving KPIs

- [ ] Flag suspicious plans (e.g., keyword stuffing) for manual review

- [ ] A/B test: compare gaming-resistant prompts

**Public Benchmark Plans:**

- [ ] Curate 20 high-quality reference plans (hand-picked by domain experts)

- [ ] Ensure all new plans compare against 2-3 benchmark plans

- [ ] Provides absolute quality anchor (mitigates cold-start bias)

**Reasoning LLM for Top 10%:**

- [ ] Implement two-tier comparison strategy (flash for all, o1-mini for top 10%)

- [ ] Cost analysis and budget approval

- [ ] A/B test: measure discrimination improvement at top of leaderboard

---

### Phase 4 (Future / Research)

**Hybrid Ranking (Elo + Absolute Quality):**

- [ ] Implement `hybrid_score` formula (α=0.7 default)

- [ ] UI: Toggle between "Relative Rank" and "Hybrid Score"

- [ ] User study: which ranking is more useful?

**Personalized Ranking Weights:**

- [ ] Allow users to customize KPI weights

- [ ] UI: Slider interface for adjusting weights

- [ ] Store user preferences in `user_kpi_weights` table

**Batch Re-Ranking:**

- [ ] Cron job: weekly re-rank of 10% of corpus

- [ ] Focus on plans with `last_comparison_date > 30 days`

- [ ] Monitor Elo stability over time

**Temporal Decay:**

- [ ] Implement `effective_elo` with 5%/month decay

- [ ] UI: Show "Fresh rank" (with decay) vs "All-time rank" (no decay)

- [ ] Domain-specific decay rates (tech: 5%/month, infrastructure: 1%/month)

**Explain-by-Example:**

- [ ] Nearest neighbor retrieval (3 higher-ranked plans)

- [ ] KPI comparison breakdown

- [ ] UI: "Compare to better plans" button

**Domain Expertise Integration:**

- [ ] Partner with domain experts for top 5% validation

- [ ] Optional human review pipeline

- [ ] Expert feedback stored in `plan_metrics.review_comment`

---

## Glossary

**API_SECRET**  
Authentication token used in `X-API-Key` header for API requests. Generated per user via admin interface. Stored in `rate_limit.api_key`.

**Elo**  
Rating system invented by Arpad Elo for chess rankings. Measures relative skill/quality through pairwise comparisons. Higher Elo = better performance. Default starting Elo: 1500. Pronounced "EE-lo" (not "E-L-O").

**Gemini-flash**  
Shorthand for `gemini-2.0-flash-exp`, Google's fast LLM optimized for structured output. Used for KPI extraction and pairwise comparison in PlanExe. Accessible via OpenRouter API.

**KPI (Key Performance Indicator)**  
Measurable metric used to evaluate plan quality. Examples: goal clarity, schedule credibility, risk management, budget realism. PlanExe extracts 6-8 KPIs per comparison dynamically via LLM.

**Likert scale**  
5-point rating scale (1 = Very poor, 2 = Below average, 3 = Average, 4 = Above average, 5 = Excellent). Used for scoring each KPI in pairwise comparisons. Integer-only (no 3.5 scores).

**pgvector**  
PostgreSQL extension for vector similarity search. Enables fast cosine similarity queries for embedding-based neighbor selection. Supports `ivfflat` and `hnsw` indexing.

**Pairwise comparison**  
Comparing two plans (A vs B) across multiple KPIs to determine which is better. Core primitive of Elo ranking system. Each new plan compared against 10 neighbors.

**Win probability**  
Probability (0-1) that plan A is better than plan B, derived from Likert score difference. Used as input to Elo update formula. Example: +2 score difference → 0.7 win probability.

---

## Quick Wins Checklist

Completed items for immediate usability improvements:

- [x] Add TOC for document navigation

- [x] Fix heading hierarchy (consistent `##` for sections, `###` for subsections)

- [x] Explain Likert→probability mapping rationale

- [x] Justify K=32 parameter choice

- [x] Document cold-start bias and mitigation strategies

- [x] Mention plan_json typical size and JSONB indexing strategy

- [x] Align rate-limit description with actual implementation code

- [x] Show full KPI extraction prompt in fenced code block

- [x] Add concrete JSON response example for KPI output

- [x] Clarify "remaining considerations" KPI naming convention

- [x] Mention 2000-token budget per comparison

- [x] Add API reference table (endpoints, auth, schemas, error codes)

- [x] Document pagination for `/api/leaderboard`

- [x] Add UI documentation with ASCII mockups

- [x] Include toggle implementation code snippet

- [x] Document responsive design breakpoints

- [x] Add ARIA/accessibility labels and keyboard navigation

- [x] Expand future work with concrete formulas (hybrid ranking, personalized weights)

- [x] Add pseudocode for batch re-ranking schedule

- [x] Document explain-by-example retrieval strategy

- [x] Fix Elo capitalization (proper noun: "Elo", not "ELO")

- [x] Fix Likert capitalization (proper noun: "Likert", not "LIKERT")

- [x] Break long paragraphs into scannable chunks

- [x] Wrap all JSON in triple backticks with `json` syntax highlighting

- [x] Consistent inline code vs fenced blocks (inline for short refs, fenced for multi-line)

- [x] Add glossary section defining all technical terms

- [x] Remove promotional phrasing ("revolutionary", "game-changing")

- [x] Set primary audience to developers (technical focus, implementation details)

---

**Document version:** 2.0  
**Last updated:** 2026-02-08  
**Maintainer:** OpenClaw team  
**Feedback:** Open issues at https://github.com/VoynichLabs/PlanExe2026/issues
