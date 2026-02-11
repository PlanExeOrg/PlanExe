---
title: Plan Explain API - Natural Language Summaries
date: 2026-02-09
status: proposal
author: Larry the Laptop Lobster
---

# Plan Explain API - Natural Language Summaries

## Overview

PlanExe generates detailed, comprehensive business plans that can be 50-100 pages long. Users often need quick summaries for:

- Email updates to stakeholders

- Dashboard previews

- Customer support responses

- Social media posts about plan progress

This proposal introduces a `/api/plan/{id}/explain` endpoint that returns natural-language summaries of any plan using a fast LLM (Gemini 2.0 Flash).

## Problem

- Plans are too long to read in full for quick updates

- No programmatic way to get "executive summary" or "elevator pitch" version

- External tools (email automation, dashboards) can't easily consume plan content

- Manual summarization is slow and inconsistent

## Proposed Solution

### API Endpoint

```http
GET /api/plan/{plan_id}/explain
Authorization: Bearer <api_key>
Query Parameters:
  - length: short|medium|long (default: short)
  - audience: technical|business|general (default: business)
  - format: text|markdown|json (default: text)

Response (200 OK):
{
  "plan_id": "550e8400-e29b-41d4-a716-446655440000",
  "title": "Coffee Shop Expansion - Portland, OR",
  "summary": "A 12-month plan to open a second location in Portland's Pearl District, targeting specialty coffee enthusiasts with a budget of $150K. The plan covers market analysis, site selection, equipment procurement, staffing, and financial projections showing break-even at month 18.",
  "key_points": [
    "Target market: Specialty coffee consumers in Pearl District",
    "Investment: $150K initial capital",
    "Timeline: 12 months to opening",
    "Break-even: Month 18"
  ],
  "generated_at": "2026-02-09T18:30:00Z",
  "model": "gemini-2.0-flash-001",
  "cached": false
}
```

### Implementation

**LLM Selection:** Gemini 2.0 Flash

- Cost: ~$0.02 per summary (2K input tokens, 500 output tokens)

- Latency: 2-3 seconds

- Quality: Good enough for summaries, not critical content

**Caching Strategy:**
```python
# Cache summaries for 12 hours
cache_key = f"plan_explain:{plan_id}:{length}:{audience}"
cached = redis.get(cache_key)
if cached:
    return json.loads(cached)

# Generate new summary
summary = generate_summary(plan_id, length, audience)
redis.setex(cache_key, 43200, json.dumps(summary))  # 12h TTL
return summary
```

**Prompt Template:**
```python
EXPLAIN_PROMPT = """
You are summarizing a business plan for {audience} audience.

Plan Title: {title}
Plan Length: {word_count} words
Target Length: {target_length}

Full Plan:
{plan_content}

Instructions:
- Write a {target_length} summary (short=2-3 sentences, medium=1 paragraph, long=3-5 paragraphs)
- Focus on: goal, target market, key strategies, timeline, budget
- Tone: {audience} ({technical/business/general})
- Format: {format}

Summary:
"""
```

## Use Cases

### 1. Email Automation
```python
# Send daily plan update emails
plan = get_plan(plan_id)
summary = requests.get(f"/api/plan/{plan_id}/explain?length=short").json()

send_email(
    to=user.email,
    subject=f"Plan Update: {plan.title}",
    body=f"Your plan is ready!\n\n{summary['summary']}\n\nView full plan: {plan.url}"
)
```

### 2. Dashboard Widgets
```jsx
// React component showing plan preview
function PlanCard({ planId }) {
  const { data } = useSWR(`/api/plan/${planId}/explain?length=medium`);
  
  return (
    <Card>
      <h3>{data.title}</h3>
      <p>{data.summary}</p>
      <ul>
        {data.key_points.map(point => <li key={point}>{point}</li>)}
      </ul>
      <Link to={`/plan/${planId}`}>View Full Plan →</Link>
    </Card>
  );
}
```

### 3. Customer Support
```python
# Support agent gets quick plan overview
def handle_support_ticket(ticket):
    plan_id = ticket.metadata.get('plan_id')
    if plan_id:
        explanation = get_plan_explanation(plan_id, audience='general')
        return f"This customer's plan: {explanation['summary']}"
```

### 4. Social Sharing
```python
# Generate tweet-length summary
summary = requests.get(f"/api/plan/{plan_id}/explain?length=short&format=text").json()
tweet = f"Just created a business plan with @PlanExe: {summary['summary']} 🚀"
post_to_twitter(tweet)
```

## Implementation Plan

### Week 1: Core Endpoint

- Build `/api/plan/{id}/explain` route

- Integrate Gemini 2.0 Flash API

- Implement basic prompt template

- Add response caching (Redis)

### Week 2: Length & Audience Options

- Add `length` parameter handling (short/medium/long)

- Add `audience` parameter (technical/business/general)

- Tune prompts for each combination

- A/B test summary quality

### Week 3: Advanced Features

- Add `format` parameter (text/markdown/json)

- Extract structured key points (bullets)

- Add confidence score (how well summary captures plan)

- Rate limiting (10 requests/minute per user)

### Week 4: Integration & Polish

- Update API docs with examples

- Build SDK helpers for common use cases

- Add to PlanExe web UI (show summary before full plan)

- Monitor cache hit rate and optimize TTL

## Cost Analysis

**Per-request cost:** ~$0.02 (Gemini Flash input + output)
**With caching (12h TTL):**

- Cache hit rate: 70-80% (most users view same plan multiple times)

- Effective cost per unique plan: $0.02 (first request) + $0.00 (cached hits)

**Monthly estimate for 1,000 active plans:**

- Unique summarizations: 1,000 × $0.02 = $20

- Cached requests: ~7,000 × $0.00 = $0

- **Total: ~$20/month**

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Summary quality varies | Human review top 100 summaries, tune prompts |
| LLM hallucination | Cross-reference summary with plan content, flag mismatches |
| Cache staleness | Invalidate cache when plan is edited |
| API abuse | Rate limit 10 req/min per user, 100/day for free tier |
| Cost explosion | Cap at 1K summaries/day, alert if exceeded |

## Success Metrics

- 80%+ of users view summary before full plan

- Cache hit rate > 70%

- Average summary generation time < 3 seconds

- User feedback: "summary accurately represents my plan" > 4/5 stars

## Future Enhancements

- **Multi-language summaries** (translate to Spanish, French, etc.)

- **Voice summaries** (TTS integration for audio version)

- **Comparison summaries** ("How does this plan differ from my previous one?")

- **Sentiment analysis** (is the plan optimistic, cautious, ambitious?)

## References

- Gemini 2.0 Flash pricing: https://ai.google.dev/pricing

- Prompt engineering best practices: Anthropic prompt guide

- Caching strategies: Redis best practices
