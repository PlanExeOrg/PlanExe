---
title: Plans as LLM Templates - Parameterized Prompt Export
date: 2026-02-09
status: proposal
author: Larry (OpenClaw)
---

# Plans as LLM Templates - Parameterized Prompt Export

## Overview

PlanExe generates comprehensive business plans, but they're currently opaque artifacts. External agents and automation tools can't easily consume plan logic or adapt plans to new contexts.

This proposal treats **completed plans as reusable LLM templates** with parameterized sections, enabling:
- Export as Jinja2-style templates
- API endpoint for template rendering with custom variables
- Plan remixing and few-shot learning for downstream agents

## Problem

- Plans are one-shot artifacts with no reuse mechanism
- Agents can't easily say "give me a plan like X but for industry Y"
- No structured way to extract the prompt logic that created a good plan

## Proposed Solution

### Plan Template Format

Export plans as structured templates with:

```jinja2
---
template_id: restaurant-expansion-v1
base_plan_id: {{ plan_uuid }}
variables:
  - industry: string (required)
  - location: string (required)
  - budget: number (optional, default: 50000)
  - timeline_months: number (optional, default: 12)
---

# {{ industry | title }} Expansion Plan - {{ location }}

## Executive Summary

This plan outlines a {{ timeline_months }}-month expansion strategy for a {{ industry }} business in {{ location }} with a budget of ${{ budget | number_format }}.

{% if budget < 100000 %}
**Budget Constraint Noted**: Lean startup approach recommended given capital limitations.
{% endif %}

## Market Analysis

{% block market_analysis %}
[Market research for {{ industry }} in {{ location }}]
{% endblock %}

...
```

### API Endpoint

```http
POST /api/plan/template/render
Authorization: Bearer <api_key>
Content-Type: application/json

{
  "template_id": "restaurant-expansion-v1",
  "variables": {
    "industry": "coffee shop",
    "location": "Portland, OR",
    "budget": 75000,
    "timeline_months": 8
  }
}
```

**Response:**
```json
{
  "rendered_plan": "# Coffee Shop Expansion Plan - Portland, OR\n\n...",
  "estimated_tokens": 12500,
  "template_version": "1.0.0"
}
```

### Storage Schema

Add `plan_templates` table:

```sql
CREATE TABLE plan_templates (
  id UUID PRIMARY KEY,
  source_plan_id UUID REFERENCES plans(id),
  template_name TEXT UNIQUE,
  template_body TEXT,  -- Jinja2 template
  variables JSONB,     -- Variable schema
  created_at TIMESTAMPTZ DEFAULT now(),
  downloads INTEGER DEFAULT 0
);
```

## Use Cases

1. **Agent Few-Shot Learning**: "Generate a plan like template X but for domain Y"
2. **Customer Self-Service**: Browse template library, fill in variables, instant draft
3. **Plan Remixing**: Combine sections from multiple templates
4. **API Integration**: External tools can request plans programmatically

## Benefits

- **Plan reuse** - Good plans become templates for future work
- **Faster generation** - Template rendering is instant (no LLM call for structure)
- **Consistency** - Templates enforce proven structures
- **Monetization** - Premium template library for subscribers

## Implementation Plan

### Phase 1: Template Export (Week 1-2)
- Add "Export as Template" button in plan UI
- Generate Jinja2 from plan HTML/markdown
- Store in `plan_templates` table

### Phase 2: Rendering Engine (Week 3)
- Build Jinja2 renderer with variable validation
- Add `/api/plan/template/render` endpoint
- Rate limit: 10 renders/hour for free tier

### Phase 3: Template Library (Week 4-5)
- Public template browse UI
- Search and filter by industry/domain
- User ratings and favorites

### Phase 4: Advanced Features (Future)
- Template versioning (v1, v2, etc.)
- Diff view between template versions
- Collaborative template editing

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Template quality varies | Curate "verified" templates from high-rated plans |
| Variable validation complexity | Start with simple types (string, number, boolean) |
| Jinja2 injection attacks | Sandbox rendering, whitelist allowed filters |
| Templates go stale | Track usage, deprecate low-download templates |

## Success Metrics

- 50+ templates published in first month
- 20% of new plans start from a template
- Template renders account for 15%+ of API usage
- User feedback: "faster than starting from scratch"

## References

- Jinja2 documentation: https://jinja.palletsprojects.com/
- Similar pattern: Terraform modules, Helm charts, AWS CloudFormation templates
