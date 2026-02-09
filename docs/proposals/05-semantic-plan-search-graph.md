---
title: Semantic Plan Search Graph - pgvector Similarity
date: 2026-02-09
status: proposal
author: Larry (OpenClaw)
---

# Semantic Plan Search Graph - pgvector Similarity

## Overview

PlanExe has generated thousands of business plans across diverse domains. This corpus is valuable for:
- Finding similar plans ("show me plans like this one")
- Few-shot learning (use similar plans as examples for new generation)
- Discovery ("I want to open a coffee shop - what plans exist?")

This proposal adds **semantic search** across the entire plan corpus using pgvector (PostgreSQL extension) and sentence embeddings.

## Problem

- No way to search plans by meaning/topic (only exact text match)
- Can't find "plans similar to mine" for inspiration
- Agents can't leverage existing plans as few-shot examples
- Plan library feels like a black box instead of a knowledge graph

## Proposed Solution

### Architecture

```
┌──────────────────────────────────┐
│  User Query                      │
│  "coffee shop expansion plan"    │
└────────────────┬─────────────────┘
                 │
                 v
┌──────────────────────────────────┐
│  Embedding Model                 │
│  sentence-transformers/          │
│  all-mpnet-base-v2               │
└────────────────┬─────────────────┘
                 │ [768-dim vector]
                 v
┌──────────────────────────────────┐
│  pgvector Similarity Search      │
│  SELECT * FROM plan_corpus       │
│  ORDER BY embedding <=> $1       │
│  LIMIT 10                        │
└────────────────┬─────────────────┘
                 │
                 v
┌──────────────────────────────────┐
│  Ranked Results                  │
│  1. Coffee Shop - Portland       │
│  2. Café Expansion - Seattle     │
│  3. Specialty Coffee Roastery    │
└──────────────────────────────────┘
```

### Database Schema

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Plan corpus table with embeddings
CREATE TABLE plan_corpus (
  id UUID PRIMARY KEY,
  title TEXT NOT NULL,
  prompt TEXT,
  summary TEXT,
  domain TEXT,  -- e.g., "food_beverage", "tech_startup", "retail"
  embedding vector(768),  -- sentence-transformers/all-mpnet-base-v2
  created_at TIMESTAMPTZ DEFAULT now(),
  plan_url TEXT,
  word_count INTEGER
);

-- Index for fast similarity search
CREATE INDEX ON plan_corpus USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);
```

### Embedding Generation

**Model:** `sentence-transformers/all-mpnet-base-v2`
- Dimension: 768
- Speed: ~100 sentences/second on CPU
- Quality: State-of-the-art for semantic search
- Cost: Free (run locally or serverless)

**Embed on Insert:**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-mpnet-base-v2')

def index_plan(plan_id, title, prompt, summary):
    # Combine title + prompt + summary for rich embedding
    text = f"{title}\n\n{prompt}\n\n{summary}"
    embedding = model.encode(text)
    
    cursor.execute("""
        INSERT INTO plan_corpus (id, title, prompt, summary, embedding)
        VALUES (%s, %s, %s, %s, %s)
    """, (plan_id, title, prompt, summary, embedding.tolist()))
```

### Search API

```http
GET /api/plans/search
Query Parameters:
  - q: Search query (e.g., "coffee shop expansion")
  - limit: Number of results (default: 10, max: 50)
  - domain: Filter by domain (optional)
  - min_similarity: Minimum cosine similarity (0-1, default: 0.5)

Response:
{
  "query": "coffee shop expansion",
  "results": [
    {
      "plan_id": "550e8400-e29b-41d4-a716-446655440000",
      "title": "Coffee Shop Expansion - Portland, OR",
      "similarity": 0.89,
      "summary": "12-month plan to open second location...",
      "url": "/plan/550e8400-e29b-41d4-a716-446655440000",
      "domain": "food_beverage"
    },
    ...
  ]
}
```

**Query Implementation:**
```python
def search_plans(query, limit=10, min_similarity=0.5):
    query_embedding = model.encode(query)
    
    results = cursor.execute("""
        SELECT id, title, summary, domain, plan_url,
               1 - (embedding <=> %s::vector) AS similarity
        FROM plan_corpus
        WHERE 1 - (embedding <=> %s::vector) > %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (query_embedding.tolist(), query_embedding.tolist(), 
          min_similarity, query_embedding.tolist(), limit))
    
    return results.fetchall()
```

## Use Cases

### 1. Plan Discovery
```python
# User: "Show me plans for opening a restaurant"
results = search_plans("opening a restaurant", limit=5)
# Returns: restaurant plans, café plans, food truck plans (semantically similar)
```

### 2. Few-Shot Learning
```python
# Agent generating new plan
def generate_plan_with_examples(prompt):
    # Find 3 similar plans to use as examples
    similar = search_plans(prompt, limit=3, min_similarity=0.7)
    
    few_shot_context = "\n\n".join([
        f"Example {i+1}: {plan['title']}\n{plan['summary']}"
        for i, plan in enumerate(similar)
    ])
    
    # Include in LLM prompt
    return generate_plan(prompt, few_shot_examples=few_shot_context)
```

### 3. Plan Recommendations
```jsx
// After user completes a plan
function RelatedPlans({ currentPlanId }) {
  const { data } = useSWR(`/api/plans/${currentPlanId}/similar?limit=5`);
  
  return (
    <section>
      <h3>Plans Like Yours</h3>
      <ul>
        {data.results.map(plan => (
          <li key={plan.plan_id}>
            <a href={plan.url}>{plan.title}</a>
            <span>({Math.round(plan.similarity * 100)}% similar)</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
```

### 4. Trend Analysis
```python
# What domains are growing?
def trending_domains(days=30):
    recent_plans = get_plans_since(days_ago=days)
    embeddings = [p.embedding for p in recent_plans]
    
    # Cluster embeddings to find topic clusters
    clusters = cluster_embeddings(embeddings, n_clusters=10)
    
    return [
        {
            "topic": get_cluster_label(cluster),
            "count": len(cluster.plans),
            "example_titles": cluster.plans[:3]
        }
        for cluster in clusters
    ]
```

## Implementation Plan

### Week 1: Core Infrastructure
- Add pgvector extension to PostgreSQL
- Create `plan_corpus` table with vector column
- Set up sentence-transformers model (serverless or Railway service)
- Build embedding generation pipeline

### Week 2: Indexing Existing Plans
- Batch process existing plans (embed title + summary)
- Insert into `plan_corpus` table
- Create similarity search index (ivfflat)
- Benchmark query performance

### Week 3: Search API
- Build `/api/plans/search` endpoint
- Add filtering (domain, min_similarity)
- Implement pagination
- Add response caching for common queries

### Week 4: UI Integration
- Add search bar to plan library
- Show "Plans like this" on plan detail page
- Add domain filters to search UI
- Display similarity scores visually

## Performance Optimization

**Indexing Strategy:**
- Use `ivfflat` index for sub-linear search time
- Trade-off: ~95% recall at 10x speed improvement
- Tune `lists` parameter based on corpus size (100 lists for 10K plans)

**Batch Embedding:**
```python
# Process 1000 plans at once
texts = [f"{p.title}\n{p.summary}" for p in plans]
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
```

**Caching:**
```python
# Cache frequent queries (e.g., "restaurant plan")
cache_key = f"search:{query_hash}:{limit}"
cached = redis.get(cache_key)
if cached:
    return json.loads(cached)

results = search_plans(query, limit)
redis.setex(cache_key, 3600, json.dumps(results))  # 1h TTL
```

## Cost Analysis

**Embedding Model:**
- Hosting: $20/month (Railway CPU service, always-on)
- Alternative: AWS Lambda (serverless, pay-per-request)

**pgvector:**
- Storage: ~1KB per plan (768-dim vector)
- 10K plans = 10MB (negligible)
- Index overhead: ~2x storage

**Query Cost:**
- Compute: Minimal (vector similarity is fast)
- No external API calls (model runs locally)

**Total:** ~$20-30/month for 10K-100K plans

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Embedding quality varies by domain | Fine-tune model on PlanExe corpus |
| Index size grows large | Shard by domain, archive old plans |
| Stale embeddings after plan edits | Re-embed on update, queue for batch processing |
| pgvector index rebuild is slow | Use incremental updates, rebuild offline |

## Success Metrics

- Search returns relevant results 80%+ of the time (user feedback)
- Average query time < 100ms (p95)
- 30%+ of users use "find similar plans" feature
- Few-shot plan generation quality improves (measured by ratings)

## Future Enhancements

- **Multi-modal embeddings** (include plan images, charts)
- **Temporal search** ("plans created in last 6 months")
- **User preference learning** (personalize search based on history)
- **Graph visualization** (show plan similarity network)

## References

- pgvector documentation: https://github.com/pgvector/pgvector
- sentence-transformers: https://www.sbert.net/
- Semantic search best practices: https://www.pinecone.io/learn/semantic-search/
