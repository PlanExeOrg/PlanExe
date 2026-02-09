---
title: Distributed Plan Execution - Worker Pool Parallelism
date: 2026-02-09
status: proposal
author: Larry (OpenClaw)
---

# Distributed Plan Execution - Worker Pool Parallelism

## Overview

PlanExe's plan generation pipeline currently runs sequentially on a single worker. For complex, multi-stage plans (research → outline → expand → review), this creates bottlenecks and wastes compute when stages could run in parallel.

This proposal introduces a **distributed execution model** with worker pool parallelism and DAG-based scheduling for compute-heavy plan stages.

## Problem

- Single-threaded execution = slow generation for complex plans
- Wasted compute: Outline stage could start while research continues
- No horizontal scaling: Can't throw more workers at the problem
- Railway infrastructure supports multi-worker deployments but pipeline doesn't use it

## Proposed Solution

### Architecture

```
┌──────────────────────┐
│  Plan Request        │
│  (HTTP API)          │
└──────────┬───────────┘
           │
           v
┌──────────────────────┐
│  DAG Scheduler       │  ← Determines stage dependencies
│  (Coordinator)       │     and dispatches to workers
└──────────┬───────────┘
           │
     ┌─────┴─────┬─────────┬─────────┐
     v           v         v         v
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│Worker 1 │ │Worker 2 │ │Worker 3 │ │Worker N │
│(Research)│ │(Outline)│ │(Expand) │ │(Review) │
└─────────┘ └─────────┘ └─────────┘ └─────────┘
     │           │         │         │
     └───────────┴─────────┴─────────┘
                   │
                   v
           ┌───────────────┐
           │  Redis Queue  │  ← Job state + results
           └───────────────┘
```

### Stage Dependency DAG

```python
# Example DAG for standard business plan
plan_dag = {
    "research": {
        "depends_on": [],
        "parallelizable": True,
        "subtasks": ["market_research", "competitor_analysis", "regulatory_research"]
    },
    "outline": {
        "depends_on": ["research"],
        "parallelizable": False
    },
    "expand_sections": {
        "depends_on": ["outline"],
        "parallelizable": True,
        "subtasks": ["exec_summary", "market_analysis", "operations", "financial"]
    },
    "review": {
        "depends_on": ["expand_sections"],
        "parallelizable": False
    },
    "format": {
        "depends_on": ["review"],
        "parallelizable": False
    }
}
```

### Worker Pool Management

**Railway Configuration:**
```yaml
# railway.toml
[workers]
  plan_worker:
    build:
      dockerfile: Dockerfile.worker
    replicas: 5  # Scale based on load
    env:
      REDIS_URL: ${REDIS_URL}
      WORKER_POOL: plan_execution
```

**Task Queue (Celery-style):**
```python
from celery import Celery

app = Celery('planexe', broker='redis://localhost:6379/0')

@app.task(name='stage.research')
def execute_research_stage(plan_id, prompt_context):
    # Run research subtasks in parallel
    results = group([
        research_market.s(plan_id, prompt_context),
        research_competitors.s(plan_id, prompt_context),
        research_regulatory.s(plan_id, prompt_context)
    ])()
    return results.get()

@app.task(name='stage.outline')
def execute_outline_stage(plan_id, research_results):
    # Depends on research completion
    return generate_outline(plan_id, research_results)
```

## Implementation Plan

### Phase 1: DAG Scheduler (Week 1-2)
- Define stage dependency graph schema (YAML config)
- Build coordinator service that parses DAG and dispatches tasks
- Add Redis for job state management
- Single worker proof-of-concept

### Phase 2: Worker Pool (Week 3)
- Deploy 3-5 workers on Railway
- Implement task routing and load balancing
- Add retry logic and failure handling
- Monitor queue depth and worker utilization

### Phase 3: Parallel Stages (Week 4)
- Enable parallel execution for research subtasks
- Enable parallel execution for section expansion
- Add progress reporting (% complete across all workers)
- Optimize stage chunking for latency

### Phase 4: Auto-Scaling (Week 5+)
- Dynamic worker scaling based on queue depth
- Cost optimization (scale down during off-hours)
- Priority queues (premium users get dedicated workers)

## Benefits

- **3-5x faster plan generation** for complex plans
- **Horizontal scaling** - add more workers as load increases
- **Better resource utilization** - multiple stages run concurrently
- **Resilience** - worker failure doesn't kill entire plan generation
- **Cost efficiency** - pay for compute only when queue is deep

## Technical Stack

- **Task Queue:** Celery + Redis (battle-tested, Python-native)
- **DAG Engine:** Custom lightweight scheduler (simpler than Airflow for our use case)
- **Worker Runtime:** Docker containers on Railway
- **State Storage:** Redis (job metadata) + PostgreSQL (completed plans)

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Added complexity | Start with simple DAG, expand gradually |
| Redis becomes bottleneck | Use Redis cluster, cache subtask results |
| Worker coordination overhead | Keep DAG shallow (max 5 stages), minimize inter-worker communication |
| Cost increase | Monitor worker utilization, scale down aggressively |
| Debugging harder | Centralized logging (Sentry), trace IDs across workers |

## Success Metrics

- Average plan generation time decreases by 50%+
- Worker CPU utilization stays 60-80% (not idle, not maxed)
- Task retry rate < 2% (most jobs succeed first try)
- P95 latency under 10 minutes for standard business plan

## Future Enhancements

- **GPU workers** for vision/multimodal stages
- **Speculative execution** (start likely next stage before deps finish)
- **Agent-specific worker pools** (specialized workers for finance plans vs. tech plans)

## References

- Celery documentation: https://docs.celeryq.dev/
- Railway multi-service deploys: https://docs.railway.app/
- DAG scheduling patterns: Apache Airflow, Prefect, Temporal
