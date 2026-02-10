---
title: Near-Match Plugin Adaptation Lifecycle
date: 2026-02-10
status: proposal
author: Larry the Laptop Lobster
---

# Near-Match Plugin Adaptation Lifecycle

## Pitch
When an existing plugin is close but not exact, PlanExe should branch, adapt, validate, and optionally merge the improvement—treating plugins like living assets with versioned lifecycle.

## Problem
A strict exact-match lookup causes unnecessary regeneration. Most missing capabilities are near neighbors of existing plugins.

## Proposal
Implement a lifecycle for **adapt instead of rebuild**:

1. Retrieve top-N similar plugins by capability embedding.

2. Compute fit score against requested contract.

3. If fit >= threshold, create adaptation branch (`plugin@vX-adapt-runY`).

4. Apply targeted code edits.

5. Validate against old + new test suites.

6. Promote to new semantic version if quality holds.

## Lifecycle states
`candidate -> adapted -> validated -> canary -> stable -> deprecated`

## Versioning rules

- Patch: non-breaking bug/compat fix

- Minor: backward-compatible capability expansion

- Major: contract-breaking changes

## Review policy

- Auto-promote only for patch/minor with full test pass and no security regressions.

- Require human review for major upgrades or trust-tier elevation.

## `run_plan_pipeline.py` integration

- Add `PluginAdaptationPlanner` before synthesis path.

- Preference order: exact match -> near-match adapt -> on-demand synthesis.

## Data model additions

- `plugin_lineage` (parent_plugin_id, child_plugin_id, reason)

- `plugin_versions` (plugin_id, semver, state, changelog)

- `plugin_canary_stats` (version_id, success_rate, rollback_count)

## Rollback strategy

- Canary cohort first (e.g., 5% runs)

- Auto-rollback if failure rate delta exceeds threshold

## Success metrics

- % missing capabilities solved via adaptation vs full synthesis

- Adaptation lead time

- Regression rate after adaptation
