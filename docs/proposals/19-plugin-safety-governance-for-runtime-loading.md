# Safety + Governance for Runtime Plugin Loading

## Pitch
If PlanExe can code and load plugins dynamically, governance must be first-class: trust tiers, policy gates, signed artifacts, and audit trails before execution in `run_plan_pipeline.py`.

## Problem
Dynamic plugin ecosystems introduce security, compliance, and reliability risks unless runtime loading is policy-controlled.

## Proposal
Introduce **policy-governed runtime loading**:

- Every plugin has trust tier (`experimental`, `verified`, `trusted`)

- Stage-level policy defines allowed tiers

- Load only signed plugins with valid checksum and provenance

- Log full execution trace for every plugin invocation

## Runtime checks in `run_plan_pipeline.py`
Before plugin execution:

1. Verify signature + checksum

2. Validate declared capabilities vs requested stage contract

3. Enforce policy (tier + owner + allowlist)

4. Enforce resource limits (CPU/memory/timeouts)

## Governance controls

- Quarantine mode for new plugins

- Kill switch per plugin/version

- Mandatory re-validation on dependency updates

- Security scan (SAST + dependency CVE check)

## Data model additions

- `plugin_policy` (stage_name, allowed_tiers, owners, max_runtime_ms)

- `plugin_audit_log` (run_id, plugin_id, version, decision, reason)

- `plugin_security_reports` (plugin_id, scan_result, vuln_count)

## Incident response

- One-click disable plugin/version

- Backfill affected runs via audit lookup

- Auto-notify maintainers on abnormal failure spikes

## Success metrics

- Blocked unsafe plugin load attempts

- Mean time to contain plugin incident

- % plugin executions with complete provenance
