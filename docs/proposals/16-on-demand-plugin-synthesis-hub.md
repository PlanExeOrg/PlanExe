# 16) On-Demand Plugin Synthesis + Plugin Hub for `run_plan_pipeline.py`

## Pitch
When the pipeline lacks a capability, PlanExe should auto-generate a focused plugin, validate it, and publish it to a shared Plugin Hub so future plans reuse it instead of re-solving the same gap.

## Problem
Today, `run_plan_pipeline.py` can only execute built-in stages. For novel domains, the plan quality drops if a required transformation or validator does not exist.

## Proposal
Add a **missing-capability loop**:
1. Detect unresolved task capability in pipeline stage execution.
2. Generate plugin spec (inputs/outputs/contracts).
3. Code plugin on demand in a sandbox.
4. Run tests + security checks.
5. If passed, register plugin in Plugin Hub.
6. Retry pipeline stage with new plugin.

## Where it plugs in
- Add `PluginResolver` before stage execution in `run_plan_pipeline.py`.
- Add `CapabilityNotFoundError` handling path that calls `PluginSynthesizer`.
- Add `PluginHubClient` for publish/fetch.

## Minimal interfaces
```python
class PluginSpec(BaseModel):
    capability: str
    stage_name: str
    input_schema: dict
    output_schema: dict
    constraints: list[str]

class PluginRecord(BaseModel):
    plugin_id: str
    capability: str
    version: str
    checksum: str
    trust_tier: str
```

## Data model additions
- `plugin_hub_plugins` (id, capability, version, checksum, owner, trust_tier, created_at)
- `plugin_hub_usage` (plugin_id, run_id, success, latency_ms, error_type)
- `plugin_hub_test_reports` (plugin_id, report_json, pass_rate)

## Rollout
- **Phase 1:** detect + suggest plugin (no auto-code)
- **Phase 2:** auto-code in dry-run mode (no publish)
- **Phase 3:** auto-code + gated publish + reuse

## Risks & controls
- Risk: low-quality generated plugins
  - Control: required contract tests + static analysis + sandbox execution
- Risk: supply-chain poisoning
  - Control: signed plugins + checksums + trust tiers + allowlist policy

## Success metrics
- % of failed stages recovered via plugin synthesis
- Plugin reuse rate across runs
- Median time-to-capability (first missing capability to successful rerun)
