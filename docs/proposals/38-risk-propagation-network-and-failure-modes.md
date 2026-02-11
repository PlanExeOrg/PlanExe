# 38) Risk Propagation Network + Failure Mode Manifestation

## Pitch
Model how local risks propagate through dependencies to system-level failure, then simulate manifestation paths across 10,000 runs.

## Why
Teams often track risks independently, but major failures emerge from interacting risks across domains.

## Proposal
Create a risk propagation graph:
- nodes: risks, tasks, milestones
- edges: causal amplification links
- edge weights: propagation strength and delay

Simulate cascading failures:
- technical delays -> procurement impacts -> financing stress -> schedule collapse
- legal blockers -> redesign -> cost spiral

## Outputs
- top failure pathways by probability
- expected loss by pathway
- intervention points with highest leverage

## Integration
- Attach propagation score to plan ranking (works with ELO post-filtering)
- Trigger mitigation playbooks automatically for high-probability cascades

## Success metrics
- Reduced surprise compound failures
- Increased mitigation effectiveness vs baseline static risk logs
