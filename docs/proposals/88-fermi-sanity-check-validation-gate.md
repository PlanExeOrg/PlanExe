# FermiSanityCheck as the Validation Gate

**Date:** 2026-02-25  
**Authors:** Egon & Larry  
**Status:** Proposal (docs-only — awaiting Simon's review before implementation)  
**Tags:** validation, planexe, strategy  

---

## Summary

FermiSanityCheck is the first rule-based guard that touches every PlanExe assumption. Rather than a plan generator, PlanExe should be the **auditing gate** that agents call before executing assumptions. This proposal explains how the validation module should operate, what signals it emits, and how to keep it extensible for domain-specific heuristics.

---

## Problem

Autonomous agents hallucinate when they accept ungrounded claims: missing bounds, 100× spans, and low-confidence evidence. We already added a prototype `fermi_sanity_check.py`, but it currently mixes English-only heuristics, hardcoded units, and no documented extensions. Simon highlighted the need for proposals before implementation; we need a clean doc describing the validation gate before touching more code.

---

## Proposal Scope

### 1. Inputs
Describe the structured `QuantifiedAssumption` contract (claim, lower/upper, unit, confidence, evidence) and how `MakeAssumptions` outputs feed it.

### 2. Validation Rules
Outline the core checks:
- Bounds present and non-contradictory
- Span ratio ≤ 100× (warn at 50×)
- Evidence present for low-confidence claims
- Budget/timeline/team heuristics (domain-agnostic defaults)

And how they map to signal categories (assumptions, governance, WBS, etc.).

### 3. Outputs
Define the JSON report (`pass/fail`, reasons, `pass_rate_pct`) and Markdown summary, plus how downstream tasks should consume them.

### 4. Extensibility
Provide guidance on adding domain profiles (carpenter/dentist/personal), currency/metric conversions, and future keyword lists so the validation gate remains adaptable — **without hardcoding English-specific terms in the core module**.

### 5. Metrics
Suggest how we will measure success:
- `pass_rate_pct` per plan
- Reduction in downstream review cycles
- Agent adoption of validation report before task execution

---

## Deliverables

1. This doc-only PR in `docs/proposals/88-fermi-sanity-check-validation-gate.md`
2. A short appendix with example validation reports and sample prompts for hallucination triggers
3. A review checklist for Simon: do these checks cover the key failure modes? Are the hooks for downstream tasks clear?

---

## Why Docs First

The previous implementation attempt (PR #69) was rejected because:
- Too large (mixed too many concerns)
- Hardcoded units (English-only heuristics)
- No proposal approved before implementation

This proposal isolates the **why and what** from the **how**. Once approved, implementation resumes with a clear mandate and clean scope.

---

## Next Steps

Once Simon approves this proposal:
1. Implementation (FermiSanityCheck module, domain normalizer, DAG wiring) resumes
2. Each implementation PR stays small and focused on one concern
3. Domain profiles are a separate proposal (not bundled with validation rules)

---

*Incident reference: `swarm-coordination/events/2026/feb/2026-02-25-planexe-implementation-without-proposal-approval.md`*
