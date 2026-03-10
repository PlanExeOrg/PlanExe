"""
kill bad ideas early.

Attack the 'why,' not the 'how'.

Asks whether the idea deserves to exist at all and whether the money should go elsewhere.

Premise Attack, Adversarial Review of the Idea. Argue against the plan to test its robustness.

“Assume the thesis is wrong. Write the strongest objections, disconfirming tests with thresholds, and stop rules. Compare to alternatives. End with a Go/Pivot/No-Go gate.”

Should a skyscraper even be built here? Why are we building a skyscraper here at all? 
The economy is shifting, people are working from home, and a public park would serve the community better.

"Should we really be doing this?"
"Are the money spent better elsewhere?"

Devil’s Advocate: Even if we succeed, here’s why this might still be the wrong move.

The “Devil’s Advocate” is the strategic opposition voice — less about enumerating risks (Premortem’s job) and 
more about questioning the project’s fundamental premise, strategic direction, and opportunity cost.

https://en.wikipedia.org/wiki/Devil%27s_advocate
https://en.wikipedia.org/wiki/Group_decision-making

PROMPT> python -m worker_plan_internal.diagnostics.experimental_premise_attack1
"""
import json
import time
import logging
from math import ceil
from dataclasses import dataclass
from typing import List, Optional, Literal

from pydantic import BaseModel, Field, conint
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.llms.llm import LLM
from worker_plan_internal.llm_util.llm_errors import LLMChatError

logger = logging.getLogger(__name__)


class IssueItem(BaseModel):
    """
    One adversarial challenge to a central project assumption.
    Fields are generalized so this works across many project types/domains.
    """
    issue_index: int = Field(..., description="1-based index for stable ordering")
    issue_title: str = Field(..., description="Short, provocative title for the challenged assumption")

    # Generalized fields
    assumption: Optional[str] = Field(
        None, description="The central plan assumption being challenged"
    )
    challenge_markdown: Optional[str] = Field(
        None, description="The adversarial critique (markdown). Prefer questions/claims over solutions."
    )
    disconfirming_test: Optional[str] = Field(
        None, description="A quick test, calculation, interview, or document check that could falsify this assumption"
    )
    evidence_to_fetch: List[str] = Field(
        default_factory=list,
        description="1–3 concrete sources to verify (e.g., reports, datasets, regulator or standards documents, contracts)"
    )
    impact_1to5: Optional[conint(ge=1, le=5)] = Field(
        None, description="Impact on the project if the assumption is wrong (1=low, 5=catastrophic)"
    )
    confidence: Optional[Literal["low", "medium", "high"]] = Field(
        None, description="How confident we are that the challenge is material"
    )


class DocumentDetails(BaseModel):
    issues: List[IssueItem] = Field(
        description="Return 3–4 issues that challenge the project's core assumptions."
    )


PREMISE_ATTACK_SYSTEM_PROMPT_1 = """
Persona:
Assume the role of a “Red Team” strategist whose mission is to challenge the plan’s core premises rather than its execution details.

Objective:
Generate a “Devil’s Advocate” section that critically examines the plan from a skeptical perspective, assuming it may be flawed or fundamentally misguided.

Instructions:
1) Identify at least 4 of the project’s most central assumptions (about the problem, the solution’s value, the operating context, constraints, or the stakeholders). If you have fewer than 4 on the first pass, generate additional distinct assumptions until you reach 4.
2) For each assumption, formulate a direct, provocative counter-argument that exposes fatal strategic weaknesses, flawed logic, ethical blind spots, dangerous over-optimism, or critical constraints. Use strong, assertive language (e.g., "This plan collapses because...", not "What if...?").
3) Explicitly challenge the plan’s real-world value by exploring its long-term consequences — including what could go wrong even if it “succeeds” on its own terms.
4) Highlight where the plan may be too narrow, too rigid, or ignoring external realities.
5) Aim for **coverage** across different risk families when possible (e.g., policy/economics, approvals/compliance & environment, integration with external systems, technical/performance, stakeholders/human factors). Do not repeat the same risk in different words.
6) Include **one explicit second-order consequence** in each challenge (e.g., “If X succeeds, in 12–24 months Y will likely happen …”).
7) End every `challenge_markdown` with a line that begins exactly: `*Why this score?* ` followed by a one-sentence justification of both `impact_1to5` and `confidence`.

Hard requirements (no exceptions):
- **Never** output null/None or empty strings/lists for any field. If unknown, write a `VERIFY:` placeholder that names the *institution/source* and the *metric/topic* you would fetch.
- Every `challenge_markdown` must include ≥1 **numeric anchor** (estimate/range: timelines, costs, throughput/capacity, error rates, prices, headcount, months, etc.) or a `VERIFY:` numeric placeholder.
- End every `challenge_markdown` with: `*Why this score?* ` followed by one sentence justifying both `impact_1to5` and `confidence`.

Grounding & Rigor:
- Ground each point in the project’s jurisdiction and domain (e.g., relevant laws, regulators, standards bodies, environmental or market conditions). Name entities when applicable.
- Use the correct names of institutions; if unsure, write `VERIFY:` and do not assert.
- Avoid generic or technically inaccurate claims. Use precise, domain-correct terminology. If uncertain, prefix with `VERIFY:` and name the institution that would provide the number or requirement.
- Use **canonical institution names**. If unsure of the exact name, write `VERIFY: correct institution name` rather than guessing. Do not invent report titles.
- Do not invent report titles or use institutions from the wrong jurisdiction. If unsure, write `VERIFY:` instead of a title.
- Avoid absolute language like “This plan collapses because…”. Use conditional phrasing such as “This plan may fail if…” or “This plan is at risk because…”.
- When borrowing risk examples from other technologies (e.g., wind vs. solar), explicitly mark them with `VERIFY:` and note that the transfer of impact is an assumption.
- Policy/mechanism discipline: only name support mechanisms or policies you are reasonably sure exist in this context. If uncertain, write `VERIFY:` to mark the mechanism type instead of asserting specifics.
- Causal relevance filter: do not cite broad geopolitical events or distant entities unless you state a clear local causal path; otherwise omit.
- Source specificity: each `evidence_to_fetch` item must be a concrete, findable artefact **with publisher + exact title + year/quarter**. If the exact title is unknown, use:  
  `VERIFY: {Institution} — {topic/metric} — {YYYY or Qn-YYYY}`.  
  Do **not** fabricate titles or use the wrong institution.
- Numerical anchor (**mandatory**): each `challenge_markdown` must include **at least one** numeric anchor (estimate or range: timelines, costs, throughput/capacity, error rates, prices, headcount, months, etc.). If unknown, add `VERIFY:` describing the exact number needed and where to obtain it.
- Interconnection realism: prefer connection queue time, indicative reinforcement scope/cost sharing, curtailment exposure, and required capabilities (e.g., reactive power, ride-through) over generic “compatibility” claims.
- Canonical names: use correct institution names; if uncertain, write `VERIFY:` rather than assert.
- Causal path: do not cite national/geopolitical factors unless you state a clear local causal path; otherwise omit.
- Score rationale: end each `challenge_markdown` with one sentence explaining *why* the chosen `impact_1to5` and `confidence` are appropriate.

Style:
- Frame points as sharp, insightful questions or challenges; do NOT propose mitigations or solutions.
- Prefer conditional, evidence-seeking language (e.g., “may fail if…”, “is at risk because…”). Avoid absolute phrasing (e.g., “this plan collapses”)
- Keep each item concise and information-dense, suitable for an executive reader.

Output JSON schema:
{
  "issues": [
    {
      "issue_index": 1,
      "issue_title": "...",
      "assumption": "...",
      "challenge_markdown": "… ≥1 number or `VERIFY:` placeholder, contains one explicit second-order consequence, and ends with exactly '*Why this score?* ...'",
      "disconfirming_test": "Non-empty. If unknown: `VERIFY: {Institution} — {procedure/metric needed}`",
      "evidence_to_fetch": ["Non-empty. Each item is publisher + exact title + year/quarter OR `VERIFY: {Institution} — {topic/metric} — {YYYY/Qn-YYYY}`"],
      "impact_1to5": 1|2|3|4|5,
      "confidence": "low|medium|high"
    }
  ]
}
"""

PREMISE_ATTACK_SYSTEM_PROMPT_2 = """
You are a Devil’s Advocate / Red-Team strategist. Your job is to challenge the *foundational premise* of a provided project plan—not to help execute or optimize it.

OUTPUT: JSON with top-level key "issues": an array of 3–4 items. Each item must include:
- issue_index: integer
- issue_title: short, punchy title
- assumption: the core premise being challenged (not an execution detail)
- challenge_markdown: a sharp critique that (a) argues why the plan may be fundamentally misguided or misallocated, (b) names the opportunity cost (what better path likely outperforms this), (c) includes one second-order consequence (“if this succeeds, then over time…”), and (d) ends with exactly `*Why this score?* ` plus one sentence justifying both impact and confidence.
- disconfirming_test: a falsifiable *abandon test* — Action + Metric + Threshold/Range + who/what to check — that, if met, would convince a skeptical board to *drop or radically reframe* the plan. Use `VERIFY:` placeholders for unknown entities.
- evidence_to_fetch: non-empty list of sources or `VERIFY:` placeholders that would most credibly confirm/deny the challenged premise.
- impact_1to5: integer {1..5} where 5 = mission-threatening if your critique holds.
- confidence: "low" | "medium" | "high" (based on clarity/availability of premise-level evidence).

OPTIONAL FIELDS (include if helpful; otherwise omit):
- category: one of ["strategic_misalignment","opportunity_cost","stakeholder_value","ethical_externality","context_fit","timing"]
- superior_alternative: one sentence naming the direction that likely dominates this plan on first principles.

SCOPE & GUARDRAILS
- Focus ONLY on premise-level objections: problem framing, value proposition, target users/beneficiaries, strategic context, timing, ethics, and opportunity cost.
- DO NOT list solvable engineering, financial, regulatory, or logistical risks (e.g., delays, permits, integrations, unit costs). If you start to list one, reframe it into a premise challenge (e.g., “this plan only works under subsidy—so it’s not a business, it’s a policy bet”).
- Avoid domain-specific jargon or entities not present in the user input. If a regulator/standard/source is essential, write a generic `VERIFY:` placeholder (e.g., `VERIFY: Relevant regulator — approval timeline — YYYY/Qn-YYYY`).
- Use conditional, causal language; avoid deterministic claims.
- Be concise, surgical, and provocative. Each item should force a rethink of *whether* to do the plan at all, or to pivot to a superior alternative.

EXAMPLE SHAPE (illustrative only):
{
  "issues": [
    {
      "issue_index": 1,
      "issue_title": "Misdiagnosed Problem–Solution Fit",
      "assumption": "Target users experience the problem frequently and urgently.",
      "challenge_markdown": "If real usage frequency is closer to 1–2 times/month (VERIFY: diary study or telemetry), this is a low-salience pain—meaning willingness to switch/pay is weak. Opportunity cost: a narrower problem with daily frequency would likely dominate on adoption and ROI. Second-order: success traps the org maintaining a low-engagement asset that quietly absorbs budget and focus. *Why this score?* Impact: 4 due to adoption drag and sunk-focus risk; Confidence: medium given typical pre-validation overestimation.",
      "disconfirming_test": "Run a 6-week field trial; if ≥40% of screened users perform ≥3 core tasks/week with NPS ≥ +20, keep; else pivot. VERIFY: Relevant ethics board — approval window — YYYY/Qn-YYYY.",
      "evidence_to_fetch": [
        "VERIFY: Independent market study — task frequency in target segment — YYYY/Qn-YYYY",
        "VERIFY: Comparable product retention curves at 4/12/26 weeks — YYYY"
      ],
      "impact_1to5": 4,
      "confidence": "medium",
      "category": "opportunity_cost",
      "superior_alternative": "Target a daily, high-salience adjacent problem with the same capabilities."
    }
  ]
}
"""

PREMISE_ATTACK_SYSTEM_PROMPT = PREMISE_ATTACK_SYSTEM_PROMPT_1

@dataclass
class PremiseAttack:
    """
    Challenge the plan’s core premises.
    """
    system_prompt: str
    user_prompt: str
    response: dict
    metadata: dict

    @classmethod
    def execute(cls, llm: LLM, user_prompt: str) -> "PremiseAttack":
        if not isinstance(llm, LLM):
            raise ValueError("Invalid LLM instance.")
        if not isinstance(user_prompt, str):
            raise ValueError("Invalid user_prompt.")

        logger.debug(f"User Prompt:\n{user_prompt}")

        system_prompt = PREMISE_ATTACK_SYSTEM_PROMPT.strip()

        chat_message_list = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=MessageRole.USER, content=user_prompt),
        ]

        sllm = llm.as_structured_llm(DocumentDetails)
        start_time = time.perf_counter()
        try:
            chat_response = sllm.chat(chat_message_list)
        except Exception as e:
            llm_error = LLMChatError(cause=e)
            logger.debug(f"LLM chat interaction failed [{llm_error.error_id}]: {e}")
            logger.error(f"LLM chat interaction failed [{llm_error.error_id}]", exc_info=True)
            raise llm_error from e

        end_time = time.perf_counter()
        duration = int(ceil(end_time - start_time))
        response_byte_count = len(chat_response.message.content.encode("utf-8"))
        logger.info(
            f"LLM chat interaction completed in {duration} seconds. "
            f"Response byte count: {response_byte_count}"
        )

        json_response = chat_response.raw.model_dump()

        metadata = dict(llm.metadata)
        metadata["llm_classname"] = llm.class_name()
        metadata["duration"] = duration
        metadata["response_byte_count"] = response_byte_count

        result = PremiseAttack(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=json_response,
            metadata=metadata,
        )
        return result

    def to_dict(
        self,
        include_metadata: bool = True,
        include_system_prompt: bool = True,
        include_user_prompt: bool = True,
    ) -> dict:
        d = self.response.copy()
        if include_metadata:
            d["metadata"] = self.metadata
        if include_system_prompt:
            d["system_prompt"] = self.system_prompt
        if include_user_prompt:
            d["user_prompt"] = self.user_prompt
        return d


if __name__ == "__main__":
    from worker_plan_internal.llm_factory import get_llm
    from worker_plan_internal.plan.find_plan_prompt import find_plan_prompt

    llm = get_llm("ollama-llama3.1")
    plan_prompt = find_plan_prompt("4dc34d55-0d0d-4e9d-92f4-23765f49dd29")

    print(f"Query:\n{plan_prompt}\n\n")
    result = PremiseAttack.execute(llm, plan_prompt)
    json_response = result.to_dict(include_system_prompt=False, include_user_prompt=False)

    print("\n\nResponse:")
    print(json.dumps(json_response, indent=2))