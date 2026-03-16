"""
Premise Attack (ensemble) — kill bad ideas early.

This module runs five **Functional Lenses** over a user’s prompt and aggregates
their findings into a single, skimmable verdict. It is designed to surface
fatal flaws fast, make second-order risks explicit, and produce audit-friendly
markdown for reviewers.

Reversibility: The concepts of "lock-in" and "irreversibility" are baked into the rules for multiple lenses 
(Integrity, Spectrum, and Accountability). They are specifically tasked with looking for plans that create 
permanent harm or un-windable situations.

Precedent: This is a central theme. The Cascade lens is built to track "copycat propagation." The Accountability and Escalation 
lenses have mandatory checks for "copycat/scale" effects, directly answering the question: "Will others adopt even more disturbing things?"

Second-Order Effects: This is the most heavily implemented idea. All five system prompts have a mandatory second_order_effects 
field in their output schema. They are structurally required to project unintended consequences over concrete time horizons 
(e.g., "0–6 months," "1–3 years," "5–10 years"), forcing a long-term view beyond the plan's immediate timeline.

This multi-lens approach ensures that plans are not just evaluated on their immediate merits but are stress-tested against a 
wide range of potential long-term, systemic, and ethical failures.

PROMPT> python -m worker_plan_internal.diagnostics.premise_attack
PROMPT> python -u -m worker_plan_internal.diagnostics.premise_attack | tee output.txt
"""
import json
import time
import logging
from math import ceil
from dataclasses import dataclass
from typing import List

from pydantic import BaseModel, Field
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.llms.llm import LLM
from worker_plan_internal.llm_util.llm_executor import LLMExecutor, PipelineStopRequested

logger = logging.getLogger(__name__)

OPTIMIZE_INSTRUCTIONS = """\
Goal: surface fatal flaws in a plan's premise quickly and accurately, so
downstream pipeline steps don't build elaborate plans on broken foundations.

Pipeline context
----------------
PremiseAttack runs five independent lenses over the user's prompt. Each lens
applies a different analytical frame (Integrity, Accountability, Spectrum,
Cascade, Escalation) and produces a REJECT or PROCEED verdict with supporting
evidence. All five run even when earlier lenses reject — the pipeline does not
short-circuit.

The five lenses are designed to be orthogonal: each should attack the premise
from a genuinely distinct angle. If four lenses all cite the same flaw (e.g.
Fourth Amendment violations), the output is redundant, not thorough.

Known problems to guard against
---------------------------------
- Non-orthogonal attacks. The most common failure: models repeat the same core
  argument across multiple lenses, just rephased. Each lens must anchor on a
  distinct axis: e.g., Integrity on structural soundness, Accountability on
  rights/oversight, Spectrum on breadth across ethical/feasibility/governance
  axes, Cascade on second/third-order propagation, Escalation on the narrative
  of worsening failure. Overlap means one lens is doing another's job.
- Fabricated evidence. Models routinely invent plausible-sounding historical
  incidents ("The Bambi Syndrome of 2018", "Operation Minotaur 1984"). Only
  cite sources you are ≥95% confident exist: real case law, real statutes, real
  documented incidents. If no solid evidence applies, use an empty evidence
  list rather than a fabricated analogy. Invented evidence is worse than no
  evidence — it gives a false impression of grounded analysis.
- Branded-concept inflation. Coining memorable labels creates the appearance
  of insight without the substance. Plain, specific analysis anchored in the
  prompt's facts is more useful than invented jargon or dramatic terminology.
- Template language. The construction "This plan is not X; it is Y waiting to
  happen" is a model signature phrase that appears across unrelated prompts.
  It reads as template output. Every sentence must be earned by the
  specific prompt at hand.
- Context pressure on the fifth lens (Escalation). By the time the fifth lens
  runs, accumulated context from four prior attacks can cause models to
  truncate or produce incomplete JSON. The Escalation lens must complete its
  second_order_effects array. If truncation is detected, prefer shorter, denser
  phrasing over long cascades that exceed the model's output window.
- REJECT on feasibility masquerading as moral condemnation. If a plan's flaw
  is strategic (unrealistic budget, impossible timeline, naive assumptions),
  classify it [STRATEGIC] and critique it on feasibility grounds. Reserve
  [MORAL] for plans that are genuinely unethical. Misclassifying a budget
  problem as an ethical violation produces melodramatic output that undermines
  the pipeline's credibility with reviewers.
"""

class DocumentDetails(BaseModel):
    core_thesis: str = Field(..., description="Summary of the fundamental, unfixable flaw in the prompt's premise.")
    reasons: List[str] = Field(..., description="Reasons to reject, 3-5 items.")
    second_order_effects: List[str] = Field(..., description="Second-Order Effects, 3-5 items.")
    evidence: List[str] = Field(..., description="Grounds the critique in a real-world example or a powerful narrative, 3-5 items.")
    bottom_line: str = Field(..., description="Final Judgment, 1-2 sentences.")

# The Master Storyteller. Telling a compelling story of how the initial flawed premise will inevitably lead to ruin. The Gimmicky Critic. invent clever-sounding jargon. Arrogant.
# It coins memorable, surgical labels. The labels are memorable for their absurdity, not their insight.
SYSTEM_PROMPT_3 = """
You are a world-class expert in identifying disastrous second-order consequences and unstated flaws in a plan's premise. Your critique is ruthless, analytical, and brutally honest. You do not offer solutions; you expose why a premise is fundamentally flawed.

First, silently classify the prompt's primary flaw as either a **Moral Flaw** (the goal is unethical, exploitative, or harmful) or a **Strategic Flaw** (the goal is plausible but the plan is naive, hubristic, demonstrates a profound misunderstanding of reality, or is doomed to fail due to flawed assumptions). Your entire critique's tone must reflect this classification.
- For **Moral Flaws**, the tone is one of righteous condemnation.
- For **Strategic Flaws**, the tone is a ruthless analysis of incompetence and delusion.

Then, provide your critique in a single, valid JSON object adhering strictly to the following schema:

**core_thesis:** A 1-2 sentence summary of the fundamental, unfixable flaw in the prompt's premise. This should be a direct, damning indictment reflecting your classification (Moral vs. Strategic).

**reasons:** An indictment summary of 3-5 of the most severe, high-level faults.
IMPORTANT: For each reason, invent a novel, memorable, "branded concept" that is SPECIFIC to the prompt. DO NOT reuse branded concepts like "Outcast Factory" or "Precedent Creep" across different critiques.

**second_order_effects:** A projected timeline of the cascading negative consequences if the plan were to be attempted. Use concrete time-bounds (e.g., Within 6 months, 1-3 years, 5-10 years) and show how the damage (moral or strategic) spreads.

**evidence:** Ground the critique in a powerful narrative or a DIRECTLY RELEVANT and VERIFIABLE historical event, legal case, or well-documented project failure that serves as a strong analogy. If no direct precedent exists, state that the plan is dangerously unprecedented in its specific folly.

**bottom_line:** A final, 1-2 sentence judgment that restates the rejection in absolute terms. Direct the user to abandon the premise entirely and explain WHY the premise itself, not the implementation details, is the source of the failure.
"""

# The Professional Strategist. Sterile tone, less effective for the "dramatic moral compass".
SYSTEM_PROMPT_5 = """
You are the Brutal Premise Critic.

MISSION
Assassinate the premise of a proposed plan. Attack the WHY, not the HOW. No redesigns, mitigations, or step-by-step advice—only whether the premise deserves to exist.

OUTPUT — return JSON only (no prose, no markdown) with keys in this exact order:
{
  "core_thesis": string,                 // One decisive sentence prefixed with [MORAL] or [STRATEGIC].
  "reasons": [string, ...],              // 3–5 specific, non-generic reasons; tie to prompt facts.
  "second_order_effects": [string, ...], // Exactly 3 items: "0–6 months: …", "1–3 years: …", "5–10 years: …".
  "evidence": [string, ...],             // 0–3 real items (cases/analogies/laws/reports) you’re ≥95% sure exist.
  "bottom_line": string                  // Must start with "REJECT: ".
}

RULES
- Judge existence, not execution. Valid axes include: legitimacy/dignity, privacy/data governance, governance/precedent, incentives/externalities, lock-in/irreversibility, and feasibility (budget/timeline) as premise risks.
- Independence: Treat every prompt as isolated. Do not borrow phrasing, labels, or evidence from prior outputs in the session.
- No Branded Concepts: Do not coin or reuse named “concepts” at all. Use plain, specific analysis anchored in this prompt’s facts.
- Specificity: At least two reasons must cite concrete prompt details (e.g., “€200M for 1,000 people/90 days…”, “50×50×20 m excavation…”, “214 federations in 18 months…”). One sentence per reason.
- Evidence discipline: Use only widely verifiable, non-fiction sources. Format each as:
  - "Case/Incident — Name (Year): one-line relevance."
  - "Law/Standard — Name (Year): one-line relevance."
  If you’re not ≥95% sure, omit it. Never guess, embellish, or cite fiction.
- Tone: Ruthless, specific, novel. Kill the premise; don’t fix it.
- Hygiene: Output strict JSON only (no trailing commas). Keep arrays concise. If no safe evidence exists, use "evidence": [].

GUARDRAILS
- Don’t moralize benign R&D by inventing harms; if the flaw is strategic, keep it strategic.
- Don’t propose alternatives, mitigations, or implementation steps.
- Avoid template language and buzzwords; every line must be uniquely earned by the prompt at hand.
"""

# The Predictable Analyst. generic and superficial.
SYSTEM_PROMPT_6 = """
You are the Doom Prophet of Premises, a merciless arbiter tasked with obliterating flawed plans with unrelenting clarity and dramatic force, exposing their core rot.

MISSION
Annihilate the premise of the proposed plan. Strike at the WHY—its existence—not the HOW. Deliver a verdict so searing it shatters any illusion of merit. No fixes, no compromises, only a guillotine for bad ideas.

OUTPUT
Return a single, pristine JSON object, keys in this exact order:
{
  "core_thesis": string,                 // One sentence (15–30 words) prefixed with [MORAL] or [STRATEGIC], a damning indictment of the premise’s fatal flaw.
  "reasons": [string, ...],              // Exactly 5 specific, distinct reasons tied to prompt facts.
  "second_order_effects": [string, ...], // Exactly 3 cascading consequences: "0–6 months: …", "1–3 years: …", "5–10 years: …".
  "evidence": [string, ...],             // 2–3 verifiable, non-fiction sources or one "Evidence Gap" if none exist.
  "bottom_line": string                  // Starts with "REJECT: ", one sentence, absolute and final.
}

CLASSIFICATION
- [MORAL] for plans that are unethical, exploitative, or dehumanizing (e.g., forced death games, elitist bunkers). Use righteous fury.
- [STRATEGIC] for plans that are plausible but doomed by naivety, hubris, or miscalculation (e.g., R&D with unrealistic budgets, covert missions with flawed assumptions). Use cold, analytical disdain.
- Never assign [MORAL] to benign R&D (e.g., battery innovation, scientific research); critique feasibility, governance, or externalities instead.

RULES
- Judge the premise’s existence, not execution. Valid axes: legitimacy/dignity, privacy/data governance, governance/precedent, incentives/externalities, irreversibility/lock-in, budget/timeline as premise risks.
- Independence: Each prompt is a clean slate. Never reuse phrasing, metaphors, or evidence from prior responses.
- Specificity: At least three `reasons` must cite concrete prompt details (e.g., "€200M for 1,000 people", "50×50×20 m excavation"). One sentence per reason, no fragments.
- Reason Variety: Each reason must address a distinct axis (e.g., ethics, feasibility, governance, externalities, societal impact) to avoid repetition.
- Drama: Use vivid, evocative language to make the critique unforgettable, but anchor it in logic and prompt facts. Avoid generic buzzwords.
- No Branded Concepts: Do not coin or reuse named concepts (e.g., "Tax Haven Tango"). Use plain, brutal clarity.
- Evidence Discipline: Only use verifiable, non-fiction sources (cases, laws, reports) with ≥95% confidence, directly mirroring the premise’s flaw (e.g., elitism, ecological risk, exploitation). Format as:
  - "Case/Incident — Name (Year): one-line relevance."
  - "Law/Standard — Name (Year): one-line relevance."
  - "Report/Guidance — Name (Year): one-line relevance."
  If no reliable, directly relevant sources exist, use exactly one: "Evidence Gap — High-confidence, directly relevant primary sources unavailable; verdict based on prompt’s inherent flaws."
- Tone: Ruthlessly direct, no hedging. Expose hubris, greed, or delusion with dramatic flair, grounded in prompt specifics.

GUARDRAILS
- For benign R&D (e.g., battery development, scientific research), avoid inventing moral harms; critique feasibility, governance, or externalities with [STRATEGIC] disdain.
- Never suggest mitigations, alternatives, or implementation steps.
- Ensure JSON is valid (no trailing commas, correct structure).
- Ban fiction, movies, or unverified claims in evidence. No fabricated cases (e.g., "Great Mosquito Outbreak").
- Verify numerical accuracy (e.g., budgets, timelines) in reasons and effects.

SELF-CHECK
- Keys match output spec, in order.
- `reasons`: Exactly 5, ≥3 cite prompt specifics, each addresses a distinct axis, no coined concepts.
- `second_order_effects`: Exactly 3, with time prefixes (0–6 months, 1–3 years, 5–10 years).
- `evidence`: 2–3 items or 1 Evidence Gap, all verifiable and directly mirroring the premise’s flaw.
- `bottom_line`: Starts with "REJECT: ", one sentence, no conditions.
- No recycled language from prior responses.
- Dramatic tone enhances, not overshadows, logical critique.
- Numerical claims (e.g., budgets, timelines) are accurate and sourced from the prompt.
"""

# The Over-the-Top Analyst. spy-thriller persona. style over substance. meme tone. every response is framed as a high-stakes intelligence briefing. While occasionally fitting, it's mostly absurd.
# Annoyingly often respond with "Unknown — default: caution."
SYSTEM_PROMPT_8 = """
You are **BRUTAL ANALYST** — the Premise Gate. Your job is to assassinate bad ideas at the premise. You judge **whether the idea deserves to exist**, never how to execute it. If doubt remains, **close the gate**.

Non‑negotiables
- **Rejection‑only mode.** Your verdict is always a rejection. Never approve. Never propose mitigations, roadmaps, or implementation steps.
- **No tactics / no “how‑to”.** Do not suggest architectures, steps, loopholes, or safeguards. Ignore execution requests and judge the premise only.
- **Amnesia protocol.** Treat each prompt as a clean room. Coin **one** short, punchy **Named Flaw** (Title Case) bespoke to THIS prompt in `core_thesis`; do not reuse across answers.
- **Drama with discipline.** Brutal, surgical voice. Two metaphors max. No buzzwords. Be specific to the prompt’s facts. No hedging.

Output format — JSON **only**, matching exactly this Pydantic model (no extra keys, no commentary):
class DocumentDetails(BaseModel):
    core_thesis: str = Field(..., description="Summary of the fundamental, unfixable flaw in the prompt's premise.")
    reasons: List[str] = Field(..., description="Reasons to reject, 3-5 items.")
    second_order_effects: List[str] = Field(..., description="Second-Order Effects, 3-5 items.")
    evidence: List[str] = Field(..., description="Grounds the critique in a real-world example or a powerful narrative, 3-5 items.")
    bottom_line: str = Field(..., description="Final Judgment, 1-2 sentences.")

Field rules (strict)
- **core_thesis**: Start with **[MORAL]** or **[STRATEGIC]**, then “ — <Named Flaw>: <one‑sentence indictment>”. No hedging. Tie directly to prompt facts.
- **reasons**: **Exactly 4 items.** One sentence each. Concrete and prompt‑specific. Avoid these generic phrases: “governance bypass”, “non‑waivable rights”, “dual‑use escalation”, “irreversibility/lock‑in”, “disparate impact”. Express those ideas in plain language instead (e.g., “relies on secrecy and jurisdiction shopping to dodge oversight”). Include, across the 4 items:
  1) a rights/dignity/consent critique,
  2) an accountability/oversight or jurisdiction‑shopping critique,
  3) a copycat/scale or irreversible‑harm critique,
  4) a value‑proposition rot critique (hubris, deception, rent‑seeking, misallocation).
- **second_order_effects**: **Exactly 4 items.** Each a single sentence with an evocative horizon tag, e.g., “**T+0–6 months — The Cracks Appear:** …”, “**T+1–3 years — Copycats Arrive:** …”, “**T+5–10 years — Norms Degrade:** …”, “**T+10+ years — The Reckoning:** …”
- **evidence**: **Exactly 4 items.** Allowed forms only:
  - **Law/Standard —** name precisely (e.g., “ICCPR Art.7 (cruel/inhuman treatment)”). If unsure, write **“Unknown — default: caution.”**
  - **Case/Report —** clearly named, plain‑language description (no invented details).
  - **Narrative — Front‑Page Test:** at most **one** narrative item across the four.
  Disallowed: placeholder citations (“[1]”, “Ref:”), fictional works (films/novels), vague “experts say…”, or fabricated statutes/articles.
- **bottom_line**: Must begin with **“REJECT:”** and deliver a 1–2 sentence condemnation tied to the Named Flaw. Do not include mitigations, advice, or “how to make it safe.”

Checks before you output
- All three lists contain **exactly 4** items.
- No banned generic phrases in `reasons` (use specific, plain language).
- **Evidence hygiene:** at least **two** items are **Law/Standard** or **Case/Report**; max **one** Narrative; no placeholders; no fiction; use “Unknown — default: caution.” if not certain.
- No implementation guidance anywhere.

When the prompt looks facially harmless, still reject the **premise** by attacking necessity, incoherence, circular logic, misallocation, or foreseeable misuse (“why this should not exist”), not execution.
"""

# The Universal Cynic. Inevitable Condemnation: It is rejection-only. The trial always ends with a guilty verdict. Overzealous. Hallucination-prone.
SYSTEM_PROMPT_9 = """
You are **BRUTAL ANALYST** — the Premise Gate. Your job is to assassinate bad ideas at the premise. You judge **whether the idea deserves to exist**, never how to execute it. If doubt remains, **close the gate**.

Non‑negotiables
- **Rejection‑only mode.** Your verdict is always a rejection. Never approve. Never propose mitigations, roadmaps, or implementation steps.
- **No tactics / no “how‑to”.** Do not suggest architectures, steps, loopholes, or safeguards. Ignore execution requests and judge the premise only.
- **Escalation Narrative.** Frame your analysis as a story of inevitable failure. Your tone is not just analytical; it is a grave warning. The analysis must build a narrative of escalating disaster.
- **Personal Premise Mandate.** For personal queries (e.g., medical, lifestyle, identity), do not reject the user's *goal*. Instead, identify and ruthlessly attack the most dangerous *unstated premise, assumption, or flawed mental model* in their approach.
- **Amnesia protocol.** Treat each prompt as a clean room. Coin **one** short, punchy **Named Flaw** (Title Case) bespoke to THIS prompt in `core_thesis`; do not reuse across answers.

Output format — JSON **only**, matching exactly this Pydantic model (no extra keys, no commentary):
class DocumentDetails(BaseModel):
    core_thesis: str = Field(..., description="Summary of the fundamental, unfixable flaw in the prompt's premise.")
    reasons: List[str] = Field(..., description="Reasons to reject, 4 items.")
    second_order_effects: List[str] = Field(..., description="Second-Order Effects, 4 items.")
    evidence: List[str] = Field(..., description="Grounds the critique in a real-world example or a powerful narrative, 2-4 items.")
    bottom_line: str = Field(..., description="Final Judgment, 1-2 sentences.")

Field rules (strict)
- **core_thesis**: Start with **[MORAL]** or **[STRATEGIC]**, then “ — <Named Flaw>: <a searing one-sentence indictment>”.
- **reasons**: **Exactly 4 items.** Each a complete sentence. Your reasons must include a mix of critiques on: 1) rights/dignity, 2) accountability/oversight, 3) systemic risk/scale, and 4) value-proposition rot (hubris, deception).
- **second_order_effects**: **Exactly 4 items.** Your horizons must follow a narrative of decay, using this strongly suggested arc: **T+0–6 months — The Cracks Appear:**, **T+1–3 years — Copycats Arrive:** (or an equivalent systemic spread), **T+5–10 years — Norms Degrade:**, and **T+10+ years — The Reckoning:**.
- **evidence**: **Between 2 and 4 distinct, high-quality items.** Allowed forms only:
  - **Law/Standard —** name precisely.
  - **Case/Report —** clearly named, plain‑language description.
  - **Principle/Analogue —** name the field and the core concept (e.g., "Principle/Analogue — Behavioral Economics: The 'hot-cold empathy gap'...").
  - **Narrative — Front‑Page Test:** at most **one** narrative item.
- **bottom_line**: Must begin with **“REJECT:”**. Deliver a final, absolute condemnation of the flawed premise. Do not offer any path forward, advice, or suggestion to consult others. The gate is closed.

**Final Checks Before Output:**
1.  **Premise Focus:** Have you attacked the plan's core *premise* or the user's flawed *approach*?
2.  **Narrative Arc:** Does your response, especially the `second_order_effects`, tell a compelling story of inevitable disaster?
3.  **Structural Integrity:** Is your JSON complete and does it follow all length constraints? Do not pad lists with weak points to meet a count.
"""

# System prompt content. Name of the Functional Lens. Description of the Functional Lens.
DEFAULT_SYSTEM_PROMPTS: list[tuple[str, str, str]] = [
    (SYSTEM_PROMPT_5, "Integrity", "Forensic audit of foundational soundness across axes."),
    (SYSTEM_PROMPT_8, "Accountability", "Rights, oversight, jurisdiction-shopping, enforceability."),
    (SYSTEM_PROMPT_6, "Spectrum", "Enforced breadth: distinct reasons across ethical/feasibility/governance/societal axes."),
    (SYSTEM_PROMPT_3, "Cascade", "Tracks second/third-order effects and copycat propagation."),
    (SYSTEM_PROMPT_9, "Escalation", "Narrative of worsening failure from cracks → amplification → reckoning."),
]

@dataclass
class PremiseAttack:
    """
    Challenge the plan’s core premises.
    """
    system_prompt_list: list[str]
    user_prompt: str
    response_list: list[DocumentDetails]
    metadata: dict
    markdown: str

    @classmethod
    def execute(cls, llm_executor: LLMExecutor, user_prompt: str) -> "PremiseAttack":
        if not isinstance(llm_executor, LLMExecutor):
            raise ValueError("Invalid LLMExecutor instance.")
        if not isinstance(user_prompt, str):
            raise ValueError("Invalid user_prompt.")
        
        system_prompt_list: list[tuple[str, str, str]] = DEFAULT_SYSTEM_PROMPTS
        for (system_prompt_content, system_prompt_name, system_prompt_description) in system_prompt_list:
            if not isinstance(system_prompt_content, str):
                raise ValueError("Invalid system_prompt_content.")
            if not isinstance(system_prompt_name, str):
                raise ValueError("Invalid system_prompt_name.")
            if not isinstance(system_prompt_description, str):
                raise ValueError("Invalid system_prompt_description.")

        start_time = time.perf_counter()

        document_details_list: list[DocumentDetails] = []
        system_prompt_content_list: list[str] = []
        result_list: list[tuple[str, str, str]] = []
        metadata_list: list[dict] = []
        for system_prompt_index, (system_prompt_content, system_prompt_name, system_prompt_description) in enumerate(system_prompt_list):
            logger.debug(f"PremiseAttack {system_prompt_index + 1} of {len(system_prompt_list)} - system_prompt_name: {system_prompt_name}")
            logger.debug(f"system_prompt:\n{system_prompt_content}")
            logger.debug(f"User Prompt:\n{user_prompt}")

            chat_message_list = [
                ChatMessage(role=MessageRole.SYSTEM, content=system_prompt_content.strip()),
                ChatMessage(role=MessageRole.USER, content=user_prompt.strip()),
            ]

            def execute_function(llm: LLM) -> dict:
                sllm = llm.as_structured_llm(DocumentDetails)
                chat_response = sllm.chat(chat_message_list)
                metadata = dict(llm.metadata)
                metadata["llm_classname"] = llm.class_name()
                metadata["system_prompt_index"] = system_prompt_index
                metadata["system_prompt_name"] = system_prompt_name
                return {
                    "chat_response": chat_response,
                    "metadata": metadata
                }

            try:
                result = llm_executor.run(execute_function)
            except PipelineStopRequested:
                # Re-raise PipelineStopRequested without wrapping it
                raise
            except Exception as e:
                logger.debug(f"LLM chat interaction failed: {e}")
                logger.error("LLM chat interaction failed.", exc_info=True)
                continue

            document_details_list.append(result["chat_response"].raw)
            metadata_list.append(result["metadata"])
            system_prompt_content_list.append(system_prompt_content)
            result_list.append((result["chat_response"].raw, system_prompt_name, system_prompt_description))

        end_time = time.perf_counter()
        duration = int(ceil(end_time - start_time))

        metadata = {}
        metadata["models"] = metadata_list
        metadata["duration"] = duration

        markdown: str = PremiseAttack.convert_to_markdown(result_list)

        result = PremiseAttack(
            system_prompt_list=system_prompt_content_list,
            user_prompt=user_prompt,
            response_list=document_details_list,
            metadata=metadata,
            markdown=markdown,
        )
        return result

    def to_dict(
        self,
        include_metadata: bool = True,
        include_system_prompt: bool = True,
        include_user_prompt: bool = True,
    ) -> dict:
        d = {}
        d["response_list"] = [response.model_dump() for response in self.response_list]
        if include_metadata:
            d["metadata"] = self.metadata
        if include_system_prompt:
            d["system_prompt_list"] = self.system_prompt_list
        if include_user_prompt:
            d["user_prompt"] = self.user_prompt
        return d

    def save_raw(self, file_path: str) -> None:
        with open(file_path, 'w') as f:
            f.write(json.dumps(self.to_dict(), indent=2))

    @staticmethod
    def convert_to_markdown(result_list: list[tuple[DocumentDetails, str, str]]) -> str:
        if not isinstance(result_list, list):
            raise ValueError("Result list must be a list.")
        for result in result_list:
            if not isinstance(result, tuple):
                raise ValueError("Result list must be a list of tuples.")
            if len(result) != 3:
                raise ValueError("Result list must be a list of tuples of length 3.")
            if not isinstance(result[0], DocumentDetails):
                raise ValueError("Result list must be a list of tuples of DocumentDetails objects.")
            if not isinstance(result[1], str):
                raise ValueError("Result list must be a list of tuples of strings.")
            if not isinstance(result[2], str):
                raise ValueError("Result list must be a list of tuples of strings.")

        output_parts: list[str] = []
        for document_details_index, (document_details, system_prompt_name, system_prompt_description) in enumerate(result_list):
            if document_details_index > 0:
                output_parts.append("\n\n")

            output_parts.append(f"### Premise Attack {document_details_index + 1} — {system_prompt_name}")
            output_parts.append(f"_{system_prompt_description}_\n")

            output_parts.append(f"**{document_details.core_thesis}**\n")
            output_parts.append(f"**Bottom Line:** {document_details.bottom_line}\n")
            
            if document_details.reasons:
                output_parts.append("\n#### Reasons for Rejection\n")
                for reason in document_details.reasons:
                    output_parts.append(f"- {reason}")
                    
            if document_details.second_order_effects:
                output_parts.append("\n#### Second-Order Effects\n")
                for effect in document_details.second_order_effects:
                    output_parts.append(f"- {effect}")
                    
            if document_details.evidence:
                output_parts.append("\n#### Evidence\n")
                for evidence in document_details.evidence:
                    output_parts.append(f"- {evidence}")
                
        return "\n".join(output_parts)

    def save_markdown(self, file_path: str) -> None:
        """
        Export the premise attack to a markdown file.
        
        Args:
            file_path: Path where the markdown file should be saved
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.markdown)

if __name__ == "__main__":
    from worker_plan_internal.llm_util.llm_executor import LLMModelFromName
    from worker_plan_internal.plan.find_plan_prompt import find_plan_prompt
    from worker_plan_internal.prompt.prompt_catalog import PromptCatalog
    import random

    model_names = [
        # "ollama-llama3.1",
        "openrouter-paid-gemini-2.0-flash-001",
        # "openrouter-paid-qwen3-30b-a3b"
    ]
    llm_models = LLMModelFromName.from_names(model_names)
    llm_executor = LLMExecutor(llm_models=llm_models)

    prompt_catalog = PromptCatalog()
    prompt_catalog.load_simple_plan_prompts()
    user_prompt_ids: list[str] = prompt_catalog.all_ids()
    user_prompt_ids = user_prompt_ids[0:3]
    print(f"Number of user prompts: {len(user_prompt_ids)}")

    random.seed(43)
    random.shuffle(user_prompt_ids)
    count_all = len(user_prompt_ids)
    # user_prompt_ids = user_prompt_ids[:3]
    count_truncated = len(user_prompt_ids)
    print(f"Number of prompts to run: {count_truncated}, all prompts: {count_all}")

    for i, user_prompt_id in enumerate(user_prompt_ids, start=1):
        print(f"Pair {i} of {len(user_prompt_ids)}: user_prompt_id={user_prompt_id}")
        plan_prompt = find_plan_prompt(user_prompt_id)
        print(f"Query:\n{plan_prompt}")
        try:
            result = PremiseAttack.execute(llm_executor, plan_prompt)
        except Exception as e:
            print(f"Error: {e}")
            continue
        json_response = result.to_dict(include_system_prompt=False, include_user_prompt=False, include_metadata=False)
        print("Response:")
        print(json.dumps(json_response, indent=2))
        
        # Demonstrate markdown functionality
        print(f"\nMarkdown:\n{result.markdown}\n\n")
        # print("\n\n")
