"""
Distill verbose report sections into a compact modelling digest.

This is intended as a pre-extraction compression stage for sections such as
Strategic Decisions, Review Plan, Premortem, and Expert Criticism. The output
keeps Monte Carlo-relevant data: denominators, thresholds, gates, uncertainty
drivers, shocks, dependencies, missing estimates, and calculation seeds.
"""
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from math import ceil
from pathlib import Path
from typing import Any, Literal, Optional

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.llms.llm import LLM
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ReportSectionTypeEnum(str, Enum):
    STRATEGIC_DECISIONS = "strategic_decisions"
    REVIEW_PLAN = "review_plan"
    PREMORTEM = "premortem"
    EXPERT_CRITICISM = "expert_criticism"
    UNKNOWN = "unknown"


class DigestItemKindEnum(str, Enum):
    DECISION = "decision"
    GATE = "gate"
    ASSUMPTION = "assumption"
    RISK = "risk"
    SHOCK = "shock"
    CONSTRAINT = "constraint"
    MISSING_DATA = "missing_data"
    CALCULATION_SEED = "calculation_seed"
    DEPENDENCY = "dependency"
    CRITIQUE = "critique"
    ACTION = "action"


class ModellingCategoryEnum(str, Enum):
    BUDGET = "budget"
    COST = "cost"
    DEMAND = "demand"
    CONVERSION = "conversion"
    CAPACITY = "capacity"
    TIME = "time"
    COVERAGE = "coverage"
    IMPACT = "impact"
    OPERATIONAL = "operational"
    RISK = "risk"
    FUNDING_GATE = "funding_gate"
    STAFFING = "staffing"
    LOGISTICS = "logistics"
    COMPLIANCE = "compliance"
    OUTCOME = "outcome"
    SENSITIVITY = "sensitivity"
    OTHER = "other"


class ValueTypeEnum(str, Enum):
    EXPLICIT = "explicit"
    DERIVED = "derived"
    INFERRED = "inferred"
    MISSING_BUT_NEEDED = "missing_but_needed"


class ModellingPriorityEnum(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class UncertaintyEnum(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class NumericRoleEnum(str, Enum):
    DENOMINATOR = "denominator"
    THRESHOLD = "threshold"
    BUDGET = "budget"
    COST = "cost"
    RATE = "rate"
    CAPACITY = "capacity"
    DEADLINE = "deadline"
    SHOCK = "shock"
    PROBABILITY = "probability"
    SCORE = "score"
    OTHER = "other"


class NumericAnchor(BaseModel):
    label: str = Field(description="Short label for the numeric value.")
    value_text: str = Field(description="Exact value or range as written or normalized, such as '15%' or '300000 DKK'.")
    unit: Optional[str] = Field(None, description="Unit such as fraction, DKK, people, days, months, hours, or unknown.")
    role: Literal[
        "denominator",
        "threshold",
        "budget",
        "cost",
        "rate",
        "capacity",
        "deadline",
        "shock",
        "probability",
        "score",
        "other",
    ] = Field(description="The modelling role this number plays.")
    source_text: str = Field(description="Source quote or close paraphrase, at most 20 words.")


class ModelDriver(BaseModel):
    id: str = Field(description="Stable snake_case id suitable for downstream modelling.")
    label: str = Field(description="Short human-readable variable name.")
    category: Literal[
        "budget",
        "cost",
        "demand",
        "conversion",
        "capacity",
        "time",
        "coverage",
        "impact",
        "operational",
        "risk",
        "funding_gate",
        "staffing",
        "logistics",
        "compliance",
        "outcome",
        "sensitivity",
        "other",
    ] = Field(description="Modelling category.")
    value_type: Literal["explicit", "derived", "inferred", "missing_but_needed"] = Field(description="How directly this value appears in the source.")
    unit: str = Field(description="Expected modelling unit, or unknown.")
    value_text: Optional[str] = Field(None, description="Known value text if explicit; otherwise null.")
    modelling_priority: Literal["critical", "high", "medium", "low"] = Field(description="Priority for Monte Carlo modelling.")
    uncertainty: Literal["low", "medium", "high"] = Field(description="Expected uncertainty if modelled.")
    reason: str = Field(description="Why this variable materially affects viability, at most 25 words.")


class GateOrThreshold(BaseModel):
    id: str = Field(description="Stable snake_case id for the gate or threshold.")
    label: str = Field(description="Short gate name.")
    pass_fail_rule: str = Field(description="Measurable rule if available, preserving operators and thresholds.")
    consequence_if_failed: str = Field(description="Compact consequence for viability, budget, schedule, or operations.")
    depends_on: list[str] = Field(description="Declared model driver or missing estimate ids needed to evaluate the rule.")


class RiskShock(BaseModel):
    id: str = Field(description="Stable snake_case id for the risk or shock.")
    label: str = Field(description="Short risk or shock name.")
    trigger: str = Field(description="Observable trigger, tripwire, or cause.")
    impact: str = Field(description="Quantified or model-relevant impact.")
    probability_hint: Optional[str] = Field(None, description="Likelihood, score, or probability text if present.")
    affected_driver_ids: list[str] = Field(description="Model driver ids this risk should perturb.")


class MissingValueEstimate(BaseModel):
    id: str = Field(description="Stable snake_case id for the missing input.")
    label: str = Field(description="Short human-readable name.")
    unit: str = Field(description="Expected modelling unit, or unknown.")
    why_needed: str = Field(description="Why this estimate is needed for bounds or Monte Carlo.")
    suggested_estimation_method: str = Field(description="How to estimate or bound the value.")


class CalculationSeed(BaseModel):
    id: str = Field(description="Stable snake_case id for the calculation.")
    label: str = Field(description="Short human-readable calculation name.")
    formula_hint: str = Field(description="Executable formula hint using declared ids.")
    depends_on: list[str] = Field(description="Declared ids used on the formula right-hand side.")
    output_unit: str = Field(description="Expected output unit, or unknown.")
    why_needed: str = Field(description="What viability claim this calculation tests.")


class DigestItem(BaseModel):
    item_id: str = Field(description="Stable id such as D1, G2, R3, or C4.")
    source_heading: str = Field(description="Original heading or closest local heading.")
    kind: Literal[
        "decision",
        "gate",
        "assumption",
        "risk",
        "shock",
        "constraint",
        "missing_data",
        "calculation_seed",
        "dependency",
        "critique",
        "action",
    ] = Field(description="Type of modelling-relevant item.")
    statement: str = Field(description="Compressed statement, at most 30 words.")
    modelling_relevance: str = Field(description="How this item affects the model, at most 25 words.")
    source_text: str = Field(description="Source quote or close paraphrase, at most 20 words.")


class DistilledReportSection(BaseModel):
    section_type: Literal["strategic_decisions", "review_plan", "premortem", "expert_criticism", "unknown"] = Field(
        description="Source report section type."
    )
    section_title: str = Field(description="Short display title for the section.")
    modelling_summary: str = Field(description="Five sentences or fewer describing the modelling value of this section.")
    digest_items: list[DigestItem] = Field(description="At most 12 high-value compressed items.")
    numeric_anchors: list[NumericAnchor] = Field(description="At most 10 numeric anchors worth preserving.")
    model_drivers: list[ModelDriver] = Field(description="At most 10 candidate modelling variables.")
    gates_and_thresholds: list[GateOrThreshold] = Field(description="At most 6 executable gates or thresholds.")
    risk_shocks: list[RiskShock] = Field(description="At most 6 shocks, failure paths, or risk drivers.")
    missing_values_to_estimate: list[MissingValueEstimate] = Field(description="At most 6 missing values needed for modelling.")
    recommended_calculations: list[CalculationSeed] = Field(description="At most 6 formula-backed calculations.")
    omitted_content_rationale: list[str] = Field(description="At most 5 short notes about what was intentionally dropped.")


SECTION_TYPE_BY_STEM = {
    "strategic_decisions": ReportSectionTypeEnum.STRATEGIC_DECISIONS.value,
    "review_plan": ReportSectionTypeEnum.REVIEW_PLAN.value,
    "premortem": ReportSectionTypeEnum.PREMORTEM.value,
    "expert_criticism": ReportSectionTypeEnum.EXPERT_CRITICISM.value,
}

SECTION_GUIDANCE = {
    ReportSectionTypeEnum.STRATEGIC_DECISIONS.value: (
        "Strategic Decisions: keep decision title, core choice, trade-off, lever id, "
        "scenario axis, risk driver, capacity constraint, and numeric thresholds. "
        "Drop long synergy/conflict prose unless it identifies a dependency."
    ),
    ReportSectionTypeEnum.REVIEW_PLAN.value: (
        "Review Plan: keep fragile assumptions, validation questions, pass/fail criteria, "
        "stakeholder gates, data-quality concerns, budget clarifications, and KPI thresholds. "
        "Drop review-objective prose that does not affect modelling."
    ),
    ReportSectionTypeEnum.PREMORTEM.value: (
        "Premortem: keep assumptions to kill, failure modes, tripwires, stop rules, "
        "likelihood/impact scores, contingency drawdowns, shocks, and causal chains. "
        "Compress stories into model-relevant triggers and impacts."
    ),
    ReportSectionTypeEnum.EXPERT_CRITICISM.value: (
        "Expert Criticism: keep hidden assumptions, missing denominators, unmodelled costs, "
        "mitigations with thresholds, root causes, quantified consequences, and expert tags. "
        "Drop repeated expert biographies and search terms unless they affect estimation."
    ),
    ReportSectionTypeEnum.UNKNOWN.value: (
        "Unknown section: keep only model-relevant denominators, thresholds, risks, "
        "dependencies, missing values, and calculation seeds."
    ),
}

DISTILL_REPORT_SECTION_SYSTEM_PROMPT = """
You are a PlanExe modelling digest transformer.

Your task is to compress one verbose PlanExe report section for later
parameter extraction, deterministic calculations, scenario analysis, and
Monte Carlo simulation.

The goal is not a human summary. Preserve the items that can become model
inputs, bounds, pass/fail gates, shock variables, dependencies, or calculation
formulas.

Preserve:
- explicit numeric values, ranges, percentages, dates, budgets, counts, scores
- denominators and internal-vs-real-world denominator distinctions
- assumptions that can be false
- uncertainty drivers and sensitivity axes
- capacity, staffing, throughput, budget, deadline, funding, compliance gates
- risks, shocks, tripwires, stop rules, and failure paths
- missing values and suggested ways to estimate them
- formulas or formula-ready relationships

Compress or omit:
- repeated narrative framing
- persuasive prose
- generic synergy/conflict paragraphs unless they define a dependency
- biographies, search terms, and role descriptions unless needed for ownership
- duplicate recommendations

Hard limits:
- digest_items: at most 12
- numeric_anchors: at most 10
- model_drivers: at most 10
- gates_and_thresholds: at most 6
- risk_shocks: at most 6
- missing_values_to_estimate: at most 6
- recommended_calculations: at most 6
- omitted_content_rationale: at most 5

Formula discipline:
- recommended_calculations.formula_hint must be non-empty and executable as a
  simple formula hint.
- depends_on must list input ids used on the right-hand side, not outputs.
- If a needed input is absent, add it to missing_values_to_estimate.
- Use stable snake_case ids.

Return only the structured object requested by the schema. Do not add markdown
or prose outside the object.
"""


def infer_section_type_from_path(file_path: str | Path) -> str:
    """Infer report section type from a PlanExe intermediary filename."""
    stem = Path(file_path).stem
    return SECTION_TYPE_BY_STEM.get(stem, ReportSectionTypeEnum.UNKNOWN.value)


def normalize_section_type(section_type: str | ReportSectionTypeEnum | None) -> str:
    """Normalize an optional section type to a schema value."""
    if section_type is None:
        return ReportSectionTypeEnum.UNKNOWN.value
    if isinstance(section_type, ReportSectionTypeEnum):
        return section_type.value
    section_type_str = str(section_type).strip().lower().replace("-", "_").replace(" ", "_")
    if section_type_str in SECTION_GUIDANCE:
        return section_type_str
    return SECTION_TYPE_BY_STEM.get(section_type_str, ReportSectionTypeEnum.UNKNOWN.value)


def build_user_prompt(section_markdown: str, section_type: str, section_title: Optional[str] = None) -> str:
    """Build the user prompt for one section."""
    title = section_title or section_type.replace("_", " ").title()
    guidance = SECTION_GUIDANCE.get(section_type, SECTION_GUIDANCE[ReportSectionTypeEnum.UNKNOWN.value])
    return "\n".join(
        [
            f"Section type: {section_type}",
            f"Section title: {title}",
            "",
            guidance,
            "",
            "Distill the following Markdown section:",
            "[START_SECTION_MARKDOWN]",
            section_markdown.strip(),
            "[END_SECTION_MARKDOWN]",
        ]
    )


def _table_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("\n", " ").replace("|", "\\|").strip()


def _append_table(rows: list[str], headers: list[str], values: list[list[Any]]) -> None:
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in values:
        rows.append("| " + " | ".join(_table_cell(value) for value in row) + " |")
    rows.append("")


@dataclass
class DistillReportSection:
    system_prompt: str
    user_prompt: str
    response: dict
    metadata: dict
    markdown: str

    @classmethod
    def execute(
        cls,
        llm: LLM,
        section_markdown: str,
        section_type: str | ReportSectionTypeEnum | None = None,
        section_title: Optional[str] = None,
        **kwargs: Any,
    ) -> "DistillReportSection":
        """Invoke the LLM to compress one report section into a modelling digest."""
        if not isinstance(llm, LLM):
            raise ValueError("Invalid LLM instance.")
        if not isinstance(section_markdown, str):
            raise ValueError("Invalid section_markdown.")

        normalized_section_type = normalize_section_type(section_type)
        system_prompt = kwargs.get("system_prompt", DISTILL_REPORT_SECTION_SYSTEM_PROMPT.strip())
        if not isinstance(system_prompt, str):
            raise ValueError("Invalid system_prompt.")

        user_prompt = build_user_prompt(
            section_markdown=section_markdown,
            section_type=normalized_section_type,
            section_title=section_title,
        )
        logger.debug(f"System Prompt:\n{system_prompt}")
        logger.debug(f"User Prompt:\n{user_prompt}")

        chat_message_list = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=MessageRole.USER, content=user_prompt),
        ]

        sllm = llm.as_structured_llm(DistilledReportSection)

        logger.debug("Starting LLM chat interaction.")
        start_time = time.perf_counter()
        chat_response = sllm.chat(chat_message_list)
        end_time = time.perf_counter()
        duration = int(ceil(end_time - start_time))
        response_byte_count = len(chat_response.message.content.encode("utf-8"))
        logger.info(f"LLM chat interaction completed in {duration} seconds. Response byte count: {response_byte_count}")

        distilled_section: DistilledReportSection = chat_response.raw
        if distilled_section is None:
            raise ValueError(
                "Structured LLM returned None for DistilledReportSection. "
                "The model likely echoed the schema instead of producing values."
            )

        metadata = dict(llm.metadata)
        metadata["llm_classname"] = llm.class_name()
        metadata["duration"] = duration
        metadata["response_byte_count"] = response_byte_count

        json_response = distilled_section.model_dump()
        markdown = cls.convert_to_markdown(distilled_section)

        return DistillReportSection(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=json_response,
            metadata=metadata,
            markdown=markdown,
        )

    @classmethod
    def execute_from_file(
        cls,
        llm: LLM,
        file_path: str | Path,
        section_type: str | ReportSectionTypeEnum | None = None,
        section_title: Optional[str] = None,
        **kwargs: Any,
    ) -> "DistillReportSection":
        """Read a Markdown file and distill it."""
        path = Path(file_path)
        markdown = path.read_text(encoding="utf-8")
        inferred_section_type = normalize_section_type(section_type) if section_type else infer_section_type_from_path(path)
        inferred_title = section_title or path.stem.replace("_", " ").title()
        return cls.execute(
            llm=llm,
            section_markdown=markdown,
            section_type=inferred_section_type,
            section_title=inferred_title,
            **kwargs,
        )

    def to_dict(self, include_metadata: bool = True, include_system_prompt: bool = True, include_user_prompt: bool = True) -> dict:
        d = self.response.copy()
        if include_metadata:
            d["metadata"] = self.metadata
        if include_system_prompt:
            d["system_prompt"] = self.system_prompt
        if include_user_prompt:
            d["user_prompt"] = self.user_prompt
        return d

    def save_raw(self, file_path: str | Path) -> None:
        path = Path(file_path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def save_markdown(self, file_path: str | Path) -> None:
        path = Path(file_path)
        path.write_text(self.markdown, encoding="utf-8")

    @staticmethod
    def convert_to_markdown(distilled_section: DistilledReportSection) -> str:
        """Convert the structured digest to compact Markdown."""
        rows: list[str] = []
        rows.append(f"# {distilled_section.section_title}")
        rows.append("")
        rows.append(distilled_section.modelling_summary)
        rows.append("")

        if distilled_section.digest_items:
            rows.append("## Modelling Digest")
            _append_table(
                rows,
                ["ID", "Kind", "Source", "Statement", "Model Relevance"],
                [
                    [
                        item.item_id,
                        item.kind,
                        item.source_heading,
                        item.statement,
                        item.modelling_relevance,
                    ]
                    for item in distilled_section.digest_items
                ],
            )

        if distilled_section.numeric_anchors:
            rows.append("## Numeric Anchors")
            _append_table(
                rows,
                ["Label", "Value", "Unit", "Role", "Source"],
                [
                    [anchor.label, anchor.value_text, anchor.unit, anchor.role, anchor.source_text]
                    for anchor in distilled_section.numeric_anchors
                ],
            )

        if distilled_section.model_drivers:
            rows.append("## Model Drivers")
            _append_table(
                rows,
                ["ID", "Category", "Value Type", "Unit", "Value", "Priority", "Uncertainty", "Reason"],
                [
                    [
                        driver.id,
                        driver.category,
                        driver.value_type,
                        driver.unit,
                        driver.value_text,
                        driver.modelling_priority,
                        driver.uncertainty,
                        driver.reason,
                    ]
                    for driver in distilled_section.model_drivers
                ],
            )

        if distilled_section.gates_and_thresholds:
            rows.append("## Gates And Thresholds")
            _append_table(
                rows,
                ["ID", "Rule", "Depends On", "Consequence"],
                [
                    [
                        gate.id,
                        gate.pass_fail_rule,
                        ", ".join(gate.depends_on),
                        gate.consequence_if_failed,
                    ]
                    for gate in distilled_section.gates_and_thresholds
                ],
            )

        if distilled_section.risk_shocks:
            rows.append("## Risks And Shocks")
            _append_table(
                rows,
                ["ID", "Trigger", "Impact", "Probability", "Affected Drivers"],
                [
                    [
                        risk.id,
                        risk.trigger,
                        risk.impact,
                        risk.probability_hint,
                        ", ".join(risk.affected_driver_ids),
                    ]
                    for risk in distilled_section.risk_shocks
                ],
            )

        if distilled_section.missing_values_to_estimate:
            rows.append("## Missing Values To Estimate")
            _append_table(
                rows,
                ["ID", "Unit", "Why Needed", "Estimation Method"],
                [
                    [
                        missing.id,
                        missing.unit,
                        missing.why_needed,
                        missing.suggested_estimation_method,
                    ]
                    for missing in distilled_section.missing_values_to_estimate
                ],
            )

        if distilled_section.recommended_calculations:
            rows.append("## Recommended Calculations")
            _append_table(
                rows,
                ["ID", "Formula", "Depends On", "Output Unit", "Why Needed"],
                [
                    [
                        calculation.id,
                        calculation.formula_hint,
                        ", ".join(calculation.depends_on),
                        calculation.output_unit,
                        calculation.why_needed,
                    ]
                    for calculation in distilled_section.recommended_calculations
                ],
            )

        if distilled_section.omitted_content_rationale:
            rows.append("## Omitted Content")
            rows.extend(f"- {item}" for item in distilled_section.omitted_content_rationale)
            rows.append("")

        return "\n".join(rows).strip() + "\n"
