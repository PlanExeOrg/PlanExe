from worker_plan_internal.parameter_extraction.distill_report_section import (
    CalculationSeed,
    DigestItem,
    DistilledReportSection,
    DistillReportSection,
    GateOrThreshold,
    MissingValueEstimate,
    ModelDriver,
    NumericAnchor,
    RiskShock,
    infer_section_type_from_path,
    normalize_section_type,
)


def test_infer_section_type_from_path() -> None:
    assert infer_section_type_from_path("strategic_decisions.md") == "strategic_decisions"
    assert infer_section_type_from_path("/tmp/review_plan.md") == "review_plan"
    assert infer_section_type_from_path("/tmp/not_a_known_section.md") == "unknown"


def test_normalize_section_type() -> None:
    assert normalize_section_type("Strategic Decisions") == "strategic_decisions"
    assert normalize_section_type("premortem") == "premortem"
    assert normalize_section_type("Review-Plan") == "review_plan"
    assert normalize_section_type("expert_criticism") == "expert_criticism"
    assert normalize_section_type("garbage") == "unknown"
    assert normalize_section_type(None) == "unknown"


def test_convert_to_markdown_preserves_monte_carlo_fields() -> None:
    distilled = DistilledReportSection(
        section_type="review_plan",
        section_title="Review Plan",
        modelling_summary="The section preserves gates, shocks, and missing inputs for modelling.",
        digest_items=[
            DigestItem(
                item_id="G1",
                source_heading="Review 6: Key Performance Indicators",
                kind="gate",
                statement="Off-peak revenue must exceed 75% of direct utility overhead.",
                modelling_relevance="Tests whether rentals cover variable utility cost.",
                source_text="revenue must exceed 75% of direct variable utility overhead",
            )
        ],
        numeric_anchors=[
            NumericAnchor(
                label="Utility overhead coverage target",
                value_text="75%",
                unit="fraction",
                role="threshold",
                source_text="exceed 75% of direct variable utility overhead",
            )
        ],
        model_drivers=[
            ModelDriver(
                id="off_peak_revenue",
                label="Off-peak revenue",
                category="budget",
                value_type="missing_but_needed",
                unit="DKK",
                value_text=None,
                modelling_priority="critical",
                uncertainty="high",
                reason="Revenue shortfall drives contingency drawdown.",
            )
        ],
        gates_and_thresholds=[
            GateOrThreshold(
                id="off_peak_overhead_coverage_gate",
                label="Off-peak overhead coverage gate",
                pass_fail_rule="off_peak_coverage_ratio >= 0.75",
                consequence_if_failed="Contingency pays routine operating costs.",
                depends_on=["off_peak_revenue", "direct_utility_overhead"],
            )
        ],
        risk_shocks=[
            RiskShock(
                id="utility_price_spike",
                label="Utility price spike",
                trigger="Monthly utility invoice exceeds 30% of variable expenses.",
                impact="Gross margin erosion and surcharge activation.",
                probability_hint="High likelihood",
                affected_driver_ids=["direct_utility_overhead"],
            )
        ],
        missing_values_to_estimate=[
            MissingValueEstimate(
                id="direct_utility_overhead",
                label="Direct utility overhead",
                unit="DKK",
                why_needed="Needed to compute off-peak coverage ratio.",
                suggested_estimation_method="Use metered energy cost during pricing trial.",
            )
        ],
        recommended_calculations=[
            CalculationSeed(
                id="off_peak_coverage_ratio",
                label="Off-peak coverage ratio",
                formula_hint="off_peak_coverage_ratio = off_peak_revenue / direct_utility_overhead",
                depends_on=["off_peak_revenue", "direct_utility_overhead"],
                output_unit="fraction",
                why_needed="Tests the operating coverage gate.",
            )
        ],
        omitted_content_rationale=["Dropped repeated review objectives."],
    )

    markdown = DistillReportSection.convert_to_markdown(distilled)

    assert "## Numeric Anchors" in markdown
    assert "off_peak_coverage_ratio = off_peak_revenue / direct_utility_overhead" in markdown
    assert "direct_utility_overhead" in markdown
    assert "Dropped repeated review objectives." in markdown
