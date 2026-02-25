# QuantifiedAssumption Schema Reference

| Field | Type | Description |
| --- | --- | --- |
| `assumption_id` | `str` | Unique stable identifier for the assumption (use `assumption-<index>` when not provided). |
| `question` | `str` | The source question that prompted the assumption. |
| `claim` | `str` | Normalized assumption text with the `Assumption:` prefix removed. |
| `lower_bound` | `float?` | Parsed lower numeric bound (if present). |
| `upper_bound` | `float?` | Parsed upper numeric bound (mirror of lower_bound when none explicitly provided). |
| `unit` | `str?` | Detected unit token (e.g., `mw`, `days`, `usd`, `%`). |
| `confidence` | `ConfidenceLevel` (`high` / `medium` / `low`) | Estimated confidence level inferred from hedging words. |
| `evidence` | `str` | Text excerpt used as evidence (currently same as `claim` but can be overridden with extracted snippets). |
| `extracted_numbers` | `List[float]` | All numeric values found in the assumption for further heuristics. |
| `raw_assumption` | `str` | Original string returned by `MakeAssumptions` (includes prefix). |

## Confidence Enum Values

| Level | Detection Signals |
| --- | --- |
| `high` | Contains strong modality ("will", "must", "ensure", "guarantee"). |
| `medium` | Default when no strong signal is detected. |
| `low` | Contains hedging words ("estimate", "approx", "may", "likely"). |

## Unit Examples

- Financial: `usd`, `eur`, `million`, `billion`
- Capacity/Scale: `mw`, `kw`, `tonnes`, `sqft`, `people`
- Time: `days`, `weeks`, `months`, `years` (expressed as words following the range)
- Percentage/Ratio: `%`, `bps`

Units are extracted by scanning the text around the numeric range or first detected unit word after the numbers.

## Evidence Expectations by Confidence

- `high`: sentence should include explicit value statements or commitments (e.g., "We will deliver 30 MW") and the evidence string can be the same sentence.
- `medium`: treat as the default; evidence is the claim text itself.
- `low`: must cite qualifiers and ideally pair the claim with supporting context (e.g., "~8 months" followed by "assuming no permit delays"). Evidence may include surrounding context when available.

Use this reference when wiring FermiSanityCheck so the validation functions know what fields exist, what values they expect, and how to treat the evidence for confidence levels.
