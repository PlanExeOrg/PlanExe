# Proposal: Location-Aware Resource & Cost Data Service

## Problem Statement

PlanExe generates plans that involve real-world costs — labor, materials, services, equipment. Currently, cost assumptions are whatever the LLM hallucinates. For PlanExe to be a **trusted auditing layer**, it needs access to real cost data:

- "What does a carpenter cost in Nairobi, Kenya?"
- "What does mahogany wood cost per board-foot in that region?"
- "What's the hourly rate for a dentist in Copenhagen?"
- "What does commercial rent cost per sqm in downtown São Paulo?"

These questions span every country, every profession, every material. The data is massive, always changing, and often incomplete.

## Requirements

- Location-aware cost lookups (country, city, region)
- Labor costs by profession/skill level
- Material costs by type and specification
- Service costs (rent, utilities, insurance, etc.)
- Graceful handling of missing data (interpolation, estimates)
- Currency-aware (integrates with Currency Service proposal)
- Data freshness indicators (how old is this data point?)
- Confidence levels (exact data vs interpolated vs LLM-estimated)

## The Scale Challenge

This is potentially a massive database:
- ~200 countries × ~1000 professions × ~10,000 materials = billions of data points
- Most of these combinations don't have direct data
- Data goes stale at different rates (labor costs: yearly, commodities: daily, rent: monthly)

## Data Architecture

### Tier 1: Authoritative Data Sources (high confidence)

- **ILO (International Labour Organization)**: Labor costs by country/sector
- **World Bank Open Data**: Economic indicators, PPP adjustments
- **UN COMTRADE**: Commodity trade data
- **BLS (US Bureau of Labor Statistics)**: Detailed US labor/material costs
- **Eurostat**: European cost data
- **Numbeo**: Cost of living data (crowdsourced but large coverage)
- **Trading Economics**: Economic indicators by country

### Tier 2: Interpolation & Estimation (medium confidence)

When exact data isn't available for a location:

**Geographic interpolation:**
- If we don't have carpenter costs for Nairobi specifically, use Kenya national average
- If Kenya data is missing, use East African regional average
- Weight by economic similarity (GDP per capita, urbanization rate)

**PPP (Purchasing Power Parity) adjustment:**
- If we know US carpenter costs and the PPP ratio between US and Kenya, we can estimate
- World Bank publishes PPP conversion factors for most countries

**Economic similarity clustering:**
- Group countries by GDP per capita, economic structure, urbanization
- Use known costs from similar countries to estimate missing ones
- Example: If we know costs in Ghana and Tanzania, interpolate for Uganda

### Tier 3: LLM-Assisted Estimation (low confidence)

- For truly obscure combinations, ask the LLM but flag confidence as "estimated"
- Use the LLM's training data as a rough guide, validated against Tier 1/2 where possible
- Always flag these as "LLM estimate — verify before using in financial projections"

## Data Model

```
CostDataPoint:
  category: str          # "labor", "material", "service", "equipment"
  item: str              # "carpenter", "mahogany_lumber", "commercial_rent"
  location:
    country: str         # ISO 3166-1 alpha-2 (e.g., "KE")
    region: str?         # State/province if available
    city: str?           # City if available
  value: float
  currency: str          # ISO 4217 (e.g., "KES")
  unit: str              # "per_hour", "per_board_foot", "per_sqm_month"
  source: str            # "ilo", "worldbank", "interpolated", "llm_estimate"
  confidence: float      # 0.0 - 1.0
  date: date             # When this data was collected/computed
  freshness: str         # "live", "monthly", "annual", "stale"
```

## Interpolation Strategy

### Step 1: Exact Match

Look up exact (item, country, city) combination. If found, return with high confidence.

### Step 2: Geographic Fallback

1. Try (item, country, region) → medium-high confidence
2. Try (item, country) national average → medium confidence
3. Try (item, economic_region) using similar countries → medium-low confidence

### Step 3: PPP Adjustment

1. Find the item cost in a reference country (usually US or nearest country with data)
2. Apply PPP conversion factor
3. Return with low-medium confidence, flagged as "PPP-adjusted estimate"

### Step 4: LLM Estimation

1. Ask the LLM with structured prompting: "Based on economic conditions in [country], estimate the cost of [item]"
2. Cross-reference against Tier 1/2 bounds if available
3. Return with low confidence, flagged as "LLM estimate"

## Implementation Approach

### Phase 1: Core Framework + Static Data

- Data model and query API
- Bundled static dataset covering major economies (US, EU, China, India, Brazil, Nigeria, Kenya, etc.)
- PPP adjustment using World Bank data (updated annually)
- Basic geographic interpolation
- Tests with realistic queries

### Phase 2: Dynamic Data Fetching

- API integrations (ILO, World Bank, Numbeo)
- Caching layer with TTL per data source
- Data freshness tracking
- Incremental updates

### Phase 3: LLM-Assisted Gap Filling

- Structured prompts for cost estimation
- Confidence scoring
- Validation against known data points
- Human review workflow for flagged estimates

### Phase 4: Community Data

- Allow users to contribute local cost data
- Verification/moderation pipeline
- Regional experts can validate estimates

## Integration with PlanExe Pipeline

```
Plan Generation → Extract Cost Assumptions → Query Cost Service → Compare/Validate → Flag Discrepancies
```

The cost service would be called by:
1. **FermiSanityCheck** — "Is this cost assumption reasonable for this location?"
2. **DomainNormalizer** — "Convert this cost to standard units and currency"
3. **RiskAssessment** — "How volatile are these cost inputs over the plan's time horizon?"

## Open Questions

1. How much static data should we bundle vs fetch dynamically?
2. Should the service be synchronous (query per assumption) or batch (validate all assumptions at once)?
3. What's the minimum viable dataset for Phase 1?
4. How do we handle informal economies where official statistics don't reflect reality?
5. Should material costs include supply chain factors (shipping to location)?
6. How do we handle time-varying costs (seasonal labor, commodity cycles)?

## Risks

- **Data quality**: Crowdsourced data (Numbeo) may be unreliable
- **Coverage gaps**: Many developing countries lack detailed cost data
- **Staleness**: Economic conditions change rapidly (inflation, crises)
- **Complexity creep**: This could become an entire product on its own
- **Scope**: Need to define "good enough" for Phase 1
