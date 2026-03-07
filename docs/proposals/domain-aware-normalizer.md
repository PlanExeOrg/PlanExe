# Proposal: Domain-Aware Assumption Normalizer

*Preserving and evolving the vision from PR #72, built on the Currency Service (PR #147) and Resource/Cost Data Service (PR #146) foundations.*

## Why This Matters

PlanExe generates business plans for anything — a carpentry workshop in Copenhagen, a dental clinic in Denver, a community farm in Nairobi, a tech startup in Seoul. Each of these plans lives in a completely different economic reality. A $50,000 budget means something very different for a lemonade stand than for a data center. "Three months" is a tight timeline for construction but generous for a software prototype. "I've done this before" carries different weight from a 30-year master carpenter than from a first-time entrepreneur.

Today, PlanExe treats all assumptions the same. When the LLM generates a plan that says "labor costs will be approximately $15-25 per hour," PlanExe has no way to know whether that's reasonable for a carpenter in rural Kenya (where it would be wildly high) or a software contractor in San Francisco (where it would be impossibly low). The numbers pass through unchecked, and downstream tasks like budgeting, scheduling, and risk assessment build on potentially hallucinated foundations.

The Domain-Aware Assumption Normalizer changes this. It sits in the pipeline between assumption generation and assumption review, and it does four things: it figures out what kind of plan this is, it converts all the numbers to a standard representation, it checks whether those numbers make sense in the real world, and it flags anything that looks wrong — including plans that rely on exploiting workers or ignoring safety standards.

## The Ethical Dimension: A Moral Compass for Planning

This is not just a technical normalization problem. When PlanExe helps someone plan a business, it has a responsibility not to optimize for human exploitation.

### The Problem

Consider a plan that says: "We'll hire 40 textile workers in Dhaka at $2/day to handle chemical dyes without protective equipment, saving 85% on labor compared to European manufacturing." From a pure cost-optimization perspective, this plan might score well — the numbers are internally consistent, the budget works out, the timeline is achievable. But it describes a sweatshop. PlanExe should not help build sweatshops.

Or consider: "Source rare earth minerals from artisanal mines in the DRC using child labor, reducing material costs by 60%." The economics might work. The plan is monstrous.

These aren't edge cases. Labor exploitation is one of the most common ways businesses cut costs, and any planning tool that optimizes for cost without considering human welfare becomes complicit in that exploitation.

### The Solution: Ethical Guardrails in the Normalizer

The Domain-Aware Normalizer will include an ethical assessment layer that evaluates labor and safety assumptions against established standards. This is not about being preachy or blocking legitimate low-cost operations in developing economies — it's about catching plans that cross clearly defined ethical lines.

**Fair Wage Floors:**
The normalizer will compare proposed labor costs against living wage data for the specified location. Not minimum wage (which is often exploitative itself), but living wage — the amount a worker needs to afford basic necessities including housing, food, healthcare, and education for their family. Data sources include the Global Living Wage Coalition, WageIndicator Foundation, and ILO datasets. If a plan proposes paying workers significantly below the living wage for their location, it gets flagged — not silently passed through.

**Workplace Safety Standards:**
When a plan involves manufacturing, construction, agriculture, mining, or any physical labor, the normalizer will check whether the plan accounts for safety equipment, training, and compliance with ILO conventions on occupational safety. A plan that mentions hazardous materials but doesn't budget for protective equipment gets flagged. A plan that proposes construction timelines impossible without cutting safety corners gets flagged.

**Child Labor and Ethical Concerns — Offering Alternatives Instead of Blocking:**
When the normalizer detects plans that propose ethically problematic assumptions — including child labor, sub-living wages, missing safety provisions, or exploitative conditions — it does not refuse to create the plan. Instead, it generates **ethical alternative recommendations** alongside the flagged assumptions. This steers plan creators toward fairness without blocking their work.

*Example:* A plan proposes "Hire 20 workers in rural Cambodia at $1.50/day to manufacture garments without protective equipment." The normalizer:
1. **Flags the ethical issues:** Sub-living wages + no safety budget + hazardous work environment
2. **Shows the original numbers:** This approach saves ~$18,000/year in labor costs
3. **Presents an ethical alternative:** "Fair-wage version at $6/day with safety equipment and training changes labor costs from $10,950/year to $43,800/year. This reduces reputational risk, improves worker retention by ~30% (based on industry benchmarks), and qualifies for fair-trade certification, which may unlock premium pricing and larger contracts."
4. **Lets the creator decide:** The plan creator sees their original assumptions, understands the consequences and risks, and can choose to adopt the ethical alternative, adjust it, or proceed with the original numbers (now informed of the tradeoffs).

This approach respects the creator's autonomy while providing the information needed to make responsible choices. When plans explicitly describe conditions amounting to forced labor or involve workers clearly under the ILO minimum age (under 15 in most countries, under 18 in hazardous work), these are flagged as **critical risks** with strong emphasis on the alternative — but the system does not refuse to process them. The creator gets a better plan *and* avoids pitfalls.

**Working Conditions Assessment & ILO vs. Local Law:**
The normalizer evaluates proposed working hours, conditions, and benefits against the *higher* of local labor law or ILO Decent Work standards. This is critical because some countries' own laws fall below international standards.

**The max(local_law, ILO) principle:** When a country's local minimum wage, maximum working hours, or safety requirements fall below ILO recommendations, the normalizer uses the ILO standard as the threshold and flags the gap as a risk factor. For example:
- A plan complies with Country X's local minimum wage of $1.50/day, but ILO living wage guidelines for that country recommend $6/day. The normalizer flags this as **reputational and supply chain audit risk** — major international buyers (retailers, manufacturers) now audit supply chains for ILO compliance, and operating at local minimum creates exposure.
- A plan proposes 16-hour shifts in a jurisdiction where local law allows it but ILO conventions recommend 8 hours. The normalizer flags this with alternatives: "Your timeline assumes 16-hour days, which local law permits but creates worker burnout risk and may trigger international audit failures. A revised timeline with 10-hour days would extend the project by X weeks but reduces reputational risk."

**Key principle:** Local legality is not the same as ethical approval. Many developing nations have weak labor laws by international standards, sometimes deliberately (to attract foreign investment), sometimes due to lack of enforcement capacity. PlanExe should help creators understand when they're operating in a legal-but-risky zone and offer alternatives that protect both workers and the business's reputation.

**How This Works in Practice:**

A plan for a textile factory in Bangladesh that budgets $8/day for workers (above local minimum wage but below living wage) with proper safety equipment and reasonable hours would receive an **informational flag** — noting that wages fall below ILO living wage estimates for the region. The report presents an alternative: "Fair-wage version budgeting $12/day (closer to ILO guidelines) with the same safety provisions would increase labor costs by ~$80,000/year. This reduces supply chain audit risk and improves worker retention."

A plan for the same factory at $2/day with no safety budget would receive a **critical flag with strong alternatives**. The report shows: original cost ($2/day, no safety = $35,000/year), the ethical alternative ($8/day with safety equipment and training = $140,000/year), and a middle option ($5/day with basic safety = $87,500/year). It explains the reputational, audit, and retention risks of each approach. The creator chooses.

A plan that explicitly describes forced labor or involves workers clearly under the ILO minimum age (under 15 in most countries, under 18 in hazardous work) receives a **critical risk flag** with strong emphasis on the ethical alternative — but the system still generates the plan, showing both the original and the revised approach. The creator has autonomy, but with full information about the risks and consequences.

**The key principle:** PlanExe should help people build legitimate businesses, including in developing economies where costs are genuinely lower. It should not help people exploit vulnerable workers. When costs are being cut through exploitation, PlanExe shows the creator a better way — not by blocking the plan, but by offering a version that's both more ethical AND more profitable in the long term (reduced audit risk, better retention, access to premium markets).

## Inclusive Workforce Planning: Beyond the Stereotypical Worker

The ethical framework above focuses on preventing exploitation and ensuring fair treatment of workers. But there is another form of discrimination that planning systems often fail to address: the implicit assumption that the workforce should look a specific way.

When business plans are created, they often implicitly assume a narrow "ideal worker" — typically a healthy, young-to-middle-aged male with no disabilities, no caregiving responsibilities, and full-time availability. This assumption is so embedded in planning conventions that it often goes unexamined. A plan that budgets for "full-time" positions assumes workers can work 40 hours a week every week. A timeline that requires employees to "ramp up quickly" assumes no need for mentoring or gradual onboarding. A safety plan that mentions "standard equipment" often omits accommodations for workers with specific needs. These assumptions are not malicious, but they are discriminatory in effect — they systematically exclude large portions of the population from consideration, without explicit justification.

PlanExe should not reinforce these biases. When the normalizer detects workforce assumptions that exclude or disadvantage specific groups, it should identify the exclusion, explain why it matters, and offer inclusive alternatives that often perform better both ethically and economically.

### The Specific Forms of Discrimination

**Gender Discrimination and the Pregnancy Penalty:** Many business plans, especially in fields like construction or manufacturing, implicitly assume an all-male or predominantly male workforce. This assumption often stems from outdated perceptions about physical capability or workplace culture. More pernicious is the explicit "pregnancy penalty" — the practice of avoiding hiring women of childbearing age because of the costs associated with parental leave. A plan that assumes 100% male workforce without justification, or that explicitly factors in "pregnancy risk" as a cost reduction, is engaging in gender discrimination. The normalizer should flag these assumptions and present an alternative: budget for parental leave as a standard business cost (it is one), and note that diverse teams consistently outperform homogeneous ones in innovation, decision-making, and customer understanding. If a plan budgets labor at $8/hour for male workers but would only hire women at $5/hour to avoid pregnancy-related costs, that disparity gets flagged with an inclusive alternative showing the true cost of treating parental leave as a one-time business expense rather than avoiding it through discrimination.

**Disability Exclusion and Reasonable Accommodations:** A significant portion of the population has disabilities — visible or invisible — that do not prevent them from performing valuable work. Yet workforce plans routinely assume all workers must be fully able-bodied and available for the full standard schedule. A person with mobility constraints might be a gifted software developer who needs a modified desk setup. A person with severe anxiety disorder might be an excellent specialist who works most effectively with a flexible schedule or partial remote work. A person who can only work three days a week due to chronic illness might deliver more focused, high-quality output than a full-time worker spreading themselves thin. The normalizer should flag plans that assume all workers must be full-time, on-site, and without accommodation needs. It should then present inclusive alternatives: suggest flexible work arrangements, reasonable accommodations (which often cost far less than assumed), and the business benefits of accessing talent pools competitors overlook. When a plan's timeline assumes continuous full-time availability but could accommodate four-day weeks or split shifts at minimal cost change, the normalizer surfaces this alternative with explicit cost comparisons and retention benefits.

**Mental Health as a Business Cost:** Workers with depression, anxiety, PTSD, or other mental health conditions are often excluded from workforce planning as though mental health support were a luxury rather than a cost of doing business. Yet mental health conditions are common, treatable, and compatible with excellent work performance when properly supported. A plan that does not account for mental health support — whether through EAP (Employee Assistance Programs), mental health days, flexible scheduling during mental health crises, or therapeutic support — is implicitly assuming workers do not have mental health needs. This assumption is both statistically incorrect (approximately 1 in 5 adults experience mental illness in a given year) and economically unsound (untreated mental health issues drive absenteeism and turnover). The normalizer should flag plans that don't budget for mental health support, then present an inclusive alternative: add a mental health budget line item (often 2-3% of labor costs for programs like EAP, mental health days, and flexible scheduling). The alternative shows how these investments reduce turnover, decrease unplanned absences, and improve overall productivity — making inclusion cost-positive for the business.

**Racial and Ethnic Discrimination in Pay and Hiring:** Some plans implicitly or explicitly assume different pay rates for workers of different ethnicities, or target hiring from specific ethnic groups to access cheaper labor markets or stereotyped skills. Others may propose pay structures that, while not explicitly discriminatory, correlate with demographic assumptions — assuming "entry-level workers" (coded as young and potentially from particular backgrounds) will earn less than "senior workers," then staffing the entry-level roles with workers from specific ethnic backgrounds. The normalizer should flag pay disparities that correlate with demographic assumptions, and it should identify any hiring assumptions that target specific ethnic groups or regions as a way to reduce labor costs. The inclusive alternative is straightforward: standardized pay scales based on role, experience, and demonstrated capability, not identity. When a plan proposes different wage tracks for "local hires" versus "immigrant workers" in the same role, or assumes workers from certain countries will accept lower wages, the normalizer surfaces this as discrimination and presents an alternative with unified pay scales, showing how this reduces legal risk, improves retention, and strengthens workplace culture.

**Age Discrimination and Knowledge Transfer:** Business plans often target workers in a narrow age range — typically 25 to 45 — explicitly or implicitly excluding younger workers (perceived as needing too much training) and older workers (perceived as past their prime or less adaptable). This assumption ignores the reality that mixed-age teams have significant advantages: younger workers bring current skills and energy, older workers bring experience and stability, and mentoring structures between them improve knowledge transfer and reduce single-points-of-failure. A plan that assumes a 25-45 age band without justification, or that explicitly budgets extra training costs for workers outside this range, is engaging in age discrimination. The normalizer should flag unrealistic age-range assumptions and present an inclusive alternative: design the team with explicit age diversity and mentoring structures. The alternative shows how this approach reduces knowledge loss when experienced workers leave (a major cost center in knowledge-intensive fields), improves innovation through perspective diversity, and accesses talent in both the entry-level and experienced-worker markets competitors might neglect.

### Why This Matters

The underlying principle is simple: PlanExe should help create plans that give ALL people better work conditions, not narrowly optimize for the stereotypical "ideal worker." When the normalizer detects assumptions that exclude or disadvantage certain groups, it should offer inclusive alternatives that show the business benefits of diversity. These benefits are not merely ethical — they are economic:

- **Broader talent pool:** When you exclude entire categories of people, you're choosing to hire from a smaller, more competitive pool. Inclusive hiring expands the pool and reduces recruitment costs.
- **Lower turnover:** Inclusive workplaces retain employees better. Research across industries shows that employees from historically underrepresented groups stay longer in inclusive environments, reducing the turnover cost (estimated at 50-200% of annual salary per replacement).
- **Innovation and problem-solving:** Diverse teams make better decisions, solve problems more creatively, and catch blind spots that homogeneous teams miss. This translates directly to better products, fewer costly mistakes, and faster adaptation to market changes.
- **Legal and compliance risk reduction:** Plans that explicitly or implicitly discriminate create legal exposure. Inclusive alternatives reduce this risk while improving the business's position if audited or challenged.
- **Access to government incentives:** Many jurisdictions offer tax credits, grants, or preferential bidding for businesses that demonstrate inclusive hiring practices, particularly for hiring workers with disabilities or from disadvantaged backgrounds.
- **Customer loyalty and brand reputation:** Customers increasingly prefer to do business with companies that treat workers fairly and practice inclusion. This can translate to premium pricing, customer loyalty, and willingness to recommend the business.

### How the Normalizer Detects Exclusionary Assumptions

The normalizer checks workforce assumptions against specific inclusion indicators. When these indicators suggest exclusionary assumptions, the system flags them and generates inclusive alternatives with adjusted costs. The checks include:

**Demographic profiling:** Does the plan assume a specific demographic profile for ideal workers (age range, gender, disability status, family structure)? If so, the plan is flagged with a note explaining which groups are implicitly excluded and why this assumption warrants reconsideration.

**Accommodation budgeting:** When a plan involves workers, does it budget for accommodations that some workers might need (modified workspace, assistive technology, flexible scheduling support)? If not, the plan assumes an able-bodied, fully available workforce.

**Parental leave and family support:** Is parental leave included as a standard business cost, or is it absent from labor cost calculations? Absence suggests an implicit assumption that workers have no caregiving responsibilities.

**Flexible work provisions:** Does the plan allow for flexible hours, remote work, part-time arrangements, or other non-standard schedules where operationally feasible? If all assumptions are "full-time on-site," the plan excludes workers who need flexibility.

**Pay scales and demographics:** Do pay scales vary by role and demonstrated capability, or do they correlate with demographic assumptions (age, ethnicity, gender)? The normalizer compares proposed pay against role-based benchmarks to detect implicit demographic-based pricing.

**Team composition and diversity:** Does the plan include explicit attention to hiring or team diversity, or are hiring decisions framed in purely functional terms without demographic consideration? Inclusion means actively designing for diversity, not just happening to be diverse.

When gaps are found, the normalizer generates inclusive alternatives with adjusted costs and timeline implications. For example, if a plan assumes a full-time workforce but could accommodate flexible arrangements at no schedule cost change, that alternative is shown. If a plan excludes people with disabilities due to assumed accommodation costs, the alternative calculates reasonable accommodations (often <$5K per person annually) and shows the net benefit of accessing that talent. If a plan implicitly assumes a male-only workforce, the alternative shows the cost of genuine gender-neutral hiring and the performance benefits of diverse teams.

The creator sees the original assumptions, the inclusive alternatives, and the quantified tradeoffs — and chooses how to proceed. PlanExe does not force inclusion. It makes the case for inclusion visible, financially quantified, and grounded in business benefits, not just ethics.

## How the Normalizer Works

### Step 1: Domain Detection

When a plan enters the normalizer, the first task is figuring out what kind of plan it is. A carpentry business operates in a fundamentally different domain than a SaaS startup, and the normalizer needs to know which domain's rules to apply.

Domain detection works by analyzing the plan text for signals: currencies mentioned (DKK suggests Scandinavia), units used (board-feet suggests woodworking), and keywords that indicate the industry (mentions of "patients" and "appointments" suggest healthcare). Each signal contributes to a score, and the highest-scoring domain profile wins.

Domain profiles are stored as external YAML configuration files — not hardcoded in Python. This means new domains can be added without changing source code, and existing domains can be tuned without redeploying. Each profile defines the domain's default currency, preferred units, confidence keyword mappings, and plausibility ranges.

### Step 2: Assumption Extraction

Before assumptions can be normalized, they need to be structured. The LLM produces free-text assumptions like "The project will cost between $50,000 and $75,000 over 6-8 months with a team of 3-5 people." The QuantifiedAssumptionExtractor parses these into typed objects:

```python
@dataclass
class QuantifiedAssumption:
    assumption_id: str
    claim: str                  # "The project will cost $50,000-$75,000"
    lower_bound: Decimal        # Decimal("50") — exact, no floating-point drift
    upper_bound: Decimal        # Decimal("75") — exact, no floating-point drift
    exponent: int               # 3 (actual value = mantissa × 10^exponent)
    unit: str                   # "USD"
    region: str                 # "KE" (ISO 3166-1 alpha-2 country code)
    confidence: str             # "medium"
    evidence: str               # Supporting text from the plan
    span_ratio: float           # upper/lower = 1.5
```

**Why Decimal, Not Float:**

Financial values must be stored as `Decimal` (from Python's `decimal` module), not `float`. A float cannot represent $25.00 exactly — it becomes 24.999999... or 25.000001... due to binary floating-point representation. This matters when comparing costs, checking thresholds, and displaying values to users. `Decimal("25.00")` is exactly 25.00, always. This is the same approach used by banks, payment processors, and financial software worldwide. The `span_ratio` field remains a `float` because it's a relative ratio, not a monetary value — precision drift in a ratio like 1.4999 vs 1.5000 is inconsequential.

**Why the Exponent Field Matters:**

The exponent field is critical for handling the enormous range of monetary values that real-world plans encounter. Rather than storing raw numbers — which quickly become unwieldy in hyperinflation economies or meaninglessly small for cryptocurrency — the assumption extractor separates every numeric value into a mantissa and a power-of-ten exponent. This is essentially scientific notation for business planning.

A carpentry project costing $50,000–$75,000 is stored as mantissa 50–75 with exponent 3. The same project priced in Venezuelan Bolívares at 50,000,000–75,000,000 becomes mantissa 50–75 with exponent 6. A Bitcoin-denominated project at 0.0050–0.0075 BTC becomes mantissa 50–75 with exponent -4. In all three cases, the mantissa is the same — only the scale changes.

This separation has several practical benefits. First, it makes cross-currency comparison meaningful: the normalizer can compare mantissas directly once currencies are converted, without dealing with strings of zeros. Second, it handles hyperinflation gracefully — when Zimbabwe's dollar inflated, prices went from hundreds to trillions, but the underlying economic relationships (relative costs of labor vs materials) remained expressible in small mantissa values at different exponents. Third, it future-proofs the system for whatever monetary forms emerge — whether a country adopts Bitcoin, a regional digital currency, or a new post-crisis denomination, the exponent simply adjusts.

Each assumption also carries a `region` tag — an ISO 3166-1 country code identifying where the assumption applies. This is critical because the same exponent and unit combination means very different things in different regions. A labor cost of mantissa 50 with exponent 0 in USD means $50/hour in the US (reasonable for skilled trades) but $50/hour in rural Kenya (wildly high). The region tag lets downstream pipeline tasks — governance, risk assessment, ethical checks, FermiSanityCheck — apply location-appropriate plausibility ranges, living wage comparisons, and regulatory context without re-parsing the original claim text. The region is stored directly in the assumption JSON so every downstream consumer has immediate access to it.

The extractor determines the appropriate exponent automatically by analyzing the currency, location, and context. For stable currencies like USD, EUR, or CHF, the exponent typically ranges from -2 to 6. For currencies experiencing active hyperinflation, the exponent might reach 9 or higher. For cryptocurrencies, negative exponents are common. The exponent is always stored alongside the value so downstream tasks can reconstruct the actual amount when needed: `actual_value = mantissa × 10^exponent`.

**Examples of Exponent-Based Representation:**

```
# Standard USD project
claim: "Labor costs $50,000-$75,000"
lower_bound: 50, upper_bound: 75, exponent: 3, unit: USD
# actual: $50,000 - $75,000

# Venezuelan Bolívares (hyperinflation)
claim: "Office rent 50,000,000-75,000,000 VES/month"
lower_bound: 50, upper_bound: 75, exponent: 6, unit: VES
# actual: 50,000,000 - 75,000,000 VES

# Bitcoin micro-transaction
claim: "Server costs 0.0050-0.0075 BTC/month"
lower_bound: 50, upper_bound: 75, exponent: -4, unit: BTC
# actual: 0.0050 - 0.0075 BTC

# Small item (individual nails)
claim: "Nails cost $0.005-$0.008 each"
lower_bound: 5, upper_bound: 8, exponent: -3, unit: USD
# actual: $0.005 - $0.008

# Post-denomination (country resets currency)
claim: "After redenomination, costs 50-75 new ZWL"
lower_bound: 50, upper_bound: 75, exponent: 0, unit: ZWL
# actual: 50 - 75 ZWL
```

This extraction handles multiple formats: explicit ranges ("10-20"), approximate values ("about $50K"), percentage ranges ("15-25% growth"), and duration estimates ("3-6 months"). It also detects currencies from context — "$" is USD in an American plan but might be AUD, CAD, or SGD depending on other signals. The extractor automatically identifies the optimal exponent by analyzing the magnitude of extracted numbers and the currency context, ensuring mantissas always fall within a reasonable range (typically 1–999) for clarity and precision.

### Step 3: Fermi Sanity Check

Named after Enrico Fermi's famous estimation technique, this step asks: "Are these numbers physically plausible?" It's not checking whether the business plan is good — it's checking whether the numbers describe something that could exist in reality.

The sanity check operates at several levels:

**Magnitude checks** compare proposed costs against real-world benchmarks from the Resource/Cost Data Service. If a plan says a carpenter in Nairobi charges $500/hour, that's flagged — real rates are $5-15/hour. If a plan says commercial rent in Manhattan is $5/sqft/month, that's also flagged — real rates are $60-150/sqft/year.

**Span ratio checks** look at the range between lower and upper bounds. A span ratio of 1.5 (upper is 50% more than lower) is normal uncertainty. A span ratio of 100 (upper is 100× the lower bound) suggests the assumption is meaningless — the planner has no idea what the real number is. These are flagged for review.

**Internal consistency checks** compare assumptions against each other within the same plan. If one assumption says the team is 3 people and another says labor costs are $2M/year for entry-level work, something doesn't add up.

**Ethical checks** (described above) evaluate labor costs, safety provisions, and working conditions against international standards.

### Step 4: Currency and Unit Normalization

Once assumptions are extracted and sanity-checked, they're normalized to a standard representation. All monetary values are converted using the Currency Service (live exchange rates, not hardcoded) to the domain's default currency plus a EUR equivalent for cross-plan comparison. All physical measurements are converted to metric. Duration estimates are standardized to a common format.

This normalization makes it possible to compare plans across domains and regions. A carpentry workshop in Copenhagen and one in Nairobi can be meaningfully compared when their costs are expressed in the same currency at current exchange rates, rather than in whatever currency the LLM happened to generate.

### Step 5: Confidence Calibration

Different domains have different relationships with certainty. When a master carpenter says "this will take three weeks," that estimate is grounded in decades of experience — it deserves high confidence. When a first-time startup founder says "we'll achieve product-market fit in three months," that's aspirational at best — it deserves low confidence regardless of how certain they sound.

The normalizer applies domain-specific confidence calibration using keyword analysis. Each domain profile defines which phrases indicate high, medium, and low confidence in that domain's context. "Standard practice in the industry" means high confidence in an established trade but medium confidence in a novel technology sector. "Based on comparable projects" means high confidence if those projects are cited, medium if they're vague.

## Example Domain Profile

Domain profiles are external YAML configuration files that define domain-specific rules, currency preferences, units, and confidence mappings. Below is a real-world example: the Carpentry & Woodworking domain.

```yaml
# Domain Profile: Carpentry / Woodworking
# File: domain-profiles/carpentry.yaml

profile:
  id: carpentry
  name: "Carpentry & Woodworking"
  description: >
    Covers woodworking businesses from small workshops to furniture
    manufacturers. Includes custom cabinetry, construction carpentry,
    and artisan woodcraft.

  detection:
    keywords:
      - wood
      - lumber
      - timber
      - cabinet
      - furniture
      - joinery
      - carpentry
      - woodwork
      - plywood
      - veneer
      - hardwood
      - softwood
    currency_signals:
      - DKK
      - EUR
      - USD
      - SEK
      - NOK
    unit_signals:
      - board_feet
      - linear_feet
      - cubic_meters
      - sheets

  currency:
    default: EUR
    display_precision:
      material_cost: 2        # €12.50 per board foot
      labor_hourly: 0         # €45/hour (no cents needed)
      project_total: 0        # €15,000 (round numbers)
    # Some markets use a different operational currency than official
    dual_currency_awareness:
      - country: TR           # Turkey
        official: TRY
        operational: [EUR, USD]
        note: "High inflation — many B2B transactions in EUR/USD"
      - country: ZW           # Zimbabwe
        official: ZWL
        operational: [USD]
        note: "USD is de facto currency for most business transactions"

  units:
    prefer_metric: true
    conversions:
      - from: board_feet
        to: cubic_meters
        factor: 0.00236
      - from: linear_feet
        to: meters
        factor: 0.3048
      - from: inches
        to: centimeters
        factor: 2.54
      - from: square_feet
        to: square_meters
        factor: 0.0929

  confidence:
    high:
      - "standard practice"
      - "I've done this"
      - "always costs"
      - "industry standard"
      - "based on 20 years"
      - "we quoted"
    medium:
      - "typically"
      - "usually"
      - "expect"
      - "comparable projects"
      - "market rate"
    low:
      - "rough estimate"
      - "maybe"
      - "not sure"
      - "guessing"
      - "could be"

  plausibility:
    labor:
      hourly_rate:
        global_range: [3, 150]          # USD equivalent
        developed_market: [25, 150]
        developing_market: [3, 40]
      living_wage_source: "Global Living Wage Coalition"
    materials:
      # Precision varies: lumber is 2 decimals, nails might be 4
      precision_rules:
        - category: lumber
          min_decimals: 2
          typical_range_usd: [2.00, 50.00]    # per board foot
        - category: hardware
          min_decimals: 3
          typical_range_usd: [0.005, 25.00]   # single nail to hinge
        - category: finish
          min_decimals: 2
          typical_range_usd: [15.00, 80.00]   # per gallon/liter
    project_budget:
      small_workshop: [500, 50000]            # EUR
      medium_business: [50000, 500000]
      large_operation: [500000, 5000000]

  ethical:
    labor_standards:
      # Use the HIGHER of local law or ILO recommendation
      standard_source: "max(local_law, ilo_recommendation)"
      flag_when_local_below_ilo: true
      flag_message: >
        Local labor laws in {country} fall below ILO recommendations
        for {standard}. This creates reputational and supply chain
        audit risks. Consider adopting ILO standards.
    safety:
      required_for:
        - chemical_finishing
        - power_tools
        - heavy_machinery
        - dust_exposure
      missing_safety_flag: >
        Plan involves {hazard} but does not budget for protective
        equipment or safety training. Recommended addition: {recommendation}
```

This example shows:
- **Domain detection:** Keywords, currency signals, and units that identify carpentry plans
- **Dual currency awareness:** Turkey and Zimbabwe use official currencies that aren't operationally realistic; the profile notes the currencies actually used in business transactions
- **Numeric precision rules:** Not all costs are rounded to the nearest dollar. Hardware items (nails) need 3 decimal places ($0.005), while project totals are rounded to whole dollars
- **Plausibility ranges:** Labor rates and material costs vary by market; the profile defines realistic ranges
- **Ethical standards:** Uses the *higher* of local law or ILO recommendations, and flags when local law is itself inadequate

## Numeric Precision and Currency Reality

A common mistake in planning systems is assuming all costs are nice round numbers in major currencies. Reality is messier.

**Arbitrary Precision in Costs:**
Not all costs are in the thousands. A single sheet of paper costs $0.01. A single nail costs $0.005. A bottle of epoxy finish costs $25.50. These are not rounding errors — they're real costs that compound across a project. The normalizer must handle arbitrary decimal precision, not force everything into dollars-and-cents. This is why the domain profiles include `precision_rules` that specify the minimum decimals for different categories.

**Hyperinflation and Currency Reality:**
In some countries, official currency values bear no relationship to real purchasing power. A bottle of water in Venezuela might cost 1,000,000 Bolívares — not because of a data entry error, but because hyperinflation has made the currency nearly worthless for real transactions. The normalizer must recognize hyperinflation contexts and flag them, often recommending that costs be expressed in a more stable reference currency (USD, EUR) instead.

**Dual-Currency Operations:**
Many countries maintain an official currency while actually operating on a different, more stable currency. Examples:
- **Zimbabwe:** Official ZWL (Zimbabwean Dollar), but USD is the de facto currency for most business transactions — stable, trusted, widely accepted
- **Venezuela:** Official VES (Bolívar), but USD circulates widely for B2B transactions due to hyperinflation of the local currency
- **Turkey:** Official TRY (Turkish Lira), but many B2B and international transactions use EUR or USD due to persistent high inflation
- **Cuba:** Official CUP (Cuban Peso), but CUC (convertible peso) was historically used for tourism/business; now largely unified but the legacy of dual currency remains
- **Lebanon:** Official LBP (Lebanese Pound), but USD has been the operational currency for most real transactions since the 2019 financial crisis

The normalizer should detect these dual-currency contexts and present costs in both currencies, with a clear note about which one reflects real purchasing decisions. A plan costing 500,000 TRY sounds expensive; the same plan costing €25,000 is more honest about its real cost in Turkey's economy.

**Handling Precision and Reality Together:**
When the normalizer encounters a cost in a hyperinflated currency, it should:
1. Recognize the hyperinflation context from domain/location signals
2. Offer a conversion to a stable reference currency (USD or EUR at current rates)
3. Flag this as a currency reality issue for the plan creator
4. Include both in reports, with the stable-currency version as the primary number

This ensures plans are grounded in economic reality, not in currency fantasy.

## Integration with PlanExe Pipeline

The normalizer integrates as a Luigi task in the existing DAG:

```
MakeAssumptions → [QuantifiedAssumptionExtractor] → [FermiSanityCheck] → [DomainNormalizer] → DistillAssumptions
```

The three new tasks (in brackets) are inserted between the existing MakeAssumptions and DistillAssumptions tasks. Each produces output files following PlanExe's standard naming convention:

- `003-12-fermi_sanity_check_report.json` — detailed per-assumption verdicts
- `003-13-fermi_sanity_check_summary.md` — human-readable summary of findings
- `003-14-normalized_assumptions.json` — all assumptions in standard representation

The FermiSanityCheck report includes a section on ethical flags, making it visible to both the downstream pipeline tasks and human reviewers.

## Implementation Phases

**Phase 1** builds the QuantifiedAssumptionExtractor as a standalone component. This is pure text parsing and structuring — no external service dependencies, easy to test, and immediately useful even without the other components.

**Phase 2** creates the domain profile system. YAML schema, profile loader, default profiles for common domains (construction/trades, healthcare, technology, food service, non-profit, personal/lifestyle). Detection algorithm with scoring. This is also standalone and testable.

**Phase 3** implements FermiSanityCheck. This component benefits from the Resource/Cost Data Service but works without it — falling back to heuristic thresholds when real-world benchmark data isn't available. The ethical assessment layer is part of this phase.

**Phase 4** brings it all together with the DomainNormalizer. Currency normalization via the Currency Service, unit conversion, confidence calibration, and full pipeline integration. This phase depends on all previous phases plus the Currency Service implementation.

Each phase will be submitted as a separate, focused PR.

## What Changed from PR #72

PR #72 attempted to deliver all of this in a single 1,260-line PR with hardcoded currency rates, a fragile file path for domain profiles, and no ethical framework. It was closed because the foundation wasn't right.

This proposal redesigns the same vision with proper prerequisites (Currency Service and Resource/Cost Data Service, both now merged as proposals), external configuration instead of hardcoded values, phased delivery instead of a monolithic PR, and — critically — an ethical framework that ensures PlanExe's cost optimization doesn't become a tool for labor exploitation.

The technical problems (hardcoded rates, fragile paths, uppercase loggers) were symptoms. The real issue was that cost normalization needs to be built on real data services and ethical principles, not hardcoded approximations. That foundation is now in place.

## Open Questions

1. Should the ethical assessment be configurable (strictness levels) or fixed to ILO standards?
2. How should PlanExe handle plans that are borderline — legal in the target jurisdiction but below ILO recommendations?
3. Should the FermiSanityCheck block pipeline progression (hard gate) or annotate and continue (soft gate)?
4. What's the minimum set of domain profiles for a useful v1?
5. How do we source living wage data reliably? The Global Living Wage Coalition covers ~30 countries — what about the rest?
6. Should the ethical flags be visible to the end user or only to the pipeline operator?
