# Proposal: Currency Exchange Rate Service for PlanExe

## Problem Statement

PlanExe generates plans for any country and currency. The current approach (PR #72) hardcoded a handful of currency conversion rates (USD, EUR, DKK, GBP, CAD) which is fundamentally wrong:

- Plans can involve ANY currency the LLM knows about
- Exchange rates change daily
- Multi-year plans need volatility data (a 5-year plan in Turkish Lira vs Swiss Franc has very different risk profiles)
- Geopolitical events (wars, tariffs, currency unions) affect long-term currency assumptions
- Future currencies may be digital/crypto

## Requirements

- Daily exchange rate updates (or on-demand)
- Support for ALL ISO 4217 currencies (not a hardcoded subset)
- Historical rate data for volatility calculations
- Graceful degradation when offline
- No hardcoded rates in PlanExe source code
- Cache layer to avoid hammering APIs during pipeline runs

## Approach Analysis

### Approach A: Python Module with Local Cache

**Description:** A standalone Python module within worker_plan that fetches rates from a free API (ECB, exchangerate.host, Open Exchange Rates) and caches them locally (JSON file or SQLite).

**Pros:**
- Simplest to implement and deploy — no extra services to run
- Works in single-user/local mode without Docker
- Easy to test (mock the API, test cache logic)
- No infrastructure overhead
- Can bundle a fallback rates file for offline use
- Fits existing PlanExe architecture (just another module)

**Cons:**
- Tightly coupled to worker_plan
- Cache management is the module's responsibility
- No shared cache between multiple worker instances
- Rate limiting on free APIs could be an issue in CI/testing
- Volatility calculations need historical data (more complex caching)

### Approach B: Docker Microservice

**Description:** A separate Docker container that exposes a REST API for currency conversion. Runs alongside PlanExe's existing Docker Compose stack. Fetches and caches rates internally.

**Pros:**
- Clean separation of concerns — currency is its own service
- Shared cache across all PlanExe instances
- Can be scaled/replaced independently
- Could serve other services beyond PlanExe
- Natural fit for the existing Docker Compose deployment
- Can include a proper database for historical rates

**Cons:**
- Adds infrastructure complexity (another container to manage)
- Doesn't work for single-user/local-only mode without Docker
- Network latency between services (minor)
- More complex deployment and testing
- Overkill for simple single-plan runs

### Approach C: MCP Tool

**Description:** Implement currency conversion as an MCP (Model Context Protocol) tool that the LLM can call directly during plan generation.

**Pros:**
- LLM-native — the model can request conversions when it needs them
- Flexible — LLM decides when and what to convert
- Fits PlanExe's existing MCP infrastructure
- Could combine with other financial data tools
- No preprocessing step needed — conversion happens during generation

**Cons:**
- Depends on LLM tool-calling reliability (local models may not call tools correctly)
- Less predictable — LLM might not call the tool when it should
- Harder to test deterministically
- Currency data still needs to come from somewhere (API/cache)
- Doesn't solve the pipeline-level normalization need (post-generation validation)
- Each LLM call adds latency

### Hybrid Approach: Python Module + MCP Tool

**Description:** Python module for pipeline-level normalization (reliable, deterministic) PLUS an MCP tool for LLM-initiated conversions during generation.

**Pros:**
- Best of both worlds — deterministic pipeline normalization + LLM flexibility
- MCP tool is optional (degrades gracefully)
- Python module handles the critical path
- Can share the same underlying rate cache

**Cons:**
- More code to maintain
- Two interfaces to the same data

## Ranking

1. **Python Module with Local Cache** — Start here. Simplest, works everywhere, no infrastructure overhead. Covers 90% of the need.
2. **Hybrid (Module + MCP Tool)** — Natural evolution once the module exists. Add MCP tool later for LLM-initiated conversions.
3. **Docker Microservice** — Only if PlanExe moves to a multi-service architecture. Overkill for current state.
4. **MCP Tool alone** — Too unreliable as the sole approach. LLM tool-calling isn't deterministic enough for critical financial data.

## Recommended Implementation Plan

### Phase 1: Python Module (this proposal)

- Fetch rates from ECB (free, no API key, 60+ currencies)
- Local JSON cache with TTL (24h default)
- Fallback rates file bundled for offline use
- ISO 4217 currency code validation
- Simple API: `get_rate(from_currency, to_currency, date=None)`
- Historical rates for volatility calculation: `get_volatility(currency_pair, years=5)`

### Phase 2: MCP Tool (future)

- Wraps the Python module as an MCP tool
- LLM can request conversions during plan generation
- Optional — pipeline still works without it

### Phase 3: Enhanced Data (future)

- Historical volatility modeling
- Geopolitical risk factors
- Digital/crypto currency support

## Data Sources (Free, No API Key)

- **ECB (European Central Bank)**: https://data.ecb.europa.eu — 60+ currencies, daily, free, reliable
- **exchangerate.host**: https://exchangerate.host — 170+ currencies, free tier
- **Frankfurter**: https://frankfurter.app — ECB data via REST API, no key needed

## Open Questions

1. Should the cache be per-plan or global?
2. What's the acceptable staleness for rates? (24h? 1 week for offline mode?)
3. Should volatility data be pre-computed or calculated on demand?
