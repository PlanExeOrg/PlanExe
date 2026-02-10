# 26) News Intake + Opportunity Sensing Grid for Autonomous Bidding

## Pitch
Build a continuous news-intake grid that detects project opportunities (bridge, IT infrastructure, utilities, public procurement) and turns them into structured planning prompts at scale.

## Why
If an autonomous AI organization generates ~1000 plans/day, the bottleneck is not planning—it is **finding high-value opportunities early** and classifying them correctly.

## Proposal
Implement a multi-source intake pipeline:
1. Ingest signals from procurement feeds, industry media, government notices, and infrastructure newsletters.
2. Normalize each item to an `opportunity_event` schema.
3. Score urgency + bidability + strategic fit.
4. Auto-generate candidate prompts for plan creation.

## Source categories to monitor
- Public procurement portals (national + regional)
- Government transport/infrastructure bulletins
- Utility/telecom modernization notices
- Construction/engineering trade publications
- Press wires (major project announcements)
- Local/regional news for early non-centralized opportunities

## Core schema
```json
{
  "event_id": "...",
  "source": "...",
  "domain": "bridge|it_infra|energy|...",
  "region": "...",
  "estimated_budget": "...",
  "deadline_hint": "...",
  "confidence": 0.0,
  "raw_text": "..."
}
```

## Success metrics
- Opportunity recall vs known project announcements
- Time-to-detection after first public signal
- % opportunities converted to high-quality planning prompts
