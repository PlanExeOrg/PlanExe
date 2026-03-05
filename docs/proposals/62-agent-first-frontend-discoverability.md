# Agent-First Frontend Discoverability for PlanExe

## Goal

Make PlanExe discoverable and usable by AI agents (OpenClaw, OpenAI Agents, Codebuff, Claude Code, etc.) via standard protocols and metadata. Enable AI agents to autonomously discover, understand, and integrate with PlanExe as a planning and orchestration tool.

## Problem Statement

PlanExe is a powerful planning orchestrator, but AI agents cannot easily:
1. **Discover** PlanExe's capabilities (what it does, what endpoints exist)
2. **Understand** how to use it (API schemas, tool signatures, authentication)
3. **Integrate** with it (function calling, MCP servers, or direct API calls)
4. **Trust** it (pricing, credit models, rate limits)

This limits PlanExe's reach to agents and orchestrators that require manual integration or discovery via documentation.

## Implementation Status Snapshot (2026-02-21)

This section records what is currently implemented in the repository and what still needs work.

### In Place

- Canonical discovery file exists at `public/llms.txt` (single source of truth).
- `mcp_cloud` serves:
  - `GET /llms.txt` (canonical)
  - `GET /llm.txt` (legacy alias redirect)
- `frontend_multi_user` serves:
  - `GET /llms.txt` (from `public/llms.txt`)
  - `GET /llm.txt` (legacy alias redirect)
- Docker packaging copies `public/llms.txt` into both `mcp_cloud` and `frontend_multi_user` images.
- Near-duplicate files under `mcp_cloud/` were removed to avoid drift.
- `public/llms.txt` now reflects current production guidance:
  - MCP endpoint is `/mcp` (not `/sse`)
  - MCP auth uses `X-API-Key`
  - Tool names match current MCP tools (`example_prompts`, `task_create`, `task_status`, `task_stop`, `task_file_info`)
  - Pricing/cost docs point to `https://docs.planexe.org/costs_and_models/`
  - Support contact includes Discord: `https://planexe.org/discord`
- Current positioning is documented:
  - `home.planexe.org` is human-facing (account/billing/docs links)
  - `mcp.planexe.org` is the AI-facing API surface
  - PlanExe supports self-hosted/offline scenarios with local model runtimes.

### Missing or Not Yet Verified

- Production verification for:
  - `https://home.planexe.org/llms.txt`
  - `https://mcp.planexe.org/llms.txt`
- Recommendation 2 (`/openapi.json` on home domain): currently not aligned with the stated architecture that `home.planexe.org` is human-facing only.
- Recommendation 3 (`/.well-known/mcp.json`): not implemented in `mcp_cloud` currently.
- Recommendation 4 (`robots.txt` updates): no `public/robots.txt` found in this repository.
- Recommendation 5 (JSON-LD/meta tags): not verified here.
- Recommendation 6 (README agent section): not verified here.

### Proposal Text That Is Now Outdated

- OpenAPI examples in this proposal remain illustrative and should not be interpreted as current production host routing.
- Any numeric pricing/rate-limit examples in this document are placeholders unless explicitly verified against live product policy docs.

## Recommendations

### 1. `llms.txt` – LLM Agent Discovery

**What:** Add `/llms.txt` to both `home.planexe.org` and `mcp.planexe.org` (cloud) and locally for Docker deployments.

**Why:** The `llms.txt` standard (emerging from orchestrator patterns) tells AI agents:
- What PlanExe does (elevator pitch)
- How to generate a plan (API endpoint, MCP endpoint, or local Docker)
- Authentication (API key, MCP auth, or open local)
- Available tools/capabilities (plan generation, status polling, artifact retrieval)
- Credit model and pricing
- Rate limits and quotas

**Content (sample, aligned with current state):**

```text
# PlanExe - AI Project Planning for Agents

PlanExe turns broad goals into structured strategic-plan drafts and downloadable artifacts.

## Service Endpoints

- Human-facing site: https://home.planexe.org
- AI-facing MCP: https://mcp.planexe.org/mcp
- Agent discovery files:
  - https://home.planexe.org/llms.txt
  - https://mcp.planexe.org/llms.txt
  - https://mcp.planexe.org/llm.txt (legacy alias redirect)

## MCP Tools

- example_prompts
- task_create
- task_status
- task_stop
- task_file_info

Recommended flow:
1) example_prompts
2) task_create
3) task_status (poll every 5 minutes)
4) task_file_info

## Authentication

1) Create account at https://home.planexe.org
2) Generate API key in Account -> API Keys
3) Send header: X-API-Key: <API_KEY>

## Cost and runtime notes

- Default runs are typically ~10-20 minutes.
- Higher-quality runs can take significantly longer and cost more.
- Cost depends on model choice and token usage.
- Pricing and billing policy: https://docs.planexe.org/costs_and_models/

## Support

- Docs: https://docs.planexe.org
- GitHub issues: https://github.com/PlanExeOrg/PlanExe/issues
- Discord: https://planexe.org/discord
```

**Implementation:**
- Add `public/llms.txt` to repo (example above)
- Serve from `home.planexe.org/llms.txt` (HTTP static)
- Serve from `mcp.planexe.org/llms.txt`
- Update Docker entrypoint to serve `/llms.txt` locally on port 3000

**Reference:** Orchestrator patterns gist notes that `llms.txt` is how agents discover available skills and MCP servers without manual integration.

---

### 2. OpenAPI / LLM-Optimized Tool Schema

**STATUS: NOT YET IMPLEMENTED** — This recommendation is aspirational. The `/openapi.json` endpoint does not currently exist on either domain.

**What:** Expose machine-readable schemas for PlanExe's AI-facing interfaces (MCP-first), and optionally OpenAPI for any public REST surfaces.

**Why:** OpenAI's function-calling guide and Responses API expect structured function schemas. LLMs perform better with:
- Clear, detailed descriptions (not just docstring summaries)
- Input/output examples
- Enum values for constrained choices
- Required vs optional fields explicitly marked
- Error codes documented

**Content (sample):**

```json
{
  "openapi": "3.1.0",
  "info": {
    "title": "PlanExe API",
    "description": "Multi-step plan generation, execution, and orchestration for AI agents",
    "version": "1.0.0",
    "contact": {
      "url": "https://github.com/PlanExeOrg/PlanExe"
    }
  },
  "servers": [
    {
      "url": "https://api.planexe.example/v1",
      "description": "Example public REST host (if exposed)"
    },
    {
      "url": "http://localhost:3000/api/v1",
      "description": "Local Docker"
    }
  ],
  "paths": {
    "/plans": {
      "post": {
        "summary": "Generate a multi-step plan",
        "description": "Break down a goal into executable steps, estimate resources, and track dependencies. Use this when you need to decompose a complex task into subtasks with clear success criteria.",
        "operationId": "generatePlan",
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "goal": {
                    "type": "string",
                    "description": "The high-level goal to plan for (e.g., 'Build a REST API for a social media platform')"
                  },
                  "context": {
                    "type": "string",
                    "description": "Optional context about constraints, existing systems, or success criteria"
                  },
                  "constraints": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "List of constraints (e.g., ['must use TypeScript', 'budget < $5000', 'timeline 2 weeks'])"
                  },
                  "model": {
                    "type": "string",
                    "enum": ["gpt-4o", "claude-3-opus", "o1"],
                    "description": "Which planning model to use (affects cost and reasoning depth)"
                  }
                },
                "required": ["goal"],
                "additionalProperties": false
              }
            }
          }
        },
        "responses": {
          "201": {
            "description": "Plan successfully generated",
            "content": {
              "application/json": {
                "schema": {
                  "type": "object",
                  "properties": {
                    "planId": { "type": "string", "format": "uuid" },
                    "goal": { "type": "string" },
                    "steps": {
                      "type": "array",
                      "items": {
                        "type": "object",
                        "properties": {
                          "stepId": { "type": "string" },
                          "description": { "type": "string" },
                          "dependencies": { "type": "array", "items": { "type": "string" } },
                          "estimatedMinutes": { "type": "integer" }
                        }
                      }
                    },
                    "createdAt": { "type": "string", "format": "date-time" }
                  }
                }
              }
            }
          },
          "400": {
            "description": "Invalid goal or constraints"
          }
        }
      }
    },
    "/plans/{planId}": {
      "get": {
        "summary": "Get plan details and execution trace",
        "description": "Fetch the current state of a plan, including completed steps, failures, and execution trace for debugging",
        "operationId": "getPlan",
        "parameters": [
          {
            "name": "planId",
            "in": "path",
            "required": true,
            "schema": { "type": "string", "format": "uuid" }
          }
        ],
        "responses": {
          "200": {
            "description": "Plan details"
          }
        }
      }
    }
  }
}
```

**Implementation:**
- Keep `/mcp/tools` accurate and stable as the primary machine-readable contract.
- If maintaining OpenAPI, generate from code and publish from docs or the relevant API host.
- Include examples for common use cases (planning quality vs speed, artifact retrieval).
- Validate against OpenAPI 3.1 when OpenAPI is published.

**Reference:** OpenAI function-calling guide emphasizes that detailed descriptions help models understand when and how to use tools. The Responses API docs recommend OpenAPI 3.1 as the source of truth for function schemas.

---

### 3. MCP Well-Known Endpoint (`.well-known/mcp.json`)

**STATUS: NOT YET IMPLEMENTED** — This endpoint does not currently exist at `mcp.planexe.org/.well-known/mcp.json`. This is a future enhancement.

**What:** Add `/.well-known/mcp.json` describing PlanExe as an MCP server, compatible with OpenAI's MCP connector and orchestrators.

**Why:** OpenAI's tools-connectors-mcp guide and orchestrator patterns show that MCP servers need a discoverable manifest listing available tools, auth requirements, and rate limits.

**Content (sample):**

```json
{
  "version": "1.0",
  "mcp_server": {
    "name": "PlanExe",
    "description": "Strategic plan generation via MCP tools",
    "tools": [
      {"name": "example_prompts"},
      {"name": "task_create"},
      {"name": "task_status"},
      {"name": "task_stop"},
      {"name": "task_file_info"}
    ]
  },
  "auth": {
    "cloud": {
      "type": "api_key",
      "header": "X-API-Key",
      "obtain_key_at": "https://home.planexe.org/account/api-keys"
    },
    "local": {
      "type": "configurable"
    }
  },
  "endpoints": {
    "cloud": "https://mcp.planexe.org/mcp",
    "cloud_trailing_slash": "https://mcp.planexe.org/mcp/",
    "local_default": "http://localhost:8001/mcp"
  },
  "discovery": {
    "llms": [
      "https://home.planexe.org/llms.txt",
      "https://mcp.planexe.org/llms.txt"
    ]
  }
}
```

**Implementation:**
- Serve on the AI-facing host (e.g., `mcp.planexe.org/.well-known/mcp.json`).
- Keep in sync with `/mcp/tools` and `public/llms.txt`.
- Document ownership/update process in maintainers' docs.

**Reference:** OpenClaw architecture shows how Gateway/MCP servers expose their capabilities via manifest endpoints. The MCP spec defines `/.well-known/mcp.json` as the discovery entry point.

---

### 4. `robots.txt` for AI Crawlers

**What:** Update `robots.txt` to allow major AI crawlers and LLM training pipelines, while protecting sensitive routes.

**Why:** Agents like Claude Web, GPTBot, and PerplexityBot use `robots.txt` to understand which content they can index and integrate. This makes PlanExe findable in AI-augmented search and agent training pipelines.

**Content (sample):**

```
# Allow AI crawlers to discover and understand PlanExe
User-agent: GPTBot
Disallow: /admin/
Disallow: /user/*/private/
Allow: /api/
Allow: /docs/
Allow: /llms.txt
Allow: /openapi.json
Allow: /.well-known/mcp.json

User-agent: Claude-Web
Disallow: /admin/
Disallow: /user/*/private/
Allow: /api/
Allow: /docs/
Allow: /llms.txt

User-agent: PerplexityBot
Disallow: /admin/
Allow: /docs/
Allow: /llms.txt

# Standard crawlers
User-agent: *
Disallow: /admin/
Disallow: /user/*/private/
Allow: /api/
Allow: /docs/
```

**Implementation:**
- Update `public/robots.txt`
- Consider allowing `/api/` for schema crawling (safe since no auth-sensitive data)
- Test with Google Search Console and ChatGPT crawlers

**Reference:** Orchestrator patterns note that explicit allow rules in `robots.txt` help agents understand what's available.

---

### 5. Structured Data (JSON-LD) & SEO Meta Tags

**What:** Add schema.org SoftwareApplication markup to homepage and service pages.

**Why:** JSON-LD makes PlanExe discoverable via AI-augmented search engines and helps agents understand what it is without full page parsing.

**Content (sample):**

```html
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  "name": "PlanExe",
  "description": "Multi-step plan generation and orchestration for AI agents. Decomposes complex goals into executable steps with resource tracking and dependency management.",
  "url": "https://home.planexe.org",
  "applicationCategory": "DeveloperApplication",
  "offers": {
    "@type": "Offer",
    "price": "10",
    "priceCurrency": "USD",
    "description": "Per plan generation (cloud). Free tier: 100 credits/month"
  },
  "featureList": [
    "Multi-step plan generation",
    "API and MCP integration",
    "Execution tracking",
    "Artifact generation",
    "OpenAI Agents compatible",
    "OpenClaw compatible"
  ],
  "operatingSystem": "Any",
  "softwareVersion": "1.0",
  "downloadUrl": "https://github.com/PlanExeOrg/PlanExe",
  "author": {
    "@type": "Organization",
    "name": "PlanExe"
  },
  "inLanguage": "en"
}
</script>

<!-- Meta tags for social/AI preview -->
<meta property="og:title" content="PlanExe - AI Plan Orchestrator" />
<meta property="og:description" content="Generate, execute, and track multi-step plans via API or MCP. Integrates with OpenAI Agents, Claude, and Codebuff." />
<meta property="og:url" content="https://home.planexe.org" />
<meta property="og:type" content="website" />
<meta name="description" content="PlanExe is an AI-native planning orchestrator that decomposes complex goals into executable multi-step plans with full execution tracking and resource management." />
<meta name="keywords" content="AI planning, orchestration, MCP, API, OpenAI agents, Codebuff, task decomposition" />
```

**Implementation:**
- Add to base layout/template in Next.js/React app
- Update homepage, docs, and API pages
- Use Yoast or similar to validate OpenGraph + structured data

**Reference:** OpenAI's new tools blog emphasizes that agents need clear metadata to understand service capabilities.

---

### 6. Agent-Readable GitHub README Section

**What:** Add a dedicated machine-readable section to the main README for agent integration.

**Why:** Agents often clone/read repos to understand integrations. A clear "For AI Agents" section with tool names, schemas, and auth instructions speeds up discovery.

**Content (sample):**

```markdown
## For AI Agents & Orchestrators

This section is optimized for AI agents (OpenAI Agents, Claude Code, Codebuff, OpenClaw, etc.) to quickly understand PlanExe's capabilities.

### Quick Integration

**Cloud (Hosted):**
```bash
curl -X POST https://mcp.planexe.org/mcp/tools/call \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "task_create",
    "arguments": {
      "prompt": "Create a plan for launching a B2B SaaS product",
      "speed_vs_detail": "fast"
    }
  }'
```

**Local Docker:**
```bash
docker compose up mcp_cloud
# MCP endpoint at http://localhost:8001/mcp
```

**MCP (OpenAI Responses API):**
```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-4o",
    tools=[{
        "type": "mcp",
        "server_label": "planexe",
        "server_url": "https://mcp.planexe.org/mcp",
        "headers": {"X-API-Key": "YOUR_API_KEY"},
    }],
    input="Generate a plan for deploying a microservices app"
)
print(response.output_text)
```

### Tools

| Tool | Input | Output | Use Case |
|------|-------|--------|----------|
| `example_prompts` | none | prompt samples | Bootstrap high-quality prompts |
| `task_create` | prompt, speed_vs_detail | task_id | Start plan generation |
| `task_status` | task_id | state, progress | Poll long-running execution |
| `task_file_info` | task_id, artifact | download metadata/url | Retrieve output files |
| `task_stop` | task_id | stop result | Cancel a running task |

### OpenAPI & Schemas

- Agent discovery: `GET /llms.txt`
- MCP tool discovery: `GET /mcp/tools`

### Authentication

**Cloud:** `X-API-Key` (create key at https://home.planexe.org/account/api-keys)
**Local Docker:** Often none for local development, configurable by deployment.

### Rate Limits

- 100 plans/hour (cloud)
- 1000 artifact downloads/day
- No limits on local Docker

### Pricing (Cloud)

See https://docs.planexe.org/costs_and_models/ for current billing and model-cost guidance.
```

**Implementation:**
- Add to main README.md after "Features" section
- Include in CONTRIBUTING.md for agent/MCP developers
- Link from docs homepage

**Reference:** Orchestrator patterns gist notes that README agent sections help developers integrate without digging through docs.

---

## Implementation Priority

1. **Keep llms.txt accurate and versioned** (Ongoing)
   - Treat `public/llms.txt` as canonical and update whenever endpoints/tool names/auth change.
   - Verify hosted endpoints regularly (`home` + `mcp`).

2. **robots.txt update** (Next)
   - Quick addition
   - Enables AI crawler indexing

3. **MCP well-known/discovery metadata** (Next)
   - Add consistent machine-readable MCP manifest and keep it synced with tools.

4. **README agent section** (Near term)
   - Publish concise MCP-first integration guidance in repo root docs.

5. **Structured data + meta tags** (Later)
   - Low complexity
   - Improves AI-augmented search discoverability

---

## How Agents Will Use This

1. **Discovery Phase:** Agent scans `/llms.txt` and learns MCP endpoint + auth model.
2. **Tool Phase:** Agent calls `/mcp/tools` to inspect tool signatures.
3. **Execution Phase:** Agent runs `task_create`, polls `task_status` (every ~5 minutes), and fetches artifacts via `task_file_info`.
4. **Iteration Phase:** Agent refines prompts and reruns to improve output quality/cost tradeoff.

---

## Expected Outcomes

- **OpenAI Agents** can natively discover and call PlanExe tools via function calling
- **Claude Code** can integrate PlanExe for multi-step code generation planning
- **Codebuff** can use PlanExe for workflow decomposition
- **OpenClaw** can add PlanExe as an MCP capability
- **AI-augmented search** (Perplexity, Claude Web) can index PlanExe as a planning resource
- **Developers** can quickly integrate PlanExe without extensive documentation reading

---

## Risks & Mitigations

### Risk: Malicious agents calling expensive endpoints
**Mitigation:** Implement rate limiting per API key, credit-based billing, approval gates for MCP calls.

### Risk: Sensitive data exposure via schema
**Mitigation:** Don't expose user-specific fields in OpenAPI; only include public tool schemas.

### Risk: MCP server compromise
**Mitigation:** Require approval for MCP calls by default (OpenAI's default); document in security guide.

**Reference:** OpenAI's MCP connector docs emphasize: "By default, OpenAI will request your approval before any data is shared with a connector or remote MCP server."

---

## References

- **Orchestrator Patterns:** https://gist.github.com/championswimmer/bd0a45f0b1482cb7181d922fd94ab978
- **OpenAI New Tools for Building Agents:** https://openai.com/blog/new-tools-for-building-agents
- **OpenAI Function Calling Guide:** https://platform.openai.com/docs/guides/function-calling
- **OpenAI MCP Connectors Guide:** https://platform.openai.com/docs/guides/tools-connectors-mcp
- **OpenClaw Gateway Architecture:** https://docs.openclaw.ai/concepts/architecture
- **MCP Specification:** https://modelcontextprotocol.io/introduction
- **llms.txt Standard:** (emerging standard for agent discovery)
- **OpenAPI 3.1 Spec:** https://spec.openapis.org/oas/v3.1.0
- **schema.org SoftwareApplication:** https://schema.org/SoftwareApplication

---

## Success Metrics

- Agent discovery via web search / MCP crawler
- OpenAI Agents integration time (should be <30 min after this)
- MCP tool call volume from orchestrators
- Developer integration time reduction
