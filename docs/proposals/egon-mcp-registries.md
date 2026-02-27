# MCP Registry Submissions

**Author:** Egon  
**Date:** 2026-02-27  
**Status:** Ready for submission  

---

## Overview

Submit PlanExe MCP to major MCP registries to increase visibility and adoption.

## Registries

### 1. mcp.so

**Submission URL:** https://mcp.so/submit

**Form fields:**
- **Type:** Server
- **Name:** PlanExe
- **URL:** https://github.com/PlanExeOrg/PlanExe
- **Description:** Turn your idea into a comprehensive plan in minutes using AI. Premier planning tool for AI agents that generates 40-page strategic plans with executive summaries, Gantt charts, governance structures, risk registers, and SWOT analyses.
- **Server Config:**
```json
{
  "mcpServers": {
    "planexe": {
      "url": "https://mcp.planexe.org/mcp",
      "headers": {
        "X-API-Key": "pex_your_api_key_here"
      }
    }
  }
}
```

---

### 2. Smithery

**Submission URL:** https://smithery.ai/

**Form fields (TBD - need to check):**
- Server name: PlanExe
- Repository: https://github.com/PlanExeOrg/PlanExe
- Description: AI-powered business planning tool
- MCP config: Same as above

---

### 3. Glama.ai

**Submission URL:** https://glama.ai/mcp-servers

**Form fields (TBD - need to check):**
- Server name: PlanExe
- Repository: https://github.com/PlanExeOrg/PlanExe
- Description: AI-powered business planning tool
- Website: https://mcp.planexe.org

---

## MCP Server Config Reference

### Option A: Remote MCP (fastest path)

```json
{
  "mcpServers": {
    "planexe": {
      "url": "https://mcp.planexe.org/mcp",
      "headers": {
        "X-API-Key": "pex_your_api_key_here"
      }
    }
  }
}
```

### Option B: Local proxy (for artifact downloads)

```json
{
  "mcpServers": {
    "planexe": {
      "command": "uv",
      "args": [
        "run",
        "--with",
        "mcp",
        "/absolute/path/to/PlanExe/mcp_local/planexe_mcp_local.py"
      ],
      "env": {
        "PLANEXE_URL": "https://mcp.planexe.org/mcp",
        "PLANEXE_MCP_API_KEY": "pex_your_api_key_here"
      }
    }
  }
}
```

---

## Next Steps

1. Submit to mcp.so (primary)
2. Submit to Smithery
3. Submit to Glama.ai
4. Verify all listings appear correctly