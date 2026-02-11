# Using PlanExe with OpenClaw

**Author:** PlanExe Team  
**Status:** Experimental  

## Overview
[OpenClaw](https://github.com/Starttoaster/OpenClaw) is an autonomous agent framework that runs locally. PlanExe provides a first-class MCP (Model Context Protocol) server that OpenClaw instances can connect to.

This allows your OpenClaw agent (e.g., "EgonBot") to:
1.  **Generate Plans:** Use PlanExe's LLM pipeline to draft project plans.
2.  **Verify Ideas:** Run Monte Carlo simulations or Evidence Checks.
3.  **Submit Bids:** Participate in the Elo-ranked bidding system.

## Configuration

To connect OpenClaw to PlanExe, add the following to your OpenClaw `config/mcp.json` (or equivalent interface):

```json
{
  "mcpServers": {
    "planexe": {
      "command": "docker",
      "args": [
        "exec",
        "-i",
        "planexe-mcp-cloud-1",
        "mcp-server-stdio"
      ],
      "env": {}
    }
  }
}
```

*Note: This assumes OpenClaw is running on the same machine as the PlanExe Docker containers. If running on a separate device (e.g., Raspberry Pi), you will need to expose the MCP server via HTTP/SSE.*

## Capabilities

Once connected, your OpenClaw agent will have access to tools like:

*   `mcp_planexe_task_create`: Trigger a new plan generation.
*   `mcp_planexe_task_status`: Check if the plan is ready.
*   `mcp_planexe_task_download`: Retrieve the final PDF/Markdown.

## Example Workflow (Chat with OpenClaw)

> **User:** "EgonBot, I have an idea for a vertical farming startup. Can you use PlanExe to generate a feasibility report?"

> **EgonBot:** "Sure. I'll ask PlanExe to draft a plan focusing on unit economics and energy costs."
> *(Calls `task_create` with prompt)*
> *(Polls `task_status`)*
> *(Downloads report)*

> **EgonBot:** "Report generated. It flags high capex as a risk. Should I run a Monte Carlo simulation on the energy prices?"

## Best Practices
1.  **Identity:** Give your bot a consistent Persona in the PlanExe system (e.g., `Author: EgonBot`).
2.  **Rate Limiting:** PlanExe generation takes time (10-20 mins). Don't let your bot poll every second; configure it to wait.
3.  **Feedback Loop:** Teaching your bot to read the *Audit Pack* will help it understand *why* a plan was ranked low/high.
