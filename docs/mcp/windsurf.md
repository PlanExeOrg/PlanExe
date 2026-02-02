---
title: Windsurf - MCP integration
---

# Windsurf

[Windsurf](https://windsurf.com/).

[Windsurf MCP documentation](https://docs.windsurf.com/windsurf/cascade/mcp)

[Windsurf MCP tutorial](https://windsurf.com/university/tutorials/configuring-first-mcp-server)

## Interaction

My interaction history:

1. get planexe example prompts
2. I want you to suggest 5 prompts, based on the example prompts
3. suggest something that fixes real world problems
4. 5 more
5. I'm in europe. make 5 suggestions that fixes serious issues in europe
6. I like your Heatwave mortality reduction idea. I want you to make a full prompt ala the planexe example prompts, and show me the prompt
7. remove the heading "Full PlanExe-style prompt: Heatwave mortality reduction (Europe)". what do you think about the prompt?
8. go ahead create this plan
9. status
10. status
    *Here windsurf went ahead and downloaded the created HTML report*
11. compare the created plan with the prompt you formulated
12. also download the zip

I had to manually ask about `check status` to get details how the plan creation was going. It's not something that Windsurf can do.


## Prerequisites

A working installation of PlanExe.

- The recommended way is to install PlanExe by following the [Getting Started](../getting_started.md) instructions.
  Make sure that `docker compose up` is running, in order to connect to PlanExe.
- Alternatively: Run PlanExe on another server and port.
- Alternatively: If you are a developer run PlanExe inside a python virtual environment.

Double check that PlanExe can take a prompt and create a plan. Since it doesn't make sense to start configuring Windsurf if the PlanExe installation is incomplete.

## Configuring Windsurf

To configure Windsurf to use PlanExe, you need to add the MCP server configuration.

![Instructions for opening Windsurf's MCP config file](windsurf_mcp_config.jpg)

1. Open Windsurf
2. Click the "..." icon at the top of the Agent panel, this opens a menu.
3. Click the "Open MCP Config File" icon at the bottom of the menu.
4. This opens the `mcp_config.json` file.

Add the following `planexe` dictionary to your `mcpServers` configuration:

```json
{
  "mcpServers": {
    "planexe": {
      "command": "uv",
      "args": [
        "run",
        "--with",
        "mcp",
        "/path/to/PlanExe/mcp_local/planexe_mcp_local.py"
      ],
      "env": {
        "PLANEXE_URL": "http://localhost:8001/mcp",
        "PLANEXE_PATH": "/Users/your-name/Desktop"
      }
    }
  }
}
```

Make these adjustments to the `planexe` snippet:

- Make adjustments to `/path/to/PlanExe` so it points to where PlanExe is located on your computer.
- Make adjustments to `/Users/your-name/Desktop` so it points to the directory where PlanExe is allowed to write to, so the plan can be downloaded.
- Optional: Make adjustments to `http://localhost:8001/mcp` if you have PlanExe running on another port.

Once you have saved the `mcp_config.json`. Then go to the `Manage MCP Servers` and click the refresh icon.

If it doesn't work then ask on the [PlanExe Discord](https://planexe.org/discord) for help.
