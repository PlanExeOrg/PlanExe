---
title: MCP troubleshooting
---

# MCP troubleshooting

Common MCP integration issues and fixes.

---

## Cannot create a plan

- Ensure your prompt is detailed (300–800 words).
- Some topics may be refused by the model (harmful, unethical, or dangerous requests).
- Try a smaller model or a more reliable paid model.
- Confirm the MCP server is reachable from your client.

---

## Status never changes

- Long‑running plans are normal; retry after a few minutes.
- If it stalls, create a new task and compare behavior.

---

## Download fails

- Confirm the download URL is reachable from your network.
- If you run locally, make sure any proxy or base URL is correct.
- Ensure `PLANEXE_PATH` is a valid directory and that you have write permissions.

---

## Low‑quality output

- Increase prompt detail and constraints.
- Use a stronger model.
- Avoid “vague” or conflicting requirements.

---

## Quota or billing errors

If you see errors like:

> You exceeded your current quota, please check your plan and billing details

Then:

- Verify your provider has active billing.
- Check usage limits or rate limits.
- Try a different model or provider.
