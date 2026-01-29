# Troubleshooting a stuck pipeline

The gradio app (`app_text2plan.py`) starts the `run_plan_pipeline` process via a `Popen` call. 

- **Environment**, if the gradio app runs in a slightly different environment than when running via commandline `python -m worker_plan_internal.plan.run_plan_pipeline`, then the child process may behave differently. I have verified that the parent process and child process runs with the same environment variables.
- **Buffering**, if the parent process isn't reading stdout/stderr fast enough, the child process may freeze. I have reworked the `Popen` code so the stdout/stderr goes to `/dev/null`.
- **Other issues**, if the pipeline still hangs, let me know, it may be some issue I'm not aware of.

## Manually resuming a stuck pipeline

In the UI copy/paste the run_id that is stuck, eg: `20250209_030626`

Insert it on commandline, and run the pipeline, like this:

```bash
PROMPT> RUN_ID=20250209_030626 python -m worker_plan_internal.plan.run_plan_pipeline
```

## Why does the pipeline get stuck?

The `log.txt` contains the output from the logger with `DEBUG` level, the most detailed.
Alas the `log.txt` have little info about what exactly went wrong. 
The exceptions rarely have useful info.

- **Censorship**, if it's a sensitive topic, then the LLM may refuse to answer.
- **Timeout**, that happens often when using AI providers in the cloud.
- **Invalid json**, responds from the server that doesn't adhere to the json schema. Too high a temperature setting may cause the LLM to be too creative and diverge from the json schema. Try use a lower temperature.
- **Too long answer**, if the respond from the server gets too long so it gets truncated, so it's invalid json.
- **Other**, there may be other reasons that I'm not aware of, please let me know if you encounter such a scenario.
