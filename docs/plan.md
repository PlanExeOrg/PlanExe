# Future plan for PlanExe

Using the "5 Whys" method:
> I want multiple-agents talking with each other, via zulip.

Why?

> Can I “execute” a plan from start to finish, with the agents doing all the labor.
> Zulip is open source. So I’m not dependent on Discord/Teams/Slack.

Why?

> When humans are doing the labor, they have to decompose the problem into tasks. 
> In my experience with PlanExe, AI can decompose sometimes better/worse than humans.
> Possible via MCP to interface with issue tracker.
> delegate parts of the plan to humans.

Why?

> Companies spend lots of effort on planning and getting the right people to communicate with meetings, emails. Something I dislike about working in coding jobs.
> Wasting time and money on planning.

Why?

> Cut cost and optimize speed.

Why?

> To satisfy my own curiosity. I’m curious to what kind of outcome it is. An AI organization/company, possible distributed network.
> Is it a global organism as seen in scifi movies that are controlled by AIs, that takes power away from politicians.
> My concerns are:
> will it be able to adapt to a changing world. Re-plan in real-time when a shipment is delayed, a machine breaks down, or an unexpected storm hits.
> quiet, compounding errors, security oversights, and cost blowouts.


## Execute the plan

Currently it's up to humans to execute a plan. How can this be automated?

Ideally take an entire plan and go with it.


## Improve plan

**Prompt optimizing with A/B testing:** Make tiny tweaks to one system prompt at a time, and see how it compares to baseline.
If most generated plans gets improved, then keep the new system prompt.
Verify across multiple LLMs/reasoning models, that the new system prompt makes an improvement.
Store the new system prompt in the repo.
Find weaknesses that are common for the generated plans.
Pick the earliest task in the pipeline that impact this weakness.
Schedule this weakness for the next A/B test improvement iteration.

**Boost initial prompt:** The `initial prompt` has the biggest impact on the generated plan, if it's bad then the final plan is bad.
If it's well written, concise, there is a higher chance for a realistic/feasible plan.
Using the current PlanExe MCP interface, and the initial prompts are of high quality, where the LLM looks at some reference prompts of what does a well formatted plan look like, and asks the user follow up questions to gather sufficient details.
However when using the UI and users entering the prompt manually, the initial prompts are of low quality.
Before MCP, I use AIs to write the initial prompt for me by first having a long conversation about the topic,
and showing examples of other initial prompts that have worked well in the past.
It may by a small tweak to the initial prompt and it yields a better plan.
It may be an entire rewrite of the initial prompt.
The user may have specified a vague prompt, or the user may not be domain expert, the prompt may be non-sense,
or the prompt may be overly specific so PlanExe attends to the wrong things.
Suggest changes to the initial prompt. This can be by picking a bigger budget, a different technology,
a different set of levers, fixing typos.

- User specifies a budget of 0..100 USD. Which is unrealistic, when the plan is to hire a team, and work on it for months.
- User leaves out physical location(s). So PlanExe picks a random location in a different part of the world.

**Dynamic plugins:** Have AI's rewrite PlanExe as they see fit, depending on what the user have prompted it with. So if it's a software project, it writes PlanExe plugins that are going to be needed. And then proceeds to creating the plan. In the middle of the plan creation, it may be necessary to create more PlanExe plugins as issues shows up.

**Grid search:** Currently PlanExe only generates a plan for 1 permutation of levers.
A plan may have 10 levers with 3-5 settings. Here it could be interesting to create 
100 full plans, each with a different combination of levers. Compare the generated plans against each other 
and pick the most 3 promising plans.

**Multiple refinements:** Currently PlanExe generates the first iteration of the plan.
Usually issues arises when making the first iteration, that have to be incorporated into the timeline.
In the future I want to do multiple iterations, until the plan is of a reasonable quality.

**Validate the plan with deep research:** Currently there is no validation.
It's up to humans to be skeptic about the plan, does this make sense, check everything.
There may be issues with: assumptions, numbers, flaws.

**Money:** Currently the LLMs make up numbers.
Alternate between these: Tweak the plan. Tweak the budget. Repeat.
Obtain latest market data.
Obtain info about what resources the user has available.
Populate a Cost-Breakdown-Structure.

**Gantt in parallel:** Currently the Gantt is waterfall.
For a team with several people it's possible to do tasks in parallel.
Obtain info about what resources the user has available, and if they are willing to do tasks in parallel.

---

# Secondary issues

## MCP - Multiple API keys

As of 2026-mar-02, the user can only have 1 api key.
Now I have quite a few places where PlanExe is integrated, and 
I want to see stats on what api key is being used, and cost.
Resetting the api key, and all the integrations dies.


## MCP - Polishing of MCP flow via planexe.org

As of 2026-mar-02, I'm focusing on improving MCP.
PlanExe is already working with OpenClaw.
But it's not as smooth as I would like.

The user adds credits here. Start with 5 USD, so you can create around 3 plans.
[https://home.planexe.org/](https://home.planexe.org/)

The agents use the api here. When AI agents connect to the MCP interface, the credits are consumed. Between 1-2 USD per plan creation.
[https://mcp.planexe.org/mcp](https://mcp.planexe.org/mcp)


## MCP - BYOK

Doing inference in the cloud cost money.
Users can BYOK (Bring your own key), and choose what models they want to use.

---

# Tertiary issues

## Capture reasoning response

Currently I only capture the final response, without any reasoning.
I want to capture the reasoning, since it may be helpful for troubleshooting.
Or for other AIs to assess the reasoning steps leading up to the response.


## Debugging

Get step-by-step debugging working again.
Now that I have switched to Docker, I have multiple python projects in the same repo, that use different incompatible packages.


## Table of content

Currently the generated report has expandable/collapsible sections. There is an overwhelming amount of content inside each sections.
I'm considering having a table of content in the left sidebar, similar to this:
[Railway Dockerfiles guide](https://docs.railway.com/guides/dockerfiles)
It uses Docusaurus which uses React. I'm no fan of React.
I'm considering using mkdocs instead.


## Eliminate redundant user prompts in the log file

Get rid of some of the many user prompt logging statements, so the log.txt is less noisy.
These user prompts are saved to the `track_activity.jsonl` file already. So having them in the log.txt is redundant.


## SSL when connecting with the database

I can't afford the pro plan to have a dedicated Postgres server.
Currently when connecting to Railway, it's via a TCP Proxy and it's unencrypted.
Either upgrade to pro, or use SSL certificates within the "database_postgres" Dockerfile.
