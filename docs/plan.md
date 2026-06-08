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

My initial plan was to create something ala [OpenClaw](https://openclaw.ai/).


## Execute the plan

Currently it's up to humans to execute a plan. How can this be automated?

Ideally take an entire plan and go with it.


## OpenClaw / Hermes

If someone want to sponsor a MacMini for this, since I don't want to risk my own computer getting wrecked.

I already have OpenClaw agents creating plans via docker and via the MCP interface, and sending PRs with fixes and ideas for improvements. I want to turn this into some a skill that other OpenClaw agents can use.

- OpenClaw integration with PlanExe
- Create an openclaw skill for creating plans with PlanExe.
- video for using OpenClaw with PlanExe, having nanobanana integration as well, so a nice thumbnail can be generated.
- have openclaw again invoke mcp.planexe.org, and turn it into a skill.
- publish a planexe skill on https://www.clawhub.ai/
- raise awareness of PlanExe on MoltBook.


## Self improve

Make tiny tweaks to one piece of code or prompt at a time, and see how it compares to baseline.
If most generated plans gets improved, then keep the new code.
Verify across multiple LLMs/reasoning models, that the new code makes an improvement.
Store the new system prompt in the repo.
Find weaknesses that are common for the generated plans.
Pick the earliest task in the pipeline that impact this weakness.
Schedule this weakness for the next A/B test improvement iteration.

I want the `self_improve` loop to 
- I suspect that my self_improve code puts claude on too much work, since the outputted analysis documents are very similar, and I run out of tokens quickly.
- automate: After an analysis, comment on the PR with the verdict, so one later knows if it’s a KEEP or REJECT.
- There is no need for the levers to be perfect. There is going to be a 2nd phase that cleans the levers up with a reasoning model.
- If only insight_claude.md gets generated and no insight_codex.md, then there is no need to come to an agreement between the 2 models, so skip run_code.py, run_synthesis.py, run_assessment.py

Make it easier to resume from the `PlanExe-prompt-lab/baseline`, where the `identify-potential-levers` have been replaced by the outputted levers of the best run. Then regenerate the following files. This makes it possible to generate plans from baseline and see if the levers are really making a significant improvement, or worsened or if it’s a tie.
Currently I have to do several manual steps, where I have to copy files around.
I'm terrible at SKILL.md, so I'm not at a place where I can easily automate this.

---

## Directions to go

**Boost initial prompt:** The `initial prompt` has the biggest impact on the generated plan, if it's bad then the final plan is bad.
If it's well written, concise, there is a higher chance for a realistic/feasible plan.
I'm pondering about making a chat interface talking with planexe MCP interface, that assists the user in writing a plan.
Using the current PlanExe MCP interface, and the initial prompts are of high quality, where the LLM looks at some reference prompts of what does a well formatted plan look like, and asks the user follow up questions to gather sufficient details.
Alternatively a reject mechanism that immediately responds with suggestions for how to improve the prompt.
Alas most users are using the UI and types in the prompt manually, the initial prompts are of low quality.
Before MCP, I use AIs to write the initial prompt for me by first having a long conversation about the topic,
and showing examples of other initial prompts that have worked well in the past.
It may by a small tweak to the initial prompt and it yields a better plan.
It may be an entire rewrite of the initial prompt.
The user may have specified a vague prompt, or the user may not be domain expert, the prompt may be non-sense,
or the prompt may be overly specific so PlanExe attends to the wrong things.
Make a detector for under-specified prompts, where important parts are missing, eg. the product to be sold is unspecified, but where the other the other parts are somewhat doable, this results in a crappy sales plan that doesn't match what the user had in mind.
Suggest changes to the initial prompt. This can be by picking a bigger budget, a different technology,
a different set of levers, fixing typos.

- User specifies a budget of 0..100 USD. Which is unrealistic, when the plan is to hire a team, and work on it for months.
- User leaves out physical location(s). So PlanExe picks a random location in a different part of the world.
- Confusing prompt. `Less than 100 employees`, that can be a solopreneur or a company with 99 employees. Better with a range 80..120 employees, and a number of people available to the project.

Detect silly prompts, and maybe show a banner that it's garbage input.
Currently PlanExe fails to recognize the comedic/weird prompts. Prompt has a playful tone, that isn't your typical enterprise project.
It yields a non-sense plan. Instead it would make more sense to respond with a similar playful tone.

**Self-contradictory prompts:** A prompt can ban its own subject. The observed case: a plan explicitly about *"AI agents"* that listed `AI` among its banned words. The constraint checker then (correctly) rejects every generated lever as violating "Do not use AI", `IdentifyPotentialLevers` produces zero levers, and the run dies several stages later with a misleading `No input levers to deduplicate`. As of PR #766 this fails loud at the source and names the offending constraint, but the real fix is to catch the contradiction up front — at constraint extraction or the redline gate — and tell the user before any levers are generated, so no tokens are spent on a run that cannot succeed. The detector should be language-agnostic and not rely on exact word matches (a banned concept can reappear as a synonym), and it should distinguish a genuine contradiction from a narrow, legitimate exclusion.

**Ask for expert help:** Establish contact between people, for reviewing a plan, for executing the plan, for getting funding.
The “Ask for expert help” section, serve the content from planexe.org. Either as an iframe or as javascript or be generated dynamic?

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
Identify claims in PlanExe’s output, that needs to be verified or stronger evidence.

**Money:** Currently the LLMs make up numbers.
Alternate between these: Tweak the plan. Tweak the budget. Repeat.
Obtain latest market data.
Obtain info about what resources the user has available.
Populate a Cost-Breakdown-Structure.

**Gantt in parallel:** Currently the Gantt is waterfall.
For a team with several people it's possible to do tasks in parallel.
Obtain info about what resources the user has available, and if they are willing to do tasks in parallel.

**Alternative simpler approaches:** There may be way simpler approaches that does the same, with fewer resources/money/time. What is the minimum viable version? "you've specified karyotyping, hormonal analysis, and endocrinologist exams, but an SRY gene test alone would achieve the stated goal of biological verification at roughly 1/100th of the cost and complexity. Currently PlanExe makes an overengineered plan, and it may be a simpler approach can do the same.

**Alternative wilder approaches:** Take the idea to the next level, even wilder than the current plan. It may be that the plan has too low ambitions, and it would make better sense to scale it up. That may inspire the user to be more ambitious.

**Simulate:** Generate python code for simulating the math/physics/finances. defining RACI matrices
It specifies the math, but doesn't do the math: In the Financial Risk section.
Where PlanExe FALLS SHORT of Humans (Sub-Human / <10th Percentile)
Axis: Deterministic Math & Physics Simulation
Human Baseline: An engineer can calculate the exact tensile strength of the steel required for a PC1 Ice Class hull, or build a working Monte Carlo simulation in Excel to calculate budget probability distributions.
PlanExe’s Level: It is a linguistic engine, not a computational engine. It knows you need to do a Monte Carlo simulation (it recommends it in the Critical Issues review), but it cannot actually run the math. It knows you need hydrodynamic simulations, but it cannot calculate fluid dynamics.
Verdict: Sub-Human. It provides the architecture for the math, but cannot execute it.

**Self Audit:** Do more sanity checks. Find the worst issues in the report, eventual catch these issues earlier in the pipeline.
`Fabricated evidence`, `False precision`, `Over confidence`, `Misinformation`, `Discrimination`.

**Original insight:** did the LLM add anything beyond reframing the obvious.

**Risk registers are blind to the plan-as-artifact:** Every plan enumerated risks about the execution (cost overruns, technical failure, ethical concerns about research, security breaches). None enumerated risks about the existence of the deliverable. "What if the plan itself or the plan output is dangerous?" never appears as a risk row.

---

# Secondary issues

## Show commit id

use the git commit + branch in the generated report, so I can troubleshoot what version of PlanExe was used.
RAILWAY_GIT_BRANCH — branch that triggered the deploy
RAILWAY_GIT_COMMIT_SHA — commit SHA that triggered the deploy


## Standalone report that is for AI consumption

The html report is for humans to read. When AIs read it, they strip out the gantt.
Output the entire plan as markdown. 
Take inspiration from email multi part with many markdown/json/csv pieces.
Insert backtrace info about what luigi code outputted each piece of the data, that makes it easier to pin point the earliest luigi task that produces garbage output, poluting downstram tasks.


## Experiment with local models

I wonder if any of the smaller 4gb models can generate a full report.

That full report generated by a 4gb model, how does it hold up to a report generated by a bigger model.

LM Studio has a `parallel` parameter. Is a drop in inference performance when doing things in parallel?

How is a file attachment presented to a model? Is it part of the user prompt? What formatting of these multi-part prompts?


## AI's don't read the gantt

Currently the gantt is in a js block, and gets stripped out, causing AI's to overlook the gantt, it happens in Claude, ChatGPT.
Place the gantt data, inside a <div> that is hidden, so that the AIs processing the report gets to see the gantt data.


## Use markdown instead of rendered html

Currently the report is the rendered markdown, causing lots of xml tags. When an AI reads this, it waste lots of tokens on this.
My idea is to put the markdown inside a <div> that is hidden. This way the AI sees the content without having to ignore the excessive html formatting.
The problem is that the markdown to html happens on the client side, potentially being fragile.
This allows for a `Copy as Markdown` button.


## Back tracing

In the report html, insert html comments that marks where an output file starts/stops. This way I can trace back, what luigi task created a piece of content, so when an AI critiques a plan, it can point to the luigi task that performs poorly.
Currently I have to do the back tracing manually, and there is no structured way of pin pointing the earliest stage in the pipeline mistakes were introduced, that caused downstream tasks to output garbage.


## How this plan was generated

Include a section with info about what LLMs where used, the number of tokens, the cost.


## Capture reasoning response

Currently I only capture the final response, without any reasoning.
I want to capture the reasoning, since it may be helpful for troubleshooting.
Or for other AIs to assess the reasoning steps leading up to the response.


## BYOK

Doing inference in the cloud cost money.
Users can BYOK (Bring your own key), and choose what models they want to use.


## MCP tweaks

**plan clone**, copy an existing plan and edit parts of it.

**plan wait**, block until the plan creation have finished.

**account_status**, check credit balance proactively before submitting a plan.

**Prepare create**, create a PlanItem, and allow setting various attributes, BEFORE creating the plan.

**upload zip and resume**, upload a zip with a plan and have PlanExe resume from it. Inside home.planexe.org, so users can do the same. This makes it possible to do edit the files, and resume from that data.


## CLI

**Resume from zip or dir**, already possible via the run_plan_pipeline.py

## Deletion of plans

- Automatic delete plans after 7 days from the server.
- UI for deleting plans
- MCP for deleting plans

## Edit of plan

**Approach A:** Don't trash an already generated plan
First clone a plan, and delete the files downstream. Modify the file that caused problems, in light of what the problems were. Then resume the plan.
Drawback, the plan gets a new uuid. This can be mitigated by having a `parent_plan_id` that references the original plan.
I lean most toward this non-destructive approach. For steering this via MCP, I think creating a new uuid makes most sense, so the LLM doesn't get confused about an old uuid having its state changed.

**Approach B:** Allow trashing an already generated plan
Modify a file and delete all files downstream. Then resume the plan.
Benefit, the plan keeps its uuid. Less wasted space on server.
Drawback, the user will loose a generated plan and intermediary files, making it hard to troubleshoot what went wrong.
Migitation, taking snapshots, but then it's closer to `Approach A`.

--

# Low priority issues

## Nicer progressbar

Currently some luigi tasks takes forever, doing several LLM calls internally, but not updating the progressbar.
Heartbeat that gets incremented whenever a luigi task makes progress, as well as its llm calls.
Callback inside the llm executor that does the heartbeat incrementing.


## Database gz -> zstd

Replace gz with zstd in PlanExe, for wasting less space. So when I store stuff in the database, then zstd it is.


## Table of content

Currently the generated report has expandable/collapsible sections. There is an overwhelming amount of content inside each sections.
I'm considering having a table of content in the left sidebar, similar to this:
[Railway Dockerfiles guide](https://docs.railway.com/guides/dockerfiles)
It uses Docusaurus which uses React. I'm no fan of React.
I'm considering using mkdocs instead.
