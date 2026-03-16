"""
WBS Level 3: Create a Work Breakdown Structure (WBS) from a project plan.

https://en.wikipedia.org/wiki/Work_breakdown_structure

The "progressive elaboration" technique is used to decompose big tasks into smaller, more manageable subtasks.
"""
import os
import json
import time
from dataclasses import dataclass
from math import ceil
from uuid import uuid4
from pydantic import BaseModel, Field
from llama_index.core.llms.llm import LLM
from worker_plan_internal.format_json_for_use_in_query import format_json_for_use_in_query

OPTIMIZE_INSTRUCTIONS = """\
Goal: decompose project phases into 3-5 concrete, actionable subtasks that
a real project manager could schedule and resource — not generic filler.

Pipeline context
----------------
CreateWBSLevel3Task runs on every WBS level 2 phase (typically 6-12 phases),
making 6-12 serial LLM calls in sequence. This makes it one of the most
resource-intensive tasks in the pipeline and a known failure point for
smaller local models (structured output validation errors accumulate across
the serial call sequence).

Each call receives: the project plan, the specific phase being decomposed,
and the existing level 2 structure. The output is a list of 3-5 subtasks
with names, descriptions, and resources needed.

Known problems to guard against
---------------------------------
- Over-decomposition. Merge related micro-steps into meaningful deliverables.
  Each subtask should represent work that takes days or weeks, not hours.
- Generic subtasks that apply to any phase. "Conduct initial research",
  "document findings", "review and approve" are templates, not subtasks.
  Each subtask must be tied to the specific phase being decomposed and
  reflect the actual work described in the project plan.
- Resource list inflation. The resources_needed field should name the actual
  humans and materials required (e.g., "licensed electrician", "concrete
  mixer", "structural engineer"). Generic entries like "team members",
  "tools", "budget" provide no planning value.
- Serial call degradation. Later calls in the WBS level 3 sequence run with
  more accumulated context than earlier ones. If subtask quality degrades
  across phases (generic late phases, specific early phases), the model is
  exhibiting context fatigue. Prefer concise, dense descriptions to reduce
  context pressure.
- Phase-context mismatch. Each decomposition call receives the specific
  phase to break down. Do not decompose a different phase than the one
  provided, and do not repeat subtasks from previous phases.
"""


class WBSSubtask(BaseModel):
    """
    A subtask.
    """
    name: str = Field(
        description="Short name of the subtask, such as: Prepare necessary documentation for permits, Conduct interviews with potential contractors, Secure final approval of funds."
    )
    description: str = Field(
        description="Longer text that describes the subtask in more detail, such as: Objective, Scope, Steps, Deliverables."
    )
    resources_needed: list[str] = Field(
        description="List of resources needed to complete the subtask. Example: ['Project manager', 'Architect', 'Engineer', 'Construction crew']."
    )

class WBSTaskDetails(BaseModel):
    """
    A big task in the project decomposed into a few smaller tasks.
    """
    subtasks: list[WBSSubtask] = Field(
        description="List of subtasks."
    )

QUERY_PREAMBLE = """
Decompose a big task into smaller, more manageable subtasks.

Split the task into 3 to 5 subtasks.

Pick a subtask name that is short, max 7 words or max 60 characters.

Don't enumerate the subtasks with integers or letters.

Don't assign a uuid to the subtask.

"""

@dataclass
class CreateWBSLevel3:
    """
    WBS Level 3: Creating a Work Breakdown Structure (WBS) from a project plan.
    """
    query: str
    response: dict
    metadata: dict
    tasks: list[dict]
    task_uuids: list[str]

    @classmethod
    def format_query(cls, plan_json: dict, wbs_level1_json: dict, wbs_level2_json: list, wbs_level2_task_durations_json: dict, decompose_task_id: str) -> str:
        if not isinstance(plan_json, dict):
            raise ValueError("Invalid plan_json.")
        if not isinstance(wbs_level1_json, dict):
            raise ValueError("Invalid wbs_level1_json.")
        if not isinstance(wbs_level2_json, list):
            raise ValueError("Invalid wbs_level1_json.")
        if not isinstance(wbs_level2_task_durations_json, list):
            raise ValueError("Invalid wbs_level2_task_durations_json.")
        if not isinstance(decompose_task_id, str):
            raise ValueError("Invalid decompose_task_id.")

        query = f"""
The project plan:
{format_json_for_use_in_query(plan_json)}

WBS Level 1:
{format_json_for_use_in_query(wbs_level1_json)}

WBS Level 2:
{format_json_for_use_in_query(wbs_level2_json)}

WBS Level 2 time estimates:
{format_json_for_use_in_query(wbs_level2_task_durations_json)}

Only decompose this task:
"{decompose_task_id}"
"""
        return query

    @classmethod
    def execute(cls, llm: LLM, query: str, decompose_task_id: str) -> 'CreateWBSLevel3':
        """
        Invoke LLM to decompose a big WBS level 2 task into smaller tasks.
        """
        if not isinstance(llm, LLM):
            raise ValueError("Invalid LLM instance.")
        if not isinstance(query, str):
            raise ValueError("Invalid query.")
        if not isinstance(decompose_task_id, str):
            raise ValueError("Invalid decompose_task_id.")

        start_time = time.perf_counter()

        sllm = llm.as_structured_llm(WBSTaskDetails)
        response = sllm.complete(QUERY_PREAMBLE + query)
        json_response = json.loads(response.text)

        end_time = time.perf_counter()
        duration = int(ceil(end_time - start_time))

        metadata = dict(llm.metadata)
        metadata["llm_classname"] = llm.class_name()
        metadata["duration"] = duration

        # Cleanup the json response from the LLM model, assign unique ids to each subtask.
        parent_task_id = decompose_task_id
        result_tasks = []
        result_task_uuids = []
        for subtask in json_response['subtasks']:
            name = subtask['name']
            description = subtask['description']
            resources_needed = subtask['resources_needed']
            uuid = str(uuid4())
            subtask_item = {
                "id": uuid,
                "name": name,
                "description": description,
                "resources_needed": resources_needed,
                "parent_id": parent_task_id
            }
            result_tasks.append(subtask_item)
            result_task_uuids.append(uuid)

        result = CreateWBSLevel3(
            query=query,
            response=json_response,
            metadata=metadata,
            tasks=result_tasks,
            task_uuids=result_task_uuids
        )
        return result    

    def raw_response_dict(self, include_metadata=True, include_query=True) -> dict:
        d = self.response.copy()
        if include_metadata:
            d['metadata'] = self.metadata
        if include_query:
            d['query'] = self.query
        return d

if __name__ == "__main__":
    from llama_index.llms.ollama import Ollama

    # TODO: Eliminate hardcoded paths
    basepath = '/Users/neoneye/Desktop/planexe_data'

    def load_json(relative_path: str) -> dict:
        path = os.path.join(basepath, relative_path)
        print(f"loading file: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            the_json = json.load(f)
        return the_json

    plan_json = load_json('002-project_plan.json')
    wbs_level1_json = load_json('006-wbs_level1.json')
    wbs_level2_json = load_json('008-wbs_level2.json')
    wbs_level2_task_durations_json = load_json('012-task_durations.json')
    decompose_task_id = "1c690f4a-ae8e-493d-9e47-6da58ef5b24c"

    query = CreateWBSLevel3.format_query(plan_json, wbs_level1_json, wbs_level2_json, wbs_level2_task_durations_json, decompose_task_id)

    model_name = "llama3.1:latest"
    # model_name = "qwen2.5-coder:latest"
    # model_name = "phi4:latest"
    llm = Ollama(model=model_name, request_timeout=120.0, temperature=0.5, is_function_calling_model=False)

    print(f"Query: {query}")
    result = CreateWBSLevel3.execute(llm, query, decompose_task_id)

    print("\n\nResponse:")
    print(json.dumps(result.raw_response_dict(include_query=False), indent=2))

    print("\n\nExtracted tasks:")
    print(json.dumps(result.tasks, indent=2))

