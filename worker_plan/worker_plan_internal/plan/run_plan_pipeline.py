"""
PROMPT> python -m worker_plan_internal.plan.run_plan_pipeline

In order to resume an unfinished run.
Insert the run_id_dir of the thing you want to resume.
If it's an already finished run, then remove the "999-pipeline_complete.txt" file.
PROMPT> RUN_ID_DIR=/absolute/path/to/PlanExe_20250216_150332 python -m worker_plan_internal.plan.run_plan_pipeline
"""
from dataclasses import dataclass, field
from datetime import date, datetime
import os
import logging
import json
from typing import Any, Optional
import luigi
from pathlib import Path
import sys
from llama_index.core.llms.llm import LLM
from worker_plan_internal.diagnostics.redline_gate import RedlineGate
from worker_plan_internal.diagnostics.premise_attack import PremiseAttack
from worker_plan_internal.diagnostics.premortem import Premortem
from worker_plan_internal.plan.pipeline_config import PIPELINE_CONFIG
from worker_plan_internal.lever.deduplicate_levers import DeduplicateLevers
from worker_plan_internal.lever.scenarios_markdown import ScenariosMarkdown
from worker_plan_internal.lever.strategic_decisions_markdown import StrategicDecisionsMarkdown
from worker_plan_api.filenames import FilenameEnum, ExtraFilenameEnum
from worker_plan_api.speedvsdetail import SpeedVsDetailEnum
from worker_plan_internal.utils.planexe_llmconfig import PlanExeLLMConfig
from worker_plan_internal.assume.identify_purpose import IdentifyPurpose
from worker_plan_internal.assume.identify_plan_type import IdentifyPlanType
from worker_plan_internal.assume.physical_locations import PhysicalLocations
from worker_plan_internal.assume.currency_strategy import CurrencyStrategy
from worker_plan_internal.assume.identify_risks import IdentifyRisks
from worker_plan_internal.assume.make_assumptions import MakeAssumptions
from worker_plan_internal.assume.distill_assumptions import DistillAssumptions
from worker_plan_internal.assume.review_assumptions import ReviewAssumptions
from worker_plan_internal.assume.shorten_markdown import ShortenMarkdown
from worker_plan_internal.expert.pre_project_assessment import PreProjectAssessment
from worker_plan_internal.plan.project_plan import ProjectPlan
from worker_plan_internal.document.identify_documents import IdentifyDocuments
from worker_plan_internal.document.filter_documents_to_find import FilterDocumentsToFind
from worker_plan_internal.document.filter_documents_to_create import FilterDocumentsToCreate
from worker_plan_internal.document.draft_document_to_find import DraftDocumentToFind
from worker_plan_internal.document.draft_document_to_create import DraftDocumentToCreate
from worker_plan_internal.document.markdown_with_document import markdown_rows_with_document_to_create, markdown_rows_with_document_to_find
from worker_plan_internal.governance.governance_phase1_audit import GovernancePhase1Audit
from worker_plan_internal.governance.governance_phase2_bodies import GovernancePhase2Bodies
from worker_plan_internal.governance.governance_phase3_impl_plan import GovernancePhase3ImplPlan
from worker_plan_internal.governance.governance_phase4_decision_escalation_matrix import GovernancePhase4DecisionEscalationMatrix
from worker_plan_internal.governance.governance_phase5_monitoring_progress import GovernancePhase5MonitoringProgress
from worker_plan_internal.governance.governance_phase6_extra import GovernancePhase6Extra
from worker_plan_internal.plan.related_resources import RelatedResources
from worker_plan_internal.questions_answers.questions_answers import QuestionsAnswers
from worker_plan_internal.lever.identify_potential_levers import IdentifyPotentialLevers
from worker_plan_internal.lever.enrich_potential_levers import EnrichPotentialLevers
from worker_plan_internal.lever.focus_on_vital_few_levers import FocusOnVitalFewLevers
from worker_plan_internal.lever.candidate_scenarios import CandidateScenarios
from worker_plan_internal.lever.select_scenario import SelectScenario
from worker_plan_internal.swot.swot_analysis import SWOTAnalysis
from worker_plan_internal.expert.expert_finder import ExpertFinder
from worker_plan_internal.expert.expert_criticism import ExpertCriticism
from worker_plan_internal.expert.expert_orchestrator import ExpertOrchestrator
from worker_plan_internal.plan.create_wbs_level1 import CreateWBSLevel1
from worker_plan_internal.plan.create_wbs_level2 import CreateWBSLevel2
from worker_plan_internal.plan.create_wbs_level3 import CreateWBSLevel3
from worker_plan_internal.pitch.create_pitch import CreatePitch
from worker_plan_internal.pitch.convert_pitch_to_markdown import ConvertPitchToMarkdown
from worker_plan_internal.plan.identify_wbs_task_dependencies import IdentifyWBSTaskDependencies
from worker_plan_internal.plan.estimate_wbs_task_durations import EstimateWBSTaskDurations
from worker_plan_internal.plan.data_collection import DataCollection
from worker_plan_internal.plan.review_plan import ReviewPlan
from worker_plan_internal.plan.executive_summary import ExecutiveSummary
from worker_plan_internal.team.find_team_members import FindTeamMembers
from worker_plan_internal.team.enrich_team_members_with_contract_type import EnrichTeamMembersWithContractType
from worker_plan_internal.team.enrich_team_members_with_background_story import EnrichTeamMembersWithBackgroundStory
from worker_plan_internal.team.enrich_team_members_with_environment_info import EnrichTeamMembersWithEnvironmentInfo
from worker_plan_internal.team.team_markdown_document import TeamMarkdownDocumentBuilder
from worker_plan_internal.team.review_team import ReviewTeam
from worker_plan_internal.self_audit.self_audit import SelfAudit
from worker_plan_internal.wbs.wbs_task import WBSTask, WBSProject
from worker_plan_internal.wbs.wbs_populate import WBSPopulate
from worker_plan_internal.wbs.wbs_task_tooltip import WBSTaskTooltip
from worker_plan_internal.schedule.project_schedule_populator import ProjectSchedulePopulator
from worker_plan_internal.schedule.schedule import ProjectSchedule
from worker_plan_internal.schedule.export_gantt_dhtmlx import ExportGanttDHTMLX
from worker_plan_internal.schedule.export_gantt_csv import ExportGanttCSV
from worker_plan_internal.schedule.export_gantt_mermaid import ExportGanttMermaid
from worker_plan_internal.llm_util.llm_executor import LLMExecutor, LLMModelFromName, ShouldStopCallbackParameters, PipelineStopRequested
from worker_plan_internal.llm_factory import get_llm_names_by_priority, SPECIAL_AUTO_ID, is_valid_llm_name
from worker_plan_internal.format_json_for_use_in_query import format_json_for_use_in_query
from worker_plan_internal.report.report_generator import ReportGenerator
from worker_plan_internal.luigi_util.obtain_output_files import ObtainOutputFiles
from worker_plan_internal.plan.pipeline_environment import PipelineEnvironment
from worker_plan_internal.plan.ping_llm import run_ping_llm_report

logger = logging.getLogger(__name__)
DEFAULT_LLM_MODEL = "ollama-llama3.1"

REPORT_EXECUTE_PLAN_SECTION_HIDDEN = True
# REPORT_EXECUTE_PLAN_SECTION_HIDDEN = False

class PlanTask(luigi.Task):
    # Default it to the current timestamp, eg. 19841231_235959
    # Path to the 'run/{run_id}' directory
    run_id_dir = luigi.Parameter(default=Path('run') / datetime.now().strftime("%Y%m%d_%H%M%S"))

    # By default, run everything but it's slow.
    # This can be overridden in developer mode, where a quick turnaround is needed, and the details are not important.
    speedvsdetail = luigi.EnumParameter(enum=SpeedVsDetailEnum, default=SpeedVsDetailEnum.ALL_DETAILS_BUT_SLOW)

    # List of LLM models to try, in order of priority.
    llm_models = luigi.ListParameter(default=[DEFAULT_LLM_MODEL])

    # Optional callback for updating progress bar and aborting the pipeline.
    # If the callback raises PipelineStopRequested, the pipeline will be aborted. This is the only exception that is allowed to be raised.
    # If the callback raises exceptions different than PipelineStopRequested, the pipeline will be aborted. This means that something went wrong, and we should not continue.
    # If the callback doesn't raise an exception, the pipeline will continue.
    # If the callback is not provided, the pipeline will run until completion.
    _pipeline_executor_callback = luigi.Parameter(default=None, significant=False, visibility=luigi.parameter.ParameterVisibility.PRIVATE)

    def file_path(self, filename: FilenameEnum) -> Path:
        return self.run_id_dir / filename.value
    
    def local_target(self, filename: FilenameEnum) -> luigi.LocalTarget:
        return luigi.LocalTarget(self.file_path(filename))

    def create_llm_executor(self) -> LLMExecutor:
        """
        Create an LLMExecutor instance.
        - Responsible for stopping the pipeline when the user presses Ctrl-C or closes the browser tab.
        - Fallback mechanism to try the next LLM if the current one fails.
        """
        # Redirect the callback to the pipeline_executor_callback.
        def should_stop_callback(parameters: ShouldStopCallbackParameters) -> None:
            if self._pipeline_executor_callback is None:
                return
            # The pipeline_executor_callback expects (task, duration) but we have ShouldStopCallbackParameters
            total_duration = parameters.total_duration
            self._pipeline_executor_callback(self, total_duration)

        llm_model_instances = LLMModelFromName.from_names(self.llm_models)

        return LLMExecutor(
            llm_models=llm_model_instances,
            should_stop_callback=should_stop_callback
        )

    def run(self):
        """
        Don't override this method. Instead either override the run_inner() method, or override the run_with_llm() method.
        """
        try:
            self.run_inner()
        except PipelineStopRequested as e:
            logger.debug(f"{self.__class__.__name__} -> PipelineStopRequested raised: {e}")
            # This exception is raised by the should_stop_callback
            # If we get here, it means that the pipeline was aborted by the callback, such as by the user pressing Ctrl-C or closing the browser tab.
            # Create a flag file to signal that the stop was intentional.
            flag_path = self.run_id_dir / ExtraFilenameEnum.PIPELINE_STOP_REQUESTED_FLAG.value
            logger.info(f"Creating stop flag file: {flag_path!r}")
            flag_path.touch()
            raise
        except Exception as e:
            # Re-raise the exception with a more descriptive message
            raise Exception(f"Failed to run {self.__class__.__name__} with any of the LLMs in the list: {self.llm_models!r} for run_id_dir: {self.run_id_dir!r}") from e

    def run_inner(self):
        """
        Override this method or the run_with_llm() method.
        """
        llm_executor: LLMExecutor = self.create_llm_executor()
        
        # Attempt executing this code with the first LLM, if that fails, try the next one, and so on.
        def execute_function(llm: LLM) -> None:
            self.run_with_llm(llm)

        # Make multiple attempts at running the run_with_llm() function.
        # No try/except needed here. Let PlanTask.run() handle it.
        llm_executor.run(execute_function)

    def run_with_llm(self, llm: LLM) -> None:
        """
        Override this method or the run_inner() method.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class StartTimeTask(PlanTask):
    """The timestamp when the pipeline was started."""
    def output(self):
        return self.local_target(FilenameEnum.START_TIME)

    def run(self):
        # The Gradio/Flask app that starts the luigi pipeline, must first create the `START_TIME` file inside the `run_id_dir`.
        # This code will ONLY run if the Gradio/Flask app *failed* to create the file.
        raise AssertionError(f"This code is not supposed to be run. Before starting the pipeline the '{FilenameEnum.START_TIME.value}' file must be present in the `run_id_dir`: {self.run_id_dir!r}")


class SetupTask(PlanTask):
    """The plan prompt text provided by the user."""
    def output(self):
        return self.local_target(FilenameEnum.INITIAL_PLAN)

    def run(self):
        # The Gradio/Flask app that starts the luigi pipeline, must first create the `INITIAL_PLAN` file inside the `run_id_dir`.
        # This code will ONLY run if the Gradio/Flask app *failed* to create the file.
        raise AssertionError(f"This code is not supposed to be run. Before starting the pipeline the '{FilenameEnum.INITIAL_PLAN.value}' file must be present in the `run_id_dir`: {self.run_id_dir!r}")


class RedlineGateTask(PlanTask):
    def requires(self):
        return self.clone(SetupTask)

    def output(self):
        return {
            'raw': self.local_target(FilenameEnum.REDLINE_GATE_RAW),
            'markdown': self.local_target(FilenameEnum.REDLINE_GATE_MARKDOWN)
        }

    def run_with_llm(self, llm: LLM) -> None:
        # Read inputs from required tasks.
        with self.input().open("r") as f:
            plan_prompt = f.read()

        redline_gate = RedlineGate.execute(llm, plan_prompt)

        # Write the result to disk.
        output_raw_path = self.output()['raw'].path
        redline_gate.save_raw(output_raw_path)
        output_markdown_path = self.output()['markdown'].path
        redline_gate.save_markdown(output_markdown_path)


class PremiseAttackTask(PlanTask):
    def requires(self):
        return self.clone(SetupTask)

    def output(self):
        return {
            'raw': self.local_target(FilenameEnum.PREMISE_ATTACK_RAW),
            'markdown': self.local_target(FilenameEnum.PREMISE_ATTACK_MARKDOWN)
        }

    def run_inner(self):
        llm_executor: LLMExecutor = self.create_llm_executor()

        # Read inputs from required tasks.
        with self.input().open("r") as f:
            plan_prompt = f.read()

        premise_attack = PremiseAttack.execute(llm_executor, plan_prompt)

        # Write the result to disk.
        output_raw_path = self.output()['raw'].path
        premise_attack.save_raw(output_raw_path)
        output_markdown_path = self.output()['markdown'].path
        premise_attack.save_markdown(output_markdown_path)


class IdentifyPurposeTask(PlanTask):
    """
    Determine if this is this going to be a business/personal/other plan.
    """
    def requires(self):
        return self.clone(SetupTask)

    def output(self):
        return {
            'raw': self.local_target(FilenameEnum.IDENTIFY_PURPOSE_RAW),
            'markdown': self.local_target(FilenameEnum.IDENTIFY_PURPOSE_MARKDOWN)
        }

    def run_with_llm(self, llm: LLM) -> None:
        # Read inputs from required tasks.
        with self.input().open("r") as f:
            plan_prompt = f.read()

        identify_purpose = IdentifyPurpose.execute(llm, plan_prompt)

        # Write the result to disk.
        output_raw_path = self.output()['raw'].path
        identify_purpose.save_raw(output_raw_path)
        output_markdown_path = self.output()['markdown'].path
        identify_purpose.save_markdown(output_markdown_path)


class PlanTypeTask(PlanTask):
    """
    Determine if the plan is purely digital or requires physical locations.
    """
    def requires(self):
        return {
            'setup': self.clone(SetupTask),
            'identify_purpose': self.clone(IdentifyPurposeTask)
        }

    def output(self):
        return {
            'raw': self.local_target(FilenameEnum.PLAN_TYPE_RAW),
            'markdown': self.local_target(FilenameEnum.PLAN_TYPE_MARKDOWN)
        }

    def run_with_llm(self, llm: LLM) -> None:
        # Read inputs from required tasks.
        with self.input()['setup'].open("r") as f:
            plan_prompt = f.read()
        with self.input()['identify_purpose']['markdown'].open("r") as f:
            identify_purpose_markdown = f.read()

        query = (
            f"File 'plan.txt':\n{plan_prompt}\n\n"
            f"File 'purpose.md':\n{identify_purpose_markdown}"
        )

        identify_plan_type = IdentifyPlanType.execute(llm, query)

        # Write the result to disk.
        output_raw_path = self.output()['raw'].path
        identify_plan_type.save_raw(str(output_raw_path))
        output_markdown_path = self.output()['markdown'].path
        identify_plan_type.save_markdown(str(output_markdown_path))

class PotentialLeversTask(PlanTask):
    """
    Identify potential levers that can be adjusted.
    """
    def requires(self):
        return {
            'setup': self.clone(SetupTask),
            'identify_purpose': self.clone(IdentifyPurposeTask),
            'plan_type': self.clone(PlanTypeTask)
        }

    def output(self):
        return {
            'raw': self.local_target(FilenameEnum.POTENTIAL_LEVERS_RAW),
            'clean': self.local_target(FilenameEnum.POTENTIAL_LEVERS_CLEAN),
        }

    def run_inner(self):
        llm_executor: LLMExecutor = self.create_llm_executor()

        # Read inputs from required tasks.
        with self.input()['setup'].open("r") as f:
            plan_prompt = f.read()
        with self.input()['identify_purpose']['markdown'].open("r") as f:
            identify_purpose_markdown = f.read()
        with self.input()['plan_type']['markdown'].open("r") as f:
            plan_type_markdown = f.read()

        query = (
            f"File 'plan.txt':\n{plan_prompt}\n\n"
            f"File 'purpose.md':\n{identify_purpose_markdown}\n\n"
            f"File 'plan_type.md':\n{plan_type_markdown}"
        )

        identify_potential_levers = IdentifyPotentialLevers.execute(llm_executor, query)

        # Write the result to disk.
        output_raw_path = self.output()['raw'].path
        identify_potential_levers.save_raw(str(output_raw_path))
        output_clean_path = self.output()['clean'].path
        identify_potential_levers.save_clean(str(output_clean_path))


class DeduplicateLeversTask(PlanTask):
    """
    The potential levers usually have some redundant levers.
    """
    def requires(self):
        return {
            'setup': self.clone(SetupTask),
            'identify_purpose': self.clone(IdentifyPurposeTask),
            'plan_type': self.clone(PlanTypeTask),
            'potential_levers': self.clone(PotentialLeversTask)
        }

    def output(self):
        return {
            'raw': self.local_target(FilenameEnum.DEDUPLICATED_LEVERS_RAW)
        }

    def run_inner(self):
        llm_executor: LLMExecutor = self.create_llm_executor()

        # Read inputs from required tasks.
        with self.input()['setup'].open("r") as f:
            plan_prompt = f.read()
        with self.input()['identify_purpose']['markdown'].open("r") as f:
            identify_purpose_markdown = f.read()
        with self.input()['plan_type']['markdown'].open("r") as f:
            plan_type_markdown = f.read()
        with self.input()['potential_levers']['clean'].open("r") as f:
            lever_item_list = json.load(f)

        query = (
            f"File 'plan.txt':\n{plan_prompt}\n\n"
            f"File 'purpose.md':\n{identify_purpose_markdown}\n\n"
            f"File 'plan_type.md':\n{plan_type_markdown}"
        )

        deduplicate_levers = DeduplicateLevers.execute(
            llm_executor,
            project_context=query,
            raw_levers_list=lever_item_list
        )

        # Write the result to disk.
        output_raw_path = self.output()['raw'].path
        deduplicate_levers.save_raw(str(output_raw_path))

class EnrichLeversTask(PlanTask):
    """
    Enrich potential levers with more information.
    """
    def requires(self):
        return {
            'setup': self.clone(SetupTask),
            'identify_purpose': self.clone(IdentifyPurposeTask),
            'plan_type': self.clone(PlanTypeTask),
            'deduplicate_levers': self.clone(DeduplicateLeversTask),
        }

    def output(self):
        return {
            'raw': self.local_target(FilenameEnum.ENRICHED_LEVERS_RAW)
        }

    def run_inner(self):
        llm_executor: LLMExecutor = self.create_llm_executor()

        # Read inputs from required tasks.
        with self.input()['setup'].open("r") as f:
            plan_prompt = f.read()
        with self.input()['identify_purpose']['markdown'].open("r") as f:
            identify_purpose_markdown = f.read()
        with self.input()['plan_type']['markdown'].open("r") as f:
            plan_type_markdown = f.read()
        with self.input()['deduplicate_levers']['raw'].open("r") as f:
            json_dict = json.load(f)
            lever_item_list = json_dict["deduplicated_levers"]

        query = (
            f"File 'plan.txt':\n{plan_prompt}\n\n"
            f"File 'purpose.md':\n{identify_purpose_markdown}\n\n"
            f"File 'plan_type.md':\n{plan_type_markdown}"
        )

        enrich_potential_levers = EnrichPotentialLevers.execute(
            llm_executor,
            project_context=query,
            raw_levers_list=lever_item_list
        )

        # Write the result to disk.
        output_raw_path = self.output()['raw'].path
        enrich_potential_levers.save_raw(str(output_raw_path))

class FocusOnVitalFewLeversTask(PlanTask):
    """
    Apply the 80/20 principle to the levers.
    """
    def requires(self):
        return {
            'setup': self.clone(SetupTask),
            'identify_purpose': self.clone(IdentifyPurposeTask),
            'plan_type': self.clone(PlanTypeTask),
            'enriched_levers': self.clone(EnrichLeversTask)
        }

    def output(self):
        return {
            'raw': self.local_target(FilenameEnum.VITAL_FEW_LEVERS_RAW)
        }

    def run_inner(self):
        llm_executor: LLMExecutor = self.create_llm_executor()

        # Read inputs from required tasks.
        with self.input()['setup'].open("r") as f:
            plan_prompt = f.read()
        with self.input()['identify_purpose']['markdown'].open("r") as f:
            identify_purpose_markdown = f.read()
        with self.input()['plan_type']['markdown'].open("r") as f:
            plan_type_markdown = f.read()
        with self.input()['enriched_levers']['raw'].open("r") as f:
            lever_item_list = json.load(f)["characterized_levers"]

        query = (
            f"File 'plan.txt':\n{plan_prompt}\n\n"
            f"File 'purpose.md':\n{identify_purpose_markdown}\n\n"
            f"File 'plan_type.md':\n{plan_type_markdown}\n\n"
        )

        focus_on_vital_few_levers = FocusOnVitalFewLevers.execute(
            llm_executor,
            project_context=query,
            raw_levers_list=lever_item_list
        )

        # Write the result to disk.
        output_raw_path = self.output()['raw'].path
        focus_on_vital_few_levers.save_raw(str(output_raw_path))


class StrategicDecisionsMarkdownTask(PlanTask):
    """
    Human readable markdown with the levers.
    """
    def requires(self):
        return {
            'enriched_levers': self.clone(EnrichLeversTask),
            'levers_vital_few': self.clone(FocusOnVitalFewLeversTask)
        }

    def output(self):
        return {
            'markdown': self.local_target(FilenameEnum.STRATEGIC_DECISIONS_MARKDOWN)
        }

    def run(self):
        with self.input()['enriched_levers']['raw'].open("r") as f:
            enrich_lever_list = json.load(f)["characterized_levers"]
        with self.input()['levers_vital_few']['raw'].open("r") as f:
            vital_data = json.load(f)
            vital_lever_list = vital_data["levers"]
            lever_assessments_list = vital_data.get("response", {}).get("lever_assessments", [])
            vital_levers_summary = vital_data.get("response", {}).get("summary", "")

        result = StrategicDecisionsMarkdown(enrich_lever_list, vital_lever_list, vital_levers_summary, lever_assessments_list)
        result.save_markdown(self.output()['markdown'].path)


class CandidateScenariosTask(PlanTask):
    """
    Combinations of the vital few levers.
    """
    def requires(self):
        return {
            'setup': self.clone(SetupTask),
            'identify_purpose': self.clone(IdentifyPurposeTask),
            'plan_type': self.clone(PlanTypeTask),
            'levers_vital_few': self.clone(FocusOnVitalFewLeversTask)
        }

    def output(self):
        return {
            'raw': self.local_target(FilenameEnum.CANDIDATE_SCENARIOS_RAW),
            'clean': self.local_target(FilenameEnum.CANDIDATE_SCENARIOS_CLEAN)
        }

    def run_inner(self):
        llm_executor: LLMExecutor = self.create_llm_executor()

        # Read inputs from required tasks.
        with self.input()['setup'].open("r") as f:
            plan_prompt = f.read()
        with self.input()['identify_purpose']['markdown'].open("r") as f:
            identify_purpose_markdown = f.read()
        with self.input()['plan_type']['markdown'].open("r") as f:
            plan_type_markdown = f.read()
        with self.input()['levers_vital_few']['raw'].open("r") as f:
            lever_item_list = json.load(f)["levers"]

        query = (
            f"File 'plan.txt':\n{plan_prompt}\n\n"
            f"File 'purpose.md':\n{identify_purpose_markdown}\n\n"
            f"File 'plan_type.md':\n{plan_type_markdown}\n\n"
        )

        scenarios = CandidateScenarios.execute(
            llm_executor=llm_executor,
            project_context=query,
            raw_vital_levers=lever_item_list
        )

        # Write the result to disk.
        output_raw_path = self.output()['raw'].path
        scenarios.save_raw(str(output_raw_path))
        output_clean_path = self.output()['clean'].path
        scenarios.save_clean(str(output_clean_path))


class SelectScenarioTask(PlanTask):
    """
    Pick the best fitting scenario to make a plan for.
    """
    def requires(self):
        return {
            'setup': self.clone(SetupTask),
            'identify_purpose': self.clone(IdentifyPurposeTask),
            'plan_type': self.clone(PlanTypeTask),
            'levers_vital_few': self.clone(FocusOnVitalFewLeversTask),
            'candidate_scenarios': self.clone(CandidateScenariosTask)
        }

    def output(self):
        return {
            'raw': self.local_target(FilenameEnum.SELECTED_SCENARIO_RAW),
            'clean': self.local_target(FilenameEnum.SELECTED_SCENARIO_CLEAN)
        }

    def run_inner(self):
        llm_executor: LLMExecutor = self.create_llm_executor()

        # Read inputs from required tasks.
        with self.input()['setup'].open("r") as f:
            plan_prompt = f.read()
        with self.input()['identify_purpose']['markdown'].open("r") as f:
            identify_purpose_markdown = f.read()
        with self.input()['plan_type']['markdown'].open("r") as f:
            plan_type_markdown = f.read()
        with self.input()['levers_vital_few']['raw'].open("r") as f:
            lever_item_list = json.load(f)["levers"]
        with self.input()['candidate_scenarios']['clean'].open("r") as f:
            scenarios_list = json.load(f).get('scenarios', [])

        query = (
            f"File 'plan.txt':\n{plan_prompt}\n\n"
            f"File 'purpose.md':\n{identify_purpose_markdown}\n\n"
            f"File 'plan_type.md':\n{plan_type_markdown}\n\n"
            f"File 'levers_vital_few.json':\n{format_json_for_use_in_query(lever_item_list)}\n\n"
            f"File 'candidate_scenarios.json':\n{format_json_for_use_in_query(scenarios_list)}"
        )

        select_scenario = SelectScenario.execute(
            llm_executor=llm_executor,
            project_context=query,
            scenarios=scenarios_list
        )

        # Write the result to disk.
        output_raw_path = self.output()['raw'].path
        select_scenario.save_raw(str(output_raw_path))
        output_clean_path = self.output()['clean'].path
        select_scenario.save_clean(str(output_clean_path))


class ScenariosMarkdownTask(PlanTask):
    """
    Present the scenarios in a human readable format.
    """
    def requires(self):
        return {
            'candidate_scenarios': self.clone(CandidateScenariosTask),
            'selected_scenario': self.clone(SelectScenarioTask)
        }

    def output(self):
        return {
            'markdown': self.local_target(FilenameEnum.SCENARIOS_MARKDOWN)
        }

    def run(self):
        with self.input()['candidate_scenarios']['clean'].open("r") as f:
            scenarios_list = json.load(f).get('scenarios', [])
        with self.input()['selected_scenario']['clean'].open("r") as f:
            selected_scenario_dict = json.load(f)

        # Extract the required data from the selected scenario
        plan_characteristics = selected_scenario_dict.get('plan_characteristics', {})
        scenario_assessments = selected_scenario_dict.get('scenario_assessments', [])
        final_choice = selected_scenario_dict.get('final_choice', {})

        result = ScenariosMarkdown(scenarios_list, plan_characteristics, scenario_assessments, final_choice)
        result.save_markdown(self.output()['markdown'].path)


class PhysicalLocationsTask(PlanTask):
    """
    Identify/suggest physical locations for the plan.
    """
    def requires(self):
        return {
            'setup': self.clone(SetupTask),
            'identify_purpose': self.clone(IdentifyPurposeTask),
            'plan_type': self.clone(PlanTypeTask),
            'strategic_decisions_markdown': self.clone(StrategicDecisionsMarkdownTask),
            'scenarios_markdown': self.clone(ScenariosMarkdownTask)
        }

    def output(self):
        return {
            'raw': self.local_target(FilenameEnum.PHYSICAL_LOCATIONS_RAW),
            'markdown': self.local_target(FilenameEnum.PHYSICAL_LOCATIONS_MARKDOWN)
        }

    def run_with_llm(self, llm: LLM) -> None:
        logger.info("Identify/suggest physical locations for the plan...")

        # Read inputs from required tasks.
        with self.input()['setup'].open("r") as f:
            plan_prompt = f.read()
        with self.input()['identify_purpose']['markdown'].open("r") as f:
            identify_purpose_markdown = f.read()
        with self.input()['plan_type']['raw'].open("r") as f:
            plan_type_dict = json.load(f)
        with self.input()['plan_type']['markdown'].open("r") as f:
            plan_type_markdown = f.read()
        with self.input()['strategic_decisions_markdown']['markdown'].open("r") as f:
            strategic_decisions_markdown = f.read()
        with self.input()['scenarios_markdown']['markdown'].open("r") as f:
            scenarios_markdown = f.read()

        output_raw_path = self.output()['raw'].path
        output_markdown_path = self.output()['markdown'].path

        plan_type = plan_type_dict.get("plan_type")
        if plan_type == "physical":
            query = (
                f"File 'plan.txt':\n{plan_prompt}\n\n"
                f"File 'purpose.md':\n{identify_purpose_markdown}\n\n"
                f"File 'plan_type.md':\n{plan_type_markdown}\n\n"
                f"File 'strategic_decisions.md':\n{strategic_decisions_markdown}\n\n"
                f"File 'scenarios.md':\n{scenarios_markdown}"
            )

            physical_locations = PhysicalLocations.execute(llm, query)

            # Write the physical locations to disk.
            physical_locations.save_raw(str(output_raw_path))
            physical_locations.save_markdown(str(output_markdown_path))
        else:
            # Write an empty file to indicate that there are no physical locations.
            data = {
                "comment": "The plan is purely digital, without any physical locations."
            }
            with open(output_raw_path, "w") as f:
                json.dump(data, f, indent=2)
            
            with open(output_markdown_path, "w", encoding='utf-8') as f:
                f.write("The plan is purely digital, without any physical locations.")

class CurrencyStrategyTask(PlanTask):
    """
    Identify/suggest what currency to use for the plan, depending on the physical locations.
    """
    def requires(self):
        return {
            'setup': self.clone(SetupTask),
            'identify_purpose': self.clone(IdentifyPurposeTask),
            'plan_type': self.clone(PlanTypeTask),
            'physical_locations': self.clone(PhysicalLocationsTask),
            'strategic_decisions_markdown': self.clone(StrategicDecisionsMarkdownTask),
            'scenarios_markdown': self.clone(ScenariosMarkdownTask)
        }

    def output(self):
        return {
            'raw': self.local_target(FilenameEnum.CURRENCY_STRATEGY_RAW),
            'markdown': self.local_target(FilenameEnum.CURRENCY_STRATEGY_MARKDOWN)
        }

    def run_with_llm(self, llm: LLM) -> None:
        # Read inputs from required tasks.
        with self.input()['setup'].open("r") as f:
            plan_prompt = f.read()
        with self.input()['identify_purpose']['markdown'].open("r") as f:
            identify_purpose_markdown = f.read()
        with self.input()['plan_type']['markdown'].open("r") as f:
            plan_type_markdown = f.read()
        with self.input()['strategic_decisions_markdown']['markdown'].open("r") as f:
            strategic_decisions_markdown = f.read()
        with self.input()['scenarios_markdown']['markdown'].open("r") as f:
            scenarios_markdown = f.read()
        with self.input()['physical_locations']['markdown'].open("r") as f:
            physical_locations_markdown = f.read()

        query = (
            f"File 'plan.txt':\n{plan_prompt}\n\n"
            f"File 'purpose.md':\n{identify_purpose_markdown}\n\n"
            f"File 'plan_type.md':\n{plan_type_markdown}\n\n"
            f"File 'strategic_decisions.md':\n{strategic_decisions_markdown}\n\n"
            f"File 'scenarios.md':\n{scenarios_markdown}\n\n"
            f"File 'physical_locations.md':\n{physical_locations_markdown}"
        )

        currency_strategy = CurrencyStrategy.execute(llm, query)

        # Write the result to disk.
        output_raw_path = self.output()['raw'].path
        currency_strategy.save_raw(str(output_raw_path))
        output_markdown_path = self.output()['markdown'].path
        currency_strategy.save_markdown(str(output_markdown_path))


class IdentifyRisksTask(PlanTask):
    """
    Identify risks for the plan, depending on the physical locations.
    """
    def requires(self):
        return {
            'setup': self.clone(SetupTask),
            'identify_purpose': self.clone(IdentifyPurposeTask),
            'plan_type': self.clone(PlanTypeTask),
            'strategic_decisions_markdown': self.clone(StrategicDecisionsMarkdownTask),
            'scenarios_markdown': self.clone(ScenariosMarkdownTask),
            'physical_locations': self.clone(PhysicalLocationsTask),
            'currency_strategy': self.clone(CurrencyStrategyTask)
        }

    def output(self):
        return {
            'raw': self.local_target(FilenameEnum.IDENTIFY_RISKS_RAW),
            'markdown': self.local_target(FilenameEnum.IDENTIFY_RISKS_MARKDOWN)
        }

    def run_with_llm(self, llm: LLM) -> None:
        # Read inputs from required tasks.
        with self.input()['setup'].open("r") as f:
            plan_prompt = f.read()
        with self.input()['identify_purpose']['markdown'].open("r") as f:
            identify_purpose_markdown = f.read()
        with self.input()['plan_type']['markdown'].open("r") as f:
            plan_type_markdown = f.read()
        with self.input()['strategic_decisions_markdown']['markdown'].open("r") as f:
            strategic_decisions_markdown = f.read()
        with self.input()['scenarios_markdown']['markdown'].open("r") as f:
            scenarios_markdown = f.read()
        with self.input()['physical_locations']['markdown'].open("r") as f:
            physical_locations_markdown = f.read()
        with self.input()['currency_strategy']['markdown'].open("r") as f:
            currency_strategy_markdown = f.read()

        query = (
            f"File 'plan.txt':\n{plan_prompt}\n\n"
            f"File 'purpose.md':\n{identify_purpose_markdown}\n\n"
            f"File 'plan_type.md':\n{plan_type_markdown}\n\n"
            f"File 'strategic_decisions.md':\n{strategic_decisions_markdown}\n\n"
            f"File 'scenarios.md':\n{scenarios_markdown}\n\n"
            f"File 'physical_locations.md':\n{physical_locations_markdown}\n\n"
            f"File 'currency_strategy.md':\n{currency_strategy_markdown}"
        )

        identify_risks = IdentifyRisks.execute(llm, query)

        # Write the result to disk.
        output_raw_path = self.output()['raw'].path
        identify_risks.save_raw(str(output_raw_path))
        output_markdown_path = self.output()['markdown'].path
        identify_risks.save_markdown(str(output_markdown_path))


class MakeAssumptionsTask(PlanTask):
    """
    Make assumptions about the plan.
    """
    def requires(self):
        return {
            'setup': self.clone(SetupTask),
            'identify_purpose': self.clone(IdentifyPurposeTask),
            'plan_type': self.clone(PlanTypeTask),
            'strategic_decisions_markdown': self.clone(StrategicDecisionsMarkdownTask),
            'scenarios_markdown': self.clone(ScenariosMarkdownTask),
            'physical_locations': self.clone(PhysicalLocationsTask),
            'currency_strategy': self.clone(CurrencyStrategyTask),
            'identify_risks': self.clone(IdentifyRisksTask)
        }

    def output(self):
        return {
            'raw': self.local_target(FilenameEnum.MAKE_ASSUMPTIONS_RAW),
            'clean': self.local_target(FilenameEnum.MAKE_ASSUMPTIONS_CLEAN),
            'markdown': self.local_target(FilenameEnum.MAKE_ASSUMPTIONS_MARKDOWN)
        }

    def run_with_llm(self, llm: LLM) -> None:
        # Read inputs from required tasks.
        with self.input()['setup'].open("r") as f:
            plan_prompt = f.read()
        with self.input()['identify_purpose']['markdown'].open("r") as f:
            identify_purpose_markdown = f.read()
        with self.input()['plan_type']['markdown'].open("r") as f:
            plan_type_markdown = f.read()
        with self.input()['strategic_decisions_markdown']['markdown'].open("r") as f:
            strategic_decisions_markdown = f.read()
        with self.input()['scenarios_markdown']['markdown'].open("r") as f:
            scenarios_markdown = f.read()
        with self.input()['physical_locations']['markdown'].open("r") as f:
            physical_locations_markdown = f.read()
        with self.input()['currency_strategy']['markdown'].open("r") as f:
            currency_strategy_markdown = f.read()
        with self.input()['identify_risks']['markdown'].open("r") as f:
            identify_risks_markdown = f.read()

        query = (
            f"File 'plan.txt':\n{plan_prompt}\n\n"
            f"File 'purpose.md':\n{identify_purpose_markdown}\n\n"
            f"File 'plan_type.md':\n{plan_type_markdown}\n\n"
            f"File 'strategic_decisions.md':\n{strategic_decisions_markdown}\n\n"
            f"File 'scenarios.md':\n{scenarios_markdown}\n\n"
            f"File 'physical_locations.md':\n{physical_locations_markdown}\n\n"
            f"File 'currency_strategy.md':\n{currency_strategy_markdown}\n\n"
            f"File 'identify_risks.md':\n{identify_risks_markdown}"
        )

        make_assumptions = MakeAssumptions.execute(llm, query)

        # Write the result to disk.
        output_raw_path = self.output()['raw'].path
        make_assumptions.save_raw(str(output_raw_path))
        output_clean_path = self.output()['clean'].path
        make_assumptions.save_assumptions(str(output_clean_path))
        output_markdown_path = self.output()['markdown'].path
        make_assumptions.save_markdown(str(output_markdown_path))


class DistillAssumptionsTask(PlanTask):
    """
    Distill raw assumption data.
    """
    def requires(self):
        return {
            'setup': self.clone(SetupTask),
            'identify_purpose': self.clone(IdentifyPurposeTask),
            'strategic_decisions_markdown': self.clone(StrategicDecisionsMarkdownTask),
            'scenarios_markdown': self.clone(ScenariosMarkdownTask),
            'make_assumptions': self.clone(MakeAssumptionsTask)
        }

    def output(self):
        return {
            'raw': self.local_target(FilenameEnum.DISTILL_ASSUMPTIONS_RAW),
            'markdown': self.local_target(FilenameEnum.DISTILL_ASSUMPTIONS_MARKDOWN)
        }

    def run_with_llm(self, llm: LLM) -> None:
        # Read inputs from required tasks.
        with self.input()['setup'].open("r") as f:
            plan_prompt = f.read()
        with self.input()['identify_purpose']['markdown'].open("r") as f:
            identify_purpose_markdown = f.read()
        with self.input()['strategic_decisions_markdown']['markdown'].open("r") as f:
            strategic_decisions_markdown = f.read()
        with self.input()['scenarios_markdown']['markdown'].open("r") as f:
            scenarios_markdown = f.read()
        make_assumptions_target = self.input()['make_assumptions']['clean']
        with make_assumptions_target.open("r") as f:
            assumptions_raw_data = json.load(f)

        query = (
            f"File 'plan.txt':\n{plan_prompt}\n\n"
            f"File 'purpose.md':\n{identify_purpose_markdown}\n\n"
            f"File 'strategic_decisions.md':\n{strategic_decisions_markdown}\n\n"
            f"File 'scenarios.md':\n{scenarios_markdown}\n\n"
            f"File 'assumptions.json':\n{format_json_for_use_in_query(assumptions_raw_data)}"
        )

        distill_assumptions = DistillAssumptions.execute(llm, query)

        # Write the result to disk.
        output_raw_path = self.output()['raw'].path
        distill_assumptions.save_raw(str(output_raw_path))
        output_markdown_path = self.output()['markdown'].path
        distill_assumptions.save_markdown(str(output_markdown_path))


class ReviewAssumptionsTask(PlanTask):
    """
    Find issues with the assumptions.
    """
    def requires(self):
        return {
            'identify_purpose': self.clone(IdentifyPurposeTask),
            'plan_type': self.clone(PlanTypeTask),
            'strategic_decisions_markdown': self.clone(StrategicDecisionsMarkdownTask),
            'scenarios_markdown': self.clone(ScenariosMarkdownTask),
            'physical_locations': self.clone(PhysicalLocationsTask),
            'currency_strategy': self.clone(CurrencyStrategyTask),
            'identify_risks': self.clone(IdentifyRisksTask),
            'make_assumptions': self.clone(MakeAssumptionsTask),
            'distill_assumptions': self.clone(DistillAssumptionsTask)
        }

    def output(self):
        return {
            'raw': self.local_target(FilenameEnum.REVIEW_ASSUMPTIONS_RAW),
            'markdown': self.local_target(FilenameEnum.REVIEW_ASSUMPTIONS_MARKDOWN)
        }

    def run_with_llm(self, llm: LLM) -> None:
        # Define the list of (title, path) tuples
        title_path_list = [
            ('Purpose', self.input()['identify_purpose']['markdown'].path),
            ('Plan Type', self.input()['plan_type']['markdown'].path),
            ('Strategic Decisions', self.input()['strategic_decisions_markdown']['markdown'].path),
            ('Scenarios', self.input()['scenarios_markdown']['markdown'].path),
            ('Physical Locations', self.input()['physical_locations']['markdown'].path),
            ('Currency Strategy', self.input()['currency_strategy']['markdown'].path),
            ('Identify Risks', self.input()['identify_risks']['markdown'].path),
            ('Make Assumptions', self.input()['make_assumptions']['markdown'].path),
            ('Distill Assumptions', self.input()['distill_assumptions']['markdown'].path)
        ]

        # Read the files and handle exceptions
        markdown_chunks = []
        for title, path in title_path_list:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    markdown_chunk = f.read()
                markdown_chunks.append(f"# {title}\n\n{markdown_chunk}")
            except FileNotFoundError:
                logger.warning(f"Markdown file not found: {path} (from {title})")
                markdown_chunks.append(f"**Problem with document:** '{title}'\n\nFile not found.")
            except Exception as e:
                logger.error(f"Error reading markdown file {path} (from {title}): {e}")
                markdown_chunks.append(f"**Problem with document:** '{title}'\n\nError reading markdown file.")

        # Combine the markdown chunks
        full_markdown = "\n\n".join(markdown_chunks)

        review_assumptions = ReviewAssumptions.execute(llm, full_markdown)

        # Write the result to disk.
        output_raw_path = self.output()['raw'].path
        review_assumptions.save_raw(str(output_raw_path))
        output_markdown_path = self.output()['markdown'].path
        review_assumptions.save_markdown(str(output_markdown_path))


class ConsolidateAssumptionsMarkdownTask(PlanTask):
    """
    Combines multiple small markdown documents into a single big document.
    """
    def requires(self):
        return {
            'identify_purpose': self.clone(IdentifyPurposeTask),
            'plan_type': self.clone(PlanTypeTask),
            'physical_locations': self.clone(PhysicalLocationsTask),
            'currency_strategy': self.clone(CurrencyStrategyTask),
            'identify_risks': self.clone(IdentifyRisksTask),
            'make_assumptions': self.clone(MakeAssumptionsTask),
            'distill_assumptions': self.clone(DistillAssumptionsTask),
            'review_assumptions': self.clone(ReviewAssumptionsTask)
        }

    def output(self):
        return {
            'full': self.local_target(FilenameEnum.CONSOLIDATE_ASSUMPTIONS_FULL_MARKDOWN),
            'short': self.local_target(FilenameEnum.CONSOLIDATE_ASSUMPTIONS_SHORT_MARKDOWN)
        }

    def run_inner(self):
        llm_executor: LLMExecutor = self.create_llm_executor()

        # Define the list of (title, path) tuples
        title_path_list = [
            ('Purpose', self.input()['identify_purpose']['markdown'].path),
            ('Plan Type', self.input()['plan_type']['markdown'].path),
            ('Physical Locations', self.input()['physical_locations']['markdown'].path),
            ('Currency Strategy', self.input()['currency_strategy']['markdown'].path),
            ('Identify Risks', self.input()['identify_risks']['markdown'].path),
            ('Make Assumptions', self.input()['make_assumptions']['markdown'].path),
            ('Distill Assumptions', self.input()['distill_assumptions']['markdown'].path),
            ('Review Assumptions', self.input()['review_assumptions']['markdown'].path)
        ]

        # Read the files and handle exceptions
        full_markdown_chunks = []
        short_markdown_chunks = []
        for title, path in title_path_list:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    markdown_chunk = f.read()
                full_markdown_chunks.append(f"# {title}\n\n{markdown_chunk}")
            except FileNotFoundError:
                logger.warning(f"Markdown file not found: {path} (from {title})")
                full_markdown_chunks.append(f"**Problem with document:** '{title}'\n\nFile not found.")
                short_markdown_chunks.append(f"**Problem with document:** '{title}'\n\nFile not found.")
                continue
            except Exception as e:
                logger.error(f"Error reading markdown file {path} (from {title}): {e}")
                full_markdown_chunks.append(f"**Problem with document:** '{title}'\n\nError reading markdown file.")
                short_markdown_chunks.append(f"**Problem with document:** '{title}'\n\nError reading markdown file.")
                continue

            # IDEA: If the chunk file already exist, then there is no need to run the LLM again.
            def execute_shorten_markdown(llm: LLM) -> ShortenMarkdown:
                return ShortenMarkdown.execute(llm, markdown_chunk)

            try:
                shorten_markdown = llm_executor.run(execute_shorten_markdown)
                short_markdown_chunks.append(f"# {title}\n{shorten_markdown.markdown}")
            except PipelineStopRequested:
                # Re-raise PipelineStopRequested without wrapping it
                raise
            except Exception as e:
                logger.error(f"Error shortening markdown file {path} (from {title}): {e}")
                short_markdown_chunks.append(f"**Problem with document:** '{title}'\n\nError shortening markdown file.")
                continue
            
        # Combine the markdown chunks
        full_markdown = "\n\n".join(full_markdown_chunks)
        short_markdown = "\n\n".join(short_markdown_chunks)

        # Write the result to disk.
        output_full_markdown_path = self.output()['full'].path
        with open(output_full_markdown_path, "w", encoding="utf-8") as f:
            f.write(full_markdown)

        output_short_markdown_path = self.output()['short'].path
        with open(output_short_markdown_path, "w", encoding="utf-8") as f:
            f.write(short_markdown)


class PreProjectAssessmentTask(PlanTask):
    def requires(self):
        return {
            'setup': self.clone(SetupTask),
            'strategic_decisions_markdown': self.clone(StrategicDecisionsMarkdownTask),
            'scenarios_markdown': self.clone(ScenariosMarkdownTask),
            'consolidate_assumptions_markdown': self.clone(ConsolidateAssumptionsMarkdownTask)
        }

    def output(self):
        return {
            'raw': self.local_target(FilenameEnum.PRE_PROJECT_ASSESSMENT_RAW),
            'clean': self.local_target(FilenameEnum.PRE_PROJECT_ASSESSMENT)
        }

    def run_with_llm(self, llm: LLM) -> None:
        logger.info("Conducting pre-project assessment...")

        # Read the plan prompt from the SetupTask's output.
        with self.input()['setup'].open("r") as f:
            plan_prompt = f.read()

        with self.input()['strategic_decisions_markdown']['markdown'].open("r") as f:
            strategic_decisions_markdown = f.read()
        with self.input()['scenarios_markdown']['markdown'].open("r") as f:
            scenarios_markdown = f.read()

        with self.input()['consolidate_assumptions_markdown']['short'].open("r") as f:
            consolidate_assumptions_markdown = f.read()

        # Build the query.
        query = (
            f"File 'plan.txt':\n{plan_prompt}\n\n"
            f"File 'strategic_decisions.md':\n{strategic_decisions_markdown}\n\n"
            f"File 'scenarios.md':\n{scenarios_markdown}\n\n"
            f"File 'assumptions.md':\n{consolidate_assumptions_markdown}"
        )

        # Execute the pre-project assessment.
        pre_project_assessment = PreProjectAssessment.execute(llm, query)

        # Save raw output.
        raw_path = self.file_path(FilenameEnum.PRE_PROJECT_ASSESSMENT_RAW)
        pre_project_assessment.save_raw(str(raw_path))

        # Save cleaned pre-project assessment.
        clean_path = self.file_path(FilenameEnum.PRE_PROJECT_ASSESSMENT)
        pre_project_assessment.save_preproject_assessment(str(clean_path))


class ProjectPlanTask(PlanTask):
    def requires(self):
        return {
            'setup': self.clone(SetupTask),
            'strategic_decisions_markdown': self.clone(StrategicDecisionsMarkdownTask),
            'scenarios_markdown': self.clone(ScenariosMarkdownTask),
            'consolidate_assumptions_markdown': self.clone(ConsolidateAssumptionsMarkdownTask),
            'preproject': self.clone(PreProjectAssessmentTask)
        }

    def output(self):
        return {
            'raw': self.local_target(FilenameEnum.PROJECT_PLAN_RAW),
            'markdown': self.local_target(FilenameEnum.PROJECT_PLAN_MARKDOWN)
        }

    def run_with_llm(self, llm: LLM) -> None:
        logger.info("Creating plan...")

        # Read the plan prompt from SetupTask's output.
        setup_target = self.input()['setup']
        with setup_target.open("r") as f:
            plan_prompt = f.read()

        with self.input()['strategic_decisions_markdown']['markdown'].open("r") as f:
            strategic_decisions_markdown = f.read()
        with self.input()['scenarios_markdown']['markdown'].open("r") as f:
            scenarios_markdown = f.read()

        # Load the consolidated assumptions.
        with self.input()['consolidate_assumptions_markdown']['short'].open("r") as f:
            consolidate_assumptions_markdown = f.read()

        # Read the pre-project assessment from its file.
        pre_project_assessment_file = self.input()['preproject']['clean']
        with pre_project_assessment_file.open("r") as f:
            pre_project_assessment_dict = json.load(f)

        # Build the query.
        query = (
            f"File 'plan.txt':\n{plan_prompt}\n\n"
            f"File 'strategic_decisions.md':\n{strategic_decisions_markdown}\n\n"
            f"File 'scenarios.md':\n{scenarios_markdown}\n\n"
            f"File 'assumptions.md':\n{consolidate_assumptions_markdown}\n\n"
            f"File 'pre-project-assessment.json':\n{format_json_for_use_in_query(pre_project_assessment_dict)}"
        )

        # Execute the plan creation.
        project_plan = ProjectPlan.execute(llm, query)
        
        # Save raw output
        project_plan.save_raw(self.output()['raw'].path)
        
        # Save markdown output
        project_plan.save_markdown(self.output()['markdown'].path)

        logger.info("Project plan created and saved")


class GovernancePhase1AuditTask(PlanTask):
    def requires(self):
        return {
            'setup': self.clone(SetupTask),
            'strategic_decisions_markdown': self.clone(StrategicDecisionsMarkdownTask),
            'scenarios_markdown': self.clone(ScenariosMarkdownTask),
            'consolidate_assumptions_markdown': self.clone(ConsolidateAssumptionsMarkdownTask),
            'project_plan': self.clone(ProjectPlanTask)
        }

    def output(self):
        return {
            'raw': self.local_target(FilenameEnum.GOVERNANCE_PHASE1_AUDIT_RAW),
            'markdown': self.local_target(FilenameEnum.GOVERNANCE_PHASE1_AUDIT_MARKDOWN)
        }

    def run_with_llm(self, llm: LLM) -> None:
        # Read inputs from required tasks.
        with self.input()['setup'].open("r") as f:
            plan_prompt = f.read()
        with self.input()['strategic_decisions_markdown']['markdown'].open("r") as f:
            strategic_decisions_markdown = f.read()
        with self.input()['scenarios_markdown']['markdown'].open("r") as f:
            scenarios_markdown = f.read()
        with self.input()['consolidate_assumptions_markdown']['short'].open("r") as f:
            consolidate_assumptions_markdown = f.read()
        with self.input()['project_plan']['markdown'].open("r") as f:
            project_plan_markdown = f.read()

        # Build the query.
        query = (
            f"File 'initial-plan.txt':\n{plan_prompt}\n\n"
            f"File 'strategic_decisions.md':\n{strategic_decisions_markdown}\n\n"
            f"File 'scenarios.md':\n{scenarios_markdown}\n\n"
            f"File 'assumptions.md':\n{consolidate_assumptions_markdown}\n\n"
            f"File 'project-plan.md':\n{project_plan_markdown}"
        )

        # Execute.
        try:
            governance_phase1_audit = GovernancePhase1Audit.execute(llm, query)
        except Exception as e:
            logger.error("GovernancePhase1Audit failed: %s", e)
            raise

        # Save the results.
        governance_phase1_audit.save_raw(self.output()['raw'].path)
        governance_phase1_audit.save_markdown(self.output()['markdown'].path)


class GovernancePhase2BodiesTask(PlanTask):
    def requires(self):
        return {
            'setup': self.clone(SetupTask),
            'strategic_decisions_markdown': self.clone(StrategicDecisionsMarkdownTask),
            'scenarios_markdown': self.clone(ScenariosMarkdownTask),
            'consolidate_assumptions_markdown': self.clone(ConsolidateAssumptionsMarkdownTask),
            'project_plan': self.clone(ProjectPlanTask),
            'governance_phase1_audit': self.clone(GovernancePhase1AuditTask)
        }

    def output(self):
        return {
            'raw': self.local_target(FilenameEnum.GOVERNANCE_PHASE2_BODIES_RAW),
            'markdown': self.local_target(FilenameEnum.GOVERNANCE_PHASE2_BODIES_MARKDOWN)
        }

    def run_with_llm(self, llm: LLM) -> None:
        # Read inputs from required tasks.
        with self.input()['setup'].open("r") as f:
            plan_prompt = f.read()
        with self.input()['strategic_decisions_markdown']['markdown'].open("r") as f:
            strategic_decisions_markdown = f.read()
        with self.input()['scenarios_markdown']['markdown'].open("r") as f:
            scenarios_markdown = f.read()
        with self.input()['consolidate_assumptions_markdown']['short'].open("r") as f:
            consolidate_assumptions_markdown = f.read()
        with self.input()['project_plan']['markdown'].open("r") as f:
            project_plan_markdown = f.read()
        with self.input()['governance_phase1_audit']['markdown'].open("r") as f:
            governance_phase1_audit_markdown = f.read()

        # Build the query.
        query = (
            f"File 'initial-plan.txt':\n{plan_prompt}\n\n"
            f"File 'strategic_decisions.md':\n{strategic_decisions_markdown}\n\n"
            f"File 'scenarios.md':\n{scenarios_markdown}\n\n"
            f"File 'assumptions.md':\n{consolidate_assumptions_markdown}\n\n"
            f"File 'project-plan.md':\n{project_plan_markdown}\n\n"
            f"File 'governance-phase1-audit.md':\n{governance_phase1_audit_markdown}"
        )

        # Execute.
        try:
            governance_phase2_bodies = GovernancePhase2Bodies.execute(llm, query)
        except Exception as e:
            logger.error("GovernancePhase2Bodies failed: %s", e)
            raise

        # Save the results.
        governance_phase2_bodies.save_raw(self.output()['raw'].path)
        governance_phase2_bodies.save_markdown(self.output()['markdown'].path)


class GovernancePhase3ImplPlanTask(PlanTask):
    def requires(self):
        return {
            'setup': self.clone(SetupTask),
            'strategic_decisions_markdown': self.clone(StrategicDecisionsMarkdownTask),
            'scenarios_markdown': self.clone(ScenariosMarkdownTask),
            'consolidate_assumptions_markdown': self.clone(ConsolidateAssumptionsMarkdownTask),
            'project_plan': self.clone(ProjectPlanTask),
            'governance_phase2_bodies': self.clone(GovernancePhase2BodiesTask)
        }

    def output(self):
        return {
            'raw': self.local_target(FilenameEnum.GOVERNANCE_PHASE3_IMPL_PLAN_RAW),
            'markdown': self.local_target(FilenameEnum.GOVERNANCE_PHASE3_IMPL_PLAN_MARKDOWN)
        }

    def run_with_llm(self, llm: LLM) -> None:
        # Read inputs from required tasks.  
        with self.input()['setup'].open("r") as f:
            plan_prompt = f.read()
        with self.input()['strategic_decisions_markdown']['markdown'].open("r") as f:
            strategic_decisions_markdown = f.read()
        with self.input()['scenarios_markdown']['markdown'].open("r") as f:
            scenarios_markdown = f.read()
        with self.input()['consolidate_assumptions_markdown']['short'].open("r") as f:
            consolidate_assumptions_markdown = f.read()
        with self.input()['project_plan']['raw'].open("r") as f:
            project_plan_dict = json.load(f)
        with self.input()['governance_phase2_bodies']['raw'].open("r") as f:
            governance_phase2_bodies_dict = json.load(f)

        # Build the query.
        query = (
            f"File 'initial-plan.txt':\n{plan_prompt}\n\n"
            f"File 'strategic_decisions.md':\n{strategic_decisions_markdown}\n\n"
            f"File 'scenarios.md':\n{scenarios_markdown}\n\n"
            f"File 'assumptions.md':\n{consolidate_assumptions_markdown}\n\n"
            f"File 'project-plan.json':\n{format_json_for_use_in_query(project_plan_dict)}\n\n"
            f"File 'governance-phase2-bodies.json':\n{format_json_for_use_in_query(governance_phase2_bodies_dict)}"
        )

        # Execute.
        try:
            governance_phase3_impl_plan = GovernancePhase3ImplPlan.execute(llm, query)
        except Exception as e:
            logger.error("GovernancePhase3ImplPlan failed: %s", e)
            raise

        # Save the results.
        governance_phase3_impl_plan.save_raw(self.output()['raw'].path)
        governance_phase3_impl_plan.save_markdown(self.output()['markdown'].path)

class GovernancePhase4DecisionEscalationMatrixTask(PlanTask):
    def requires(self):
        return {
            'setup': self.clone(SetupTask),
            'strategic_decisions_markdown': self.clone(StrategicDecisionsMarkdownTask),
            'scenarios_markdown': self.clone(ScenariosMarkdownTask),
            'consolidate_assumptions_markdown': self.clone(ConsolidateAssumptionsMarkdownTask),
            'project_plan': self.clone(ProjectPlanTask),
            'governance_phase2_bodies': self.clone(GovernancePhase2BodiesTask),
            'governance_phase3_impl_plan': self.clone(GovernancePhase3ImplPlanTask)
        }

    def output(self):
        return {
            'raw': self.local_target(FilenameEnum.GOVERNANCE_PHASE4_DECISION_ESCALATION_MATRIX_RAW),
            'markdown': self.local_target(FilenameEnum.GOVERNANCE_PHASE4_DECISION_ESCALATION_MATRIX_MARKDOWN)
        }

    def run_with_llm(self, llm: LLM) -> None:
        # Read inputs from required tasks.  
        with self.input()['setup'].open("r") as f:
            plan_prompt = f.read()
        with self.input()['strategic_decisions_markdown']['markdown'].open("r") as f:
            strategic_decisions_markdown = f.read()
        with self.input()['scenarios_markdown']['markdown'].open("r") as f:
            scenarios_markdown = f.read()
        with self.input()['consolidate_assumptions_markdown']['short'].open("r") as f:
            consolidate_assumptions_markdown = f.read()
        with self.input()['project_plan']['raw'].open("r") as f:
            project_plan_dict = json.load(f)
        with self.input()['governance_phase2_bodies']['raw'].open("r") as f:
            governance_phase2_bodies_dict = json.load(f)
        with self.input()['governance_phase3_impl_plan']['raw'].open("r") as f:
            governance_phase3_impl_plan_dict = json.load(f)

        # Build the query.
        query = (
            f"File 'initial-plan.txt':\n{plan_prompt}\n\n"
            f"File 'strategic_decisions.md':\n{strategic_decisions_markdown}\n\n"
            f"File 'scenarios.md':\n{scenarios_markdown}\n\n"
            f"File 'assumptions.md':\n{consolidate_assumptions_markdown}\n\n"
            f"File 'project-plan.json':\n{format_json_for_use_in_query(project_plan_dict)}\n\n"
            f"File 'governance-phase2-bodies.json':\n{format_json_for_use_in_query(governance_phase2_bodies_dict)}\n\n"
            f"File 'governance-phase3-impl-plan.json':\n{format_json_for_use_in_query(governance_phase3_impl_plan_dict)}"
        )

        # Execute.
        try:
            governance_phase4_decision_escalation_matrix = GovernancePhase4DecisionEscalationMatrix.execute(llm, query)
        except Exception as e:
            logger.error("GovernancePhase4DecisionEscalationMatrix failed: %s", e)
            raise

        # Save the results.
        governance_phase4_decision_escalation_matrix.save_raw(self.output()['raw'].path)
        governance_phase4_decision_escalation_matrix.save_markdown(self.output()['markdown'].path)

class GovernancePhase5MonitoringProgressTask(PlanTask):
    def requires(self):
        return {
            'setup': self.clone(SetupTask),
            'strategic_decisions_markdown': self.clone(StrategicDecisionsMarkdownTask),
            'scenarios_markdown': self.clone(ScenariosMarkdownTask),
            'consolidate_assumptions_markdown': self.clone(ConsolidateAssumptionsMarkdownTask),
            'project_plan': self.clone(ProjectPlanTask),
            'governance_phase2_bodies': self.clone(GovernancePhase2BodiesTask),
            'governance_phase3_impl_plan': self.clone(GovernancePhase3ImplPlanTask),
            'governance_phase4_decision_escalation_matrix': self.clone(GovernancePhase4DecisionEscalationMatrixTask)
        }

    def output(self):
        return {
            'raw': self.local_target(FilenameEnum.GOVERNANCE_PHASE5_MONITORING_PROGRESS_RAW),
            'markdown': self.local_target(FilenameEnum.GOVERNANCE_PHASE5_MONITORING_PROGRESS_MARKDOWN)
        }

    def run_with_llm(self, llm: LLM) -> None:
        # Read inputs from required tasks.  
        with self.input()['setup'].open("r") as f:
            plan_prompt = f.read()
        with self.input()['strategic_decisions_markdown']['markdown'].open("r") as f:
            strategic_decisions_markdown = f.read()
        with self.input()['scenarios_markdown']['markdown'].open("r") as f:
            scenarios_markdown = f.read()
        with self.input()['consolidate_assumptions_markdown']['short'].open("r") as f:
            consolidate_assumptions_markdown = f.read()
        with self.input()['project_plan']['raw'].open("r") as f:
            project_plan_dict = json.load(f)
        with self.input()['governance_phase2_bodies']['raw'].open("r") as f:
            governance_phase2_bodies_dict = json.load(f)
        with self.input()['governance_phase3_impl_plan']['raw'].open("r") as f:
            governance_phase3_impl_plan_dict = json.load(f)
        with self.input()['governance_phase4_decision_escalation_matrix']['raw'].open("r") as f:
            governance_phase4_decision_escalation_matrix_dict = json.load(f)
        
        # Build the query.
        query = (
            f"File 'initial-plan.txt':\n{plan_prompt}\n\n"
            f"File 'strategic_decisions.md':\n{strategic_decisions_markdown}\n\n"
            f"File 'scenarios.md':\n{scenarios_markdown}\n\n"
            f"File 'assumptions.md':\n{consolidate_assumptions_markdown}\n\n"
            f"File 'project-plan.json':\n{format_json_for_use_in_query(project_plan_dict)}\n\n"
            f"File 'governance-phase2-bodies.json':\n{format_json_for_use_in_query(governance_phase2_bodies_dict)}\n\n"
            f"File 'governance-phase3-impl-plan.json':\n{format_json_for_use_in_query(governance_phase3_impl_plan_dict)}\n\n"
            f"File 'governance-phase4-decision-escalation-matrix.json':\n{format_json_for_use_in_query(governance_phase4_decision_escalation_matrix_dict)}"
        )

        # Execute.
        try:
            governance_phase5_monitoring_progress = GovernancePhase5MonitoringProgress.execute(llm, query)
        except Exception as e:
            logger.error("GovernancePhase5MonitoringProgress failed: %s", e)
            raise

        # Save the results.
        governance_phase5_monitoring_progress.save_raw(self.output()['raw'].path)
        governance_phase5_monitoring_progress.save_markdown(self.output()['markdown'].path)

class GovernancePhase6ExtraTask(PlanTask):
    def requires(self):
        return {
            'setup': self.clone(SetupTask),
            'strategic_decisions_markdown': self.clone(StrategicDecisionsMarkdownTask),
            'scenarios_markdown': self.clone(ScenariosMarkdownTask),
            'consolidate_assumptions_markdown': self.clone(ConsolidateAssumptionsMarkdownTask),
            'project_plan': self.clone(ProjectPlanTask),
            'governance_phase1_audit': self.clone(GovernancePhase1AuditTask),
            'governance_phase2_bodies': self.clone(GovernancePhase2BodiesTask),
            'governance_phase3_impl_plan': self.clone(GovernancePhase3ImplPlanTask),
            'governance_phase4_decision_escalation_matrix': self.clone(GovernancePhase4DecisionEscalationMatrixTask),
            'governance_phase5_monitoring_progress': self.clone(GovernancePhase5MonitoringProgressTask)
        }

    def output(self):
        return {
            'raw': self.local_target(FilenameEnum.GOVERNANCE_PHASE6_EXTRA_RAW),
            'markdown': self.local_target(FilenameEnum.GOVERNANCE_PHASE6_EXTRA_MARKDOWN)
        }

    def run_with_llm(self, llm: LLM) -> None:
        # Read inputs from required tasks.
        with self.input()['setup'].open("r") as f:
            plan_prompt = f.read()
        with self.input()['strategic_decisions_markdown']['markdown'].open("r") as f:
            strategic_decisions_markdown = f.read()
        with self.input()['scenarios_markdown']['markdown'].open("r") as f:
            scenarios_markdown = f.read()
        with self.input()['consolidate_assumptions_markdown']['short'].open("r") as f:
            consolidate_assumptions_markdown = f.read()
        with self.input()['project_plan']['raw'].open("r") as f:
            project_plan_dict = json.load(f)
        with self.input()['governance_phase1_audit']['raw'].open("r") as f:
            governance_phase1_audit_dict = json.load(f)
        with self.input()['governance_phase2_bodies']['raw'].open("r") as f:
            governance_phase2_bodies_dict = json.load(f)
        with self.input()['governance_phase3_impl_plan']['raw'].open("r") as f:
            governance_phase3_impl_plan_dict = json.load(f)
        with self.input()['governance_phase4_decision_escalation_matrix']['raw'].open("r") as f:
            governance_phase4_decision_escalation_matrix_dict = json.load(f)
        with self.input()['governance_phase5_monitoring_progress']['raw'].open("r") as f:
            governance_phase5_monitoring_progress_dict = json.load(f)

        # Build the query.
        query = (
            f"File 'initial-plan.txt':\n{plan_prompt}\n\n"
            f"File 'strategic_decisions.md':\n{strategic_decisions_markdown}\n\n"
            f"File 'scenarios.md':\n{scenarios_markdown}\n\n"
            f"File 'assumptions.md':\n{consolidate_assumptions_markdown}\n\n"
            f"File 'project-plan.json':\n{format_json_for_use_in_query(project_plan_dict)}\n\n"
            f"File 'governance-phase1-audit.json':\n{format_json_for_use_in_query(governance_phase1_audit_dict)}\n\n"
            f"File 'governance-phase2-bodies.json':\n{format_json_for_use_in_query(governance_phase2_bodies_dict)}\n\n"
            f"File 'governance-phase3-impl-plan.json':\n{format_json_for_use_in_query(governance_phase3_impl_plan_dict)}\n\n"
            f"File 'governance-phase4-decision-escalation-matrix.json':\n{format_json_for_use_in_query(governance_phase4_decision_escalation_matrix_dict)}\n\n"
            f"File 'governance-phase5-monitoring-progress.json':\n{format_json_for_use_in_query(governance_phase5_monitoring_progress_dict)}"
        )

        # Execute.
        try:
            governance_phase6_extra = GovernancePhase6Extra.execute(llm, query)
        except Exception as e:
            logger.error("GovernancePhase6Extra failed: %s", e)
            raise

        # Save the results.
        governance_phase6_extra.save_raw(self.output()['raw'].path)
        governance_phase6_extra.save_markdown(self.output()['markdown'].path)

class ConsolidateGovernanceTask(PlanTask):
    def requires(self):
        return {
            'governance_phase1_audit': self.clone(GovernancePhase1AuditTask),
            'governance_phase2_bodies': self.clone(GovernancePhase2BodiesTask),
            'governance_phase3_impl_plan': self.clone(GovernancePhase3ImplPlanTask),
            'governance_phase4_decision_escalation_matrix': self.clone(GovernancePhase4DecisionEscalationMatrixTask),
            'governance_phase5_monitoring_progress': self.clone(GovernancePhase5MonitoringProgressTask),
            'governance_phase6_extra': self.clone(GovernancePhase6ExtraTask)
        }

    def output(self):
        return self.local_target(FilenameEnum.CONSOLIDATE_GOVERNANCE_MARKDOWN)

    def run_inner(self):
        # Read inputs from required tasks.
        with self.input()['governance_phase1_audit']['markdown'].open("r") as f:
            governance_phase1_audit_markdown = f.read()
        with self.input()['governance_phase2_bodies']['markdown'].open("r") as f:
            governance_phase2_bodies_markdown = f.read()
        with self.input()['governance_phase3_impl_plan']['markdown'].open("r") as f:
            governance_phase3_impl_plan_markdown = f.read()
        with self.input()['governance_phase4_decision_escalation_matrix']['markdown'].open("r") as f:
            governance_phase4_decision_escalation_matrix_markdown = f.read()
        with self.input()['governance_phase5_monitoring_progress']['markdown'].open("r") as f:
            governance_phase5_monitoring_progress_markdown = f.read()
        with self.input()['governance_phase6_extra']['markdown'].open("r") as f:
            governance_phase6_extra_markdown = f.read()

        # Build the document.
        markdown = []
        markdown.append(f"# Governance Audit\n\n{governance_phase1_audit_markdown}")
        markdown.append(f"# Internal Governance Bodies\n\n{governance_phase2_bodies_markdown}")
        markdown.append(f"# Governance Implementation Plan\n\n{governance_phase3_impl_plan_markdown}")
        markdown.append(f"# Decision Escalation Matrix\n\n{governance_phase4_decision_escalation_matrix_markdown}")
        markdown.append(f"# Monitoring Progress\n\n{governance_phase5_monitoring_progress_markdown}")
        markdown.append(f"# Governance Extra\n\n{governance_phase6_extra_markdown}")

        content = "\n\n".join(markdown)

        with self.output().open("w") as f:
            f.write(content)

class RelatedResourcesTask(PlanTask):
    def requires(self):
        return {
            'setup': self.clone(SetupTask),
            'strategic_decisions_markdown': self.clone(StrategicDecisionsMarkdownTask),
            'scenarios_markdown': self.clone(ScenariosMarkdownTask),
            'consolidate_assumptions_markdown': self.clone(ConsolidateAssumptionsMarkdownTask),
            'project_plan': self.clone(ProjectPlanTask)
        }

    def output(self):
        return {
            'raw': self.local_target(FilenameEnum.RELATED_RESOURCES_RAW),
            'markdown': self.local_target(FilenameEnum.RELATED_RESOURCES_MARKDOWN)
        }

    def run_with_llm(self, llm: LLM) -> None:
        # Read inputs from required tasks.
        with self.input()['setup'].open("r") as f:
            plan_prompt = f.read()
        with self.input()['strategic_decisions_markdown']['markdown'].open("r") as f:
            strategic_decisions_markdown = f.read()
        with self.input()['scenarios_markdown']['markdown'].open("r") as f:
            scenarios_markdown = f.read()
        with self.input()['consolidate_assumptions_markdown']['short'].open("r") as f:
            consolidate_assumptions_markdown = f.read()
        with self.input()['project_plan']['raw'].open("r") as f:
            project_plan_dict = json.load(f)

        # Build the query.
        query = (
            f"File 'initial-plan.txt':\n{plan_prompt}\n\n"
            f"File 'strategic_decisions.md':\n{strategic_decisions_markdown}\n\n"
            f"File 'scenarios.md':\n{scenarios_markdown}\n\n"
            f"File 'assumptions.md':\n{consolidate_assumptions_markdown}\n\n"
            f"File 'project-plan.json':\n{format_json_for_use_in_query(project_plan_dict)}"
        )

        # Execute.
        try:
            related_resources = RelatedResources.execute(llm, query)
        except Exception as e:
            logger.error("SimilarProjects failed: %s", e)
            raise

        # Save the results.
        related_resources.save_raw(self.output()['raw'].path)
        related_resources.save_markdown(self.output()['markdown'].path)

class FindTeamMembersTask(PlanTask):
    def requires(self):
        return {
            'setup': self.clone(SetupTask),
            'strategic_decisions_markdown': self.clone(StrategicDecisionsMarkdownTask),
            'scenarios_markdown': self.clone(ScenariosMarkdownTask),
            'consolidate_assumptions_markdown': self.clone(ConsolidateAssumptionsMarkdownTask),
            'preproject': self.clone(PreProjectAssessmentTask),
            'project_plan': self.clone(ProjectPlanTask),
            'related_resources': self.clone(RelatedResourcesTask),
        }

    def output(self):
        return {
            'raw': self.local_target(FilenameEnum.FIND_TEAM_MEMBERS_RAW),
            'clean': self.local_target(FilenameEnum.FIND_TEAM_MEMBERS_CLEAN)
        }

    def run_with_llm(self, llm: LLM) -> None:
        # Read inputs from required tasks.
        with self.input()['setup'].open("r") as f:
            plan_prompt = f.read()
        with self.input()['strategic_decisions_markdown']['markdown'].open("r") as f:
            strategic_decisions_markdown = f.read()
        with self.input()['scenarios_markdown']['markdown'].open("r") as f:
            scenarios_markdown = f.read()
        with self.input()['consolidate_assumptions_markdown']['short'].open("r") as f:
            consolidate_assumptions_markdown = f.read()
        with self.input()['preproject']['clean'].open("r") as f:
            pre_project_assessment_dict = json.load(f)
        with self.input()['project_plan']['markdown'].open("r") as f:
            project_plan_markdown = f.read()
        with self.input()['related_resources']['markdown'].open("r") as f:
            related_resources_markdown = f.read()

        # Build the query.
        query = (
            f"File 'initial-plan.txt':\n{plan_prompt}\n\n"
            f"File 'strategic_decisions.md':\n{strategic_decisions_markdown}\n\n"
            f"File 'scenarios.md':\n{scenarios_markdown}\n\n"
            f"File 'assumptions.md':\n{consolidate_assumptions_markdown}\n\n"
            f"File 'pre-project-assessment.json':\n{format_json_for_use_in_query(pre_project_assessment_dict)}\n\n"
            f"File 'project-plan.md':\n{project_plan_markdown}\n\n"
            f"File 'related-resources.md':\n{related_resources_markdown}"
        )

        # Execute.
        try:
            find_team_members = FindTeamMembers.execute(llm, query)
        except Exception as e:
            logger.error("FindTeamMembers failed: %s", e)
            raise

        # Save the raw output.
        raw_dict = find_team_members.to_dict()
        with self.output()['raw'].open("w") as f:
            json.dump(raw_dict, f, indent=2)

        # Save the cleaned up result.
        team_member_list = find_team_members.team_member_list
        with self.output()['clean'].open("w") as f:
            json.dump(team_member_list, f, indent=2)

class EnrichTeamMembersWithContractTypeTask(PlanTask):
    def requires(self):
        return {
            'setup': self.clone(SetupTask),
            'strategic_decisions_markdown': self.clone(StrategicDecisionsMarkdownTask),
            'scenarios_markdown': self.clone(ScenariosMarkdownTask),
            'consolidate_assumptions_markdown': self.clone(ConsolidateAssumptionsMarkdownTask),
            'preproject': self.clone(PreProjectAssessmentTask),
            'project_plan': self.clone(ProjectPlanTask),
            'find_team_members': self.clone(FindTeamMembersTask),
            'related_resources': self.clone(RelatedResourcesTask)
        }

    def output(self):
        return {
            'raw': self.local_target(FilenameEnum.ENRICH_TEAM_MEMBERS_CONTRACT_TYPE_RAW),
            'clean': self.local_target(FilenameEnum.ENRICH_TEAM_MEMBERS_CONTRACT_TYPE_CLEAN)
        }

    def run_with_llm(self, llm: LLM) -> None:
        # Read inputs from required tasks.
        with self.input()['setup'].open("r") as f:
            plan_prompt = f.read()
        with self.input()['strategic_decisions_markdown']['markdown'].open("r") as f:
            strategic_decisions_markdown = f.read()
        with self.input()['scenarios_markdown']['markdown'].open("r") as f:
            scenarios_markdown = f.read()
        with self.input()['consolidate_assumptions_markdown']['short'].open("r") as f:
            consolidate_assumptions_markdown = f.read()
        with self.input()['preproject']['clean'].open("r") as f:
            pre_project_assessment_dict = json.load(f)
        with self.input()['project_plan']['markdown'].open("r") as f:
            project_plan_markdown = f.read()
        with self.input()['find_team_members']['clean'].open("r") as f:
            team_member_list = json.load(f)
        with self.input()['related_resources']['markdown'].open("r") as f:
            related_resources_markdown = f.read()

        # Build the query.
        query = (
            f"File 'initial-plan.txt':\n{plan_prompt}\n\n"
            f"File 'strategic_decisions.md':\n{strategic_decisions_markdown}\n\n"
            f"File 'scenarios.md':\n{scenarios_markdown}\n\n"
            f"File 'assumptions.md':\n{consolidate_assumptions_markdown}\n\n"
            f"File 'pre-project-assessment.json':\n{format_json_for_use_in_query(pre_project_assessment_dict)}\n\n"
            f"File 'project-plan.md':\n{project_plan_markdown}\n\n"
            f"File 'team-members-that-needs-to-be-enriched.json':\n{format_json_for_use_in_query(team_member_list)}\n\n"
            f"File 'related-resources.md':\n{related_resources_markdown}"
        )

        # Execute.
        try:
            enrich_team_members_with_contract_type = EnrichTeamMembersWithContractType.execute(llm, query, team_member_list)
        except Exception as e:
            logger.error("EnrichTeamMembersWithContractType failed: %s", e)
            raise

        # Save the raw output.
        raw_dict = enrich_team_members_with_contract_type.to_dict()
        with self.output()['raw'].open("w") as f:
            json.dump(raw_dict, f, indent=2)

        # Save the cleaned up result.
        team_member_list = enrich_team_members_with_contract_type.team_member_list
        with self.output()['clean'].open("w") as f:
            json.dump(team_member_list, f, indent=2)

class EnrichTeamMembersWithBackgroundStoryTask(PlanTask):
    def requires(self):
        return {
            'setup': self.clone(SetupTask),
            'strategic_decisions_markdown': self.clone(StrategicDecisionsMarkdownTask),
            'scenarios_markdown': self.clone(ScenariosMarkdownTask),
            'consolidate_assumptions_markdown': self.clone(ConsolidateAssumptionsMarkdownTask),
            'preproject': self.clone(PreProjectAssessmentTask),
            'project_plan': self.clone(ProjectPlanTask),
            'enrich_team_members_with_contract_type': self.clone(EnrichTeamMembersWithContractTypeTask),
            'related_resources': self.clone(RelatedResourcesTask)
        }

    def output(self):
        return {
            'raw': self.local_target(FilenameEnum.ENRICH_TEAM_MEMBERS_BACKGROUND_STORY_RAW),
            'clean': self.local_target(FilenameEnum.ENRICH_TEAM_MEMBERS_BACKGROUND_STORY_CLEAN)
        }

    def run_with_llm(self, llm: LLM) -> None:
        # Read inputs from required tasks.
        with self.input()['setup'].open("r") as f:
            plan_prompt = f.read()
        with self.input()['strategic_decisions_markdown']['markdown'].open("r") as f:
            strategic_decisions_markdown = f.read()
        with self.input()['scenarios_markdown']['markdown'].open("r") as f:
            scenarios_markdown = f.read()
        with self.input()['consolidate_assumptions_markdown']['short'].open("r") as f:
            consolidate_assumptions_markdown = f.read()
        with self.input()['preproject']['clean'].open("r") as f:
            pre_project_assessment_dict = json.load(f)
        with self.input()['project_plan']['markdown'].open("r") as f:
            project_plan_markdown = f.read()
        with self.input()['enrich_team_members_with_contract_type']['clean'].open("r") as f:
            team_member_list = json.load(f)
        with self.input()['related_resources']['markdown'].open("r") as f:
            related_resources_markdown = f.read()

        # Build the query.
        query = (
            f"File 'initial-plan.txt':\n{plan_prompt}\n\n"
            f"File 'strategic_decisions.md':\n{strategic_decisions_markdown}\n\n"
            f"File 'scenarios.md':\n{scenarios_markdown}\n\n"
            f"File 'assumptions.md':\n{consolidate_assumptions_markdown}\n\n"
            f"File 'pre-project-assessment.json':\n{format_json_for_use_in_query(pre_project_assessment_dict)}\n\n"
            f"File 'project-plan.md':\n{project_plan_markdown}\n\n"
            f"File 'team-members-that-needs-to-be-enriched.json':\n{format_json_for_use_in_query(team_member_list)}\n\n"
            f"File 'related-resources.md':\n{related_resources_markdown}"
        )

        # Execute.
        try:
            enrich_team_members_with_background_story = EnrichTeamMembersWithBackgroundStory.execute(llm, query, team_member_list)
        except Exception as e:
            logger.error("EnrichTeamMembersWithBackgroundStory failed: %s", e)
            raise

        # Save the raw output.
        raw_dict = enrich_team_members_with_background_story.to_dict()
        with self.output()['raw'].open("w") as f:
            json.dump(raw_dict, f, indent=2)

        # Save the cleaned up result.
        team_member_list = enrich_team_members_with_background_story.team_member_list
        with self.output()['clean'].open("w") as f:
            json.dump(team_member_list, f, indent=2)

class EnrichTeamMembersWithEnvironmentInfoTask(PlanTask):
    def requires(self):
        return {
            'setup': self.clone(SetupTask),
            'strategic_decisions_markdown': self.clone(StrategicDecisionsMarkdownTask),
            'scenarios_markdown': self.clone(ScenariosMarkdownTask),
            'consolidate_assumptions_markdown': self.clone(ConsolidateAssumptionsMarkdownTask),
            'preproject': self.clone(PreProjectAssessmentTask),
            'project_plan': self.clone(ProjectPlanTask),
            'enrich_team_members_with_background_story': self.clone(EnrichTeamMembersWithBackgroundStoryTask),
            'related_resources': self.clone(RelatedResourcesTask)
        }

    def output(self):
        return {
            'raw': self.local_target(FilenameEnum.ENRICH_TEAM_MEMBERS_ENVIRONMENT_INFO_RAW),
            'clean': self.local_target(FilenameEnum.ENRICH_TEAM_MEMBERS_ENVIRONMENT_INFO_CLEAN)
        }

    def run_with_llm(self, llm: LLM) -> None:
        # Read inputs from required tasks.
        with self.input()['setup'].open("r") as f:
            plan_prompt = f.read()
        with self.input()['strategic_decisions_markdown']['markdown'].open("r") as f:
            strategic_decisions_markdown = f.read()
        with self.input()['scenarios_markdown']['markdown'].open("r") as f:
            scenarios_markdown = f.read()
        with self.input()['consolidate_assumptions_markdown']['short'].open("r") as f:
            consolidate_assumptions_markdown = f.read()
        with self.input()['preproject']['clean'].open("r") as f:
            pre_project_assessment_dict = json.load(f)
        with self.input()['project_plan']['markdown'].open("r") as f:
            project_plan_markdown = f.read()
        with self.input()['enrich_team_members_with_background_story']['clean'].open("r") as f:
            team_member_list = json.load(f)
        with self.input()['related_resources']['markdown'].open("r") as f:
            related_resources_markdown = f.read()

        # Build the query.
        query = (
            f"File 'initial-plan.txt':\n{plan_prompt}\n\n"
            f"File 'strategic_decisions.md':\n{strategic_decisions_markdown}\n\n"
            f"File 'scenarios.md':\n{scenarios_markdown}\n\n"
            f"File 'assumptions.md':\n{consolidate_assumptions_markdown}\n\n"
            f"File 'pre-project-assessment.json':\n{format_json_for_use_in_query(pre_project_assessment_dict)}\n\n"
            f"File 'project-plan.md':\n{project_plan_markdown}\n\n"
            f"File 'team-members-that-needs-to-be-enriched.json':\n{format_json_for_use_in_query(team_member_list)}\n\n"
            f"File 'related-resources.md':\n{related_resources_markdown}"
        )

        # Execute.
        try:
            enrich_team_members_with_background_story = EnrichTeamMembersWithEnvironmentInfo.execute(llm, query, team_member_list)
        except Exception as e:
            logger.error("EnrichTeamMembersWithEnvironmentInfo failed: %s", e)
            raise

        # Save the raw output.
        raw_dict = enrich_team_members_with_background_story.to_dict()
        with self.output()['raw'].open("w") as f:
            json.dump(raw_dict, f, indent=2)

        # Save the cleaned up result.
        team_member_list = enrich_team_members_with_background_story.team_member_list
        with self.output()['clean'].open("w") as f:
            json.dump(team_member_list, f, indent=2)

class ReviewTeamTask(PlanTask):
    def requires(self):
        return {
            'setup': self.clone(SetupTask),
            'strategic_decisions_markdown': self.clone(StrategicDecisionsMarkdownTask),
            'scenarios_markdown': self.clone(ScenariosMarkdownTask),
            'consolidate_assumptions_markdown': self.clone(ConsolidateAssumptionsMarkdownTask),
            'preproject': self.clone(PreProjectAssessmentTask),
            'project_plan': self.clone(ProjectPlanTask),
            'enrich_team_members_with_environment_info': self.clone(EnrichTeamMembersWithEnvironmentInfoTask),
            'related_resources': self.clone(RelatedResourcesTask)
        }

    def output(self):
        return self.local_target(FilenameEnum.REVIEW_TEAM_RAW)

    def run_with_llm(self, llm: LLM) -> None:
        # Read inputs from required tasks.
        with self.input()['setup'].open("r") as f:
            plan_prompt = f.read()
        with self.input()['strategic_decisions_markdown']['markdown'].open("r") as f:
            strategic_decisions_markdown = f.read()
        with self.input()['scenarios_markdown']['markdown'].open("r") as f:
            scenarios_markdown = f.read()
        with self.input()['consolidate_assumptions_markdown']['short'].open("r") as f:
            consolidate_assumptions_markdown = f.read()
        with self.input()['preproject']['clean'].open("r") as f:
            pre_project_assessment_dict = json.load(f)
        with self.input()['project_plan']['markdown'].open("r") as f:
            project_plan_markdown = f.read()
        with self.input()['enrich_team_members_with_environment_info']['clean'].open("r") as f:
            team_member_list = json.load(f)
        with self.input()['related_resources']['markdown'].open("r") as f:
            related_resources_markdown = f.read()

        # Convert the team members to a Markdown document.
        builder = TeamMarkdownDocumentBuilder()
        builder.append_roles(team_member_list, title=None)
        team_document_markdown = builder.to_string()

        # Build the query.
        query = (
            f"File 'initial-plan.txt':\n{plan_prompt}\n\n"
            f"File 'strategic_decisions.md':\n{strategic_decisions_markdown}\n\n"
            f"File 'scenarios.md':\n{scenarios_markdown}\n\n"
            f"File 'assumptions.md':\n{consolidate_assumptions_markdown}\n\n"
            f"File 'pre-project-assessment.json':\n{format_json_for_use_in_query(pre_project_assessment_dict)}\n\n"
            f"File 'project-plan.md':\n{project_plan_markdown}\n\n"
            f"File 'team-members.md':\n{team_document_markdown}\n\n"
            f"File 'related-resources.md':\n{related_resources_markdown}"
        )

        # Execute.
        try:
            review_team = ReviewTeam.execute(llm, query)
        except Exception as e:
            logger.error("ReviewTeam failed: %s", e)
            raise

        # Save the raw output.
        raw_dict = review_team.to_dict()
        with self.output().open("w") as f:
            json.dump(raw_dict, f, indent=2)

        logger.info("ReviewTeamTask complete.")

class TeamMarkdownTask(PlanTask):
    def requires(self):
        return {
            'enrich_team_members_with_environment_info': self.clone(EnrichTeamMembersWithEnvironmentInfoTask),
            'review_team': self.clone(ReviewTeamTask)
        }

    def output(self):
        return self.local_target(FilenameEnum.TEAM_MARKDOWN)

    def run_inner(self):
        logger.info("TeamMarkdownTask. Loading files...")

        # 1. Read the team_member_list from EnrichTeamMembersWithEnvironmentInfoTask.
        with self.input()['enrich_team_members_with_environment_info']['clean'].open("r") as f:
            team_member_list = json.load(f)

        # 2. Read the json from ReviewTeamTask.
        with self.input()['review_team'].open("r") as f:
            review_team_json = json.load(f)

        logger.info("TeamMarkdownTask. All files are now ready. Processing...")

        # Combine the team members and the review into a Markdown document.
        builder = TeamMarkdownDocumentBuilder()
        builder.append_team_member_subtitle()
        builder.append_roles(team_member_list)
        builder.append_separator()
        builder.append_full_review(review_team_json)
        builder.write_to_file(self.output().path)

        logger.info("TeamMarkdownTask complete.")

class SWOTAnalysisTask(PlanTask):
    def requires(self):
        return {
            'setup': self.clone(SetupTask),
            'strategic_decisions_markdown': self.clone(StrategicDecisionsMarkdownTask),
            'scenarios_markdown': self.clone(ScenariosMarkdownTask),
            'identify_purpose': self.clone(IdentifyPurposeTask),
            'consolidate_assumptions_markdown': self.clone(ConsolidateAssumptionsMarkdownTask),
            'preproject': self.clone(PreProjectAssessmentTask),
            'project_plan': self.clone(ProjectPlanTask),
            'related_resources': self.clone(RelatedResourcesTask)
        }

    def output(self):
        return {
            'raw': self.local_target(FilenameEnum.SWOT_RAW),
            'markdown': self.local_target(FilenameEnum.SWOT_MARKDOWN)
        }

    def run_with_llm(self, llm: LLM) -> None:
        # Read inputs from required tasks.
        with self.input()['setup'].open("r") as f:
            plan_prompt = f.read()
        with self.input()['strategic_decisions_markdown']['markdown'].open("r") as f:
            strategic_decisions_markdown = f.read()
        with self.input()['scenarios_markdown']['markdown'].open("r") as f:
            scenarios_markdown = f.read()
        with self.input()['identify_purpose']['raw'].open("r") as f:
            identify_purpose_dict = json.load(f)
        with self.input()['consolidate_assumptions_markdown']['short'].open("r") as f:
            consolidate_assumptions_markdown = f.read()
        with self.input()['preproject']['clean'].open("r") as f:
            pre_project_assessment_dict = json.load(f)
        with self.input()['project_plan']['markdown'].open("r") as f:
            project_plan_markdown = f.read()
        with self.input()['related_resources']['markdown'].open("r") as f:
            related_resources_markdown = f.read()

        # Build the query for SWOT analysis.
        query = (
            f"File 'initial-plan.txt':\n{plan_prompt}\n\n"
            f"File 'strategic_decisions.md':\n{strategic_decisions_markdown}\n\n"
            f"File 'scenarios.md':\n{scenarios_markdown}\n\n"
            f"File 'assumptions.md':\n{consolidate_assumptions_markdown}\n\n"
            f"File 'pre-project-assessment.json':\n{format_json_for_use_in_query(pre_project_assessment_dict)}\n\n"
            f"File 'project-plan.md':\n{project_plan_markdown}\n\n"
            f"File 'related-resources.md':\n{related_resources_markdown}"
        )

        # Execute the SWOT analysis.
        # Send the identify_purpose_dict to SWOTAnalysis, and use business/personal/other to select the system prompt
        try:
            swot_analysis = SWOTAnalysis.execute(llm=llm, query=query, identify_purpose_dict=identify_purpose_dict)
        except Exception as e:
            logger.error("SWOT analysis failed: %s", e)
            raise

        # Convert the SWOT analysis to a dict and markdown.
        swot_raw_dict = swot_analysis.to_dict()
        swot_markdown = swot_analysis.to_markdown(include_metadata=False, include_purpose=False)

        # Write the raw SWOT JSON.
        with self.output()['raw'].open("w") as f:
            json.dump(swot_raw_dict, f, indent=2)

        # Write the SWOT analysis as Markdown.
        markdown_path = self.output()['markdown'].path
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(swot_markdown)

class ExpertReviewTask(PlanTask):
    """
    Finds experts to review the SWOT analysis and have them provide criticism.
    """
    def requires(self):
        return {
            'setup': self.clone(SetupTask),
            'strategic_decisions_markdown': self.clone(StrategicDecisionsMarkdownTask),
            'scenarios_markdown': self.clone(ScenariosMarkdownTask),
            'preproject': self.clone(PreProjectAssessmentTask),
            'project_plan': self.clone(ProjectPlanTask),
            'swot_analysis': self.clone(SWOTAnalysisTask)
        }

    def output(self):
        return self.local_target(FilenameEnum.EXPERT_CRITICISM_MARKDOWN)

    def run_inner(self):
        llm_executor: LLMExecutor = self.create_llm_executor()

        logger.info("Finding experts to review the SWOT analysis, and having them provide criticism...")

        # Read inputs from required tasks.
        with self.input()['setup'].open("r") as f:
            plan_prompt = f.read()
        with self.input()['strategic_decisions_markdown']['markdown'].open("r") as f:
            strategic_decisions_markdown = f.read()
        with self.input()['scenarios_markdown']['markdown'].open("r") as f:
            scenarios_markdown = f.read()
        with self.input()['preproject']['clean'].open("r") as f:
            pre_project_assessment_dict = json.load(f)
        with self.input()['project_plan']['markdown'].open("r") as f:
            project_plan_markdown = f.read()
        swot_markdown_path = self.input()['swot_analysis']['markdown'].path
        with open(swot_markdown_path, "r", encoding="utf-8") as f:
            swot_markdown = f.read()

        # Build the query.
        query = (
            f"File 'initial-plan.txt':\n{plan_prompt}\n\n"
            f"File 'strategic_decisions.md':\n{strategic_decisions_markdown}\n\n"
            f"File 'scenarios.md':\n{scenarios_markdown}\n\n"
            f"File 'pre-project assessment.json':\n{format_json_for_use_in_query(pre_project_assessment_dict)}\n\n"
            f"File 'project_plan.md':\n{project_plan_markdown}\n\n"
            f"File 'SWOT Analysis.md':\n{swot_markdown}"
        )

        # Define callback functions.
        def phase1_post_callback(expert_finder: ExpertFinder) -> None:
            raw_path = self.run_id_dir / FilenameEnum.EXPERTS_RAW.value
            cleaned_path = self.run_id_dir / FilenameEnum.EXPERTS_CLEAN.value
            expert_finder.save_raw(str(raw_path))
            expert_finder.save_cleanedup(str(cleaned_path))

        def phase2_post_callback(expert_criticism: ExpertCriticism, expert_index: int) -> None:
            file_path = self.run_id_dir / FilenameEnum.EXPERT_CRITICISM_RAW_TEMPLATE.format(expert_index + 1)
            expert_criticism.save_raw(str(file_path))

        # Execute the expert orchestration.
        expert_orchestrator = ExpertOrchestrator()
        # IDEA: max_expert_count. don't truncate to 2 experts. Interview them all in production mode.
        # IDEA: If the expert file for expert_index already exist, then there is no need to run the LLM again.
        expert_orchestrator.phase1_post_callback = phase1_post_callback
        expert_orchestrator.phase2_post_callback = phase2_post_callback
        expert_orchestrator.execute(llm_executor, query)

        # Write final expert criticism markdown.
        expert_criticism_markdown_file = self.file_path(FilenameEnum.EXPERT_CRITICISM_MARKDOWN)
        with expert_criticism_markdown_file.open("w") as f:
            f.write(expert_orchestrator.to_markdown())


class DataCollectionTask(PlanTask):
    """
    Determine what kind of data is to be collected.
    """
    def output(self):
        return {
            'raw': self.local_target(FilenameEnum.DATA_COLLECTION_RAW),
            'markdown': self.local_target(FilenameEnum.DATA_COLLECTION_MARKDOWN)
        }
    
    def requires(self):
        return {
            'strategic_decisions_markdown': self.clone(StrategicDecisionsMarkdownTask),
            'scenarios_markdown': self.clone(ScenariosMarkdownTask),
            'consolidate_assumptions_markdown': self.clone(ConsolidateAssumptionsMarkdownTask),
            'project_plan': self.clone(ProjectPlanTask),
            'related_resources': self.clone(RelatedResourcesTask),
            'swot_analysis': self.clone(SWOTAnalysisTask),
            'team_markdown': self.clone(TeamMarkdownTask),
            'expert_review': self.clone(ExpertReviewTask)
        }
    
    def run_with_llm(self, llm: LLM) -> None:
        # Read inputs from required tasks.
        with self.input()['strategic_decisions_markdown']['markdown'].open("r") as f:
            strategic_decisions_markdown = f.read()
        with self.input()['scenarios_markdown']['markdown'].open("r") as f:
            scenarios_markdown = f.read()
        with self.input()['consolidate_assumptions_markdown']['short'].open("r") as f:
            assumptions_markdown = f.read()
        with self.input()['project_plan']['markdown'].open("r") as f:
            project_plan_markdown = f.read()
        with self.input()['related_resources']['markdown'].open("r") as f:
            related_resources_markdown = f.read()
        with self.input()['swot_analysis']['markdown'].open("r") as f:
            swot_analysis_markdown = f.read()
        with self.input()['team_markdown'].open("r") as f:
            team_markdown = f.read()
        with self.input()['expert_review'].open("r") as f:
            expert_review = f.read()

        # Build the query.
        query = (
            f"File 'strategic_decisions.md':\n{strategic_decisions_markdown}\n\n"
            f"File 'scenarios.md':\n{scenarios_markdown}\n\n"
            f"File 'assumptions.md':\n{assumptions_markdown}\n\n"
            f"File 'project-plan.md':\n{project_plan_markdown}\n\n"
            f"File 'related-resources.md':\n{related_resources_markdown}\n\n"
            f"File 'swot-analysis.md':\n{swot_analysis_markdown}\n\n"
            f"File 'team.md':\n{team_markdown}\n\n"
            f"File 'expert-review.md':\n{expert_review}"
        )

        # Invoke the LLM.
        data_collection = DataCollection.execute(llm, query)

        # Save the results.
        json_path = self.output()['raw'].path
        data_collection.save_raw(json_path)
        markdown_path = self.output()['markdown'].path
        data_collection.save_markdown(markdown_path)

class IdentifyDocumentsTask(PlanTask):
    """
    Identify documents that need to be created or found for the project.
    """
    def output(self):
        return {
            "raw": self.local_target(FilenameEnum.IDENTIFIED_DOCUMENTS_RAW),
            "markdown": self.local_target(FilenameEnum.IDENTIFIED_DOCUMENTS_MARKDOWN),
            "documents_to_find": self.local_target(FilenameEnum.IDENTIFIED_DOCUMENTS_TO_FIND_JSON),
            "documents_to_create": self.local_target(FilenameEnum.IDENTIFIED_DOCUMENTS_TO_CREATE_JSON),
        }

    def requires(self):
        return {
            'identify_purpose': self.clone(IdentifyPurposeTask),
            'strategic_decisions_markdown': self.clone(StrategicDecisionsMarkdownTask),
            'scenarios_markdown': self.clone(ScenariosMarkdownTask),
            'consolidate_assumptions_markdown': self.clone(ConsolidateAssumptionsMarkdownTask),
            'project_plan': self.clone(ProjectPlanTask),
            'related_resources': self.clone(RelatedResourcesTask),
            'swot_analysis': self.clone(SWOTAnalysisTask),
            'team_markdown': self.clone(TeamMarkdownTask),
            'expert_review': self.clone(ExpertReviewTask)
        }
    
    def run_with_llm(self, llm: LLM) -> None:
        # Read inputs from required tasks.
        with self.input()['identify_purpose']['raw'].open("r") as f:
            identify_purpose_dict = json.load(f)
        with self.input()['strategic_decisions_markdown']['markdown'].open("r") as f:
            strategic_decisions_markdown = f.read()
        with self.input()['scenarios_markdown']['markdown'].open("r") as f:
            scenarios_markdown = f.read()
        with self.input()['consolidate_assumptions_markdown']['short'].open("r") as f:
            assumptions_markdown = f.read()
        with self.input()['project_plan']['markdown'].open("r") as f:
            project_plan_markdown = f.read()
        with self.input()['related_resources']['markdown'].open("r") as f:
            related_resources_markdown = f.read()
        with self.input()['swot_analysis']['markdown'].open("r") as f:
            swot_analysis_markdown = f.read()
        with self.input()['team_markdown'].open("r") as f:
            team_markdown = f.read()
        with self.input()['expert_review'].open("r") as f:
            expert_review = f.read()

        # Build the query.
        query = (
            f"File 'strategic_decisions.md':\n{strategic_decisions_markdown}\n\n"
            f"File 'scenarios.md':\n{scenarios_markdown}\n\n"
            f"File 'assumptions.md':\n{assumptions_markdown}\n\n"
            f"File 'project-plan.md':\n{project_plan_markdown}\n\n"
            f"File 'related-resources.md':\n{related_resources_markdown}\n\n"
            f"File 'swot-analysis.md':\n{swot_analysis_markdown}\n\n"
            f"File 'team.md':\n{team_markdown}\n\n"
            f"File 'expert-review.md':\n{expert_review}"
        )

        # Invoke the LLM.
        identify_documents = IdentifyDocuments.execute(
            llm=llm,
            user_prompt=query,
            identify_purpose_dict=identify_purpose_dict
        )

        # Save the results.
        identify_documents.save_raw(self.output()["raw"].path)
        identify_documents.save_markdown(self.output()["markdown"].path)
        identify_documents.save_json_documents_to_find(self.output()["documents_to_find"].path)
        identify_documents.save_json_documents_to_create(self.output()["documents_to_create"].path)

class FilterDocumentsToFindTask(PlanTask):
    """
    The "documents to find" may be a long list of documents, some duplicates, irrelevant, not needed at an early stage of the project.
    This task narrows down to a handful of relevant documents.
    """
    def output(self):
        return {
            "raw": self.local_target(FilenameEnum.FILTER_DOCUMENTS_TO_FIND_RAW),
            "clean": self.local_target(FilenameEnum.FILTER_DOCUMENTS_TO_FIND_CLEAN)
        }

    def requires(self):
        return {
            'identify_purpose': self.clone(IdentifyPurposeTask),
            'strategic_decisions_markdown': self.clone(StrategicDecisionsMarkdownTask),
            'scenarios_markdown': self.clone(ScenariosMarkdownTask),
            'consolidate_assumptions_markdown': self.clone(ConsolidateAssumptionsMarkdownTask),
            'project_plan': self.clone(ProjectPlanTask),
            'identified_documents': self.clone(IdentifyDocumentsTask),
        }
    
    def run_with_llm(self, llm: LLM) -> None:
        # Read inputs from required tasks.
        with self.input()['identify_purpose']['raw'].open("r") as f:
            identify_purpose_dict = json.load(f)
        with self.input()['strategic_decisions_markdown']['markdown'].open("r") as f:
            strategic_decisions_markdown = f.read()
        with self.input()['scenarios_markdown']['markdown'].open("r") as f:
            scenarios_markdown = f.read()
        with self.input()['consolidate_assumptions_markdown']['short'].open("r") as f:
            assumptions_markdown = f.read()
        with self.input()['project_plan']['markdown'].open("r") as f:
            project_plan_markdown = f.read()
        with self.input()['identified_documents']['documents_to_find'].open("r") as f:
            documents_to_find = json.load(f)

        # Build the query.
        process_documents, integer_id_to_document_uuid = FilterDocumentsToFind.process_documents_and_integer_ids(documents_to_find)
        query = (
            f"File 'strategic_decisions.md':\n{strategic_decisions_markdown}\n\n"
            f"File 'scenarios.md':\n{scenarios_markdown}\n\n"
            f"File 'assumptions.md':\n{assumptions_markdown}\n\n"
            f"File 'project-plan.md':\n{project_plan_markdown}\n\n"
            f"File 'documents.json':\n{process_documents}"
        )

        # Invoke the LLM.
        filter_documents = FilterDocumentsToFind.execute(
            llm=llm,
            user_prompt=query,
            identified_documents_raw_json=documents_to_find,
            integer_id_to_document_uuid=integer_id_to_document_uuid,
            identify_purpose_dict=identify_purpose_dict
        )

        # Save the results.
        filter_documents.save_raw(self.output()["raw"].path)
        filter_documents.save_filtered_documents(self.output()["clean"].path)

class FilterDocumentsToCreateTask(PlanTask):
    """
    The "documents to create" may be a long list of documents, some duplicates, irrelevant, not needed at an early stage of the project.
    This task narrows down to a handful of relevant documents.
    """
    def output(self):
        return {
            "raw": self.local_target(FilenameEnum.FILTER_DOCUMENTS_TO_CREATE_RAW),
            "clean": self.local_target(FilenameEnum.FILTER_DOCUMENTS_TO_CREATE_CLEAN)
        }

    def requires(self):
        return {
            'identify_purpose': self.clone(IdentifyPurposeTask),
            'strategic_decisions_markdown': self.clone(StrategicDecisionsMarkdownTask),
            'scenarios_markdown': self.clone(ScenariosMarkdownTask),
            'consolidate_assumptions_markdown': self.clone(ConsolidateAssumptionsMarkdownTask),
            'project_plan': self.clone(ProjectPlanTask),
            'identified_documents': self.clone(IdentifyDocumentsTask),
        }
    
    def run_with_llm(self, llm: LLM) -> None:
        # Read inputs from required tasks.
        with self.input()['identify_purpose']['raw'].open("r") as f:
            identify_purpose_dict = json.load(f)
        with self.input()['strategic_decisions_markdown']['markdown'].open("r") as f:
            strategic_decisions_markdown = f.read()
        with self.input()['scenarios_markdown']['markdown'].open("r") as f:
            scenarios_markdown = f.read()
        with self.input()['consolidate_assumptions_markdown']['short'].open("r") as f:
            assumptions_markdown = f.read()
        with self.input()['project_plan']['markdown'].open("r") as f:
            project_plan_markdown = f.read()
        with self.input()['identified_documents']['documents_to_create'].open("r") as f:
            documents_to_create = json.load(f)

        # Build the query.
        process_documents, integer_id_to_document_uuid = FilterDocumentsToCreate.process_documents_and_integer_ids(documents_to_create)
        query = (
            f"File 'strategic_decisions.md':\n{strategic_decisions_markdown}\n\n"
            f"File 'scenarios.md':\n{scenarios_markdown}\n\n"
            f"File 'assumptions.md':\n{assumptions_markdown}\n\n"
            f"File 'project-plan.md':\n{project_plan_markdown}\n\n"
            f"File 'documents.json':\n{process_documents}"
        )

        # Invoke the LLM.
        filter_documents = FilterDocumentsToCreate.execute(
            llm=llm,
            user_prompt=query,
            identified_documents_raw_json=documents_to_create,
            integer_id_to_document_uuid=integer_id_to_document_uuid,
            identify_purpose_dict=identify_purpose_dict
        )

        # Save the results.
        filter_documents.save_raw(self.output()["raw"].path)
        filter_documents.save_filtered_documents(self.output()["clean"].path)

class DraftDocumentsToFindTask(PlanTask):
    """
    The "documents to find". Write bullet points to what each document roughly should contain.
    """
    def output(self):
        return self.local_target(FilenameEnum.DRAFT_DOCUMENTS_TO_FIND_CONSOLIDATED)

    def requires(self):
        return {
            'identify_purpose': self.clone(IdentifyPurposeTask),
            'strategic_decisions_markdown': self.clone(StrategicDecisionsMarkdownTask),
            'scenarios_markdown': self.clone(ScenariosMarkdownTask),
            'consolidate_assumptions_markdown': self.clone(ConsolidateAssumptionsMarkdownTask),
            'project_plan': self.clone(ProjectPlanTask),
            'filter_documents_to_find': self.clone(FilterDocumentsToFindTask),
        }
    
    def run_inner(self):
        llm_executor: LLMExecutor = self.create_llm_executor()

        # Read inputs from required tasks.
        with self.input()['identify_purpose']['raw'].open("r") as f:
            identify_purpose_dict = json.load(f)
        with self.input()['strategic_decisions_markdown']['markdown'].open("r") as f:
            strategic_decisions_markdown = f.read()
        with self.input()['scenarios_markdown']['markdown'].open("r") as f:
            scenarios_markdown = f.read()
        with self.input()['consolidate_assumptions_markdown']['short'].open("r") as f:
            assumptions_markdown = f.read()
        with self.input()['project_plan']['markdown'].open("r") as f:
            project_plan_markdown = f.read()
        with self.input()['filter_documents_to_find']['clean'].open("r") as f:
            documents_to_find = json.load(f)

        accumulated_documents = documents_to_find.copy()

        logger.info(f"DraftDocumentsToFindTask.speedvsdetail: {self.speedvsdetail}")
        if self.speedvsdetail == SpeedVsDetailEnum.FAST_BUT_SKIP_DETAILS:
            logger.info("FAST_BUT_SKIP_DETAILS mode, truncating to 2 chunks for testing.")
            documents_to_find = documents_to_find[:2]
        else:
            logger.info("Processing all chunks.")

        for index, document in enumerate(documents_to_find):
            logger.info(f"Document-to-find: Drafting document {index+1} of {len(documents_to_find)}...")

            # Build the query.
            query = (
                f"File 'strategic_decisions.md':\n{strategic_decisions_markdown}\n\n"
                f"File 'scenarios.md':\n{scenarios_markdown}\n\n"
                f"File 'assumptions.md':\n{assumptions_markdown}\n\n"
                f"File 'project-plan.md':\n{project_plan_markdown}\n\n"
                f"File 'document.json':\n{document}"
            )

            # IDEA: If the document already exist, then there is no need to run the LLM again.
            def execute_draft_document_to_find(llm: LLM) -> DraftDocumentToFind:
                return DraftDocumentToFind.execute(llm=llm, user_prompt=query, identify_purpose_dict=identify_purpose_dict)

            try:
                draft_document = llm_executor.run(execute_draft_document_to_find)
            except PipelineStopRequested:
                # Re-raise PipelineStopRequested without wrapping it
                raise
            except Exception as e:
                logger.error(f"Document-to-find {index+1} LLM interaction failed.", exc_info=True)
                raise ValueError(f"Document-to-find {index+1} LLM interaction failed.") from e

            json_response = draft_document.to_dict()

            # Write the raw JSON for this document using the FilenameEnum template.
            raw_filename = FilenameEnum.DRAFT_DOCUMENTS_TO_FIND_RAW_TEMPLATE.value.format(index+1)
            raw_chunk_path = self.run_id_dir / raw_filename
            with open(raw_chunk_path, 'w') as f:
                json.dump(json_response, f, indent=2)

            # Merge the draft document into the original document.
            document_updated = document.copy()
            for key in draft_document.response.keys():
                document_updated[key] = draft_document.response[key]
            accumulated_documents[index] = document_updated

        # Write the accumulated documents to the output file.
        with self.output().open("w") as f:
            json.dump(accumulated_documents, f, indent=2)

class DraftDocumentsToCreateTask(PlanTask):
    """
    The "documents to create". Write bullet points to what each document roughly should contain.
    """
    def output(self):
        return self.local_target(FilenameEnum.DRAFT_DOCUMENTS_TO_CREATE_CONSOLIDATED)

    def requires(self):
        return {
            'identify_purpose': self.clone(IdentifyPurposeTask),
            'strategic_decisions_markdown': self.clone(StrategicDecisionsMarkdownTask),
            'scenarios_markdown': self.clone(ScenariosMarkdownTask),
            'consolidate_assumptions_markdown': self.clone(ConsolidateAssumptionsMarkdownTask),
            'project_plan': self.clone(ProjectPlanTask),
            'filter_documents_to_create': self.clone(FilterDocumentsToCreateTask),
        }
    
    def run_inner(self):
        llm_executor: LLMExecutor = self.create_llm_executor()

        # Read inputs from required tasks.
        with self.input()['identify_purpose']['raw'].open("r") as f:
            identify_purpose_dict = json.load(f)
        with self.input()['strategic_decisions_markdown']['markdown'].open("r") as f:
            strategic_decisions_markdown = f.read()
        with self.input()['scenarios_markdown']['markdown'].open("r") as f:
            scenarios_markdown = f.read()
        with self.input()['consolidate_assumptions_markdown']['short'].open("r") as f:
            assumptions_markdown = f.read()
        with self.input()['project_plan']['markdown'].open("r") as f:
            project_plan_markdown = f.read()
        with self.input()['filter_documents_to_create']['clean'].open("r") as f:
            documents_to_create = json.load(f)

        accumulated_documents = documents_to_create.copy()

        logger.info(f"DraftDocumentsToCreateTask.speedvsdetail: {self.speedvsdetail}")
        if self.speedvsdetail == SpeedVsDetailEnum.FAST_BUT_SKIP_DETAILS:
            logger.info("FAST_BUT_SKIP_DETAILS mode, truncating to 2 chunks for testing.")
            documents_to_create = documents_to_create[:2]
        else:
            logger.info("Processing all chunks.")

        for index, document in enumerate(documents_to_create):
            logger.info(f"Document-to-create: Drafting document {index+1} of {len(documents_to_create)}...")

            # Build the query.
            query = (
                f"File 'strategic_decisions.md':\n{strategic_decisions_markdown}\n\n"
                f"File 'scenarios.md':\n{scenarios_markdown}\n\n"
                f"File 'assumptions.md':\n{assumptions_markdown}\n\n"
                f"File 'project-plan.md':\n{project_plan_markdown}\n\n"
                f"File 'document.json':\n{document}"
            )

            # IDEA: If the document already exist, then there is no need to run the LLM again.
            def execute_draft_document_to_create(llm: LLM) -> DraftDocumentToCreate:
                return DraftDocumentToCreate.execute(llm=llm, user_prompt=query, identify_purpose_dict=identify_purpose_dict)

            try:
                draft_document = llm_executor.run(execute_draft_document_to_create)
            except PipelineStopRequested:
                # Re-raise PipelineStopRequested without wrapping it
                raise
            except Exception as e:
                logger.error(f"Document-to-create {index+1} LLM interaction failed.", exc_info=True)
                raise ValueError(f"Document-to-create {index+1} LLM interaction failed.") from e

            json_response = draft_document.to_dict()

            # Write the raw JSON for this document using the FilenameEnum template.
            raw_filename = FilenameEnum.DRAFT_DOCUMENTS_TO_CREATE_RAW_TEMPLATE.value.format(index+1)
            raw_chunk_path = self.run_id_dir / raw_filename
            with open(raw_chunk_path, 'w') as f:
                json.dump(json_response, f, indent=2)

            # Merge the draft document into the original document.
            document_updated = document.copy()
            for key in draft_document.response.keys():
                document_updated[key] = draft_document.response[key]
            accumulated_documents[index] = document_updated

        # Write the accumulated documents to the output file.
        with self.output().open("w") as f:
            json.dump(accumulated_documents, f, indent=2)

class MarkdownWithDocumentsToCreateAndFindTask(PlanTask):
    """
    Create markdown with the "documents to create and find"
    """
    def output(self):
        return self.local_target(FilenameEnum.DOCUMENTS_TO_CREATE_AND_FIND_MARKDOWN)

    def requires(self):
        return {
            'draft_documents_to_create': self.clone(DraftDocumentsToCreateTask),
            'draft_documents_to_find': self.clone(DraftDocumentsToFindTask),
        }
    
    def run_inner(self):
        # Read inputs from required tasks.
        with self.input()['draft_documents_to_create'].open("r") as f:
            documents_to_create = json.load(f)
        with self.input()['draft_documents_to_find'].open("r") as f:
            documents_to_find = json.load(f)

        accumulated_rows = []
        accumulated_rows.append("# Documents to Create")
        for index, document in enumerate(documents_to_create, start=1):
            rows = markdown_rows_with_document_to_create(index, document)
            accumulated_rows.extend(rows)

        accumulated_rows.append("\n\n# Documents to Find")
        for index, document in enumerate(documents_to_find, start=1):
            rows = markdown_rows_with_document_to_find(index, document)
            accumulated_rows.extend(rows)

        markdown_representation = "\n".join(accumulated_rows)

        # Write the markdown to the output file.
        output_file_path = self.output().path
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(markdown_representation)

class CreateWBSLevel1Task(PlanTask):
    """
    Creates the Work Breakdown Structure (WBS) Level 1.
    Depends on:
      - ProjectPlanTask: provides the project plan as JSON.
    Produces:
      - Raw WBS Level 1 output file (xxx-wbs_level1_raw.json)
      - Cleaned up WBS Level 1 file (xxx-wbs_level1.json)
    """
    def requires(self):
        return {
            'project_plan': self.clone(ProjectPlanTask)
        }

    def output(self):
        return {
            'raw': self.local_target(FilenameEnum.WBS_LEVEL1_RAW),
            'clean': self.local_target(FilenameEnum.WBS_LEVEL1),
            'project_title': self.local_target(FilenameEnum.WBS_LEVEL1_PROJECT_TITLE)
        }

    def run_with_llm(self, llm: LLM) -> None:
        logger.info("Creating Work Breakdown Structure (WBS) Level 1...")
        
        # Read the project plan JSON from the dependency.
        with self.input()['project_plan']['raw'].open("r") as f:
            project_plan_dict = json.load(f)
        
        # Build the query using the project plan.
        query = format_json_for_use_in_query(project_plan_dict)
        
        # Execute the WBS Level 1 creation.
        create_wbs_level1 = CreateWBSLevel1.execute(llm, query)
        
        # Save the raw output.
        wbs_level1_raw_dict = create_wbs_level1.raw_response_dict()
        with self.output()['raw'].open("w") as f:
            json.dump(wbs_level1_raw_dict, f, indent=2)
        
        # Save the cleaned up result.
        wbs_level1_result_json = create_wbs_level1.cleanedup_dict()
        with self.output()['clean'].open("w") as f:
            json.dump(wbs_level1_result_json, f, indent=2)

        # Save the project title.
        with self.output()['project_title'].open("w") as f:
            f.write(create_wbs_level1.project_title)
        
        logger.info("WBS Level 1 created successfully.")

class CreateWBSLevel2Task(PlanTask):
    """
    Creates the Work Breakdown Structure (WBS) Level 2.
    Depends on:
      - ProjectPlanTask: provides the project plan as JSON.
      - CreateWBSLevel1Task: provides the cleaned WBS Level 1 result.
    Produces:
      - Raw WBS Level 2 output (007-wbs_level2_raw.json)
      - Cleaned WBS Level 2 output (008-wbs_level2.json)
    """
    def requires(self):
        return {
            'strategic_decisions_markdown': self.clone(StrategicDecisionsMarkdownTask),
            'scenarios_markdown': self.clone(ScenariosMarkdownTask),
            'project_plan': self.clone(ProjectPlanTask),
            'wbs_level1': self.clone(CreateWBSLevel1Task),
            'data_collection': self.clone(DataCollectionTask),
        }

    def output(self):
        return {
            'raw': self.local_target(FilenameEnum.WBS_LEVEL2_RAW),
            'clean': self.local_target(FilenameEnum.WBS_LEVEL2)
        }

    def run_with_llm(self, llm: LLM) -> None:
        logger.info("Creating Work Breakdown Structure (WBS) Level 2...")

        # Read inputs from required tasks.
        with self.input()['strategic_decisions_markdown']['markdown'].open("r") as f:
            strategic_decisions_markdown = f.read()
        with self.input()['scenarios_markdown']['markdown'].open("r") as f:
            scenarios_markdown = f.read()
        with self.input()['project_plan']['markdown'].open("r") as f:
            project_plan_markdown = f.read()        
        with self.input()['data_collection']['markdown'].open("r") as f:
            data_collection_markdown = f.read()
        with self.input()['wbs_level1']['clean'].open("r") as f:
            wbs_level1_result_json = json.load(f)
        
        query = (
            f"File 'strategic_decisions.md':\n{strategic_decisions_markdown}\n\n"
            f"File 'scenarios.md':\n{scenarios_markdown}\n\n"
            f"File 'project_plan.md':\n{project_plan_markdown}\n\n"
            f"File 'WBS Level 1.json':\n{format_json_for_use_in_query(wbs_level1_result_json)}\n\n"
            f"File 'data_collection.md':\n{data_collection_markdown}"
        )
        
        # Execute the WBS Level 2 creation.
        create_wbs_level2 = CreateWBSLevel2.execute(llm, query)
        
        # Retrieve and write the raw output.
        wbs_level2_raw_dict = create_wbs_level2.raw_response_dict()
        with self.output()['raw'].open("w") as f:
            json.dump(wbs_level2_raw_dict, f, indent=2)
        
        # Retrieve and write the cleaned output (e.g. major phases with subtasks).
        with self.output()['clean'].open("w") as f:
            json.dump(create_wbs_level2.major_phases_with_subtasks, f, indent=2)
        
        logger.info("WBS Level 2 created successfully.")

class WBSProjectLevel1AndLevel2Task(PlanTask):
    """
    Create a WBS project from the WBS Level 1 and Level 2 JSON files.
    
    It depends on:
      - CreateWBSLevel1Task: providing the cleaned WBS Level 1 JSON.
      - CreateWBSLevel2Task: providing the major phases with subtasks and the task UUIDs.
    """
    def output(self):
        return self.local_target(FilenameEnum.WBS_PROJECT_LEVEL1_AND_LEVEL2)
    
    def requires(self):
        return {
            'wbs_level1': self.clone(CreateWBSLevel1Task),
            'wbs_level2': self.clone(CreateWBSLevel2Task),
        }
    
    def run_inner(self):
        wbs_level1_path = self.input()['wbs_level1']['clean'].path
        wbs_level2_path = self.input()['wbs_level2']['clean'].path
        wbs_project = WBSPopulate.project_from_level1_json(wbs_level1_path)
        WBSPopulate.extend_project_with_level2_json(wbs_project, wbs_level2_path)

        json_representation = json.dumps(wbs_project.to_dict(), indent=2)
        with self.output().open("w") as f:
            f.write(json_representation)

class CreatePitchTask(PlanTask):
    """
    Create a the pitch that explains the project plan, from multiple perspectives.
    
    This task depends on:
      - ProjectPlanTask: provides the project plan JSON.
      - WBSProjectLevel1AndLevel2Task: containing the top level of the project plan.
    
    The resulting pitch JSON is written to the file specified by FilenameEnum.PITCH.
    """
    def output(self):
        return self.local_target(FilenameEnum.PITCH_RAW)
    
    def requires(self):
        return {
            'strategic_decisions_markdown': self.clone(StrategicDecisionsMarkdownTask),
            'scenarios_markdown': self.clone(ScenariosMarkdownTask),
            'project_plan': self.clone(ProjectPlanTask),
            'wbs_project': self.clone(WBSProjectLevel1AndLevel2Task),
            'related_resources': self.clone(RelatedResourcesTask)
        }
    
    def run_with_llm(self, llm: LLM) -> None:
        # Read the project plan JSON.
        with self.input()['strategic_decisions_markdown']['markdown'].open("r") as f:
            strategic_decisions_markdown = f.read()
        with self.input()['scenarios_markdown']['markdown'].open("r") as f:
            scenarios_markdown = f.read()
        with self.input()['project_plan']['markdown'].open("r") as f:
            project_plan_markdown = f.read()
        
        with self.input()['wbs_project'].open("r") as f:
            wbs_project_dict = json.load(f)
        wbs_project = WBSProject.from_dict(wbs_project_dict)
        wbs_project_json = wbs_project.to_dict()

        with self.input()['related_resources']['markdown'].open("r") as f:
            related_resources_markdown = f.read()
        
        # Build the query
        query = (
            f"File 'strategic_decisions.md':\n{strategic_decisions_markdown}\n\n"
            f"File 'scenarios.md':\n{scenarios_markdown}\n\n"
            f"File 'project_plan.md':\n{project_plan_markdown}\n\n"
            f"File 'Work Breakdown Structure.json':\n{format_json_for_use_in_query(wbs_project_json)}\n\n"
            f"File 'similar_projects.md':\n{related_resources_markdown}"
        )
        
        # Execute the pitch creation.
        create_pitch = CreatePitch.execute(llm, query)
        pitch_dict = create_pitch.raw_response_dict()
        
        # Write the resulting pitch JSON to the output file.
        with self.output().open("w") as f:
            json.dump(pitch_dict, f, indent=2)
        
        logger.info("Pitch created and written to %s", self.output().path)

class ConvertPitchToMarkdownTask(PlanTask):
    """
    Human readable version of the pitch.
    
    This task depends on:
      - CreatePitchTask: Creates the pitch JSON.
    """
    def output(self):
        return {
            'raw': self.local_target(FilenameEnum.PITCH_CONVERT_TO_MARKDOWN_RAW),
            'markdown': self.local_target(FilenameEnum.PITCH_MARKDOWN)
        }
    
    def requires(self):
        return {
            'pitch': self.clone(CreatePitchTask),
        }
    
    def run_with_llm(self, llm: LLM) -> None:
        # Read the project plan JSON.
        with self.input()['pitch'].open("r") as f:
            pitch_json = json.load(f)
        
        # Build the query
        query = format_json_for_use_in_query(pitch_json)
        
        # Execute the conversion.
        converted = ConvertPitchToMarkdown.execute(llm, query)

        # Save the results.
        json_path = self.output()['raw'].path
        converted.save_raw(json_path)
        markdown_path = self.output()['markdown'].path
        converted.save_markdown(markdown_path)


class IdentifyTaskDependenciesTask(PlanTask):
    """
    This task identifies the dependencies between WBS tasks.
    """
    def output(self):
        return self.local_target(FilenameEnum.TASK_DEPENDENCIES_RAW)
    
    def requires(self):
        return {
            'strategic_decisions_markdown': self.clone(StrategicDecisionsMarkdownTask),
            'scenarios_markdown': self.clone(ScenariosMarkdownTask),
            'project_plan': self.clone(ProjectPlanTask),
            'wbs_level2': self.clone(CreateWBSLevel2Task),
            'data_collection': self.clone(DataCollectionTask),
        }
    
    def run_with_llm(self, llm: LLM) -> None:
        logger.info("Identifying task dependencies...")
        
        # Read inputs from required tasks.
        with self.input()['strategic_decisions_markdown']['markdown'].open("r") as f:
            strategic_decisions_markdown = f.read()
        with self.input()['scenarios_markdown']['markdown'].open("r") as f:
            scenarios_markdown = f.read()
        with self.input()['project_plan']['markdown'].open("r") as f:
            project_plan_markdown = f.read()        
        with self.input()['data_collection']['markdown'].open("r") as f:
            data_collection_markdown = f.read()
        with self.input()['wbs_level2']['clean'].open("r") as f:
            major_phases_with_subtasks = json.load(f)
        
        # Build the query
        query = (
            f"File 'strategic_decisions.md':\n{strategic_decisions_markdown}\n\n"
            f"File 'scenarios.md':\n{scenarios_markdown}\n\n"
            f"File 'project_plan.md':\n{project_plan_markdown}\n\n"
            f"File 'Work Breakdown Structure.json':\n{format_json_for_use_in_query(major_phases_with_subtasks)}\n\n"
            f"File 'data_collection.md':\n{data_collection_markdown}"
        )
        
        # Execute the dependency identification.
        identify_dependencies = IdentifyWBSTaskDependencies.execute(llm, query)
        dependencies_raw_dict = identify_dependencies.raw_response_dict()
        
        # Write the raw dependencies JSON to the output file.
        with self.output().open("w") as f:
            json.dump(dependencies_raw_dict, f, indent=2)
        
        logger.info("Task dependencies identified and written to %s", self.output().path)

class EstimateTaskDurationsTask(PlanTask):
    """
    This task estimates durations for WBS tasks in chunks.
    
    It depends on:
      - ProjectPlanTask: providing the project plan JSON.
      - WBSProjectLevel1AndLevel2Task: providing the major phases with subtasks and the task UUIDs.
    
    For each chunk of 3 task IDs, a raw JSON file (e.g. "011-1-task_durations_raw.json") is written,
    and an aggregated JSON file (defined by FilenameEnum.TASK_DURATIONS) is produced.

    IDEA: 1st estimate the Tasks that have zero children.
    2nd estimate tasks that have children where all children have been estimated.
    repeat until all tasks have been estimated.
    """
    def output(self):
        return self.local_target(FilenameEnum.TASK_DURATIONS)
    
    def requires(self):
        return {
            'project_plan': self.clone(ProjectPlanTask),
            'wbs_project': self.clone(WBSProjectLevel1AndLevel2Task),
        }
    
    def run_inner(self):
        llm_executor: LLMExecutor = self.create_llm_executor()

        logger.info("Estimating task durations...")
        
        # Load the project plan JSON.
        with self.input()['project_plan']['raw'].open("r") as f:
            project_plan_dict = json.load(f)

        with self.input()['wbs_project'].open("r") as f:
            wbs_project_dict = json.load(f)
        wbs_project = WBSProject.from_dict(wbs_project_dict)

        # json'ish representation of the major phases in the WBS, and their subtasks.
        root_task = wbs_project.root_task
        major_tasks = [child.to_dict() for child in root_task.task_children]
        major_phases_with_subtasks = major_tasks

        # Don't include uuid of the root task. It's the child tasks that are of interest to estimate.
        decompose_task_id_list = []
        for task in wbs_project.root_task.task_children:
            decompose_task_id_list.extend(task.task_ids())

        logger.info(f"There are {len(decompose_task_id_list)} tasks to be estimated.")
        
        # Split the task IDs into chunks of 3.
        task_ids_chunks = [decompose_task_id_list[i:i + 3] for i in range(0, len(decompose_task_id_list), 3)]
        
        # In production mode, all chunks are processed.
        # In developer mode, truncate to only 2 chunks for fast turnaround cycle. Otherwise LOTS of tasks are to be estimated.
        logger.info(f"EstimateTaskDurationsTask.speedvsdetail: {self.speedvsdetail}")
        if self.speedvsdetail == SpeedVsDetailEnum.FAST_BUT_SKIP_DETAILS:
            logger.info("FAST_BUT_SKIP_DETAILS mode, truncating to 2 chunks for testing.")
            task_ids_chunks = task_ids_chunks[:2]
        else:
            logger.info("Processing all chunks.")

        # Process each chunk.
        accumulated_task_duration_list = []
        for index, task_ids_chunk in enumerate(task_ids_chunks, start=1):
            logger.info("Processing chunk %d of %d", index, len(task_ids_chunks))
            
            query = EstimateWBSTaskDurations.format_query(
                project_plan_dict,
                major_phases_with_subtasks,
                task_ids_chunk
            )
            
            # IDEA: If the chunk file already exist, then there is no need to run the LLM again.
            def execute_estimate_task_durations(llm: LLM) -> EstimateWBSTaskDurations:
                return EstimateWBSTaskDurations.execute(llm, query)

            try:
                estimate_durations = llm_executor.run(execute_estimate_task_durations)
            except PipelineStopRequested:
                # Re-raise PipelineStopRequested without wrapping it
                raise
            except Exception as e:
                logger.error(f"Task durations chunk {index} LLM interaction failed.", exc_info=True)
                raise ValueError(f"Task durations chunk {index} LLM interaction failed.") from e

            durations_raw_dict = estimate_durations.raw_response_dict()
            
            # Write the raw JSON for this chunk.
            filename = FilenameEnum.TASK_DURATIONS_RAW_TEMPLATE.format(index)
            raw_chunk_path = self.run_id_dir / filename
            with open(raw_chunk_path, "w") as f:
                json.dump(durations_raw_dict, f, indent=2)
            
            accumulated_task_duration_list.extend(durations_raw_dict.get('task_details', []))
        
        # Write the aggregated task durations.
        aggregated_path = self.file_path(FilenameEnum.TASK_DURATIONS)
        with open(aggregated_path, "w") as f:
            json.dump(accumulated_task_duration_list, f, indent=2)
        
        logger.info("Task durations estimated and aggregated results written to %s", aggregated_path)

class CreateWBSLevel3Task(PlanTask):
    """
    This task creates the Work Breakdown Structure (WBS) Level 3, by decomposing tasks from Level 2 into subtasks.
    
    It depends on:
      - ProjectPlanTask: provides the project plan JSON.
      - WBSProjectLevel1AndLevel2Task: provides the major phases with subtasks and the task UUIDs.
      - EstimateTaskDurationsTask: provides the aggregated task durations (task_duration_list).
    
    For each task without any subtasks, a query is built and executed using the LLM. 
    The raw JSON result for each task is written to a file using the template from FilenameEnum. 
    Finally, all individual results are accumulated and written as an aggregated JSON file.
    """
    def output(self):
        return self.local_target(FilenameEnum.WBS_LEVEL3)
    
    def requires(self):
        return {
            'project_plan': self.clone(ProjectPlanTask),
            'wbs_project': self.clone(WBSProjectLevel1AndLevel2Task),
            'task_durations': self.clone(EstimateTaskDurationsTask),
            'data_collection': self.clone(DataCollectionTask),
        }
    
    def run_inner(self):
        llm_executor: LLMExecutor = self.create_llm_executor()

        logger.info("Creating Work Breakdown Structure (WBS) Level 3...")
        
        # Read inputs from required tasks.
        with self.input()['project_plan']['raw'].open("r") as f:
            project_plan_dict = json.load(f)

        with self.input()['wbs_project'].open("r") as f:
            wbs_project_dict = json.load(f)
        wbs_project = WBSProject.from_dict(wbs_project_dict)

        with self.input()['data_collection']['markdown'].open("r") as f:
            data_collection_markdown = f.read()

        # Load the estimated task durations.
        task_duration_list_path = self.input()['task_durations'].path
        WBSPopulate.extend_project_with_durations_json(wbs_project, task_duration_list_path)

        # for each task in the wbs_project, find the task that has no children
        tasks_with_no_children = []
        def visit_task(task):
            if len(task.task_children) == 0:
                tasks_with_no_children.append(task)
            else:
                for child in task.task_children:
                    visit_task(child)
        visit_task(wbs_project.root_task)

        # for each task with no children, extract the task_id
        decompose_task_id_list = []
        for task in tasks_with_no_children:
            decompose_task_id_list.append(task.id)
        
        logger.info("There are %d tasks to be decomposed.", len(decompose_task_id_list))

        # In production mode, all chunks are processed.
        # In developer mode, truncate to only 2 chunks for fast turnaround cycle. Otherwise LOTS of tasks are to be decomposed.
        logger.info(f"CreateWBSLevel3Task.speedvsdetail: {self.speedvsdetail}")
        if self.speedvsdetail == SpeedVsDetailEnum.FAST_BUT_SKIP_DETAILS:
            logger.info("FAST_BUT_SKIP_DETAILS mode, truncating to 2 chunks for testing.")
            decompose_task_id_list = decompose_task_id_list[:2]
        else:
            logger.info("Processing all chunks.")
        
        project_plan_str = format_json_for_use_in_query(project_plan_dict)
        wbs_project_str = format_json_for_use_in_query(wbs_project.to_dict())

        # Loop over each task ID.
        wbs_level3_result_accumulated = []
        total_tasks = len(decompose_task_id_list)
        for index, task_id in enumerate(decompose_task_id_list, start=1):
            logger.info("Decomposing task %d of %d", index, total_tasks)
            
            query = (
                f"The project plan:\n{project_plan_str}\n\n"
                f"Data collection:\n{data_collection_markdown}\n\n"
                f"Work breakdown structure:\n{wbs_project_str}\n\n"
                f"Only decompose this task:\n\"{task_id}\""
            )

            # IDEA: If the chunk file already exist, then there is no need to run the LLM again.
            def execute_create_wbs_level3(llm: LLM) -> CreateWBSLevel3:
                return CreateWBSLevel3.execute(llm, query, task_id)

            try:
                create_wbs_level3 = llm_executor.run(execute_create_wbs_level3)
            except PipelineStopRequested:
                # Re-raise PipelineStopRequested without wrapping it
                raise
            except Exception as e:
                logger.error(f"WBS Level 3 task {index} LLM interaction failed.", exc_info=True)
                raise ValueError(f"WBS Level 3 task {index} LLM interaction failed.") from e

            wbs_level3_raw_dict = create_wbs_level3.raw_response_dict()
            
            # Write the raw JSON for this task using the FilenameEnum template.
            raw_filename = FilenameEnum.WBS_LEVEL3_RAW_TEMPLATE.value.format(index)
            raw_chunk_path = self.run_id_dir / raw_filename
            with open(raw_chunk_path, 'w') as f:
                json.dump(wbs_level3_raw_dict, f, indent=2)
            
            # Accumulate the decomposed tasks.
            wbs_level3_result_accumulated.extend(create_wbs_level3.tasks)
        
        # Write the aggregated WBS Level 3 result.
        aggregated_path = self.file_path(FilenameEnum.WBS_LEVEL3)
        with open(aggregated_path, 'w') as f:
            json.dump(wbs_level3_result_accumulated, f, indent=2)
        
        logger.info("WBS Level 3 created and aggregated results written to %s", aggregated_path)

class WBSProjectLevel1AndLevel2AndLevel3Task(PlanTask):
    """
    Create a WBS project from the WBS Level 1 and Level 2 and Level 3 JSON files.
    
    It depends on:
      - WBSProjectLevel1AndLevel2Task: providing the major phases with subtasks and the task UUIDs.
      - CreateWBSLevel3Task: providing the decomposed tasks.
    """
    def output(self):
        return {
            'full': self.local_target(FilenameEnum.WBS_PROJECT_LEVEL1_AND_LEVEL2_AND_LEVEL3_FULL),
            'csv': self.local_target(FilenameEnum.WBS_PROJECT_LEVEL1_AND_LEVEL2_AND_LEVEL3_CSV)
        }
    
    def requires(self):
        return {
            'wbs_project12': self.clone(WBSProjectLevel1AndLevel2Task),
            'wbs_level3': self.clone(CreateWBSLevel3Task),
        }
    
    def run_inner(self):
        wbs_project_path = self.input()['wbs_project12'].path
        with open(wbs_project_path, "r") as f:
            wbs_project_dict = json.load(f)
        wbs_project = WBSProject.from_dict(wbs_project_dict)

        wbs_level3_path = self.input()['wbs_level3'].path
        WBSPopulate.extend_project_with_decomposed_tasks_json(wbs_project, wbs_level3_path)

        json_representation = json.dumps(wbs_project.to_dict(), indent=2)
        with self.output()['full'].open("w") as f:
            f.write(json_representation)
        
        csv_representation = wbs_project.to_csv_string()
        with self.output()['csv'].open("w") as f:
            f.write(csv_representation)

class CreateScheduleTask(PlanTask):
    def output(self):
        return {
            'mermaid_html': self.local_target(FilenameEnum.SCHEDULE_GANTT_MERMAID_HTML),
            'dhtmlx_html': self.local_target(FilenameEnum.SCHEDULE_GANTT_DHTMLX_HTML),
            'machai_csv': self.local_target(FilenameEnum.SCHEDULE_GANTT_MACHAI_CSV)
        }
    
    def requires(self):
        return {
            'start_time': self.clone(StartTimeTask),
            'wbs_level1': self.clone(CreateWBSLevel1Task),
            'dependencies': self.clone(IdentifyTaskDependenciesTask),
            'durations': self.clone(EstimateTaskDurationsTask),
            'wbs_project123': self.clone(WBSProjectLevel1AndLevel2AndLevel3Task)
        }
    
    def run_inner(self):
        # For the report title, use the 'project_title' of the WBS Level 1 result.
        with self.input()['wbs_level1']['project_title'].open("r") as f:
            title = f.read()
        with self.input()['dependencies'].open("r") as f:
            dependencies_dict = json.load(f)
        with self.input()['durations'].open("r") as f:
            duration_list: list[dict[str, Any]] = json.load(f)
        wbs_project_path = self.input()['wbs_project123']['full'].path
        with open(wbs_project_path, "r") as f:
            wbs_project_dict = json.load(f)
        wbs_project = WBSProject.from_dict(wbs_project_dict)

        # Read the start time from the StartTimeTask to get the actual pipeline start date
        with self.input()['start_time'].open("r") as f:
            start_time_dict = json.load(f)

        # The start_time.server_iso_utc is in format "YYYY-MM-DDTHH:MM:SSZ"
        utc_timestamp = start_time_dict['server_iso_utc']
        # The 'Z' suffix for UTC is not supported by fromisoformat() in Python < 3.11. Replace, ensures compatibility.
        project_start_dt: datetime = datetime.fromisoformat(utc_timestamp.replace('Z', '+00:00'))
        project_start: date = project_start_dt.date()

        # logger.debug(f"dependencies_dict {dependencies_dict}")
        # logger.debug(f"duration_list {duration_list}")
        # logger.debug(f"wbs_project {wbs_project.to_dict()}")

        # Tooltips with a detailed description of each task.
        task_id_to_html_tooltip_dict: dict[str, str] = WBSTaskTooltip.html_tooltips(wbs_project)
        task_id_to_text_tooltip_dict: dict[str, str] = WBSTaskTooltip.text_tooltips(wbs_project)

        project_schedule: ProjectSchedule = ProjectSchedulePopulator.populate(
            wbs_project=wbs_project,
            duration_list=duration_list
        )

        # Export the Gantt chart to CSV.
        # Always run the CSV export so that the code gets exercised, otherwise the code will rot.
        csv_data: str = ExportGanttCSV.to_gantt_csv(
            project_schedule=project_schedule, 
            project_start=project_start,
            task_id_to_tooltip_dict=task_id_to_text_tooltip_dict
        )
        if PIPELINE_CONFIG.enable_csv_export == False:
            # When disabled, then hide the "Export to CSV" button and don't embed the CSV data in the html report.
            csv_data = None

        ExportGanttCSV.save(
            project_schedule=project_schedule, 
            path=self.output()['machai_csv'].path,
            project_start=project_start,
            task_id_to_tooltip_dict=task_id_to_text_tooltip_dict
        )

        # Identify the tasks that should be treated as project activities.
        task_ids_to_treat_as_project_activities = wbs_project.task_ids_with_one_or_more_children()

        # Export the Gantt chart to Frappe.
        # I'm disappointed by Frappe, it lacks a lot of features that are present in DHTMLX.
        # ExportGanttFrappe.save(
        #     project_schedule=project_schedule, 
        #     path=self.output()['frappe_html'].path, 
        #     project_start=project_start,
        #     task_ids_to_treat_as_project_activities=task_ids_to_treat_as_project_activities
        # )

        # Export the Gantt chart to Mermaid.
        ExportGanttMermaid.save(
            project_schedule=project_schedule, 
            path=self.output()['mermaid_html'].path, 
            project_start=project_start
        )

        # Export the Gantt chart to DHTMLX.
        ExportGanttDHTMLX.save(
            project_schedule=project_schedule, 
            path=self.output()['dhtmlx_html'].path, 
            project_start=project_start,
            task_ids_to_treat_as_project_activities=task_ids_to_treat_as_project_activities,
            task_id_to_tooltip_dict=task_id_to_html_tooltip_dict,
            title=title,
            csv_data=csv_data
        )

class ReviewPlanTask(PlanTask):
    """
    Ask questions about the almost finished plan.
    """
    def output(self):
        return {
            'raw': self.local_target(FilenameEnum.REVIEW_PLAN_RAW),
            'markdown': self.local_target(FilenameEnum.REVIEW_PLAN_MARKDOWN)
        }
    
    def requires(self):
        return {
            'strategic_decisions_markdown': self.clone(StrategicDecisionsMarkdownTask),
            'scenarios_markdown': self.clone(ScenariosMarkdownTask),
            'consolidate_assumptions_markdown': self.clone(ConsolidateAssumptionsMarkdownTask),
            'project_plan': self.clone(ProjectPlanTask),
            'data_collection': self.clone(DataCollectionTask),
            'related_resources': self.clone(RelatedResourcesTask),
            'swot_analysis': self.clone(SWOTAnalysisTask),
            'team_markdown': self.clone(TeamMarkdownTask),
            'pitch_markdown': self.clone(ConvertPitchToMarkdownTask),
            'expert_review': self.clone(ExpertReviewTask),
            'wbs_project123': self.clone(WBSProjectLevel1AndLevel2AndLevel3Task)
        }
    
    def run_inner(self):
        llm_executor: LLMExecutor = self.create_llm_executor()

        # Read inputs from required tasks.
        with self.input()['strategic_decisions_markdown']['markdown'].open("r") as f:
            strategic_decisions_markdown = f.read()
        with self.input()['scenarios_markdown']['markdown'].open("r") as f:
            scenarios_markdown = f.read()
        with self.input()['consolidate_assumptions_markdown']['short'].open("r") as f:
            assumptions_markdown = f.read()
        with self.input()['project_plan']['markdown'].open("r") as f:
            project_plan_markdown = f.read()
        with self.input()['data_collection']['markdown'].open("r") as f:
            data_collection_markdown = f.read()
        with self.input()['related_resources']['markdown'].open("r") as f:
            related_resources_markdown = f.read()
        with self.input()['swot_analysis']['markdown'].open("r") as f:
            swot_analysis_markdown = f.read()
        with self.input()['team_markdown'].open("r") as f:
            team_markdown = f.read()
        with self.input()['pitch_markdown']['markdown'].open("r") as f:
            pitch_markdown = f.read()
        with self.input()['expert_review'].open("r") as f:
            expert_review = f.read()
        with self.input()['wbs_project123']['csv'].open("r") as f:
            wbs_project_csv = f.read()

        # Build the query.
        query = (
            f"File 'strategic_decisions.md':\n{strategic_decisions_markdown}\n\n"
            f"File 'scenarios.md':\n{scenarios_markdown}\n\n"
            f"File 'assumptions.md':\n{assumptions_markdown}\n\n"
            f"File 'project-plan.md':\n{project_plan_markdown}\n\n"
            f"File 'data-collection.md':\n{data_collection_markdown}\n\n"
            f"File 'related-resources.md':\n{related_resources_markdown}\n\n"
            f"File 'swot-analysis.md':\n{swot_analysis_markdown}\n\n"
            f"File 'team.md':\n{team_markdown}\n\n"
            f"File 'pitch.md':\n{pitch_markdown}\n\n"
            f"File 'expert-review.md':\n{expert_review}\n\n"
            f"File 'work-breakdown-structure.csv':\n{wbs_project_csv}"
        )

        # Perform the review.
        review_plan = ReviewPlan.execute(llm_executor=llm_executor, document=query, speed_vs_detail=self.speedvsdetail)

        # Save the results.
        json_path = self.output()['raw'].path
        review_plan.save_raw(json_path)
        markdown_path = self.output()['markdown'].path
        review_plan.save_markdown(markdown_path)


class ExecutiveSummaryTask(PlanTask):
    """
    Create an executive summary of the plan.    
    """
    def output(self):
        return {
            'raw': self.local_target(FilenameEnum.EXECUTIVE_SUMMARY_RAW),
            'markdown': self.local_target(FilenameEnum.EXECUTIVE_SUMMARY_MARKDOWN)
        }
    
    def requires(self):
        return {
            'strategic_decisions_markdown': self.clone(StrategicDecisionsMarkdownTask),
            'scenarios_markdown': self.clone(ScenariosMarkdownTask),
            'consolidate_assumptions_markdown': self.clone(ConsolidateAssumptionsMarkdownTask),
            'project_plan': self.clone(ProjectPlanTask),
            'data_collection': self.clone(DataCollectionTask),
            'related_resources': self.clone(RelatedResourcesTask),
            'swot_analysis': self.clone(SWOTAnalysisTask),
            'team_markdown': self.clone(TeamMarkdownTask),
            'pitch_markdown': self.clone(ConvertPitchToMarkdownTask),
            'expert_review': self.clone(ExpertReviewTask),
            'wbs_project123': self.clone(WBSProjectLevel1AndLevel2AndLevel3Task),
            'review_plan': self.clone(ReviewPlanTask)
        }
    
    def run_with_llm(self, llm: LLM) -> None:
        # Read inputs from required tasks.
        with self.input()['strategic_decisions_markdown']['markdown'].open("r") as f:
            strategic_decisions_markdown = f.read()
        with self.input()['scenarios_markdown']['markdown'].open("r") as f:
            scenarios_markdown = f.read()
        with self.input()['consolidate_assumptions_markdown']['short'].open("r") as f:
            assumptions_markdown = f.read()
        with self.input()['project_plan']['markdown'].open("r") as f:
            project_plan_markdown = f.read()
        with self.input()['data_collection']['markdown'].open("r") as f:
            data_collection_markdown = f.read()
        with self.input()['related_resources']['markdown'].open("r") as f:
            related_resources_markdown = f.read()
        with self.input()['swot_analysis']['markdown'].open("r") as f:
            swot_analysis_markdown = f.read()
        with self.input()['team_markdown'].open("r") as f:
            team_markdown = f.read()
        with self.input()['pitch_markdown']['markdown'].open("r") as f:
            pitch_markdown = f.read()
        with self.input()['expert_review'].open("r") as f:
            expert_review = f.read()
        with self.input()['wbs_project123']['csv'].open("r") as f:
            wbs_project_csv = f.read()
        with self.input()['review_plan']['markdown'].open("r") as f:
            review_plan_markdown = f.read()

        # Build the query.
        query = (
            f"File 'strategic_decisions.md':\n{strategic_decisions_markdown}\n\n"
            f"File 'scenarios.md':\n{scenarios_markdown}\n\n"
            f"File 'assumptions.md':\n{assumptions_markdown}\n\n"
            f"File 'project-plan.md':\n{project_plan_markdown}\n\n"
            f"File 'data-collection.md':\n{data_collection_markdown}\n\n"
            f"File 'related-resources.md':\n{related_resources_markdown}\n\n"
            f"File 'swot-analysis.md':\n{swot_analysis_markdown}\n\n"
            f"File 'team.md':\n{team_markdown}\n\n"
            f"File 'pitch.md':\n{pitch_markdown}\n\n"
            f"File 'expert-review.md':\n{expert_review}\n\n"
            f"File 'work-breakdown-structure.csv':\n{wbs_project_csv}\n\n"
            f"File 'review-plan.md':\n{review_plan_markdown}"
        )

        # Create the executive summary.
        executive_summary = ExecutiveSummary.execute(llm, query)

        # Save the results.
        json_path = self.output()['raw'].path
        executive_summary.save_raw(json_path)
        markdown_path = self.output()['markdown'].path
        executive_summary.save_markdown(markdown_path)


class QuestionsAndAnswersTask(PlanTask):
    def output(self):
        return {
            'raw': self.local_target(FilenameEnum.QUESTIONS_AND_ANSWERS_RAW),
            'markdown': self.local_target(FilenameEnum.QUESTIONS_AND_ANSWERS_MARKDOWN),
            'html': self.local_target(FilenameEnum.QUESTIONS_AND_ANSWERS_HTML)
        }
    
    def requires(self):
        return {
            'strategic_decisions_markdown': self.clone(StrategicDecisionsMarkdownTask),
            'scenarios_markdown': self.clone(ScenariosMarkdownTask),
            'consolidate_assumptions_markdown': self.clone(ConsolidateAssumptionsMarkdownTask),
            'team_markdown': self.clone(TeamMarkdownTask),
            'related_resources': self.clone(RelatedResourcesTask),
            'consolidate_governance': self.clone(ConsolidateGovernanceTask),
            'swot_analysis': self.clone(SWOTAnalysisTask),
            'pitch_markdown': self.clone(ConvertPitchToMarkdownTask),
            'data_collection': self.clone(DataCollectionTask),
            'documents_to_create_and_find': self.clone(MarkdownWithDocumentsToCreateAndFindTask),
            'wbs_project123': self.clone(WBSProjectLevel1AndLevel2AndLevel3Task),
            'expert_review': self.clone(ExpertReviewTask),
            'project_plan': self.clone(ProjectPlanTask),
            'review_plan': self.clone(ReviewPlanTask),
        }
    
    def run_with_llm(self, llm: LLM) -> None:
        # Read inputs from required tasks.
        with self.input()['strategic_decisions_markdown']['markdown'].open("r") as f:
            strategic_decisions_markdown = f.read()
        with self.input()['scenarios_markdown']['markdown'].open("r") as f:
            scenarios_markdown = f.read()
        with self.input()['consolidate_assumptions_markdown']['short'].open("r") as f:
            assumptions_markdown = f.read()
        with self.input()['project_plan']['markdown'].open("r") as f:
            project_plan_markdown = f.read()
        with self.input()['data_collection']['markdown'].open("r") as f:
            data_collection_markdown = f.read()
        with self.input()['related_resources']['markdown'].open("r") as f:
            related_resources_markdown = f.read()
        with self.input()['swot_analysis']['markdown'].open("r") as f:
            swot_analysis_markdown = f.read()
        with self.input()['team_markdown'].open("r") as f:
            team_markdown = f.read()
        with self.input()['pitch_markdown']['markdown'].open("r") as f:
            pitch_markdown = f.read()
        with self.input()['expert_review'].open("r") as f:
            expert_review = f.read()
        with self.input()['wbs_project123']['csv'].open("r") as f:
            wbs_project_csv = f.read()
        with self.input()['review_plan']['markdown'].open("r") as f:
            review_plan_markdown = f.read()

        # Build the query.
        query = (
            f"File 'strategic_decisions.md':\n{strategic_decisions_markdown}\n\n"
            f"File 'scenarios.md':\n{scenarios_markdown}\n\n"
            f"File 'assumptions.md':\n{assumptions_markdown}\n\n"
            f"File 'project-plan.md':\n{project_plan_markdown}\n\n"
            f"File 'data-collection.md':\n{data_collection_markdown}\n\n"
            f"File 'related-resources.md':\n{related_resources_markdown}\n\n"
            f"File 'swot-analysis.md':\n{swot_analysis_markdown}\n\n"
            f"File 'team.md':\n{team_markdown}\n\n"
            f"File 'pitch.md':\n{pitch_markdown}\n\n"
            f"File 'expert-review.md':\n{expert_review}\n\n"
            f"File 'work-breakdown-structure.csv':\n{wbs_project_csv}\n\n"
            f"File 'review-plan.md':\n{review_plan_markdown}"
        )

        # Invoke the LLM
        question_answers = QuestionsAnswers.execute(llm, query)

        # Save the results.
        json_path = self.output()['raw'].path
        question_answers.save_raw(json_path)
        markdown_path = self.output()['markdown'].path
        question_answers.save_markdown(markdown_path)
        html_path = self.output()['html'].path
        question_answers.save_html(html_path)

class PremortemTask(PlanTask):
    def output(self):
        return {
            'raw': self.local_target(FilenameEnum.PREMORTEM_RAW),
            'markdown': self.local_target(FilenameEnum.PREMORTEM_MARKDOWN)
        }
    
    def requires(self):
        return {
            'strategic_decisions_markdown': self.clone(StrategicDecisionsMarkdownTask),
            'scenarios_markdown': self.clone(ScenariosMarkdownTask),
            'consolidate_assumptions_markdown': self.clone(ConsolidateAssumptionsMarkdownTask),
            'team_markdown': self.clone(TeamMarkdownTask),
            'related_resources': self.clone(RelatedResourcesTask),
            'consolidate_governance': self.clone(ConsolidateGovernanceTask),
            'swot_analysis': self.clone(SWOTAnalysisTask),
            'pitch_markdown': self.clone(ConvertPitchToMarkdownTask),
            'data_collection': self.clone(DataCollectionTask),
            'documents_to_create_and_find': self.clone(MarkdownWithDocumentsToCreateAndFindTask),
            'wbs_project123': self.clone(WBSProjectLevel1AndLevel2AndLevel3Task),
            'expert_review': self.clone(ExpertReviewTask),
            'project_plan': self.clone(ProjectPlanTask),
            'review_plan': self.clone(ReviewPlanTask),
            'questions_and_answers': self.clone(QuestionsAndAnswersTask)
        }
    
    def run_inner(self):
        llm_executor: LLMExecutor = self.create_llm_executor()

        # Read inputs from required tasks.
        with self.input()['strategic_decisions_markdown']['markdown'].open("r") as f:
            strategic_decisions_markdown = f.read()
        with self.input()['scenarios_markdown']['markdown'].open("r") as f:
            scenarios_markdown = f.read()
        with self.input()['consolidate_assumptions_markdown']['short'].open("r") as f:
            assumptions_markdown = f.read()
        with self.input()['project_plan']['markdown'].open("r") as f:
            project_plan_markdown = f.read()
        with self.input()['data_collection']['markdown'].open("r") as f:
            data_collection_markdown = f.read()
        with self.input()['related_resources']['markdown'].open("r") as f:
            related_resources_markdown = f.read()
        with self.input()['swot_analysis']['markdown'].open("r") as f:
            swot_analysis_markdown = f.read()
        with self.input()['team_markdown'].open("r") as f:
            team_markdown = f.read()
        with self.input()['pitch_markdown']['markdown'].open("r") as f:
            pitch_markdown = f.read()
        with self.input()['expert_review'].open("r") as f:
            expert_review = f.read()
        with self.input()['wbs_project123']['csv'].open("r") as f:
            wbs_project_csv = f.read()
        with self.input()['review_plan']['markdown'].open("r") as f:
            review_plan_markdown = f.read()
        with self.input()['questions_and_answers']['markdown'].open("r") as f:
            questions_and_answers_markdown = f.read()

        # Build the query.
        query = (
            f"File 'strategic_decisions.md':\n{strategic_decisions_markdown}\n\n"
            f"File 'scenarios.md':\n{scenarios_markdown}\n\n"
            f"File 'assumptions.md':\n{assumptions_markdown}\n\n"
            f"File 'project-plan.md':\n{project_plan_markdown}\n\n"
            f"File 'data-collection.md':\n{data_collection_markdown}\n\n"
            f"File 'related-resources.md':\n{related_resources_markdown}\n\n"
            f"File 'swot-analysis.md':\n{swot_analysis_markdown}\n\n"
            f"File 'team.md':\n{team_markdown}\n\n"
            f"File 'pitch.md':\n{pitch_markdown}\n\n"
            f"File 'expert-review.md':\n{expert_review}\n\n"
            f"File 'work-breakdown-structure.csv':\n{wbs_project_csv}\n\n"
            f"File 'review-plan.md':\n{review_plan_markdown}\n\n"
            f"File 'questions-and-answers.md':\n{questions_and_answers_markdown}"
        )

        # Invoke the LLM
        premortem = Premortem.execute(llm_executor=llm_executor, speed_vs_detail=self.speedvsdetail, user_prompt=query)

        # Save the results.
        json_path = self.output()['raw'].path
        premortem.save_raw(json_path)
        markdown_path = self.output()['markdown'].path
        premortem.save_markdown(markdown_path)


class SelfAuditTask(PlanTask):
    def output(self):
        return {
            'raw': self.local_target(FilenameEnum.SELF_AUDIT_RAW),
            'markdown': self.local_target(FilenameEnum.SELF_AUDIT_MARKDOWN)
        }
    
    def requires(self):
        return {
            'strategic_decisions_markdown': self.clone(StrategicDecisionsMarkdownTask),
            'scenarios_markdown': self.clone(ScenariosMarkdownTask),
            'consolidate_assumptions_markdown': self.clone(ConsolidateAssumptionsMarkdownTask),
            'team_markdown': self.clone(TeamMarkdownTask),
            'related_resources': self.clone(RelatedResourcesTask),
            'consolidate_governance': self.clone(ConsolidateGovernanceTask),
            'swot_analysis': self.clone(SWOTAnalysisTask),
            'pitch_markdown': self.clone(ConvertPitchToMarkdownTask),
            'data_collection': self.clone(DataCollectionTask),
            'documents_to_create_and_find': self.clone(MarkdownWithDocumentsToCreateAndFindTask),
            'wbs_project123': self.clone(WBSProjectLevel1AndLevel2AndLevel3Task),
            'expert_review': self.clone(ExpertReviewTask),
            'project_plan': self.clone(ProjectPlanTask),
            'review_plan': self.clone(ReviewPlanTask),
            'questions_and_answers': self.clone(QuestionsAndAnswersTask),
            'premortem': self.clone(PremortemTask)
        }
    
    def run_inner(self):
        llm_executor: LLMExecutor = self.create_llm_executor()

        # Read inputs from required tasks.
        with self.input()['strategic_decisions_markdown']['markdown'].open("r") as f:
            strategic_decisions_markdown = f.read()
        with self.input()['scenarios_markdown']['markdown'].open("r") as f:
            scenarios_markdown = f.read()
        with self.input()['consolidate_assumptions_markdown']['short'].open("r") as f:
            assumptions_markdown = f.read()
        with self.input()['project_plan']['markdown'].open("r") as f:
            project_plan_markdown = f.read()
        with self.input()['data_collection']['markdown'].open("r") as f:
            data_collection_markdown = f.read()
        with self.input()['related_resources']['markdown'].open("r") as f:
            related_resources_markdown = f.read()
        with self.input()['swot_analysis']['markdown'].open("r") as f:
            swot_analysis_markdown = f.read()
        with self.input()['team_markdown'].open("r") as f:
            team_markdown = f.read()
        with self.input()['pitch_markdown']['markdown'].open("r") as f:
            pitch_markdown = f.read()
        with self.input()['expert_review'].open("r") as f:
            expert_review = f.read()
        with self.input()['wbs_project123']['csv'].open("r") as f:
            wbs_project_csv = f.read()
        with self.input()['review_plan']['markdown'].open("r") as f:
            review_plan_markdown = f.read()
        with self.input()['questions_and_answers']['markdown'].open("r") as f:
            questions_and_answers_markdown = f.read()
        with self.input()['premortem']['markdown'].open("r") as f:
            premortem_markdown = f.read()

        # Build the query.
        user_prompt = (
            f"File 'strategic_decisions.md':\n{strategic_decisions_markdown}\n\n"
            f"File 'scenarios.md':\n{scenarios_markdown}\n\n"
            f"File 'assumptions.md':\n{assumptions_markdown}\n\n"
            f"File 'project-plan.md':\n{project_plan_markdown}\n\n"
            f"File 'data-collection.md':\n{data_collection_markdown}\n\n"
            f"File 'related-resources.md':\n{related_resources_markdown}\n\n"
            f"File 'swot-analysis.md':\n{swot_analysis_markdown}\n\n"
            f"File 'team.md':\n{team_markdown}\n\n"
            f"File 'pitch.md':\n{pitch_markdown}\n\n"
            f"File 'expert-review.md':\n{expert_review}\n\n"
            f"File 'work-breakdown-structure.csv':\n{wbs_project_csv}\n\n"
            f"File 'review-plan.md':\n{review_plan_markdown}\n\n"
            f"File 'questions-and-answers.md':\n{questions_and_answers_markdown}\n\n"
            f"File 'premortem.md':\n{premortem_markdown}"
        )

        logger.info(f"SelfAuditTask.speedvsdetail: {self.speedvsdetail}")
        max_number_of_items: Optional[int] = None
        if self.speedvsdetail == SpeedVsDetailEnum.FAST_BUT_SKIP_DETAILS:
            logger.info("FAST_BUT_SKIP_DETAILS mode, truncating to 2 items for testing a subset of the SelfAudit items.")
            max_number_of_items = 2
        else:
            logger.info("Processing all SelfAudit items.")

        # Invoke the LLM
        self_audit = SelfAudit.execute(
            llm_executor=llm_executor, 
            user_prompt=user_prompt,
            max_number_of_items=max_number_of_items,
        )

        # Save the results.
        json_path = self.output()['raw'].path
        self_audit.save_raw(json_path)
        markdown_path = self.output()['markdown'].path
        self_audit.save_markdown(markdown_path)

class ReportTask(PlanTask):
    """
    Generate a report html document.
    """
    def output(self):
        return self.local_target(FilenameEnum.REPORT)
    
    def requires(self):
        return {
            'setup': self.clone(SetupTask),
            'redline_gate': self.clone(RedlineGateTask),
            'premise_attack': self.clone(PremiseAttackTask),
            'strategic_decisions_markdown': self.clone(StrategicDecisionsMarkdownTask),
            'scenarios_markdown': self.clone(ScenariosMarkdownTask),
            'consolidate_assumptions_markdown': self.clone(ConsolidateAssumptionsMarkdownTask),
            'team_markdown': self.clone(TeamMarkdownTask),
            'related_resources': self.clone(RelatedResourcesTask),
            'consolidate_governance': self.clone(ConsolidateGovernanceTask),
            'swot_analysis': self.clone(SWOTAnalysisTask),
            'pitch_markdown': self.clone(ConvertPitchToMarkdownTask),
            'data_collection': self.clone(DataCollectionTask),
            'documents_to_create_and_find': self.clone(MarkdownWithDocumentsToCreateAndFindTask),
            'wbs_level1': self.clone(CreateWBSLevel1Task),
            'wbs_project123': self.clone(WBSProjectLevel1AndLevel2AndLevel3Task),
            'expert_review': self.clone(ExpertReviewTask),
            'project_plan': self.clone(ProjectPlanTask),
            'review_plan': self.clone(ReviewPlanTask),
            'executive_summary': self.clone(ExecutiveSummaryTask),
            'create_schedule': self.clone(CreateScheduleTask),
            'questions_and_answers': self.clone(QuestionsAndAnswersTask),
            'premortem': self.clone(PremortemTask),
            'self_audit': self.clone(SelfAuditTask)
        }
    
    def run_inner(self):
        # For the report title, use the 'project_title' of the WBS Level 1 result.
        with self.input()['wbs_level1']['project_title'].open("r") as f:
            title = f.read()

        rg = ReportGenerator()
        rg.append_markdown('Executive Summary', self.input()['executive_summary']['markdown'].path)
        rg.append_html('Gantt Overview', self.input()['create_schedule']['mermaid_html'].path)
        rg.append_html('Gantt Interactive', self.input()['create_schedule']['dhtmlx_html'].path)
        rg.append_markdown('Pitch', self.input()['pitch_markdown']['markdown'].path)
        rg.append_markdown('Project Plan', self.input()['project_plan']['markdown'].path)
        rg.append_markdown('Strategic Decisions', self.input()['strategic_decisions_markdown']['markdown'].path)
        rg.append_markdown('Scenarios', self.input()['scenarios_markdown']['markdown'].path)
        rg.append_markdown('Assumptions', self.input()['consolidate_assumptions_markdown']['full'].path)
        rg.append_markdown('Governance', self.input()['consolidate_governance'].path)
        rg.append_markdown('Related Resources', self.input()['related_resources']['markdown'].path)
        rg.append_markdown('Data Collection', self.input()['data_collection']['markdown'].path)
        rg.append_markdown('Documents to Create and Find', self.input()['documents_to_create_and_find'].path)
        rg.append_markdown('SWOT Analysis', self.input()['swot_analysis']['markdown'].path)
        rg.append_markdown('Team', self.input()['team_markdown'].path)
        rg.append_markdown('Expert Criticism', self.input()['expert_review'].path)
        rg.append_csv('Work Breakdown Structure', self.input()['wbs_project123']['csv'].path)
        rg.append_markdown('Review Plan', self.input()['review_plan']['markdown'].path)
        rg.append_html('Questions & Answers', self.input()['questions_and_answers']['html'].path)
        rg.append_markdown_with_tables('Premortem', self.input()['premortem']['markdown'].path)
        rg.append_markdown_with_tables('Self Audit', self.input()['self_audit']['markdown'].path)
        rg.append_initial_prompt_vetted(
            document_title='Initial Prompt Vetted', 
            initial_prompt_file_path=self.input()['setup'].path, 
            redline_gate_markdown_file_path=self.input()['redline_gate']['markdown'].path, 
            premise_attack_markdown_file_path=self.input()['premise_attack']['markdown'].path
        )
        rg.save_report(self.output().path, title=title, execute_plan_section_hidden=REPORT_EXECUTE_PLAN_SECTION_HIDDEN)

class FullPlanPipeline(PlanTask):
    def requires(self):
        return {
            'start_time': self.clone(StartTimeTask),
            'setup': self.clone(SetupTask),
            'redline_gate': self.clone(RedlineGateTask),
            'premise_attack': self.clone(PremiseAttackTask),
            'identify_purpose': self.clone(IdentifyPurposeTask),
            'plan_type': self.clone(PlanTypeTask),
            'potential_levers': self.clone(PotentialLeversTask),
            'deduplicate_levers': self.clone(DeduplicateLeversTask),
            'enriched_levers': self.clone(EnrichLeversTask),
            'focus_on_vital_few_levers': self.clone(FocusOnVitalFewLeversTask),
            'strategic_decisions_markdown': self.clone(StrategicDecisionsMarkdownTask),
            'candidate_scenarios': self.clone(CandidateScenariosTask),
            'select_scenario': self.clone(SelectScenarioTask),
            'scenarios_markdown': self.clone(ScenariosMarkdownTask),
            'physical_locations': self.clone(PhysicalLocationsTask),
            'currency_strategy': self.clone(CurrencyStrategyTask),
            'identify_risks': self.clone(IdentifyRisksTask),
            'make_assumptions': self.clone(MakeAssumptionsTask),
            'assumptions': self.clone(DistillAssumptionsTask),
            'review_assumptions': self.clone(ReviewAssumptionsTask),
            'consolidate_assumptions_markdown': self.clone(ConsolidateAssumptionsMarkdownTask),
            'pre_project_assessment': self.clone(PreProjectAssessmentTask),
            'project_plan': self.clone(ProjectPlanTask),
            'governance_phase1_audit': self.clone(GovernancePhase1AuditTask),
            'governance_phase2_bodies': self.clone(GovernancePhase2BodiesTask),
            'governance_phase3_impl_plan': self.clone(GovernancePhase3ImplPlanTask),
            'governance_phase4_decision_escalation_matrix': self.clone(GovernancePhase4DecisionEscalationMatrixTask),
            'governance_phase5_monitoring_progress': self.clone(GovernancePhase5MonitoringProgressTask),
            'governance_phase6_extra': self.clone(GovernancePhase6ExtraTask),
            'consolidate_governance': self.clone(ConsolidateGovernanceTask),
            'related_resources': self.clone(RelatedResourcesTask),
            'find_team_members': self.clone(FindTeamMembersTask),
            'enrich_team_members_with_contract_type': self.clone(EnrichTeamMembersWithContractTypeTask),
            'enrich_team_members_with_background_story': self.clone(EnrichTeamMembersWithBackgroundStoryTask),
            'enrich_team_members_with_environment_info': self.clone(EnrichTeamMembersWithEnvironmentInfoTask),
            'review_team': self.clone(ReviewTeamTask),
            'team_markdown': self.clone(TeamMarkdownTask),
            'swot_analysis': self.clone(SWOTAnalysisTask),
            'expert_review': self.clone(ExpertReviewTask),
            'data_collection': self.clone(DataCollectionTask),
            'identified_documents': self.clone(IdentifyDocumentsTask),
            'filter_documents_to_find': self.clone(FilterDocumentsToFindTask),
            'filter_documents_to_create': self.clone(FilterDocumentsToCreateTask),
            'draft_documents_to_find': self.clone(DraftDocumentsToFindTask),
            'draft_documents_to_create': self.clone(DraftDocumentsToCreateTask),
            'documents_to_create_and_find': self.clone(MarkdownWithDocumentsToCreateAndFindTask),
            'wbs_level1': self.clone(CreateWBSLevel1Task),
            'wbs_level2': self.clone(CreateWBSLevel2Task),
            'wbs_project12': self.clone(WBSProjectLevel1AndLevel2Task),
            'pitch_raw': self.clone(CreatePitchTask),
            'pitch_markdown': self.clone(ConvertPitchToMarkdownTask),
            'dependencies': self.clone(IdentifyTaskDependenciesTask),
            'durations': self.clone(EstimateTaskDurationsTask),
            'wbs_level3': self.clone(CreateWBSLevel3Task),
            'wbs_project123': self.clone(WBSProjectLevel1AndLevel2AndLevel3Task),
            'plan_evaluator': self.clone(ReviewPlanTask),
            'executive_summary': self.clone(ExecutiveSummaryTask),
            'create_schedule': self.clone(CreateScheduleTask),
            'questions_and_answers': self.clone(QuestionsAndAnswersTask),
            'premortem': self.clone(PremortemTask),
            'self_audit': self.clone(SelfAuditTask),
            'report': self.clone(ReportTask),
        }

    def output(self):
        return self.local_target(FilenameEnum.PIPELINE_COMPLETE)

    def run_inner(self):
        with self.output().open("w") as f:
            f.write("Full pipeline executed successfully.\n")


@dataclass
class PipelineProgress:
    progress_message: str
    progress_percentage: float


@dataclass
class HandleTaskCompletionParameters:
    task: PlanTask
    progress: PipelineProgress
    duration: float


@dataclass
class ExecutePipeline:
    run_id_dir: Path
    speedvsdetail: SpeedVsDetailEnum
    llm_models: list[str]
    full_plan_pipeline_task: Optional[FullPlanPipeline] = field(default=None)
    all_expected_filenames: list[str] = field(default_factory=list)
    luigi_build_return_value: Optional[bool] = field(default=None, init=False)

    def setup(self) -> None:
        # Check that the run_id_dir exists and contains the required files.
        if not self.run_id_dir.exists():
            raise FileNotFoundError(f"The run_id_dir does not exist: {self.run_id_dir!r}")
        if not self.run_id_dir.is_dir():
            raise NotADirectoryError(f"The run_id_dir is not a directory: {self.run_id_dir!r}")
        if not (self.run_id_dir / FilenameEnum.START_TIME.value).exists():
            raise FileNotFoundError(f"The '{FilenameEnum.START_TIME.value}' file does not exist in the run_id_dir: {self.run_id_dir!r}")
        if not (self.run_id_dir / FilenameEnum.INITIAL_PLAN.value).exists():
            raise FileNotFoundError(f"The '{FilenameEnum.INITIAL_PLAN.value}' file does not exist in the run_id_dir: {self.run_id_dir!r}")

        full_plan_pipeline_task = FullPlanPipeline(
            run_id_dir=self.run_id_dir, 
            speedvsdetail=self.speedvsdetail, 
            llm_models=self.llm_models, 
            _pipeline_executor_callback=self.callback_run_task
        )
        self.full_plan_pipeline_task = full_plan_pipeline_task

        # Obtain a list of all the expected output files of the FullPlanPipeline task and all its dependencies
        obtain_output_files = ObtainOutputFiles.execute(full_plan_pipeline_task)
        self.all_expected_filenames = obtain_output_files.get_all_filenames()

    def resolve_luigi_workers(self) -> int:
        default_workers = 1
        try:
            llm_config = PlanExeLLMConfig.load()
        except Exception as exc:
            logger.warning(f"Could not load llm_config.json; defaulting Luigi workers to {default_workers}: {exc}")
            return default_workers

        workers_candidates: list[int] = []
        for llm_name in self.llm_models:
            config = llm_config.llm_config_dict.get(llm_name)
            if not isinstance(config, dict):
                continue
            luigi_workers_value = config.get("luigi_workers")
            if luigi_workers_value is None:
                continue
            try:
                luigi_workers_int = int(luigi_workers_value)
            except (TypeError, ValueError):
                logger.warning(f"Invalid luigi_workers for {llm_name!r}: {luigi_workers_value!r}")
                continue
            if luigi_workers_int < 1:
                logger.warning(f"Invalid luigi_workers for {llm_name!r}: {luigi_workers_int!r}")
                continue
            workers_candidates.append(luigi_workers_int)

        if not workers_candidates:
            return default_workers
        return min(workers_candidates)
    
    @classmethod
    def resolve_llm_models(cls, specified_llm_model: Optional[str]) -> list[str]:
        llm_models = get_llm_names_by_priority()
        if len(llm_models) == 0:
            logger.error("No LLM models found. Please check your llm_config.json file and add 'priority' values.")
            llm_models = [DEFAULT_LLM_MODEL]

        if specified_llm_model:
            llm_model = specified_llm_model
            logger.info(f"Using the specified LLM model: {llm_model!r}")
            if llm_model != SPECIAL_AUTO_ID:
                if not is_valid_llm_name(llm_model):
                    logger.error(f"Invalid LLM model: {llm_model!r}. Please check your llm_config.json file and add the model.")
                    raise ValueError(f"Invalid LLM model: {llm_model!r}. Please check your llm_config.json file and add the model.")
                llm_models = [llm_model]

        logger.info("These are the LLM models that will be used in the pipeline:")
        for index, llm_name in enumerate(llm_models):
            logger.info(f"{index}. {llm_name!r}")
        return llm_models
    
    def get_progress_percentage(self) -> PipelineProgress:
        files = []
        try:
            if self.run_id_dir.exists() and self.run_id_dir.is_dir():
                files = [f.name for f in self.run_id_dir.iterdir()]
        except OSError as e:
            logger.warning(f"Could not list files in {run_id_dir}: {e}")

        ignore_files = [
            ExtraFilenameEnum.EXPECTED_FILENAMES1_JSON.value,
            ExtraFilenameEnum.LOG_TXT.value,
            ExtraFilenameEnum.PIPELINE_STOP_REQUESTED_FLAG.value,
            '.DS_Store',
        ]
        files = [f for f in files if f not in ignore_files]
        # logger.debug(f"Files in run_id_dir for {job.run_id}: {files}") # Debug, can be noisy
        # logger.debug(f"Number of files in run_id_dir for {job.run_id}: {len(files)}") # Debug

        # Determine the progress, by comparing the generated files with the expected_filenames1.json
        set_files = set(files)
        set_expected_files = set(self.all_expected_filenames)
        intersection_files = set_files & set_expected_files
        extra_files = set_files - set_expected_files
        # if len(extra_files) > 0:
        #     logger.debug(f"Extra files: {extra_files}")
        progress_message_long = f"{len(intersection_files)} of {len(set_expected_files)}. Extra files: {len(extra_files)}"
        progress_message_short = f"{len(intersection_files)} of {len(set_expected_files)}"
        progress_message = progress_message_short if len(extra_files) == 0 else progress_message_long
        progress_percentage: float = 0.0
        if len(set_expected_files) > 0:
            progress_percentage = (len(intersection_files) * 100.0) / len(set_expected_files)

        return PipelineProgress(progress_message=progress_message, progress_percentage=progress_percentage)

    def _handle_task_completion(self, parameters: HandleTaskCompletionParameters) -> None:
        """
        Protected hook method for custom logic after a task completes.
        This method is called by callback_run_task.
        Subclasses can override this to implement custom behaviors such as:
        - Updating a database with the latest progress.
        - Checking an external source (like a database flag) to determine if the pipeline should continue or be aborted.

        Args:
            parameters: Details about the PlanTask instance that has successfully completed.
                 The `self` of this method is the ExecutePipeline instance,
                 so you can access `self.run_id_dir`, `self.get_progress_percentage()`, etc.

        Raises:
            PipelineStopRequested: To abort the pipeline execution.
        """
        logger.debug(f"ExecutePipeline._handle_task_completion: Default behavior for task {parameters.task.task_id} in run {self.run_id_dir}. Pipeline will continue.")
        # Default implementation simply allows the pipeline to continue.
        # Subclasses will provide meaningful implementations here.

    def callback_run_task(self, task: PlanTask, duration: float) -> None:
        logger.debug(f"ExecutePipeline.callback_run_task: Current task_id: {task.task_id}. Duration: {duration:.2f} seconds")

        progress: PipelineProgress = self.get_progress_percentage()
        logger.debug(f"ExecutePipeline.callback_run_task: Current progress for run {self.run_id_dir}: {progress!r}")

        parameters = HandleTaskCompletionParameters(task=task, progress=progress, duration=duration)

        # Delegate custom handling (like DB updates or stop checks) to the hook method.
        self._handle_task_completion(parameters)

    @property
    def has_pipeline_complete_file(self) -> bool:
        file_path = self.run_id_dir / FilenameEnum.PIPELINE_COMPLETE.value
        return file_path.exists()

    @property
    def has_report_file(self) -> bool:
        file_path = self.run_id_dir / FilenameEnum.REPORT.value
        return file_path.exists()

    @property
    def has_stop_flag_file(self) -> bool:
        file_path = self.run_id_dir / ExtraFilenameEnum.PIPELINE_STOP_REQUESTED_FLAG.value
        return file_path.exists()

    def run(self):
        # Clean up any pre-existing stop flag before the run.
        stop_flag_path = self.run_id_dir / ExtraFilenameEnum.PIPELINE_STOP_REQUESTED_FLAG.value
        if stop_flag_path.exists():
            logger.debug(f"Removing pre-existing stop flag file: {stop_flag_path!r}")
            stop_flag_path.unlink()

        # create a json file with the expected filenames. Save it to the run/run_id/expected_filenames1.json
        expected_filenames_path = self.run_id_dir / ExtraFilenameEnum.EXPECTED_FILENAMES1_JSON.value
        with open(expected_filenames_path, "w") as f:
            json.dump(self.all_expected_filenames, f, indent=2)
        logger.info(f"Saved {len(self.all_expected_filenames)} expected filenames to {expected_filenames_path}")

        luigi_workers = self.resolve_luigi_workers()
        logger.info(f"Luigi workers: {luigi_workers}")
        self.luigi_build_return_value = luigi.build(
            [self.full_plan_pipeline_task],
            local_scheduler=True,
            workers=luigi_workers
        )

        # After the pipeline finishes (or fails), check for the stop flag.
        if self.has_stop_flag_file:
            logger.info("Pipeline was stopped intentionally via PipelineStopRequested exception.")

        logger.info(f"luigi_build_return_value: {self.luigi_build_return_value}")
        logger.info(f"has_pipeline_complete_file: {self.has_pipeline_complete_file}")
        logger.info(f"has_report_file: {self.has_report_file}")
        logger.info(f"has_stop_flag_file: {self.has_stop_flag_file}")

class DemoStoppingExecutePipeline(ExecutePipeline):
    """
    Exercise the pipeline stopping mechanism.
    when a task completes it raises PipelineStopRequested and causes the pipeline to stop.
    """
    def _handle_task_completion(self, parameters: HandleTaskCompletionParameters) -> None:
        logger.info("DemoStoppingExecutePipeline._handle_task_completion: Demo of stopping the pipeline.")
        raise PipelineStopRequested("Demo: Stopping the pipeline after task completion")


def configure_logging(run_id_dir: Path) -> int:
    """
    Configure logging for console (plain text) and file output.
    Returns the console log level that was applied.
    """
    level_name = os.environ.get("PLANEXE_LOG_LEVEL", "INFO").upper()
    resolved_level = getattr(logging, level_name, None)
    if not isinstance(resolved_level, int):
        resolved_level = logging.INFO

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(resolved_level)

    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(resolved_level)
    stdout_handler.setFormatter(formatter)
    root_logger.addHandler(stdout_handler)

    log_file: Path = run_id_dir / ExtraFilenameEnum.LOG_TXT.value
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    return resolved_level


if __name__ == '__main__':
    from llama_index.core.instrumentation import get_dispatcher
    from worker_plan_internal.llm_util.track_activity import TrackActivity

    pipeline_environment = PipelineEnvironment.from_env()
    try:
        run_id_dir: Path = pipeline_environment.get_run_id_dir()
    except ValueError as e:
        msg = f"RUN_ID_DIR is not set or invalid. Error getting run_id_dir: {e!r}"
        logger.error(msg)
        print(f"Exiting... {msg}")
        sys.exit(1)

    console_level = configure_logging(run_id_dir)
    logger.info(
        "pipeline_environment: %r (log_level=%s)",
        pipeline_environment,
        logging.getLevelName(console_level),
    )

    # Example logging messages
    if False:
        logger.debug("This is a debug message.")
        logger.info("This is an info message.")
        logger.warning("This is a warning message.")
        logger.error("This is an error message.")
        logger.critical("This is a critical message.")

    speedvsdetail = SpeedVsDetailEnum.ALL_DETAILS_BUT_SLOW
    speedvsdetail_value = pipeline_environment.speed_vs_detail
    if speedvsdetail_value:
        found = False
        for e in SpeedVsDetailEnum:
            if e.value == speedvsdetail_value:
                speedvsdetail = e
                found = True
                logger.info(f"Setting Speed vs Detail: {speedvsdetail}")
                break
        if not found:
            logger.error(f"Invalid value for SPEED_VS_DETAIL: {speedvsdetail_value}")
    logger.info(f"Speed vs Detail: {speedvsdetail}")

    if False:
        raise Exception("This is a test exception.")

    # logger.info("Environment variables Luigi:\n" + get_env_as_string() + "\n\n\n")

    llm_models = ExecutePipeline.resolve_llm_models(pipeline_environment.llm_model)

    if speedvsdetail == SpeedVsDetailEnum.PING_LLM:
        try:
            run_ping_llm_report(
                run_id_dir=run_id_dir,
                llm_models=LLMModelFromName.from_names(llm_models),
            )
        except Exception as e:
            logger.error("PING_LLM failed: %s", e, exc_info=True)
            sys.exit(1)
        sys.exit(0)

    if True:
        track_activity = TrackActivity(jsonl_file_path=run_id_dir / ExtraFilenameEnum.TRACK_ACTIVITY_JSONL.value, write_to_logger=False)
        get_dispatcher().add_event_handler(track_activity)

    if True:
        execute_pipeline = ExecutePipeline(run_id_dir=run_id_dir, speedvsdetail=speedvsdetail, llm_models=llm_models)
    else:
        execute_pipeline = DemoStoppingExecutePipeline(run_id_dir=run_id_dir, speedvsdetail=speedvsdetail, llm_models=llm_models)
    
    try:
        execute_pipeline.setup()
    except Exception as e:
        logger.error(f"Failed to setup pipeline: {e}")
        sys.exit(1)
    
    logger.info(f"execute_pipeline: {execute_pipeline!r}")
    
    try:
        execute_pipeline.run()
    except Exception as e:
        logger.error(f"Failed to run pipeline: {e}")
        sys.exit(1)
