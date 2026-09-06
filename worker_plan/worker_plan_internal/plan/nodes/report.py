"""ReportTask - Generates the final HTML and Markdown report documents."""
from worker_plan_internal.plan.run_plan_pipeline import PlanTask, REPORT_EXECUTE_PLAN_SECTION_HIDDEN
from worker_plan_internal.report.report_generator import ReportGenerator
from worker_plan_api.filenames import FilenameEnum
from worker_plan_internal.plan.nodes.setup import SetupTask
from worker_plan_internal.plan.nodes.redline_gate import RedlineGateTask
from worker_plan_internal.plan.nodes.premise_attack import PremiseAttackTask
from worker_plan_internal.plan.nodes.strategic_decisions_markdown import StrategicDecisionsMarkdownTask
from worker_plan_internal.plan.nodes.scenarios_markdown import ScenariosMarkdownTask
from worker_plan_internal.plan.nodes.consolidate_assumptions_markdown import ConsolidateAssumptionsMarkdownTask
from worker_plan_internal.plan.nodes.project_plan import ProjectPlanTask
from worker_plan_internal.plan.nodes.data_collection import DataCollectionTask
from worker_plan_internal.plan.nodes.related_resources import RelatedResourcesTask
from worker_plan_internal.plan.nodes.swot_analysis import SWOTAnalysisTask
from worker_plan_internal.plan.nodes.team_markdown import TeamMarkdownTask
from worker_plan_internal.plan.nodes.convert_pitch_to_markdown import ConvertPitchToMarkdownTask
from worker_plan_internal.plan.nodes.expert_review import ExpertReviewTask
from worker_plan_internal.plan.nodes.consolidate_governance import ConsolidateGovernanceTask
from worker_plan_internal.plan.nodes.markdown_documents import MarkdownWithDocumentsToCreateAndFindTask
from worker_plan_internal.plan.nodes.create_wbs_level1 import CreateWBSLevel1Task
from worker_plan_internal.plan.nodes.wbs_project_level1_level2_level3 import WBSProjectLevel1AndLevel2AndLevel3Task
from worker_plan_internal.plan.nodes.review_plan import ReviewPlanTask
from worker_plan_internal.plan.nodes.executive_summary import ExecutiveSummaryTask
from worker_plan_internal.plan.nodes.create_schedule import CreateScheduleTask
from worker_plan_internal.plan.nodes.questions_and_answers import QuestionsAndAnswersTask
from worker_plan_internal.plan.nodes.premortem import PremortemTask
from worker_plan_internal.plan.nodes.self_audit import SelfAuditTask
from worker_plan_internal.plan.nodes.prompt_adherence import PromptAdherenceTask
from worker_plan_internal.plan.nodes.screen_planning_prompt import ScreenPlanningPromptTask


class ReportTask(PlanTask):
    """Assemble all pipeline outputs into the final HTML and Markdown reports."""
    def output(self):
        return {
            'html': self.local_target(FilenameEnum.REPORT_HTML),
            'markdown': self.local_target(FilenameEnum.REPORT_MARKDOWN),
        }

    def requires(self):
        return {
            'setup': self.clone(SetupTask),
            'screen_planning_prompt': self.clone(ScreenPlanningPromptTask),
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
            'self_audit': self.clone(SelfAuditTask),
            'prompt_adherence': self.clone(PromptAdherenceTask),
        }

    def run_inner(self):
        # For the report title, use the 'project_title' of the WBS Level 1 result.
        with self.input()['wbs_level1']['project_title'].open("r") as f:
            title = f.read()

        rg = ReportGenerator()
        rg.append_markdown('Executive Summary', self.input()['executive_summary']['markdown'].path)
        rg.append_html('Gantt', self.input()['create_schedule']['dhtmlx_html'].path)
        rg.append_markdown('Pitch', self.input()['pitch_markdown']['markdown'].path)
        rg.append_markdown('Project Plan', self.input()['project_plan']['markdown'].path)
        rg.append_markdown('Strategic Decisions', self.input()['strategic_decisions_markdown']['markdown'].path)
        rg.append_markdown('Scenarios', self.input()['scenarios_markdown']['markdown'].path)
        rg.append_markdown_with_tables('Assumptions', self.input()['consolidate_assumptions_markdown']['full'].path)
        rg.append_markdown('Governance', self.input()['consolidate_governance'].path)
        rg.append_markdown('Related Resources', self.input()['related_resources']['markdown'].path)
        rg.append_markdown('Data Collection', self.input()['data_collection']['markdown'].path)
        rg.append_markdown('Documents to Create and Find', self.input()['documents_to_create_and_find'].path)
        rg.append_markdown('SWOT Analysis', self.input()['swot_analysis']['markdown'].path)
        rg.append_markdown('Team', self.input()['team_markdown'].path)
        rg.append_markdown('Expert Criticism', self.input()['expert_review'].path)
        rg.append_csv('Work Breakdown Structure', self.input()['wbs_project123']['csv'].path)
        rg.append_markdown('Review Plan', self.input()['review_plan']['markdown'].path)
        rg.append_markdown('Questions & Answers', self.input()['questions_and_answers']['markdown'].path)
        rg.append_markdown_with_tables('Premortem', self.input()['premortem']['markdown'].path)
        rg.append_markdown_with_tables('Self Audit', self.input()['self_audit']['markdown'].path)
        rg.append_initial_prompt_vetted(
            document_title='Initial Prompt Vetted',
            initial_prompt_file_path=self.input()['setup'].path,
            screen_planning_prompt_raw_file_path=self.input()['screen_planning_prompt']['raw'].path,
            screen_planning_prompt_markdown_file_path=self.input()['screen_planning_prompt']['markdown'].path,
            redline_gate_markdown_file_path=self.input()['redline_gate']['markdown'].path,
            premise_attack_markdown_file_path=self.input()['premise_attack']['markdown'].path
        )
        rg.append_markdown_with_tables('Prompt Adherence', self.input()['prompt_adherence']['markdown'].path)
        rg.save_html_report(self.output()['html'].path, title=title, execute_plan_section_hidden=REPORT_EXECUTE_PLAN_SECTION_HIDDEN)
        rg.save_markdown_report(self.output()['markdown'].path, title=title)
