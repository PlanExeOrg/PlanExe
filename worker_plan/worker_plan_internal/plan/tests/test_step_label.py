import unittest

from worker_plan_internal.plan.run_plan_pipeline import _task_class_to_step_label


class TestTaskClassToStepLabel(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(_task_class_to_step_label("ReviewPlanTask"), "Review Plan")

    def test_acronym(self):
        self.assertEqual(_task_class_to_step_label("SWOTAnalysisTask"), "SWOT Analysis")

    def test_acronym_with_digits(self):
        self.assertEqual(_task_class_to_step_label("CreateWBSLevel1Task"), "Create WBS Level 1")

    def test_numbered_phase(self):
        self.assertEqual(_task_class_to_step_label("GovernancePhase1AuditTask"), "Governance Phase 1 Audit")

    def test_no_task_suffix(self):
        self.assertEqual(_task_class_to_step_label("ExecutiveSummary"), "Executive Summary")

    def test_single_word(self):
        self.assertEqual(_task_class_to_step_label("ReportTask"), "Report")


if __name__ == "__main__":
    unittest.main()
