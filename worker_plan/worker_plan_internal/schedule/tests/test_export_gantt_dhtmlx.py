import unittest
import tempfile
from datetime import date
from pathlib import Path
from worker_plan_internal.schedule.export_gantt_dhtmlx import ExportGanttDHTMLX
from worker_plan_internal.schedule.parse_schedule_input_data import parse_schedule_input_data
from worker_plan_internal.schedule.schedule import ProjectSchedule
from worker_plan_internal.utils.dedent_strip import dedent_strip


class TestExportGanttDHTMLX(unittest.TestCase):
    def test_body_content_starts_with_waterfall_note(self):
        # Arrange
        activities = parse_schedule_input_data(dedent_strip("""
            Activity;Predecessor;Duration;Comment
            A;-;3;
            B;A;2;
        """))
        for activity in activities:
            activity.title = f"Title{activity.id}"
        project_schedule = ProjectSchedule.create(activities)

        # Act
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gantt.html"
            ExportGanttDHTMLX.save(project_schedule, str(path), date(2025, 8, 4), title="Demo")
            html = path.read_text(encoding="utf-8")

        # Assert
        note = "Unoptimized waterfall. Parallel work not modelled here."
        body_start = html.index("<!--HTML_BODY_CONTENT_START-->")
        chart_start = html.index('id="gantt_container"')
        note_pos = html.find(note)
        self.assertNotEqual(note_pos, -1, "note is missing from the exported HTML")
        self.assertTrue(body_start < note_pos < chart_start, "note must be at the top of the body content, before the chart")


if __name__ == "__main__":
    unittest.main()
