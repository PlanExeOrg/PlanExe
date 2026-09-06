import tempfile
import unittest
from pathlib import Path

from worker_plan_internal.report.report_generator import ReportGenerator


class TestReportGeneratorAppendHtml(unittest.TestCase):
    def test_subtitle_is_rendered_first_in_section_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            html_path = Path(tmp) / "widget.html"
            html_path.write_text(
                "<!--HTML_HEAD_START--><style></style><!--HTML_HEAD_END-->"
                "<!--HTML_BODY_CONTENT_START--><div id=\"widget\">chart</div><!--HTML_BODY_CONTENT_END-->"
                "<!--HTML_BODY_SCRIPT_START--><script></script><!--HTML_BODY_SCRIPT_END-->"
            )
            rg = ReportGenerator()
            rg.append_html("Gantt", html_path, subtitle="Unoptimized waterfall. Parallel work <not> modelled here.")
            html = rg.generate_html_report(title="Sample")

        button = html.index('<button class="collapsible">Gantt</button>')
        subtitle = html.index("<p>Unoptimized waterfall. Parallel work &lt;not&gt; modelled here.</p>")
        widget = html.index('id="widget"')
        self.assertTrue(button < subtitle < widget)

    def test_no_subtitle_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            html_path = Path(tmp) / "widget.html"
            html_path.write_text(
                "<!--HTML_BODY_CONTENT_START--><div id=\"widget\">chart</div><!--HTML_BODY_CONTENT_END-->"
            )
            rg = ReportGenerator()
            rg.append_html("Gantt", html_path)
            html = rg.generate_html_report(title="Sample")

        button = html.index('<button class="collapsible">Gantt</button>')
        widget = html.index('id="widget"')
        self.assertNotIn("<p>", html[button:widget])


if __name__ == "__main__":
    unittest.main()
