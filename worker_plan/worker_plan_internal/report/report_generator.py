"""
This generates the report and afterwards opens it in the browser.
PROMPT> python -m worker_plan_internal.report.report_generator /path/to/PlanExe_20250216_dir

This generates the report without opening the browser.
PROMPT> python -m worker_plan_internal.report.report_generator /path/to/PlanExe_20250216_dir --no-browser
"""
import re
import json
import logging
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import markdown
from html import escape
from typing import Dict, Any, Optional
import importlib.resources

logger = logging.getLogger(__name__)

@dataclass
class ReportDocumentItem:
    document_title: str
    document_html_content: str
    css_classes: list[str] = field(default_factory=list)

class ReportGenerator:
    def __init__(self):
        self.report_item_list: list[ReportDocumentItem] = []
        self.html_head_content: list[str] = []
        self.html_body_script_content: list[str] = []

    def read_json_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Read a JSON file and return its contents."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.warning(f"{file_path} not found")
            return None
        except json.JSONDecodeError:
            logging.warning(f"{file_path} contains invalid JSON")
            return None

    def read_markdown_file(self, file_path: Path) -> Optional[str]:
        """Read a markdown file and return its contents."""
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            logging.warning(f"{file_path} not found")
            return None

    def read_csv_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Read a CSV file and return its contents as a pandas DataFrame."""
        try:
            # First try to detect the delimiter by reading the first few lines
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                
            # Count potential delimiters
            delimiters = {
                ',': first_line.count(','),
                ';': first_line.count(';'),
                '\t': first_line.count('\t'),
                '|': first_line.count('|')
            }
            
            # Use the delimiter that appears most frequently
            delimiter = max(delimiters.items(), key=lambda x: x[1])[0]
            
            # Try reading with the detected delimiter
            try:
                df = pd.read_csv(file_path, delimiter=delimiter)
                return df
            except Exception:
                # If that fails, try with more options
                try:
                    df = pd.read_csv(file_path, delimiter=delimiter, 
                                   on_bad_lines='skip', engine='python')
                    logging.warning(f"Some lines in {file_path} were skipped due to parsing errors")
                    return df
                except Exception as e:
                    logging.error(f"Error reading CSV file {file_path}: {str(e)}")
                    return None
                
        except FileNotFoundError:
            logging.error(f"{file_path} not found")
            return None
        except Exception as e:
            logging.error(f"Error reading CSV file {file_path}: {str(e)}")
            return None

    def append_json(self, document_title: str, file_path: Path, css_classes: list[str] = []):
        """Append a JSON document to the report."""
        json_data = self.read_json_file(file_path)
        if json_data is None:
            logging.warning(f"Document: '{document_title}'. Could not read JSON file: {file_path}")
            return
        
        # Convert the JSON data to a formatted string
        json_str = json.dumps(json_data, indent=2)
        
        # Create markdown content with JSON in a code block
        markdown_content = f"```json\n{json_str}\n```"
        
        # Convert markdown to HTML and add to report (fenced_code extension needed for ``` blocks)
        html = markdown.markdown(markdown_content, extensions=['fenced_code'])
        self.report_item_list.append(ReportDocumentItem(document_title, html, css_classes=css_classes))
        
    def append_markdown(self, document_title: str, file_path: Path, css_classes: list[str] = []):
        """Append a markdown document to the report."""
        md_data = self.read_markdown_file(file_path)
        if md_data is None:
            logging.warning(f"Document: '{document_title}'. Could not read markdown file: {file_path}")
            return
        html = markdown.markdown(md_data)
        self.report_item_list.append(ReportDocumentItem(document_title, html, css_classes=css_classes))
    
    def append_markdown_with_tables(self, document_title: str, file_path: Path, css_classes: list[str] = []):
        """Append a markdown document to the report. Render markdown tables as HTML tables."""
        md_data = self.read_markdown_file(file_path)
        if md_data is None:
            logging.warning(f"Document: '{document_title}'. Could not read markdown file: {file_path}")
            return
        html = markdown.markdown(md_data, extensions=['tables'])
        self.report_item_list.append(ReportDocumentItem(document_title, html, css_classes=css_classes))
    
    def append_csv(self, document_title: str, file_path: Path, css_classes: list[str] = []):
        """Append a CSV to the report."""
        df_data = self.read_csv_file(file_path)
        if df_data is None:
            logging.warning(f"Document: '{document_title}'. Could not read CSV file: {file_path}")
            return
        # Clean up the dataframe
        # Remove any completely empty rows or columns
        df = df_data.dropna(how='all', axis=0).dropna(how='all', axis=1)
        html = df.to_html(classes='dataframe', index=False, na_rep='')
        self.report_item_list.append(ReportDocumentItem(document_title, html, css_classes=css_classes))

    def append_html(self, document_title: str, file_path: Path, css_classes: list[str] = []):
        """Append an HTML document to the report."""
        with open(file_path, 'r') as f:
            html_raw = f.read()
        
        # Extract the html_head content between <!--HTML_HEAD_START--> and <!--HTML_HEAD_END-->
        html_head_match = re.search(r'<!--HTML_HEAD_START-->(.*)<!--HTML_HEAD_END-->', html_raw, re.DOTALL)
        if html_head_match:
            html_head = html_head_match.group(1)
            self.html_head_content.append(html_head)
        else:
            logging.warning(f"Document: '{document_title}'. Could not find HTML_HEAD_START and HTML_HEAD_END in {file_path}")
        
        # Extract the html_body content between <!--HTML_BODY_CONTENT_START--> and <!--HTML_BODY_CONTENT_END-->
        html_body_match = re.search(r'<!--HTML_BODY_CONTENT_START-->(.*)<!--HTML_BODY_CONTENT_END-->', html_raw, re.DOTALL)
        if html_body_match:
            html_body = html_body_match.group(1)
            self.report_item_list.append(ReportDocumentItem(document_title, html_body))
        else:
            logging.warning(f"Document: '{document_title}'. Could not find HTML_BODY_CONTENT_START and HTML_BODY_CONTENT_END in {file_path}")
            # If no markers found, use the entire content as the body
            self.report_item_list.append(ReportDocumentItem(document_title, html_raw, css_classes=css_classes))

        # Extract the html_body_script content between <!--HTML_BODY_SCRIPT_START--> and <!--HTML_BODY_SCRIPT_END-->
        html_body_script_match = re.search(r'<!--HTML_BODY_SCRIPT_START-->(.*)<!--HTML_BODY_SCRIPT_END-->', html_raw, re.DOTALL)
        if html_body_script_match:
            html_body_script = html_body_script_match.group(1)
            self.html_body_script_content.append(html_body_script)
        else:
            logging.warning(f"Document: '{document_title}'. Could not find HTML_BODY_SCRIPT_START and HTML_BODY_SCRIPT_END in {file_path}")

    def append_initial_prompt_vetted(self, document_title: str, initial_prompt_file_path: Path, redline_gate_markdown_file_path: Path, premise_attack_markdown_file_path: Path, css_classes: list[str] = []):
        """Append the section 'Inital Prompt Vetted' to the report."""
        # The user-provided prompt can contain unsafe HTML symbols. Escape them to prevent XSS.
        with open(initial_prompt_file_path, 'r', encoding='utf-8') as f:
            initial_prompt_raw = f.read()
        if initial_prompt_raw is None:
            logging.warning(f"Document: '{document_title}'. Could not read file: {initial_prompt_file_path}")
            return
        initial_prompt_html = escape(initial_prompt_raw).replace('\n', '<br>')

        # The Redline Gate markdown contains markdown tables.
        with open(redline_gate_markdown_file_path, 'r', encoding='utf-8') as f:
            redline_gate_markdown = f.read()
        if redline_gate_markdown is None:
            logging.warning(f"Document: '{document_title}'. Could not read file: {redline_gate_markdown_file_path}")
            return
        redline_gate_html = markdown.markdown(redline_gate_markdown, extensions=['tables'])

        # The Premise Attack markdown contains markdown tables.
        with open(premise_attack_markdown_file_path, 'r', encoding='utf-8') as f:
            premise_attack_markdown = f.read()
        if premise_attack_markdown is None:
            logging.warning(f"Document: '{document_title}'. Could not read file: {premise_attack_markdown_file_path}")
            return
        premise_attack_html = markdown.markdown(premise_attack_markdown, extensions=['tables'])

        html = f"""
        <h2>Initial Prompt</h2>
        <p>{initial_prompt_html}</p>
        <h2>Redline Gate</h2>
        {redline_gate_html}
        <h2>Premise Attack</h2>
        {premise_attack_html}
        """
        self.report_item_list.append(ReportDocumentItem(document_title, html, css_classes=css_classes))

    def generate_html_report(self, title: Optional[str] = None, execute_plan_section_hidden: bool = True) -> str:
        """Generate an HTML report from the gathered data."""

        resolved_title = title if title else "PlanExe Project Report"
        escaped_title = escape(resolved_title)

        path_to_template = importlib.resources.files('worker_plan_internal.report') / 'report_template.html'
        with importlib.resources.as_file(path_to_template) as path_to_template:
            with open(path_to_template, 'r') as f:
                html_template = f.read()
        
        html_head = '\n'.join(self.html_head_content)
        html_template = html_template.replace('<!--HTML_HEAD_INSERT_HERE-->', html_head)        

        html_body_script = '\n'.join(self.html_body_script_content)
        html_template = html_template.replace('<!--HTML_BODY_SCRIPT_INSERT_HERE-->', html_body_script)

        html_template = html_template.replace('HEAD_TITLE_INSERT_HERE', escaped_title)

        if execute_plan_section_hidden:
            html_template = html_template.replace('EXECUTE_PLAN_CSS_PLACEHOLDER', 'section-execute-plan-hidden')
        else:
            html_template = html_template.replace('EXECUTE_PLAN_CSS_PLACEHOLDER', 'section-execute-plan-visible')

        html_parts = []
        # Title and Timestamp
        html_parts.append(f"""
        <h1>{escaped_title}</h1>
        <p class="planexe-report-info">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} with PlanExe. <a href="https://planexe.org/discord.html">Discord</a>, <a href="https://github.com/PlanExeOrg/PlanExe">GitHub</a></p>
        """)

        def add_section(title: str, content: str, css_classes: list[str]):
            resolved_css_classes = ['section'] + css_classes
            css_classes_str = ' '.join(resolved_css_classes)
            html_parts.append(f"""
            <div class="{css_classes_str}">
                <button class="collapsible">{title}</button>
                <div class="content">        
                    {content}
                </div>
            </div>
            """)

        for item in self.report_item_list:
            add_section(item.document_title, item.document_html_content, item.css_classes)

        html_content = '\n'.join(html_parts)

        # Replace the content between <!--CONTENT-START--> and <!--CONTENT-END--> with html_content
        pattern = re.compile(r'<!--CONTENT-START-->.*<!--CONTENT-END-->', re.DOTALL)
        
        # Escape any backslashes in the content to prevent regex escape sequence issues
        escaped_html_content = html_content.replace('\\', '\\\\')
        
        html = re.sub(
            pattern,
            f'<!--CONTENT-START-->\n{escaped_html_content}\n<!--CONTENT-END-->',
            html_template
        )

        return html

    def save_report(self, output_path: Path, title: Optional[str] = None, execute_plan_section_hidden: bool = True) -> None:
        """Generate and save the report."""
        html_report = self.generate_html_report(
            title=title, 
            execute_plan_section_hidden=execute_plan_section_hidden
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        logger.info(f"Report generated successfully: {output_path}")

def main():
    from worker_plan_internal.plan.filenames import FilenameEnum
    import argparse
    parser = argparse.ArgumentParser(description='Generate a report from PlanExe output (zip file or directory)')
    parser.add_argument('input_path', help='Path to PlanExe output zip file or directory')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser automatically')
    
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

    
    # Convert input path to absolute path
    input_path = Path(args.input_path).resolve()
    
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        return
    
    output_path = input_path / FilenameEnum.REPORT.value
    
    report_generator = ReportGenerator()
    report_generator.append_markdown('Initial Plan', input_path / FilenameEnum.INITIAL_PLAN.value)
    report_generator.append_markdown('Pitch', input_path / FilenameEnum.PITCH_MARKDOWN.value)
    report_generator.append_markdown('Assumptions', input_path / FilenameEnum.CONSOLIDATE_ASSUMPTIONS_FULL_MARKDOWN.value)
    report_generator.append_markdown('SWOT Analysis', input_path / FilenameEnum.SWOT_MARKDOWN.value)
    report_generator.append_markdown('Team', input_path / FilenameEnum.TEAM_MARKDOWN.value)
    report_generator.append_markdown('Expert Criticism', input_path / FilenameEnum.EXPERT_CRITICISM_MARKDOWN.value)
    report_generator.append_csv('Work Breakdown Structure', input_path / FilenameEnum.WBS_PROJECT_LEVEL1_AND_LEVEL2_AND_LEVEL3_CSV.value)
    report_generator.save_report(output_path, title="Demo Project Report", execute_plan_section_hidden=False)
        
    if not args.no_browser:
        # Try to open the report in the default browser
        try:
            import webbrowser
            url = f'file://{output_path.absolute()}'
            print(f"Opening report in browser: {url}")
            if not webbrowser.open(url):
                print("Could not open browser automatically.")
                print("Please open this file in your web browser:")
                print(f"  {output_path}")
        except Exception as e:
            print(f"Error opening browser: {e}")
            print("Please open this file in your web browser:")
            print(f"  {output_path}")

if __name__ == "__main__":
    main()
