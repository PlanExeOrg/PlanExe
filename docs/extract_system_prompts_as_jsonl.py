"""
Traverse all the files in the worker_plan_internal package and extract all system prompts to a "system_prompts.jsonl" file.
The "system_prompts.jsonl" file is to be used for prompt engineering.

The file should be in the following format:
{"id": "diagnostics/premortem.py:20", "prompt": "long system prompt here", name: "PREMORTEM_SYSTEM_PROMPT"}
{"id": "assume/physical_locations.py:50", "prompt": "long system prompt here", name: "PHYSICAL_LOCATIONS_SYSTEM_PROMPT"}
{"id": "assume/make_assumptions.py:28", "prompt": "long system prompt here", "name": "SYSTEM_PROMPT_1"}
{"id": "assume/make_assumptions.py:80", "prompt": "long system prompt here", "name": "SYSTEM_PROMPT_2"}
{"id": "assume/make_assumptions.py:140", "prompt": "long system prompt here", "name": "SYSTEM_PROMPT_3"}

USAGE:
python extract_system_prompts_as_jsonl.py
or
python docs/extract_system_prompts_as_jsonl.py
"""
import os
import re
import json
from pathlib import Path
from typing import List, Dict

def find_system_prompts_in_file(file_path: Path, package_path: Path) -> List[Dict]:
    """
    Extract system prompts from a single Python file.
    
    Returns a list of dictionaries with system prompt information.
    """
    prompts = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return prompts
    
    # Pattern to match system prompt variable assignments
    # Matches patterns like: VARIABLE_NAME = """ or VARIABLE_NAME = '''
    # Handles both _SYSTEM_PROMPT and SYSTEM_PROMPT_ patterns
    system_prompt_pattern = r'^([\w_]*?SYSTEM_PROMPT[\w_]*?)\s*=\s*["\']{3}(.*?)["\']{3}'
    
    # Find all matches in the file
    matches = re.finditer(system_prompt_pattern, content, re.DOTALL | re.MULTILINE)
    
    for match in matches:
        variable_name = match.group(1)
        prompt_content = match.group(2).strip()
        start_pos = match.start()
        
        # Calculate line number
        line_number = content[:start_pos].count('\n') + 1
        
        # Create relative path from worker_plan_internal directory
        relative_path = file_path.relative_to(package_path)
        
        prompt_info = {
            "id": f"{relative_path}:{line_number}",
            "prompt": prompt_content,
            "name": variable_name,
        }
        
        prompts.append(prompt_info)
    
    return prompts

def find_all_python_files(package_dir: Path) -> List[Path]:
    """
    Find all Python files in the worker_plan_internal directory recursively.
    """
    python_files = []
    
    for root, dirs, files in os.walk(package_dir):
        # Skip __pycache__, proof_of_concepts, and other common directories to ignore
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__' and d != 'proof_of_concepts']
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    
    python_files.sort()
    return python_files

def extract_all_system_prompts(package_path: Path) -> List[Dict]:
    """
    Extract all system prompts from all Python files in the worker_plan_internal directory.
    """
    if not package_path.exists():
        raise FileNotFoundError(f"Directory {package_path!r} not found")
    
    all_prompts = []
    python_files = find_all_python_files(package_path)
    
    print(f"Found {len(python_files)} Python files to process...")
    
    for file_path in python_files:
        prompts = find_system_prompts_in_file(file_path, package_path)
        all_prompts.extend(prompts)
        
        if prompts:
            print(f"Found {len(prompts)} system prompt(s) in {file_path.relative_to(package_path)}")
    
    return all_prompts

def save_to_jsonl(prompts: List[Dict], output_file: str = "system_prompts.jsonl"):
    """
    Save system prompts to a JSONL file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for prompt in prompts:
            json_line = json.dumps(prompt, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"Saved {len(prompts)} system prompts to {output_file}")

def main():
    """
    Main function to extract system prompts and save them to JSONL.
    """
    # Get the directory where this script is located, then go up one level to find worker_plan_internal
    script_dir = Path(__file__).parent
    input_dir = script_dir / ".." / "worker_plan" / "worker_plan_internal"
    try:
        print("Extracting system prompts from worker_plan_internal package...")
        prompts = extract_all_system_prompts(input_dir)
        
        if not prompts:
            print("No system prompts found!")
            return
        
        print(f"\nTotal system prompts found: {len(prompts)}")
        
        save_to_jsonl(prompts)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
