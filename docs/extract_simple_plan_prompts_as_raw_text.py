"""
Extract the plan prompts from this file
PlanExe/worker_plan/worker_plan_internal/prompt/data/simple_plan_prompts.jsonl

It has this format
{"id": "d3e10877-446f-4eb0-8027-864e923973b0", "prompt": "Construct a train bridge between Denmark and England.", "tags": ["denmark", "england", "bridge", "business"]}
{"id": "4dc34d55-0d0d-4e9d-92f4-23765f49dd29", "prompt": "Establish a solar farm in Denmark.", "tags": ["denmark", "energy", "sun", "business"]}

The extracted raw text should be in the following format, joined by a newline:
Construct a train bridge between Denmark and England.
Establish a solar farm in Denmark.

USAGE:
python extract_simple_plan_prompts_as_raw_text.py > plan_prompts.txt
or
python docs/extract_simple_plan_prompts_as_raw_text.py > plan_prompts.txt
"""

import json
import os

def extract_prompts():
    # Get the path to the JSONL file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    jsonl_path = os.path.join(script_dir, '..', 'worker_plan', 'worker_plan_internal', 'prompt', 'data', 'simple_plan_prompts.jsonl')
    
    # Check if file exists
    if not os.path.exists(jsonl_path):
        print(f"Error: File not found at {jsonl_path}")
        return
    
    prompts = []
    
    # Read and parse the JSONL file
    with open(jsonl_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    data = json.loads(line)
                    if 'prompt' in data:
                        prompts.append(data['prompt'])
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {line}")
                    print(f"JSON error: {e}")
    
    # Output the prompts, one per line
    for prompt in prompts:
        print(prompt)

if __name__ == "__main__":
    extract_prompts()
