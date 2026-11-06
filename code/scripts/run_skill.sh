#!/bin/bash
# Step 2. Skill profiling of the segments that require document-level context.
cd "$(dirname "$0")/../module"
export PYTHONPATH="$(cd .. && pwd):$PYTHONPATH"
python3 skillCreator.py -d "$1" -o "$2" -e "${3:-error_skill.jsonl}" -s "skillset.json" -g -m "gpt-4.1-mini"
