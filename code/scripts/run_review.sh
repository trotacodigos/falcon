#!/bin/bash
# Step 3. Skill scoring of system translations.
cd "$(dirname "$0")/../module"
export PYTHONPATH="$(cd .. && pwd):$PYTHONPATH"
python3 judge.py -d "$1" -o "$2" -e "${3:-error_review.jsonl}" -s "skillset.json" -m "gpt-4.1-mini" -b -t 0 -x 1024
