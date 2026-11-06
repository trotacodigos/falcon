#!/bin/bash
# Step 1. Context profiling of the English source segments.
cd "$(dirname "$0")/../module"
export PYTHONPATH="$(cd .. && pwd):$PYTHONPATH"
python3 contextCreator.py -d "$1" -o "$2" -e "${3:-error_context.jsonl}" -m "gpt-4.1-mini"
