#!/bin/bash
# Simple script to activate venv and run the vision system

# Get the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate virtual environment
source "$DIR/venv/bin/activate"

# Run the vision system with any passed arguments
python "$DIR/vectorlines_heatmap.py" "$@"


