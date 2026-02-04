#!/bin/bash

# Check if uv is installed
if ! command -v uv &> /dev/null
then
    echo "uv could not be found. Please install it first."
    exit 1
fi

# Run the python script using uv
uv run python src/main.py "$@"
