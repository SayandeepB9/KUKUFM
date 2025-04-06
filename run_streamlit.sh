#!/bin/bash

# Check if virtual environment exists and activate it
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

echo "Starting KUKUFM Story Generator App..."
streamlit run streamlit_app.py
