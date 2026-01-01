#!/bin/bash
# Run backend tests

cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate

echo "Installing test dependencies..."
pip install -r requirements.txt > /dev/null

echo "Running tests..."
# Add current directory to PYTHONPATH so 'app' module can be found
export PYTHONPATH=$PYTHONPATH:.
python3 -m pytest tests/ -v
