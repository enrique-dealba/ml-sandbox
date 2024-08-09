#!/bin/bash

# Enable command echoing
set -x

# Set up logging for the shell script
SHELL_LOG="/app/logs/shell_log.txt"
PYTHON_LOG="/app/logs/python_log.txt"

# Redirect all output to both console and shell log file
exec > >(tee -a "$SHELL_LOG") 2>&1

echo "==============================================="
echo "Starting run_and_log.sh script"
echo "==============================================="

echo "Current directory: $(pwd)"
echo "Directory contents:"
ls -la

if [ ! -d "/app/logs" ]; then
    echo "Creating log directory"
    mkdir -p /app/logs
fi

touch "$PYTHON_LOG"

echo "Starting Python log tail"
tail -f "$PYTHON_LOG" &
TAIL_PID=$!

echo "Running Python script with arguments: $@"
python -u train.py "$@" 2>&1 | tee -a "$PYTHON_LOG"
PYTHON_EXIT_CODE=${PIPESTATUS[0]}

echo "Python script finished with exit code: $PYTHON_EXIT_CODE"

echo "Stopping Python log tail"
kill $TAIL_PID

sleep 2

echo "==============================================="
echo "Full Python log contents:"
echo "==============================================="
cat "$PYTHON_LOG"

echo "==============================================="
echo "Full shell script log contents:"
echo "==============================================="
cat "$SHELL_LOG"

echo "==============================================="
echo "Script completed"
echo "==============================================="
