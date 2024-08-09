#!/bin/bash

# Enable command echoing
set -x

# Print current directory and its contents
echo "Current directory: $(pwd)"
echo "Directory contents:"
ls -la

# Check if log directory exists, create if not
if [ ! -d "/app/logs" ]; then
    echo "Creating log directory"
    mkdir -p /app/logs
fi

# Define log file path
LOG_FILE="/app/logs/training_log.txt"

# Touch the log file to ensure it exists
touch $LOG_FILE

echo "Starting log tail"
# Start tailing the log file in the background
tail -f $LOG_FILE &
TAIL_PID=$!

echo "Running Python script"
# Run the training script
python -u train.py
PYTHON_EXIT_CODE=$?

echo "Python script finished with exit code: $PYTHON_EXIT_CODE"

# Kill the tail process
echo "Stopping log tail"
kill $TAIL_PID

# Wait for a moment to ensure all logs are written
sleep 2

echo "Full log contents:"
# Cat the full log file
cat $LOG_FILE

echo "Script completed"
