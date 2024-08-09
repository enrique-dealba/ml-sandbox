#!/bin/bash

# Start tailing the log file in the background
tail -f /app/output.log &

# Run the Python script
python train.py

# Wait for both processes to finish
wait
