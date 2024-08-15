#!/bin/bash
set -e

# Function to check if we're running with NVIDIA GPU support
has_nvidia_support() {
    [ -e /dev/nvidia0 ] || [ -e /proc/driver/nvidia/version ]
}

# Check if we have NVIDIA support
if has_nvidia_support; then
    echo "SUCCESS: NVIDIA GPU support detected."
else
    echo "WARNING: No NVIDIA GPU support detected."
fi

# Add the current directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/app"

if [ "$1" = "train" ]; then
    python train.py
elif [ "$1" = "diagnostics" ]; then
    python scripts/run_diagnostics.py
else
    echo "Usage: ./start.sh [train|diagnostics]"
    exit 1
fi
