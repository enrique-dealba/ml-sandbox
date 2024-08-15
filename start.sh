#!/bin/bash

if [ "$1" = "train" ]; then
    python train.py
elif [ "$1" = "diagnostics" ]; then
    python scripts/run_diagnostics.py
else
    echo "Usage: ./start.sh [train|diagnostics]"
    exit 1
fi
