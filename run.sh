#!/bin/bash

mkdir -p /app/logs

stdbuf -oL -eL python -u train.py 2>&1 | tee -a /app/logs/output.log
