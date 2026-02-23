#!/bin/bash

# MoneyPrinterTurbo Desktop App Launcher
# Runs the native desktop window version of MoneyPrinterTurbo

# Set up CUDA/cuDNN environment (same as webui.sh)
source "$(dirname "$0")/setup_cuda_env.sh"

# Activate the virtual environment if it exists
if [ -d "$(dirname "$0")/.venv" ]; then
    source "$(dirname "$0")/.venv/bin/activate"
fi

# Launch the desktop app
python "$(dirname "$0")/desktop.py"
