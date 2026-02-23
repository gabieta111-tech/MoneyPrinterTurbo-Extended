#!/bin/bash

# If you could not download the model from the official site, you can use the mirror site.
# Just remove the comment of the following line .
# 如果你无法从官方网站下载模型，你可以使用镜像网站。
# 只需要移除下面一行的注释即可。

# export HF_ENDPOINT=https://hf-mirror.com

# Set up CUDA/cuDNN environment
source "$(dirname "$0")/setup_cuda_env.sh"

# Activate the virtual environment if it exists
if [ -d "$(dirname "$0")/.venv" ]; then
    source "$(dirname "$0")/.venv/bin/activate"
fi

streamlit run ./webui/Main.py --browser.serverAddress="0.0.0.0" --server.enableCORS=True --browser.gatherUsageStats=False