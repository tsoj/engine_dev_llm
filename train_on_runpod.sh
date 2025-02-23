#!/bin/bash

set -e

# to run script:
# cd /workspace
# git clone https://github.com/tsoj/engine_dev_llm.git
# cd engine_dev_llm
# git switch more_data
# ./train_on_runpod.sh


mkdir data

apt update && apt install nano nvtop

export HF_HOME=/workspace/hf_cache/

pip install torch
pip install transformers datasets peft bitsandbytes dataclasses-json

while true; do
    read -p "Updated constants.py? (y/n): " answer

    if [[ "$answer" =~ ^[yY]([eE][sS])?$ ]]; then
        break
    fi
done

while true; do
    read -p "Moved all the data via 'scp -r -P <your_port> /path/to/data/text/ root@<your_ip>:/workspace/engine_dev_llm/data/'? (y/n): " answer

    if [[ "$answer" =~ ^[yY]([eE][sS])?$ ]]; then
        break
    fi
done

echo "Starting training"

nohup python train.py > nohup.out &
tail -f nohup.out
