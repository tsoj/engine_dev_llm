# Setup

Nvidia:
```bash
conda create -n engine_dev_llm python=3.12
conda activate engine_dev_llm
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets peft bitsandbytes dataclasses-json
```

AMD:
```bash
conda create -n engine_dev_llm python=3.12
conda activate engine_dev_llm
pip install torch --index-url https://download.pytorch.org/whl/rocm6.1/
pip install 'https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_multi-backend-refactor/bitsandbytes-0.44.1.dev0-py3-none-manylinux_2_24_x86_64.whl'
pip install transformers datasets peft dataclasses-json
```

# Train

Needs roughly 80 GB memory per GPU.

```bash
# replace 3 with the number of GPUs you want to train on
torchrun --nproc_per_node 3 train.py
```

If one GPU is not enough to hold an entire model, then enable gradient_checkpoint again, adjust the batchsize and gradient accumulation steps.
Then run the main.py script normally with `python train.py`.

# Generate

Needs roughly 13 GB memory.

```bash
# generates 1000 tokens
python generate.py 1000

# interactive version
python generate.py interactive

# custom prompt
python generate.py interactive "Engine Programming - ataxx"
```

## Supported channel prompts

```bash
"Engine Programming - 1024challenge"
"Engine Programming - 2048"
"Engine Programming - 4ku"
"Engine Programming - ataxx"
"Engine Programming - bitboards"
"Engine Programming - bullet"
"Engine Programming - chess"
"Engine Programming - cutegames"
"Engine Programming - events"
"Engine Programming - general"
"Engine Programming - go"
"Engine Programming - machine-learning"
"Engine Programming - off-topic"
"Engine Programming - pijersi"
"Engine Programming - princhess"
"Engine Programming - programming"
"Engine Programming - releases"
"Engine Programming - reversi"
"Engine Programming - style"
"Engine Programming - texel-tuner"
"Engine Programming - uttt"
"Stockfish - chessdbcn"
"Stockfish - engine-releases"
"Stockfish - engines-dev"
"Stockfish - fishtest-dev"
"Stockfish - general-chess"
"Stockfish - hardware-discuss"
"Stockfish - memes"
"Stockfish - music"
"Stockfish - nnue-dev"
"Stockfish - off-topic"
"Stockfish - programming"
"Stockfish - sf-dev"
"Stockfish - sf-general"
```
