# engine_dev_llm

Fine-tune an LLM on Discord chat exports so it can simulate a channel: the model
writes messages as the channel's members, or you join the conversation yourself.

The model is **Gemma 4 12B** (`google/gemma-4-12B`, the pretrained base model),
fine-tuned with LoRA/QLoRA on plain language modeling of the chat logs.

# Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/). PyTorch comes in
separate builds per backend, selected with a uv extra (`cuda`, `rocm` or `cpu`).
Run everything through the `./run.sh` wrapper, which detects the backend of the
machine (NVIDIA → `cuda`, AMD ROCm → `rocm`, otherwise `cpu`) and passes the
matching extra to `uv run`:

```bash
./run.sh train.py --help          # = uv run --extra <backend> python train.py --help
./run.sh ruff check .             # other commands work too
TORCH_BACKEND=cpu ./run.sh ...    # override the detection
```

The first call installs the environment. Avoid plain `uv run` / `uv sync`
without `--extra`: they replace the GPU build of PyTorch with the default one
from PyPI.

Gemma models on Hugging Face may require accepting the license; if so, log in
with `./run.sh hf auth login` or set `HF_TOKEN`.

# Pipeline

## 1. Build the dataset

Export chats as JSON (see [How to get the data](#how-to-get-the-data)), then:

```bash
./run.sh data.py --json_dirs data/discord_json_data --output_dir data/dataset
```

- Several export directories can be combined: `--json_dirs dir1 dir2`.
- Noisy channels are skipped with `--exclude_channels` (glob patterns on
  `"Guild - channel"`; see `./run.sh data.py --help` for the defaults).
- Bot messages, joins, pins etc. are dropped; attachments become `[image]`,
  `[file]`, ... placeholders.
- The most recent 2% of every channel is held out as the eval split.
- `data/dataset/meta.json` lists token counts and the included channel names.

## 2. Train

```bash
./run.sh train.py --run_name first-try
./run.sh train.py --run_name first-try --resume  # continue after an interruption
```

Everything for a run goes to `runs/<run_name>/`: `checkpoints/`, the final
`adapter/`, and with `--merge` a standalone `merged/` model. The eval loss at
step 0 is the base model's, as a baseline.

Defaults: chunks of 2048 tokens (dozens of chat messages of context), one chunk
per optimizer step, QLoRA. Suggested settings per GPU (estimates; check peak
memory with a short `--max_steps 20` run first):

| GPU memory | Flags |
| --- | --- |
| 80 GB | `--quantization none` (bf16 LoRA, faster than 4-bit) |
| 24–48 GB | defaults |
| 20 GB | defaults, or a dataset built with `--max_length 1024` if memory runs out |

See `./run.sh train.py --help` for all options (LoRA rank, learning rate,
epochs, ...).

## 3. Compare checkpoints

Loss alone doesn't tell you whether the output reads like the real channel.
`sample.py` continues a few held-out conversations with the base model and every
checkpoint of a run, using the same random seed, and writes them side by side to
a Markdown file:

```bash
./run.sh sample.py --run_dir runs/first-try
```

## 4. Generate

```bash
# the model writes 30 messages
./run.sh generate.py --model runs/first-try/adapter --channel "Stockfish - engines-dev"

# join the conversation
./run.sh generate.py --model runs/first-try/adapter --channel "Stockfish - engines-dev" --interactive
```

In interactive mode, enter a blank line to let the model pick the next speaker,
`name` to have the model write as `name`, or `name: message` to write a message
yourself.

`--model` accepts an adapter, a checkpoint (`runs/*/checkpoints/checkpoint-*`) or
a merged model. The context window defaults to the length the run was trained
with. Weights are loaded in 4-bit by default, which needs roughly
10 GB of VRAM (an estimate). Use `--quantization 8bit` or `none` if you have
more memory.

# Switching to a different model

Everything model-specific lives in `model_spec.py`: the expected architecture and
size, the special tokens of the chat format, and the LoRA target modules. Every
script checks these before doing anything, and `train.py` additionally verifies
that TRL's loss matches the model's own loss. A different model fails these
checks on purpose: go through each assumption in `model_spec.py`, adapt it, and
only then update the expected values.

# How to get the data

Use [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter). Be
careful with your account; ideally use one that wouldn't matter much if it got
compromised. Passing the token through an environment variable keeps it out of
your shell history:

```bash
export DISCORD_TOKEN="your.token"

# list accessible servers and their IDs
/path/to/DiscordChatExporter.Cli guilds -t "$DISCORD_TOKEN"

# export a few servers
./download_data.sh /path/to/DiscordChatExporter.Cli <server_id_1> <server_id_2>

# or export everything accessible
/path/to/DiscordChatExporter.Cli exportall -t "$DISCORD_TOKEN" -f Json \
    -o "./data/discord_json_data/[%G|%C][%g|%c].json"
```

The exports must be JSON. For up-to-date details, see the DiscordChatExporter
repository.

<details>
<summary>Channels of the engine-dev dataset</summary>

```
Chess Programming Wiki - 6-hours-slowmode
Chess Programming Wiki - brainrot
Chess Programming Wiki - chess-talk
Chess Programming Wiki - engine-dev
Chess Programming Wiki - enginetest
Chess Programming Wiki - feedback
Chess Programming Wiki - general
Chess Programming Wiki - math
Chess Programming Wiki - wiki-general
Engine Programming - 1024challenge
Engine Programming - 2048
Engine Programming - 4kdotc
Engine Programming - 4ku
Engine Programming - ataxx
Engine Programming - bitboards
Engine Programming - bullet
Engine Programming - chess
Engine Programming - cutegames
Engine Programming - deep-chess
Engine Programming - dice-wars
Engine Programming - events
Engine Programming - feedback
Engine Programming - general
Engine Programming - go
Engine Programming - honse
Engine Programming - machine-learning
Engine Programming - mnk
Engine Programming - off-topic
Engine Programming - pijersi
Engine Programming - princhess
Engine Programming - programming
Engine Programming - pytteliten
Engine Programming - releases
Engine Programming - resources
Engine Programming - reversi
Engine Programming - style
Engine Programming - tak
Engine Programming - tetka
Engine Programming - texel-tuner
Engine Programming - uttt
Leela Chess Zero - dev-log
Leela Chess Zero - dev-public
Leela Chess Zero - dev
Leela Chess Zero - general
Leela Chess Zero - off-topic
Leela Chess Zero - publications-discuss
Leela Chess Zero - test-discuss
OpenBench - all-other-things
OpenBench - chess-things
OpenBench - off-topic
OpenBench - openbench-instances
OpenBench - openbench-support
OpenBench - open-rank
Stockfish - chessdbcn
Stockfish - engine-releases
Stockfish - engines-dev
Stockfish - fishtest-dev
Stockfish - general-chess
Stockfish - hardware-discuss
Stockfish - kaggle-talk
Stockfish - lawsuit-discuss
Stockfish - Livestreams
Stockfish - nnue-dev
Stockfish - off-topic
Stockfish - programming
Stockfish - sf-dev
Stockfish - sf-general
Stockfish - sf-web-and-wiki
Stockfish - top-dev-chill
```

</details>
