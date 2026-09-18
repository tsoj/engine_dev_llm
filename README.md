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

## 3. Validate on chat the model has never seen

The most honest check is chat written *after* the training export: unlike a
random split it can't share a conversation with the training data. Export the
newer messages, build an eval-only dataset from them (`--eval_fraction 1` puts
every chunk in the eval split), and score the base model and every checkpoint on
it:

```bash
./run.sh data.py --json_dirs data/newer_json_data --output_dir data/newer --eval_fraction 1
./run.sh evaluate.py --run_dir runs/first-try --dataset_dir data/newer
```

This writes `runs/first-try/eval_eval_<time>.md` (a table with loss, perplexity,
next-token accuracy and a per-channel breakdown), a `.json` with all numbers, and
a `.png` plotting the validation loss over the training loss curve. Use it to
see whether the later epochs still help or only overfit.

## 4. Compare checkpoints

Loss alone doesn't tell you whether the output reads like the real channel.
`sample.py` continues a few held-out conversations with the base model and every
checkpoint of a run, using the same random seed, and writes them side by side to
a Markdown file, next to the real continuation from the dataset:

```bash
./run.sh sample.py --run_dir runs/first-try

# prompts from newer chat instead of the run's own eval split
./run.sh sample.py --run_dir runs/first-try --dataset_dir data/newer --include_base false
```

## 5. Generate

```bash
# the model writes 30 messages
./run.sh generate.py --model runs/first-try/adapter

# pick a channel and join the conversation
./run.sh generate.py --model runs/first-try/adapter --channel "My Server - general" --interactive
```

In interactive mode, enter a blank line to let the model pick the next speaker,
`name` to have the model write as `name`, or `name: message` to write a message
yourself.

`--model` accepts an adapter, a checkpoint (`runs/*/checkpoints/checkpoint-*`) or
a merged model. `--channel` defaults to the first channel of the run's training
dataset; the full list is in that dataset's `meta.json` (and in the run's
`train_config_*.json`). The context window defaults to the length the run was
trained with. Weights are loaded in 4-bit by default, which needs roughly
10 GB of VRAM (an estimate). Use `--quantization 8bit` or `none` if you have
more memory.

## 6. Discord bot

`bot.py` runs the model as a Discord bot that writes as one member of the
dataset. It answers when it is mentioned or replied to, and with
`--reply_chance 0.1` it also joins in on its own after about every tenth message.
As long as the model predicts that the persona would write the next message too,
the bot sends up to `--max_messages` (default 5) messages in a row. It stops early
when someone else writes in the meantime. Before answering it pauses for a random
time of at least 0.2 s. The pause is most likely around 0.5 s if the bot wrote in
the last 10 seconds, and grows to around 4 s the longer it has been quiet
(`--reaction_floor_seconds`, `--reaction_peak_active_seconds`,
`--reaction_peak_idle_seconds`). It also doesn't write faster than a person:
if a message is generated faster than it could be typed at `--typing_wpm`
(default 60 words per minute), the bot keeps showing "typing…" until then.
Each Discord channel gets its own conversation, started from its most recent
messages (`--history_messages`).

Setting up the bot account:

1. Create an application at https://discord.com/developers/applications. On its
   **Bot** page, copy the token and enable **Message Content Intent**.
2. Under **OAuth2 → URL Generator**, select the scope `bot` and the permissions
   *View Channels*, *Send Messages* and *Read Message History*. Open the
   generated link to add the bot to a server.

```bash
export DISCORD_BOT_TOKEN="the.bot.token"
./run.sh bot.py --model runs/first-try/adapter --persona "nickname" \
    --channel "My Server - general" --channel_ids 123456789012345678
```

`--persona` must be spelled as in the dataset (the server nickname).
`--channel` is the trained channel whose style the bot should use, and
`--channel_ids` limits the bot to certain Discord channels (right-click →
*Copy Channel ID* with Developer Mode on).

Only imitate people who agreed to it. Discord's developer policy doesn't allow
bots that impersonate others.

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
compromised or deleted. Passing the token through an environment variable keeps it out of
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
