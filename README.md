# Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/). Pick the extra
matching your GPU vendor (exactly one):

```bash
# NVIDIA (CUDA 13.0) — e.g. the 80 GB H100 used for training
uv sync --extra cuda

# AMD (ROCm 7.2, Linux only)
uv sync --extra rocm

# CPU only (inference experiments / no GPU)
uv sync --extra cpu
```

Run any script with `uv run python <script>.py`.

The current model is `Qwen/Qwen3.6-27B`. It is a *multimodal* causal LM (text
decoder + vision encoder), but we train and sample it on **text only**: no
images are ever passed, so the vision tower never runs, and LoRA is scoped to
the language layers. This needs a recent `transformers` (5.x) — `uv sync` pulls
it in. Set a Hugging Face token with read access if the model is gated:

```bash
export HF_TOKEN=hf_...
```

# Pipeline

1. **Build the dataset** — convert the Discord JSON exports (see *How to get the
   data* below) into a ChatML JSONL file at `data/dataset.jsonl`:

   ```bash
   uv run python data.py
   ```

2. **Train** — QLoRA fine-tune tuned for a single 80 GB H100. Resumes from the
   latest checkpoint in `checkpoints/` automatically:

   ```bash
   uv run python train.py
   ```

   Set `MERGE_MODEL=1` to also export a standalone merged model (needs the base
   in bf16, ~54 GB).

3. **Generate** — sample a continuing multi-party conversation. Point
   `--adapter` at the `*_LORA` (or `*_merged`) directory train.py produced:

   ```bash
   # auto-generate 30 messages in a channel
   uv run python generate.py --adapter ./engine_dev_model_..._LORA \
       --prompt "Stockfish - engines-dev" --max-messages 30

   # interactive: you choose the next speaker each turn
   uv run python generate.py --adapter ./engine_dev_model_..._LORA \
       --prompt "Engine Programming - ataxx" --interactive
   ```

## Supported channel prompts

```bash
"Chess Programming Wiki - 6-hours-slowmode"
"Chess Programming Wiki - brainrot"
"Chess Programming Wiki - chess-talk"
"Chess Programming Wiki - counting"
"Chess Programming Wiki - engine-dev"
"Chess Programming Wiki - enginetest"
"Chess Programming Wiki - feedback"
"Chess Programming Wiki - general"
"Chess Programming Wiki - math"
"Chess Programming Wiki - memes"
"Chess Programming Wiki - wiki-general"
"Engine Programming - 1024challenge"
"Engine Programming - 2048"
"Engine Programming - 4kdotc"
"Engine Programming - 4ku"
"Engine Programming - ataxx"
"Engine Programming - bitboards"
"Engine Programming - bullet"
"Engine Programming - chess"
"Engine Programming - cutegames"
"Engine Programming - deep-chess"
"Engine Programming - dice-wars"
"Engine Programming - events"
"Engine Programming - feedback"
"Engine Programming - general"
"Engine Programming - go"
"Engine Programming - honse"
"Engine Programming - machine-learning"
"Engine Programming - mnk"
"Engine Programming - off-topic"
"Engine Programming - pijersi"
"Engine Programming - princhess"
"Engine Programming - programming"
"Engine Programming - pytteliten"
"Engine Programming - releases"
"Engine Programming - resources"
"Engine Programming - reversi"
"Engine Programming - style"
"Engine Programming - tak"
"Engine Programming - tetka"
"Engine Programming - texel-tuner"
"Engine Programming - uttt"
"Leela Chess Zero - announcements"
"Leela Chess Zero - dev-log"
"Leela Chess Zero - dev-public"
"Leela Chess Zero - dev"
"Leela Chess Zero - general"
"Leela Chess Zero - off-topic"
"Leela Chess Zero - publications-discuss"
"Leela Chess Zero - test-discuss"
"OpenBench - all-other-things"
"OpenBench - chess-things"
"OpenBench - off-topic"
"OpenBench - openbench-instances"
"OpenBench - openbench-support"
"OpenBench - open-rank"
"Stockfish - chessdbcn"
"Stockfish - engine-releases"
"Stockfish - engines-dev"
"Stockfish - fishtest-dev"
"Stockfish - general-chess"
"Stockfish - hardware-discuss"
"Stockfish - kaggle-talk"
"Stockfish - lawsuit-discuss"
"Stockfish - Livestreams"
"Stockfish - memes"
"Stockfish - music"
"Stockfish - nnue-dev"
"Stockfish - off-topic"
"Stockfish - programming"
"Stockfish - sf-dev"
"Stockfish - sf-general"
"Stockfish - sf-web-and-wiki"
"Stockfish - top-dev-chill"
```

# How to get the data

You can use https://github.com/Tyrrrz/DiscordChatExporter for that. Be careful with your account, ideally use some account that wouldn't matter much if it got compromised.

List accessible servers and their IDs:
```bash
./DiscordChatExporter.Cli guilds -t "your.token"
```

If you want download just from a single server with ID `<server_id>`, you can do:
```bash
/path/to/DiscordChatExporter.Cli exportguild -t "your.token" -g <server_id> -f Json -o "./data/discord_json_data/[%G|%C][%g|%c].json"
```

To download from a selected number of servers, run the following script.
`<server_id_N>` are the IDs of the servers you wanna download:
```bash
./download_servers.sh /path/to/DiscordChatExporter.Cli "your.token" <server_id_1> <server_id_2> <server_id_3>
```

If you want download from all accesible servers, you can do:
```bash
/path/to/DiscordChatExporter.Cli exportall -t "your.token" -f Json -o "./data/discord_json_data/[%G|%C][%g|%c].json"
```

For more detailed and potentially up to date info visit the above-mentioned github repo.
The downloaded data must be in JSON format in the path `./data/discord_json_data/`.
