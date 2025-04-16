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
pip install transformers datasets peft dataclasses-json
pip install --force-reinstall 'https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_multi-backend-refactor/bitsandbytes-0.44.1.dev0-py3-none-manylinux_2_24_x86_64.whl'
pip install --force-reinstall pytorch-triton-rocm==3.1.0 torch==2.5.1+rocm6.2 --index-url https://download.pytorch.org/whl/rocm6.2
```

Since the current model is based on Mistral-Small-24B-Base-2501 you may need to set the environment variable `HF_TOKEN` to your Hugging Face token with read access, since you need to agree to some stuff to access the Mistral Small 3 models.

You can download the fine-tuned LoRA parameters from [here](https://drive.google.com/file/d/10qyp5_XQpJcC6Nq5EbLCa8p_ed-qePHD/view?usp=drive_link). Unzip them into this repo project dir.

# Train

Needs roughly 80 GB VRAM.

```bash
python train.py
```

# Generate

Needs roughly 20 GB memory.

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
