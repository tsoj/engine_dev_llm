import os

# Model to fine-tune. Qwen3.6-27B (April 2026) is a multimodal causal LM
# (Qwen3_5ForConditionalGeneration: text decoder + vision encoder). We train it
# on text only: the vision tower never executes for text batches, and LoRA is
# scoped to the language layers (see train.py). It must be loaded with
# AutoProcessor / AutoModelForMultimodalLM, NOT AutoTokenizer / *ForCausalLM.
# Note: it is post-trained (instruct), so there is some assistant alignment that
# the chat data overrides. Requires a very recent `transformers`.
model_name = "Qwen/Qwen3.6-27B"

# Read access token for gated models. Set it in the environment:
#   export HF_TOKEN=hf_...
hf_token = os.environ.get("HF_TOKEN")

# Sequence length used for both chunking the dataset and training (tokens).
max_token_context_length = 4096

# Paths.
data_dir = "data"
discord_json_dir = "data/discord_json_data"
dataset_path = "data/dataset.jsonl"
checkpoint_dir = "checkpoints"

# ChatML delimiters. Qwen's tokenizer already knows these as special tokens,
# so no embedding resize is needed.
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
