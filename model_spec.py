"""Everything that is specific to the target model: Gemma 4 12B ("Unified").

The rest of the repo relies on the assumptions checked here (special tokens, LoRA
module names, attention implementation, ...). All scripts call these checks
before doing real work, so switching to a different model fails loudly instead
of silently training or sampling something broken.

To switch models: go through every check below, verify the assumption still
holds for the new model (or adapt the code that relies on it), and only then
update the expected values.
"""

import json
import re
from pathlib import Path
from typing import Literal

import torch
from transformers import (
    AutoConfig,
    AutoModelForMultimodalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# Base (pretrained, not instruction-tuned) model: we train plain language
# modeling on chat logs, so there is no assistant tuning to fight against. The
# instruct variant (google/gemma-4-12B-it) passes the same checks.
DEFAULT_MODEL_NAME = "google/gemma-4-12B"

EXPECTED_ARCHITECTURE = "Gemma4UnifiedForConditionalGeneration"
EXPECTED_MODEL_TYPE = "gemma4_unified"
# Pins the size (12B). Other sizes may need different memory settings.
EXPECTED_NUM_LAYERS = 48
EXPECTED_HIDDEN_SIZE = 3840

# Gemma's turn delimiters. Each is a single special token, so using them for our
# multi-party chat format needs no embedding resize. Any author name is used as
# the "role" after TURN_START.
TURN_START = "<|turn>"
TURN_END = "<turn|>"
EXPECTED_TOKEN_IDS = {
    "<pad>": 0,
    "<eos>": 1,
    "<bos>": 2,
    TURN_START: 105,
    TURN_END: 106,
}

# LoRA goes on the attention and MLP projections of the text decoder only. The
# multimodal embedders (model.embed_vision / model.embed_audio) and lm_head (tied
# to the input embeddings) stay frozen.
LORA_TARGET_REGEX = r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"

Quantization = Literal["4bit", "8bit", "none"]


class ModelSpecError(RuntimeError):
    def __init__(self, message: str):
        super().__init__(
            f"{message}\n\nThis repo is written for {EXPECTED_ARCHITECTURE} (Gemma 4 12B). "
            "If you are intentionally switching models, review the assumptions in model_spec.py first."
        )


def check_config(config) -> None:
    architectures = getattr(config, "architectures", None) or []
    if EXPECTED_ARCHITECTURE not in architectures:
        raise ModelSpecError(f"Unexpected architecture {architectures}, expected {EXPECTED_ARCHITECTURE}.")
    if config.model_type != EXPECTED_MODEL_TYPE:
        raise ModelSpecError(f"Unexpected model_type {config.model_type!r}, expected {EXPECTED_MODEL_TYPE!r}.")

    text_config = config.get_text_config()
    if text_config.num_hidden_layers != EXPECTED_NUM_LAYERS or text_config.hidden_size != EXPECTED_HIDDEN_SIZE:
        raise ModelSpecError(
            f"Unexpected model size: {text_config.num_hidden_layers} layers, hidden size "
            f"{text_config.hidden_size} (expected {EXPECTED_NUM_LAYERS} / {EXPECTED_HIDDEN_SIZE})."
        )
    # We use the "sdpa" attention implementation, which can't do attention logit
    # softcapping (Gemma 2 needed "eager" for that).
    if getattr(text_config, "attn_logit_softcapping", None):
        raise ModelSpecError("Model uses attention logit softcapping, which the sdpa attention implementation ignores.")


def check_tokenizer(tokenizer) -> None:
    for token, expected_id in EXPECTED_TOKEN_IDS.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if token_id != expected_id or encoded != [expected_id]:
            raise ModelSpecError(
                f"Token {token!r} has id {token_id} and encodes to {encoded}, expected the single id {expected_id}."
            )
    if tokenizer.bos_token_id != EXPECTED_TOKEN_IDS["<bos>"] or tokenizer.pad_token_id != EXPECTED_TOKEN_IDS["<pad>"]:
        raise ModelSpecError(
            f"Unexpected bos/pad ids {tokenizer.bos_token_id}/{tokenizer.pad_token_id}, "
            f"expected {EXPECTED_TOKEN_IDS['<bos>']}/{EXPECTED_TOKEN_IDS['<pad>']}."
        )


def tokenizer_fingerprint(tokenizer) -> str:
    """Stable hash of the vocabulary, to detect datasets built with another tokenizer."""
    import hashlib

    vocab = json.dumps(sorted(tokenizer.get_vocab().items()), ensure_ascii=False)
    return hashlib.sha256(vocab.encode()).hexdigest()


def expected_lora_targets(config) -> set[str]:
    """Names of all modules LORA_TARGET_REGEX should match, derived from the config.

    Global (full attention) layers with attention_k_eq_v reuse the keys as
    values and have no v_proj.
    """
    text_config = config.get_text_config()
    k_eq_v = getattr(text_config, "attention_k_eq_v", False)
    names = set()
    for i, layer_type in enumerate(text_config.layer_types):
        prefix = f"model.language_model.layers.{i}"
        projections = ["q", "k", "o"] if (k_eq_v and layer_type == "full_attention") else ["q", "k", "v", "o"]
        names.update(f"{prefix}.self_attn.{p}_proj" for p in projections)
        names.update(f"{prefix}.mlp.{p}_proj" for p in ["gate", "up", "down"])
    return names


def check_lora_targets(model) -> None:
    """Verify LORA_TARGET_REGEX matches exactly the decoder projections (call before wrapping with PEFT)."""
    matched = {
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and re.fullmatch(LORA_TARGET_REGEX, name)
    }
    expected = expected_lora_targets(model.config)
    if matched != expected:
        missing = sorted(expected - matched)[:5]
        unexpected = sorted(matched - expected)[:5]
        raise ModelSpecError(
            f"LoRA target regex matched {len(matched)} modules, expected {len(expected)}. "
            f"Missing (first 5): {missing}. Unexpected (first 5): {unexpected}."
        )


def load_config(name_or_path: str):
    config = AutoConfig.from_pretrained(name_or_path)
    check_config(config)
    return config


def load_tokenizer(name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    check_tokenizer(tokenizer)
    return tokenizer


def load_processor(name_or_path: str):
    processor = AutoProcessor.from_pretrained(name_or_path)
    check_tokenizer(processor.tokenizer)
    return processor


def load_model(name_or_path: str, quantization: Quantization, device_map: str | dict | None = "auto"):
    # Checks the config (a few KB) before downloading ~24 GB of weights.
    load_config(name_or_path)

    if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        # Gemma activations overflow in fp16.
        raise RuntimeError("This GPU doesn't support bfloat16, which Gemma needs (fp16 overflows).")

    quantization_config = None
    if quantization == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    return AutoModelForMultimodalLM.from_pretrained(
        name_or_path,
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="sdpa",  # works on CUDA, ROCm and CPU; handles the sliding-window mask
    )


def adapter_base_model(adapter_path: str | Path) -> str:
    """The base model a PEFT adapter directory was trained on."""
    config_path = Path(adapter_path) / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} not found; is {adapter_path} a LoRA adapter directory?")
    return json.loads(config_path.read_text())["base_model_name_or_path"]
