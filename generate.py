"""Simulate a channel with a trained model.

    # let the model write 30 messages
    ./run.sh generate.py --model runs/my-run/adapter

    # pick a channel and take part in the conversation
    ./run.sh generate.py --model runs/my-run/adapter --channel "My Server - general" --interactive

--model can be a LoRA adapter (runs/*/adapter or runs/*/checkpoints/checkpoint-*),
a merged model (runs/*/merged), or a base model name. By default the weights are
loaded in 4-bit, which needs roughly 10 GB of VRAM for Gemma 4 12B.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import HfArgumentParser, TextStreamer, set_seed

import model_spec
from chat_format import ChatFormat, Message
from model_spec import TURN_START


@dataclass
class SamplingConfig:
    # Gemma's recommended sampling settings. No repetition penalty: chat logs
    # legitimately repeat names and phrases.
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    max_new_tokens: int = field(default=256, metadata={"help": "Upper bound on the length of one message."})
    context_tokens: int | None = field(
        default=None, metadata={"help": "Defaults to the max_length the model was trained with."}
    )


@dataclass
class GenerateConfig:
    model: str = field(metadata={"help": "Adapter dir, merged model dir, or base model name."})
    channel: str | None = field(
        default=None, metadata={"help": "Defaults to the first channel the run was trained on."}
    )
    max_messages: int = 30
    interactive: bool = False
    quantization: model_spec.Quantization | None = field(
        default=None, metadata={"help": 'Defaults to "4bit" on GPU and "none" on CPU.'}
    )
    seed: int | None = None


def trained_dataset(model_path: str | Path) -> dict | None:
    """The meta of the dataset a run was trained on (channels, max_length, ...), found via the
    train_config_*.json that train.py writes into the run directory (model_path is
    runs/<run>/adapter or runs/<run>/checkpoints/checkpoint-*)."""
    path = Path(model_path)
    for directory in [path, path.parent, path.parent.parent]:
        configs = sorted(directory.glob("train_config_*.json"))
        if configs:
            return json.loads(configs[-1].read_text())["dataset"]
    return None


def trained_context_tokens(model_path: str | Path) -> int | None:
    dataset = trained_dataset(model_path)
    return dataset["max_length"] if dataset else None


def resolve_context_tokens(sampling: SamplingConfig, trained: int | None) -> None:
    if sampling.context_tokens is None:
        if trained is None:
            raise ValueError("Can't determine the training context length; pass --context_tokens.")
        sampling.context_tokens = trained
    elif trained is not None and sampling.context_tokens > trained:
        print(f"Warning: --context_tokens {sampling.context_tokens} exceeds the training length {trained}.")


def split_chunk(chat_format: ChatFormat, ids: list[int]) -> tuple[list[int], list[list[int]]]:
    """Split a data.py chunk ([bos] + header turn + message turns) into header and turns."""
    turn_start_id = chat_format.tokenizer.convert_tokens_to_ids(TURN_START)
    starts = [i for i, token in enumerate(ids) if token == turn_start_id] + [len(ids)]
    if ids[0] != chat_format.bos_id or starts[0] != 1:
        raise ValueError("ids don't start with <bos> and the channel header.")
    turns = [ids[a:b] for a, b in zip(starts[1:], starts[2:], strict=False)]
    return ids[: starts[1]], turns


def load_for_inference(path: str, quantization: model_spec.Quantization | None):
    """Load a model (with its LoRA adapter, if path is one) and its tokenizer."""
    if quantization is None:
        quantization = "4bit" if torch.cuda.is_available() else "none"
    adapter = Path(path, "adapter_config.json").exists()
    base = model_spec.adapter_base_model(path) if adapter else path
    model = model_spec.load_model(base, quantization)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, path)
    model.eval()
    has_tokenizer = Path(path, "tokenizer_config.json").exists()
    tokenizer = model_spec.load_tokenizer(path if has_tokenizer else base)
    return model, tokenizer


class Conversation:
    """A channel transcript as token ids, with a KV cache that is reused across turns."""

    def __init__(
        self,
        model,
        chat_format: ChatFormat,
        header: list[int],
        sampling: SamplingConfig,
        turns: list[list[int]] | None = None,
    ):
        if sampling.context_tokens is None:
            raise ValueError("sampling.context_tokens must be set (see resolve_context_tokens).")
        self.model = model
        self.chat_format = chat_format
        self.sampling = sampling
        self.header = header
        self.turns = list(turns or [])
        self._cache = None
        self._cached_ids: list[int] = []
        # Leave room for the message being generated.
        self._limit = sampling.context_tokens - sampling.max_new_tokens
        if self._limit <= sampling.context_tokens // 4:
            raise ValueError("max_new_tokens is too large for context_tokens.")

    @classmethod
    def from_ids(cls, model, chat_format: ChatFormat, ids: list[int], sampling: SamplingConfig):
        header, turns = split_chunk(chat_format, ids)
        return cls(model, chat_format, header, sampling, turns)

    def add_message(self, author: str, content: str) -> None:
        ids = self.chat_format.message_ids(Message(author, content), max_tokens=self._limit // 2)
        if ids is not None:
            self.turns.append(ids)

    def _prompt(self, prefix: list[int]) -> list[int]:
        def length() -> int:
            return len(self.header) + sum(map(len, self.turns)) + len(prefix)

        if length() > self._limit:
            # Drop the oldest turns with some slack, so the KV cache stays valid for a few turns.
            while self.turns and length() > self._limit * 3 // 4:
                self.turns.pop(0)
        return self.header + [t for turn in self.turns for t in turn] + prefix

    @torch.no_grad()
    def generate_turn(self, role: str = "", streamer=None) -> list[int]:
        """Generate one message. With an empty role the model also picks the speaker."""
        prefix = self.chat_format.turn_start_ids(role)
        prompt = self._prompt(prefix)

        cache_len = len(self._cached_ids)
        if self._cache is None or prompt[:cache_len] != self._cached_ids:
            self._cache, cache_len = None, 0

        stop_ids = [self.chat_format.turn_end_id, self.chat_format.tokenizer.eos_token_id]
        input_ids = torch.tensor([prompt], device=self.model.device)
        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            past_key_values=self._cache,
            do_sample=True,
            temperature=self.sampling.temperature,
            top_p=self.sampling.top_p,
            top_k=self.sampling.top_k,
            max_new_tokens=self.sampling.max_new_tokens,
            eos_token_id=stop_ids,
            pad_token_id=self.chat_format.tokenizer.pad_token_id,
            streamer=streamer,
            return_dict_in_generate=True,
        )
        sequence = output.sequences[0].tolist()
        # The cache covers every token except the last sampled one.
        self._cache = output.past_key_values
        self._cached_ids = sequence[:-1]

        new_ids = sequence[len(prompt) :]
        if new_ids and new_ids[-1] in stop_ids:
            new_ids = new_ids[:-1]
        turn = prefix + new_ids + self.chat_format.turn_end_ids()
        self.turns.append(turn)
        return turn

    def to_text(self, turns: list[list[int]] | None = None) -> str:
        """Human-readable transcript (without the special tokens)."""
        turns = self.turns if turns is None else turns
        text = self.chat_format.decode([t for turn in turns for t in turn])
        text = text.replace(self.chat_format.tokenizer.bos_token, "").replace(model_spec.TURN_START, "")
        return text.replace(model_spec.TURN_END, "\n")  # blank line between messages


def main():
    cfg, sampling = HfArgumentParser((GenerateConfig, SamplingConfig)).parse_args_into_dataclasses()
    if cfg.seed is not None:
        set_seed(cfg.seed)

    print("Using GPU:", torch.cuda.get_device_name() if torch.cuda.is_available() else "none (CPU)")
    dataset_meta = trained_dataset(cfg.model)
    if dataset_meta is None and sampling.context_tokens is None:
        sampling.context_tokens = 1024
        print("Not a runs/ directory, using --context_tokens 1024.")
    resolve_context_tokens(sampling, dataset_meta["max_length"] if dataset_meta else None)

    channels = (dataset_meta or {}).get("channels", [])
    channel = cfg.channel or next(iter(channels), None)
    if channel is None:
        raise ValueError("Pass --channel: the run directory doesn't say which channels it was trained on.")
    if channels and channel not in channels:
        print(f"Warning: {channel!r} is not one of the {len(channels)} trained channels, e.g. {channels[:3]}.")
    elif cfg.channel is None and channels:
        print(f"Using the first of {len(channels)} trained channels; pass --channel to pick another.")

    model, tokenizer = load_for_inference(cfg.model, cfg.quantization)
    chat_format = ChatFormat(tokenizer)
    conversation = Conversation(model, chat_format, chat_format.header_ids(channel), sampling)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    print(f"\n# {channel}")
    if cfg.interactive:
        print(
            "Enter: blank = model picks the next speaker, 'name' = model writes as name, "
            "'name: message' = you write as name, 'q' = quit."
        )

    for _ in range(cfg.max_messages):
        role = ""
        if cfg.interactive:
            line = input("\n> ").strip()
            if line == "q":
                break
            if ": " in line:
                author, content = line.split(": ", 1)
                conversation.add_message(author, content)
                continue
            role = line
        print(f"\n{role}" if role else "", flush=True)
        conversation.generate_turn(role, streamer=streamer)


if __name__ == "__main__":
    main()
