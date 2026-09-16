"""Simulate a channel with a trained model.

    # let the model write 30 messages
    uv run python generate.py --model runs/my-run/adapter --channel "Stockfish - engines-dev"

    # take part in the conversation
    uv run python generate.py --model runs/my-run/adapter --channel "Stockfish - engines-dev" --interactive

--model can be a LoRA adapter (runs/*/adapter or runs/*/checkpoints/checkpoint-*),
a merged model (runs/*/merged), or a base model name. By default the weights are
loaded in 4-bit, which needs roughly 10 GB of VRAM for Gemma 4 12B.
"""

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
    max_new_tokens: int = field(default=512, metadata={"help": "Upper bound on the length of one message."})
    context_tokens: int = field(default=4096, metadata={"help": "Use the training max_length."})


@dataclass
class GenerateConfig:
    model: str = field(metadata={"help": "Adapter dir, merged model dir, or base model name."})
    channel: str = "Stockfish - engines-dev"
    max_messages: int = 30
    interactive: bool = False
    quantization: model_spec.Quantization | None = field(
        default=None, metadata={"help": 'Defaults to "4bit" on GPU and "none" on CPU.'}
    )
    seed: int | None = None


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
        """Split a data.py chunk ([bos] + header turn + message turns) back into turns."""
        turn_start_id = chat_format.tokenizer.convert_tokens_to_ids(TURN_START)
        starts = [i for i, token in enumerate(ids) if token == turn_start_id] + [len(ids)]
        if ids[0] != chat_format.bos_id or starts[0] != 1:
            raise ValueError("ids don't start with <bos> and the channel header.")
        turns = [ids[a:b] for a, b in zip(starts[1:], starts[2:], strict=False)]
        return cls(model, chat_format, ids[: starts[1]], sampling, turns)

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
    model, tokenizer = load_for_inference(cfg.model, cfg.quantization)
    chat_format = ChatFormat(tokenizer)
    conversation = Conversation(model, chat_format, chat_format.header_ids(cfg.channel), sampling)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    print(f"\n# {cfg.channel}")
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
