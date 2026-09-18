"""Loading a trained model and generating chat with it, shared by generate.py, sample.py,
evaluate.py and bot.py."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

import model_spec
from chat_format import ChatFormat, Message
from model_spec import TURN_START

# Upper bound on the length of a role line like "alice replies to bob".
MAX_ROLE_TOKENS = 48


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

    def add_message(self, author: str, content: str, reply_to: str | None = None) -> None:
        ids = self.chat_format.message_ids(Message(author, content, reply_to), max_tokens=self._limit // 2)
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
    def _sample(self, prefix: list[int], max_new_tokens: int, stop_ids=(), streamer=None) -> tuple[list[int], bool]:
        """Sample a continuation of the transcript plus prefix, up to the end of the turn or one of
        stop_ids. Returns the new ids without the stop token, and whether stop_ids ended it."""
        prompt = self._prompt(prefix)
        if self._cache is None or prompt[: len(self._cached_ids)] != self._cached_ids:
            self._cache = None

        end_ids = [self.chat_format.turn_end_id, self.chat_format.tokenizer.eos_token_id]
        input_ids = torch.tensor([prompt], device=self.model.device)
        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            past_key_values=self._cache,
            do_sample=True,
            temperature=self.sampling.temperature,
            top_p=self.sampling.top_p,
            top_k=self.sampling.top_k,
            max_new_tokens=max_new_tokens,
            eos_token_id=[*end_ids, *stop_ids],
            pad_token_id=self.chat_format.tokenizer.pad_token_id,
            streamer=streamer,
            return_dict_in_generate=True,
        )
        sequence = output.sequences[0].tolist()
        # The cache covers every token except the last sampled one.
        self._cache = output.past_key_values
        self._cached_ids = sequence[:-1]

        new_ids = sequence[len(prompt) :]
        stopped = bool(new_ids) and new_ids[-1] in stop_ids
        if new_ids and (stopped or new_ids[-1] in end_ids):
            new_ids = new_ids[:-1]
        return new_ids, stopped

    def generate_turn(self, role: str = "", streamer=None) -> list[int]:
        """Generate one message. With an empty role the model also picks the speaker."""
        prefix = self.chat_format.turn_start_ids(role)
        new_ids, _ = self._sample(prefix, self.sampling.max_new_tokens, streamer=streamer)
        turn = prefix + new_ids + self.chat_format.turn_end_ids()
        self.turns.append(turn)
        return turn

    def predict_role(self) -> str:
        """Who the model expects to write next, as a role line ("alice" or "alice replies to bob").
        Only the role line is generated, so this is much cheaper than generating the message. The
        transcript is unchanged; pass the role to generate_turn to write the message itself."""
        line_breaks = self.chat_format.line_break_ids
        new_ids, complete = self._sample(self.chat_format.turn_start_ids(""), MAX_ROLE_TOKENS, stop_ids=line_breaks)
        return self.chat_format.decode(new_ids).strip() if complete else ""

    def undo_turn(self) -> list[int] | None:
        """Take back the last turn. The cache still holds its tokens and a sliding-window cache
        cannot be cropped back, so it is dropped: the next generation re-reads the prompt."""
        if not self.turns:
            return None
        self._cache, self._cached_ids = None, []
        return self.turns.pop()

    def turn_role(self, turn: list[int]) -> str:
        """The role line of a turn from generate_turn, e.g. "alice" or "alice replies to bob"."""
        text = self.chat_format.decode(turn).replace(model_spec.TURN_START, "", 1)
        return text.split("\n", 1)[0].strip()

    def turn_content(self, turn: list[int]) -> str:
        """The message text of a turn from generate_turn, without the role line."""
        text = self.chat_format.decode(turn).replace(model_spec.TURN_START, "", 1)
        return text.split("\n", 1)[1].replace(model_spec.TURN_END, "").strip() if "\n" in text else ""

    def to_text(self, turns: list[list[int]] | None = None) -> str:
        """Human-readable transcript (without the special tokens)."""
        turns = self.turns if turns is None else turns
        text = self.chat_format.decode([t for turn in turns for t in turn])
        text = text.replace(self.chat_format.tokenizer.bos_token, "").replace(model_spec.TURN_START, "")
        return text.replace(model_spec.TURN_END, "\n")  # blank line between messages
