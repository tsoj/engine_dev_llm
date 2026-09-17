"""Build a tokenized training dataset from DiscordChatExporter JSON exports.

    ./run.sh data.py --json_dirs data/discord_json_data --output_dir data/dataset

Each channel is converted to the format in chat_format.py and cut into chunks of
at most --max_length tokens on message boundaries. The last --eval_fraction of
each channel's chunks (i.e. its most recent messages) is held out for
evaluation, so eval chunks are never interleaved with training chunks.

The output is a Hugging Face DatasetDict with "train" and (unless empty) "eval" splits holding
pre-tokenized "input_ids", plus a meta.json that train.py uses to verify the
dataset matches the model it is about to train.
"""

import fnmatch
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from transformers import HfArgumentParser

import model_spec
from chat_format import ChatFormat, Message

DATASET_FORMAT_VERSION = 1

# DiscordChatExporter message types that are actual chat messages. Everything
# else (joins, pins, poll results, slash command responses, ...) is skipped.
CHAT_MESSAGE_TYPES = {"Default", "Reply"}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".avif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv"}
AUDIO_EXTENSIONS = {".mp3", ".ogg", ".wav", ".m4a", ".flac"}


@dataclass
class DataConfig:
    json_dirs: list[str] = field(
        default_factory=lambda: ["data/discord_json_data"],
        metadata={"help": "Directories containing DiscordChatExporter *.json exports."},
    )
    output_dir: str = "data/dataset"
    model_name: str = model_spec.DEFAULT_MODEL_NAME
    max_length: int = field(
        default=2048,
        metadata={"help": "Tokens per training chunk (dozens of chat messages); also the context at generation."},
    )
    eval_fraction: float = field(
        default=0.02,
        metadata={"help": "Most recent fraction of each channel held out. 0 = train on everything, 1 = eval only."},
    )
    exclude_channels: list[str] = field(
        default_factory=lambda: [
            "*counting*",
            "*memes*",
            "*music*",
            "*song-of-the-day*",
            "*bots*",
            "*rules*",
            "*welcome*",
            "*announcements*",
            "*stream-schedule*",
        ],
        metadata={"help": 'Case-insensitive glob patterns matched against "Guild - channel".'},
    )
    speaker_name: Literal["nickname", "username"] = field(
        default="nickname",
        metadata={"help": "Server nickname matches how @mentions appear in message content."},
    )
    include_bots: bool = False
    attachment_placeholders: bool = field(
        default=True, metadata={"help": 'Represent attachments/stickers as e.g. "[image]" instead of dropping them.'}
    )


def attachment_placeholder(file_name: str) -> str:
    extension = Path(file_name).suffix.lower()
    if extension in IMAGE_EXTENSIONS:
        return "[image]"
    if extension in VIDEO_EXTENSIONS:
        return "[video]"
    if extension in AUDIO_EXTENSIONS:
        return "[audio]"
    return "[file]"


def load_channel(path: Path, cfg: DataConfig) -> tuple[str, list[Message]]:
    """Read one export. Only the few fields we need are accessed, so changes to
    unrelated parts of the DiscordChatExporter format don't break this."""
    with open(path, encoding="utf-8") as f:
        export = json.load(f)
    channel = f"{export['guild']['name']} - {export['channel']['name']}"

    def speaker(author: dict) -> str:
        if cfg.speaker_name == "nickname":
            return author.get("nickname") or author["name"]
        return author["name"]

    speakers_by_id: dict[str, str] = {}
    messages = []
    for message in export["messages"]:
        if message["type"] not in CHAT_MESSAGE_TYPES:
            continue
        if message["author"].get("isBot") and not cfg.include_bots:
            continue

        content = message["content"].strip()
        if cfg.attachment_placeholders:
            extras = [attachment_placeholder(a["fileName"]) for a in message.get("attachments", [])]
            extras += [f"[sticker: {s['name']}]" for s in message.get("stickers", [])]
            content = "\n".join(part for part in [content, " ".join(extras)] if part)
        if not content:
            continue

        author = speaker(message["author"])
        speakers_by_id[message["id"]] = author
        reference = message.get("reference") or {}
        reply_to = speakers_by_id.get(reference.get("messageId")) if message["type"] == "Reply" else None
        messages.append(Message(author=author, content=content, reply_to=reply_to))

    return channel, messages


def chunk_channel(
    chat_format: ChatFormat, channel: str, messages: list[Message], max_length: int
) -> Iterator[list[int]]:
    """Yield token chunks of at most max_length, each starting with the channel header."""
    header = chat_format.header_ids(channel)
    budget = max_length - len(header)
    buffer: list[int] = []
    for message in messages:
        ids = chat_format.message_ids(message, max_tokens=budget)
        if ids is None:
            continue
        if buffer and len(buffer) + len(ids) > budget:
            yield header + buffer
            buffer = []
        buffer.extend(ids)
    if buffer:
        yield header + buffer


def is_excluded(channel: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(channel.lower(), pattern.lower()) for pattern in patterns)


def main():
    (cfg,) = HfArgumentParser(DataConfig).parse_args_into_dataclasses()
    output_dir = Path(cfg.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists; delete it or pick another --output_dir.")

    files = sorted(p for d in cfg.json_dirs for p in Path(d).glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No *.json exports found in {cfg.json_dirs}.")

    tokenizer = model_spec.load_tokenizer(cfg.model_name)
    chat_format = ChatFormat(tokenizer)

    splits: dict[str, dict[str, list]] = {
        "train": {"input_ids": [], "channel": []},
        "eval": {"input_ids": [], "channel": []},
    }
    excluded = []
    for path in tqdm(files, desc="Channels"):
        channel, messages = load_channel(path, cfg)
        if is_excluded(channel, cfg.exclude_channels):
            excluded.append(channel)
            continue
        chunks = [np.array(c, dtype=np.int32) for c in chunk_channel(chat_format, channel, messages, cfg.max_length)]
        n_eval = round(len(chunks) * cfg.eval_fraction)
        for split, split_chunks in [
            ("train", chunks[: len(chunks) - n_eval]),
            ("eval", chunks[len(chunks) - n_eval :]),
        ]:
            splits[split]["input_ids"].extend(split_chunks)
            splits[split]["channel"].extend([channel] * len(split_chunks))

    if excluded:
        print(f"Excluded {len(excluded)} channels: {', '.join(excluded)}")
    if cfg.eval_fraction > 0 and not splits["eval"]["input_ids"]:
        print("Warning: the eval split is empty (too little data for --eval_fraction).")

    # Empty splits are left out: `datasets` can't load an empty split back from disk.
    dataset = DatasetDict(
        {split: Dataset.from_dict(columns) for split, columns in splits.items() if columns["input_ids"]}
    )
    dataset.save_to_disk(output_dir)

    token_counts = {split: int(sum(len(ids) for ids in columns["input_ids"])) for split, columns in splits.items()}
    meta = {
        "format_version": DATASET_FORMAT_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "tokenizer_fingerprint": model_spec.tokenizer_fingerprint(tokenizer),
        "chunks": {split: len(columns["input_ids"]) for split, columns in splits.items()},
        "tokens": token_counts,
        "channels": sorted(set(splits["train"]["channel"]) | set(splits["eval"]["channel"])),
        **asdict(cfg),
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    for split in splits:
        print(f"{split}: {meta['chunks'][split]} chunks, {token_counts[split]:,} tokens")
    if splits["train"]["input_ids"]:
        print("\nStart of the first training chunk:\n")
        print(chat_format.decode(splits["train"]["input_ids"][0][:300].tolist()))
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
