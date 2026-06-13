"""Convert DiscordChatExporter JSON exports into a ChatML-formatted dataset.

Each Discord message becomes one ChatML block::

    <|im_start|>{author}[ replies to {other}]
    {content}<|im_end|>

Messages from a channel are concatenated and split into chunks that fit within
``constants.max_token_context_length`` tokens (measured with the model's real
tokenizer, split on message boundaries). Every chunk is prefixed with a small
system header naming the channel, and written as one row of a JSONL file with a
single ``text`` field — exactly what TRL's SFTTrainer consumes for
language-modeling (full-sequence-loss) training.
"""

import json
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json, cfg
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from tqdm import tqdm
from transformers import AutoProcessor

import constants

cfg.global_config.encoders[datetime] = datetime.isoformat
cfg.global_config.decoders[datetime] = datetime.fromisoformat


@dataclass_json
@dataclass
class Guild:
    id: str
    name: str
    iconUrl: str

@dataclass_json
@dataclass
class Channel:
    id: str
    type: str
    categoryId: Optional[str]
    category: Optional[str]
    name: str
    topic: Optional[str]

@dataclass_json
@dataclass
class DataRange:
    after: Optional[datetime]
    before: Optional[datetime]

@dataclass_json
@dataclass
class Attachment:
    id: str
    url: str
    fileName: str
    fileSizeBytes: int

@dataclass_json
@dataclass
class Image:
    url: str
    width: int
    height: int

@dataclass_json
@dataclass
class Field:
    name: str
    value: str
    isInline: bool

@dataclass_json
@dataclass
class Embed:
    title: str
    url: Optional[str]
    timestamp: Optional[datetime]
    description: str
    images: List[Image]
    fields: List[Field]
    thumbnail: Optional[Image] = field(default=None)

@dataclass_json
@dataclass
class Sticker:
    id: str
    name: str
    format: str
    sourceUrl: str

@dataclass_json
@dataclass
class Emoji:
    id: str
    name: str
    code: str
    isAnimated: bool
    imageUrl: str

@dataclass_json
@dataclass
class Role:
    id: str
    name: str
    color: Optional[str]
    position: int

@dataclass_json
@dataclass
class User:
    id: str
    name: str
    discriminator: str
    nickname: str
    isBot: bool
    avatarUrl: str
    color: Optional[str] = field(default=None)
    roles: Optional[List[Role]] = field(default=None)

@dataclass_json
@dataclass
class Reaction:
    emoji: Emoji
    count: int
    users: List[User]

@dataclass_json
@dataclass
class Reference:
    messageId: Optional[str]
    channelId: str
    guildId: Optional[str]

@dataclass_json
@dataclass
class Message:
    id: str
    type: str
    timestamp: datetime
    timestampEdited: Optional[datetime]
    callEndedTimestamp: Optional[datetime]
    isPinned: bool
    content: str
    author: User
    attachments: List[Attachment]
    embeds: List[Embed]
    stickers: List[Sticker]
    reactions: List[Reaction]
    mentions: List[User]
    reference: Optional[Reference] = field(default=None)

@dataclass_json
@dataclass
class Chat:
    guild: Guild
    channel: Channel
    dateRange: DataRange
    exportedAt: datetime
    messages: List[Message]
    messageCount: int


def sanitize(content: str) -> str:
    """Strip literal ChatML delimiters out of user content so they can't be
    confused with structural tokens during training."""
    return content.replace(constants.IM_START, "").replace(constants.IM_END, "").strip()


def message_role(message: Message, previous: Optional[Message]) -> str:
    role = message.author.name
    if previous is not None:
        role += f" replies to {previous.author.name}"
    return role


def format_block(role: str, content: str) -> str:
    return f"{constants.IM_START}{role}\n{content}{constants.IM_END}\n"


def format_message(message: Message, previous: Optional[Message]) -> str:
    return format_block(message_role(message, previous), sanitize(message.content))


def system_header(chat: Chat) -> str:
    channel = f"{chat.guild.name} - {chat.channel.name}"
    return f"{constants.IM_START}system\nChannel: {channel}{constants.IM_END}\n"


def chunk_channel(chat: Chat, tokenizer, max_tokens: int):
    """Yield ChatML text chunks for one channel, each <= max_tokens tokens."""
    header = system_header(chat)
    header_len = len(tokenizer.encode(header))
    budget = max_tokens - header_len

    id_to_message = {}
    buffer: List[str] = []
    buffer_len = 0

    def flush():
        nonlocal buffer, buffer_len
        if buffer:
            yield_text = header + "".join(buffer)
            buffer = []
            buffer_len = 0
            return yield_text
        return None

    for message in chat.messages:
        id_to_message[message.id] = message
        previous = None
        if (
            message.type == "Reply"
            and message.reference is not None
            and message.reference.messageId in id_to_message
        ):
            previous = id_to_message[message.reference.messageId]

        if not sanitize(message.content):
            continue

        block = format_message(message, previous)
        block_len = len(tokenizer.encode(block))

        # A single oversized message: truncate only its content, keeping the
        # role prefix and closing <|im_end|> intact so delimiters stay valid.
        if block_len > budget:
            role = message_role(message, previous)
            empty_len = len(tokenizer.encode(format_block(role, "")))
            if empty_len >= budget:
                continue  # role line alone doesn't fit; drop the message
            content_ids = tokenizer.encode(sanitize(message.content))[: budget - empty_len]
            content = tokenizer.decode(content_ids, skip_special_tokens=True)
            block = format_block(role, content)
            block_len = len(tokenizer.encode(block))

        if buffer_len + block_len > budget:
            text = flush()
            if text is not None:
                yield text

        buffer.append(block)
        buffer_len += block_len

    text = flush()
    if text is not None:
        yield text


def main():
    tokenizer = AutoProcessor.from_pretrained(constants.model_name, token=constants.hf_token).tokenizer

    in_path = Path(constants.discord_json_dir)
    out_path = Path(constants.dataset_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_chunks = 0
    with open(out_path, "w") as out_file:
        for file_path in sorted(in_path.glob("*.json")):
            print("Loading from", file_path)
            with open(file_path) as f:
                chat = Chat.from_json(f.read())

            for text in tqdm(
                chunk_channel(chat, tokenizer, constants.max_token_context_length),
                desc=f"{chat.guild.name} - {chat.channel.name}",
            ):
                out_file.write(json.dumps({"text": text}) + "\n")
                n_chunks += 1

    print(f"Wrote {n_chunks} chunks to {out_path}")


if __name__ == "__main__":
    main()
