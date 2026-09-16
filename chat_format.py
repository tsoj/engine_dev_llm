"""The token-level format of a multi-party channel, shared by data.py and generation.

A channel is encoded as::

    <bos><|turn>system
    Channel: {guild} - {channel}<turn|>
    <|turn>{author}[ replies to {other}]
    {content}<turn|>
    <|turn>...

Everything is built directly as token ids (never by re-tokenizing a decoded
string), so training data and generation prompts are guaranteed to match.
"""

from dataclasses import dataclass

from model_spec import TURN_END, TURN_START, check_tokenizer

SYSTEM_ROLE = "system"


@dataclass(frozen=True)
class Message:
    author: str
    content: str
    reply_to: str | None = None


class ChatFormat:
    def __init__(self, tokenizer):
        check_tokenizer(tokenizer)
        self.tokenizer = tokenizer
        self.bos_id = tokenizer.bos_token_id
        self.turn_end_id = tokenizer.convert_tokens_to_ids(TURN_END)
        self.newline_ids = self.encode("\n")
        # Longest first, so e.g. "<|image|>" is removed before a shorter token it contains.
        self._special_tokens = sorted(
            (t.content for t in tokenizer.added_tokens_decoder.values() if t.special), key=len, reverse=True
        )

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def sanitize(self, text: str) -> str:
        """Remove special token strings from user text; the tokenizer would otherwise parse
        e.g. a literal "<turn|>" or "<|image|>" in a message as the real control token."""
        for token in self._special_tokens:
            text = text.replace(token, "")
        return text.strip()

    def role(self, author: str, reply_to: str | None = None) -> str:
        def clean(name: str) -> str:
            name = " ".join(self.sanitize(name).split())  # the role must stay on one line
            return f"{name} (user)" if name.lower() == SYSTEM_ROLE else name

        role = clean(author)
        if reply_to is not None:
            role += f" replies to {clean(reply_to)}"
        return role

    def header_ids(self, channel: str) -> list[int]:
        text = f"{TURN_START}{SYSTEM_ROLE}\nChannel: {self.sanitize(channel)}{TURN_END}\n"
        return [self.bos_id] + self.encode(text)

    def turn_start_ids(self, role: str) -> list[int]:
        """The start of a turn, up to where the content begins. An empty role
        yields just the turn token, leaving the choice of speaker to the model."""
        return self.encode(f"{TURN_START}{role}\n" if role else TURN_START)

    def turn_end_ids(self) -> list[int]:
        return [self.turn_end_id] + self.newline_ids

    def message_ids(self, message: Message, max_tokens: int | None = None) -> list[int] | None:
        """Token ids of one complete turn. Content is truncated so the whole turn fits in
        max_tokens; returns None if even an empty turn wouldn't fit or there is no content."""
        content = self.sanitize(message.content)
        if not content:
            return None
        prefix = self.turn_start_ids(self.role(message.author, message.reply_to))
        suffix = self.turn_end_ids()
        content_ids = self.encode(content)
        if max_tokens is not None:
            room = max_tokens - len(prefix) - len(suffix)
            if room <= 0:
                return None
            content_ids = content_ids[:room]
        return prefix + content_ids + suffix
