"""Run a trained model as a Discord bot that writes as one member of the dataset.

    ./run.sh bot.py --model runs/my-run/adapter --persona "some nickname" --token_env MY_BOT_TOKEN

The bot answers when it is mentioned or replied to, and with --reply_chance also
joins in on its own. Like people in chat, it may send a few messages in a row
(--max_messages), as long as the model predicts the next message to be the
persona's too. Each Discord channel gets its own conversation, started from the
channel's recent history. Setting up the bot account is described in the README.
"""

import asyncio
import math
import os
import random
import re
import time
from dataclasses import dataclass, field

import discord
from transformers import HfArgumentParser, set_seed

import model_spec
from chat_format import ChatFormat, author_of, with_placeholders
from inference import Conversation, SamplingConfig, load_for_inference, resolve_context_tokens, trained_dataset

DISCORD_MAX_MESSAGE_LENGTH = 2000
# What with_placeholders writes for attachments and stickers. The bot can't send those.
PLACEHOLDER = re.compile(r"\[(image|video|audio|file|sticker: [^\]]*)\]")

# Reaction time before answering: floor + a log-normal delay (quick answers are common, slow ones
# happen too). Its peak is short while the bot is in a conversation (it wrote at most ACTIVE_SECONDS
# ago) and moves back to the idle peak with this time constant once it has been quiet.
ACTIVE_SECONDS = 10
IDLE_TIME_CONSTANT_SECONDS = 60
REACTION_SIGMA = 0.5
MAX_REACTION_PEAKS = 3  # cut off the long tail at this multiple of the idle peak


@dataclass
class BotConfig:
    model: str = field(metadata={"help": "Adapter dir, checkpoint, or merged model dir."})
    persona: str = field(metadata={"help": "The member to write as, spelled as in the dataset (server nickname)."})
    token_env: str = field(default="DISCORD_BOT_TOKEN", metadata={"help": "Environment variable with the token."})
    channel: str | None = field(
        default=None,
        metadata={"help": "Trained channel name used as the prompt header. Defaults to the first trained channel."},
    )
    channel_ids: list[int] = field(
        default_factory=list, metadata={"help": "Discord channel IDs to take part in. Default: every visible one."}
    )
    history_messages: int = field(default=50, metadata={"help": "Recent messages loaded as context per channel."})
    reply_chance: float = field(
        default=0.0, metadata={"help": "Probability of writing a message after any message, without being asked."}
    )
    max_messages: int = field(
        default=5, metadata={"help": "Most messages sent in a row, while the model keeps writing as the persona."}
    )
    reaction_floor_seconds: float = field(
        default=0.2, metadata={"help": "Shortest pause before the bot starts typing an answer."}
    )
    reaction_peak_active_seconds: float = field(
        default=0.5, metadata={"help": f"Most likely pause when the bot wrote in the last {ACTIVE_SECONDS} s."}
    )
    reaction_peak_idle_seconds: float = field(
        default=4, metadata={"help": "Most likely pause when the bot hasn't written for a long time."}
    )
    typing_wpm: float = field(
        default=60,
        metadata={
            "help": "Typing speed in words per minute. Messages the model generates faster are held back "
            "until a person would have typed them. 0 posts right away."
        },
    )
    quantization: model_spec.Quantization | None = field(
        default=None, metadata={"help": 'Defaults to "4bit" on GPU and "none" on CPU.'}
    )
    seed: int | None = None


class PersonaBot(discord.Client):
    def __init__(self, cfg: BotConfig, conversation_factory):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.cfg = cfg
        self.new_conversation = conversation_factory
        self.conversations: dict[int, Conversation] = {}
        self.latest_message: dict[int, int] = {}  # channel id -> id of the newest message seen
        self.to_answer: dict[int, int] = {}  # channel id -> id of the newest message to answer
        self.last_sent: dict[int, float] = {}  # channel id -> time.monotonic() of our last message
        # One GPU: generations run one at a time, and a conversation is only
        # changed while holding the lock, so messages stay in order.
        self.lock = asyncio.Lock()

    async def on_ready(self):
        print(f"Logged in as {self.user}, writing as {self.cfg.persona!r}.")

    def speaker(self, message: discord.Message) -> str:
        return self.cfg.persona if message.author == self.user else message.author.display_name

    def content(self, message: discord.Message) -> str:
        text = message.clean_content
        if message.guild is not None:
            # Mentions of the bot become mentions of the persona.
            text = text.replace(f"@{message.guild.me.display_name}", f"@{self.cfg.persona}")
        return with_placeholders(text, [a.filename for a in message.attachments], [s.name for s in message.stickers])

    def replied_to(self, message: discord.Message) -> discord.Message | None:
        resolved = message.reference.resolved if message.reference else None
        return resolved if isinstance(resolved, discord.Message) else None

    def add(self, conversation: Conversation, message: discord.Message) -> None:
        parent = self.replied_to(message)
        conversation.add_message(self.speaker(message), self.content(message), self.speaker(parent) if parent else None)

    async def conversation(self, channel) -> Conversation:
        if channel.id not in self.conversations:
            conversation = self.new_conversation()
            history = [m async for m in channel.history(limit=self.cfg.history_messages)]
            for message in reversed(history):  # oldest first
                if message.author == self.user or not message.author.bot:
                    self.add(conversation, message)
            self.conversations[channel.id] = conversation
        return self.conversations[channel.id]

    def reaction_seconds(self, channel_id: int) -> float:
        """A random pause before starting to answer, shorter when the bot wrote recently."""
        cfg = self.cfg
        quiet = time.monotonic() - self.last_sent.get(channel_id, -math.inf)
        idle_share = 1 - math.exp(-max(0, quiet - ACTIVE_SECONDS) / IDLE_TIME_CONSTANT_SECONDS)
        peak = cfg.reaction_peak_active_seconds + idle_share * (
            cfg.reaction_peak_idle_seconds - cfg.reaction_peak_active_seconds
        )
        mode = max(peak - cfg.reaction_floor_seconds, 1e-3)  # the log-normal part's most likely value
        delay = random.lognormvariate(math.log(mode) + REACTION_SIGMA**2, REACTION_SIGMA)
        return cfg.reaction_floor_seconds + min(delay, MAX_REACTION_PEAKS * cfg.reaction_peak_idle_seconds)

    def typing_seconds(self, text: str) -> float:
        """How long a person needs to type text (a word is counted as 5 characters, as in typing tests)."""
        if self.cfg.typing_wpm <= 0:
            return 0.0
        return len(text) / (self.cfg.typing_wpm * 5 / 60)

    def interrupted(self, message: discord.Message) -> bool:
        """Someone wrote after message: don't talk over them."""
        return self.latest_message[message.channel.id] != message.id

    async def on_message(self, message: discord.Message):
        if message.author == self.user or message.author.bot:
            return  # our own turns are already in the conversation; other bots weren't in the dataset
        if self.cfg.channel_ids and message.channel.id not in self.cfg.channel_ids:
            return

        self.latest_message[message.channel.id] = message.id
        parent = self.replied_to(message)
        addressed = self.user in message.mentions or (parent is not None and parent.author == self.user)
        speak = addressed or random.random() < self.cfg.reply_chance

        async with self.lock:
            known = message.channel.id in self.conversations
            conversation = await self.conversation(message.channel)
            if known:  # a new conversation already has the message from the history
                self.add(conversation, message)
        if not speak:
            return
        self.to_answer[message.channel.id] = message.id
        # People don't answer instantly. Messages arriving meanwhile still go into the
        # conversation (the lock is free), so the answer can take them into account.
        await asyncio.sleep(self.reaction_seconds(message.channel.id))

        async with self.lock:
            if self.to_answer[message.channel.id] != message.id:
                return  # a newer message will be answered, with this one in its context
            # Answer in the format the model saw for replies, so it knows whom it is talking to.
            persona = conversation.chat_format.role(self.cfg.persona)  # the name as the model sees it
            role = conversation.chat_format.role(self.cfg.persona, message.author.display_name if addressed else None)
            for i in range(self.cfg.max_messages):
                if i > 0:
                    # Keep going only if the model expects the persona to write the next message too.
                    # Predicting just the role line is quick, so it happens without "typing...".
                    role = await asyncio.to_thread(conversation.predict_role)
                    if author_of(role) != persona or self.interrupted(message):
                        break
                async with message.channel.typing():
                    started = time.monotonic()
                    turn = await asyncio.to_thread(conversation.generate_turn, role)
                    text = conversation.turn_content(turn)[:DISCORD_MAX_MESSAGE_LENGTH]
                    postable = bool(PLACEHOLDER.sub("", text).strip())
                    if postable:
                        # Don't write faster than a person: if the model was quicker, keep "typing" a bit.
                        await asyncio.sleep(started + self.typing_seconds(text) - time.monotonic())
                if i > 0 and self.interrupted(message):
                    conversation.undo_turn()  # their message comes next, not ours
                    break
                if not postable:
                    continue  # e.g. just "[image]": stays in the context, but there is nothing to post
                if addressed and i == 0:
                    await message.reply(text, mention_author=False)
                else:
                    await message.channel.send(text)
                self.last_sent[message.channel.id] = time.monotonic()


def main():
    cfg, sampling = HfArgumentParser((BotConfig, SamplingConfig)).parse_args_into_dataclasses()
    if cfg.seed is not None:
        set_seed(cfg.seed)
    token = os.environ.get(cfg.token_env)
    if not token:
        raise ValueError(f"Set the bot token in ${cfg.token_env} (or pick another variable with --token_env).")

    dataset_meta = trained_dataset(cfg.model)
    resolve_context_tokens(sampling, dataset_meta["max_length"] if dataset_meta else None)
    channels = (dataset_meta or {}).get("channels", [])
    channel = cfg.channel or next(iter(channels), None)
    if channel is None:
        raise ValueError("Pass --channel: the run directory doesn't say which channels it was trained on.")
    if channels and channel not in channels:
        print(f"Warning: {channel!r} is not one of the {len(channels)} trained channels, e.g. {channels[:3]}.")

    model, tokenizer = load_for_inference(cfg.model, cfg.quantization)
    chat_format = ChatFormat(tokenizer)
    header = chat_format.header_ids(channel)
    print(f"Prompt header: {channel!r}")

    bot = PersonaBot(cfg, lambda: Conversation(model, chat_format, header, sampling))
    bot.run(token)


if __name__ == "__main__":
    main()
