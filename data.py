from dataclasses import dataclass, field
from dataclasses_json import dataclass_json, config, cfg
from datetime import datetime
from typing import List, Optional
from pathlib import Path
from tqdm import tqdm

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
    categoryId: str
    category: str
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
    color: Optional[str]  = field(default=None)
    roles: Optional[List[Role]]  = field(default=None)

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


in_path = Path("data/engine_programming_discord")
out_path = Path("data/text")

for file_path in in_path.glob("*.json"):
    print("Loading from", file_path)

    with open(file_path, 'r') as file:

        chat = Chat.from_json(file.read())
        id_to_message = {}

        out_file_name = out_path / (chat.guild.name + " - " + chat.channel.name + ".txt")
        with open(out_file_name, 'w') as file:
            for message in tqdm(chat.messages):
                assert message.id not in id_to_message
                id_to_message[message.id] = message

                file.write("<|" + message.author.name)
                if message.type == "Reply" and message.reference.messageId in id_to_message:
                    previous_message = id_to_message[message.reference.messageId]
                    file.write(" replies to " + previous_message.author.name)

                file.write("|>\n")
                file.write(message.content)
                file.write("</s>\n\n")

        print("Finished", out_file_name)
