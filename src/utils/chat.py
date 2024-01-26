from enum import Enum
from typing import List
from pydantic import BaseModel
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.schema.messages import AIMessage, ChatMessage, HumanMessage, SystemMessage

# Your original Message class
class Message(BaseModel):
    role: str
    content: str

def chat_messages_otemplate(messages: List[Message]) -> List[ChatMessage]:
    for message in messages:
        message.content = f"{message.content}"
    return messages

def chat_messages_template(messages: List[Message]) -> List[ChatMessage]:
    for message in messages:
        message.content = f"<|im_start|>{message.content}<|im_end|>\n"
    return messages

def chat_messages_embeds(messages: List[Message]) -> List[ChatMessage]:
    chat_embeds = []
    for message in messages:
        if type(message) == HumanMessage:
            chat_embeds.append(Embed(
                description=f"{message.content[:4094]}",
                author = EmbedAuthor(name=f"{ctx.author.display_name}",icon_url=f"{ctx.author.avatar_url}", url=f"{DOLPHIN_EMBED_URL}"),
            ))
        elif type(message) == AIMessage:  
            if len(message.content) <= 4094:
                chat_embeds.append(Embed(
                    description=f"{message.content[:4094]}",
                    author = EmbedAuthor(name=f"{self.client.app.name}",icon_url=f"https://cdn.discordapp.com/avatars/1190425430302404650/f6e0920c4861c7aec9ed972018a5d2f8.webp?size=320", url=f"{DOLPHIN_EMBED_URL}"),
                ))
            elif len(message.content) > 4094:
                chat_embeds.append(Embed(
                    description=f"{message.content[:4094]}",
                    author = EmbedAuthor(name=f"{self.client.app.name}",icon_url=f"https://cdn.discordapp.com/avatars/1190425430302404650/f6e0920c4861c7aec9ed972018a5d2f8.webp?size=320", url=f"{DOLPHIN_EMBED_URL}"),
                ))
                chat_embeds.append(Embed(
                    description=f"{message.content[4095:6000]}",
                ))
    return chat_embeds

def chat_messages_regenerate(id: str) -> List[ChatMessage]:
    chats = []
    history = RedisChatMessageHistory(id, url="redis://localhost:6379")
    chats.extend(history.messages)
    chats.pop()
    print(chats)
    return chats