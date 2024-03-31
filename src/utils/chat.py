"""
This module contains the chat util function for bot.
"""

from typing import List
from pydantic import BaseModel
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.messages import ChatMessage

# Your original Message class


class Message(BaseModel):
    """
    A class representing a message in a conversational context.

    Attributes:
        role (str): The role of the message sender (e.g., "user", "assistant").
        content (str): The content of the message.

    This class inherits from the BaseModel class, which likely provides
    functionality for serialization/deserialization and data validation.
    """
    role: str
    content: str


def chat_messages_otemplate(messages: List[Message]) -> List[ChatMessage]:
    """
    Converts a list of Message objects to a list of ChatMessage objects.

    Args:
        messages (List[Message]): A list of Message objects.

    Returns:
        List[ChatMessage]: A list of ChatMessage objects.

    This function iterates through the provided list of Message objects
    and performs a string formatting operation on the 'content' attribute
    of each Message object. The resulting list of formatted Message objects
    is returned as a list of ChatMessage objects.

    Note: This function assumes that the ChatMessage class is compatible
    with the Message class, and that the 'content' attribute is a string.
    """
    for message in messages:
        message.content = f"{message.content}"
    return messages


def chat_messages_template(messages: List[Message]) -> List[ChatMessage]:
    """
    Converts a list of Message objects to a list of ChatMessage objects.

    Args:
        messages (List[Message]): A list of Message objects.

    Returns:
        List[ChatMessage]: A list of ChatMessage objects with ChatML format.

    This function iterates through the provided list of Message objects
    and performs a string formatting operation on the 'content' attribute
    of each Message object. The resulting list of formatted Message objects
    is returned as a list of ChatMessage objects with ChatML format.

    Note: This function assumes that the ChatMessage class is compatible
    with the Message class, and that the 'content' attribute is a string.
    """
    for message in messages:
        message.content = f"<|im_start|>{message.content}<|im_end|>\n"
    return messages


# def chat_messages_embeds(messages: List[Message]) -> List[ChatMessage]:
#     """
#     Converts a list of Message objects into a list of Discord embeds.

#     Args:
#         messages (List[Message]): A list of Message objects.

#     Returns:
#         List[ChatMessage]: A list of Discord embeds (Embed objects).

#     This function iterates through the provided list of Message objects
#     and creates Discord embeds (Embed objects) based on the message content
#     and sender information. The embeds are designed to display the message
#     content, author name, author avatar, and a URL (DOLPHIN_EMBED_URL).

#     If the message content exceeds 4094 characters (Discord's limit for
#     embed descriptions), the content is split into multiple embeds, with
#     each embed containing up to 4094 characters.

#     The function assumes the existence of HumanMessage and AIMessage classes,
#     which are subclasses of the Message class. The embeds are created differently
#     based on the message type (HumanMessage or AIMessage).

#     Note: This function requires the 'ctx' and 'self.client.app.name' objects
#     to be available in the current context. It also assumes the existence of
#     the 'Embed', 'EmbedAuthor', and 'DOLPHIN_EMBED_URL' entities.
#     """
#     chat_embeds = []
#     for message in messages:
#         if isinstance(message) == HumanMessage:
#             chat_embeds.append(Embed(
#                 description=f"{message.content[:4094]}",
#                 author = EmbedAuthor(name=f"{ctx.author.display_name}",
#                 icon_url=f"{ctx.author.avatar_url}",
#                 url=f"{DOLPHIN_EMBED_URL}")
#             ))
#         elif isinstance(message) == AIMessage:
#             if len(message.content) <= 4094:
#                 chat_embeds.append(Embed(
#                     description=f"{message.content[:4094]}",
#                     author = EmbedAuthor(name=f"{self.client.app.name}",
#                     icon_url=f"{DOLPHIN_EMBED_IMG}",
#                     url=f"{DOLPHIN_EMBED_URL}"),
#                 ))
#             elif len(message.content) > 4094:
#                 chat_embeds.append(Embed(
#                     description=f"{message.content[:4094]}",
#                     author = EmbedAuthor(name=f"{self.client.app.name}",
#                     icon_url=f"{DOLPHIN_EMBED_IMG}",
#                     url=f"{DOLPHIN_EMBED_URL}"),
#                 ))
#                 chat_embeds.append(Embed(
#                     description=f"{message.content[4095:6000]}",
#                 ))
#     return chat_embeds


def chat_messages_regenerate(chat_id: str) -> List[ChatMessage]:
    """
    Retrieves and regenerates a list of chat messages from Redis.

    Args:
        id (str): The unique identifier for the chat history.

    Returns:
        List[ChatMessage]: A list of ChatMessage objects representing the chat history.

    This function retrieves the chat history associated with the provided 'id' from
    a Redis database using the 'RedisChatMessageHistory' class. It then creates a
    list of ChatMessage objects from the retrieved history, excluding the last message.

    The chat history is stored in Redis using the 'RedisChatMessageHistory' class,
    which is assumed to be available in the current context. The function assumes
    that the Redis server is running locally on port 6379.

    Note: This function requires the 'RedisChatMessageHistory' class to be available
    and properly configured. It also assumes the existence of the 'ChatMessage' class
    to represent individual chat messages.
    """
    chats = []
    history = RedisChatMessageHistory(chat_id, url="redis://localhost:6379")
    chats.extend(history.messages)
    chats.pop()
    print(chats)
    return chats
