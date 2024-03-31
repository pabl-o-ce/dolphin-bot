"""
This module contains the main function for bot.
"""

import logging
import os
import asyncio
import traceback

from dotenv import load_dotenv
from interactions import Client, Intents, listen
from interactions.ext import prefixed_commands

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

logging.basicConfig()
cls_log = logging.getLogger("Dolphin-Logger:: ")
cls_log.setLevel(logging.DEBUG)

bot = Client(
    intents=Intents.DEFAULT,
    delete_unused_application_cmds=True,
    sync_interactions=True,
    asyncio_debug=True,
    logger=cls_log
)
prefixed_commands.setup(bot)


@listen()
async def on_ready():
    """
    On Ready
    """
    print("Ready")
    print(f"We're online! We've logged in as {bot.app.name}.")
    print(f"This bot is owned by {bot.owner}")


@listen()
async def on_guild_create(event):
    """
    Event for on guild
    """
    print(f"guild created : {event.guild.name}")


@listen()
async def on_message_create(event):
    """
    Event for on message
    """
    print(f"message received: {event.message.content}")


async def main():
    """
    Main function
    """
    try:
        bot.reload_extension("commands.dolphin")
        # bot.load_extension("commands.cognitive")
        await bot.astart(TOKEN)
    except ValueError:
        error_message = traceback.format_exc()
        print(error_message)

if __name__ == "__main__":
    asyncio.run(main())
