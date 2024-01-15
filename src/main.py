import logging
import os
import asyncio

from dotenv import load_dotenv
from interactions import Client, Intents, listen
from interactions.api.events import Component
from interactions.ext import prefixed_commands

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

logging.basicConfig()
cls_log = logging.getLogger("DolphinLogger")
cls_log.setLevel(logging.DEBUG)

bot = Client(
    intents=Intents.DEFAULT,
    sync_interactions=True,
    asyncio_debug=True,
    logger=cls_log
)
prefixed_commands.setup(bot)

@listen()
async def on_ready():
    print("Ready")
    # We can use the client "app" attribute to get information about the bot.
    print(f"We're online! We've logged in as {bot.app.name}.")
    print(f"This bot is owned by {bot.owner}")

@listen()
async def on_guild_create(event):
    print(f"guild created : {event.guild.name}")

# Message content is a privileged intent.
# Ensure you have message content enabled in the Developer Portal for this to work.
@listen()
async def on_message_create(event):
    print(f"message received: {event.message.content}")

@listen()
async def on_component(event: Component):
    ctx = event.ctx
    await ctx.edit_origin("test")

async def main():
    bot.load_extension("commands.llm")
    await bot.astart(TOKEN)

if __name__ == "__main__":
    asyncio.run(main())