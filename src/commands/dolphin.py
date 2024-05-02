"""
This module contains the dolphin command for bot.
"""

import asyncio
import os
import time
import io
import uuid

from typing import List
from dotenv import load_dotenv
from interactions import slash_command, SlashCommandChoice, slash_option, \
    SlashContext, max_concurrency, Buckets, Button, ActionRow, ButtonStyle, \
    Embed, EmbedAuthor, EmbedFooter, Extension, OptionType, listen, File
from interactions.ext.paginators import Paginator
from interactions.api.events import Component

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
from llama_index.storage.chat_store.redis import RedisChatStore
from llama_index.core.memory import ChatMemoryBuffer

from utils.chat import chat_messages_template

load_dotenv()
DOLPHIN_PATH = os.getenv('DOLPHIN_PATH')
DOLPHIN_MODELS = os.getenv('DOLPHIN_MODELS')
DOLPHIN_REDIS = os.getenv('DOLPHIN_REDIS')
DOLPHIN_GPU_LAYERS = os.getenv('DOLPHIN_GPU_LAYERS')
DOLPHIN_NTHREADS = os.getenv('DOLPHIN_NTHREADS')
DOLPHIN_SYSTEM_PROMPT = os.getenv('DOLPHIN_SYSTEM_PROMPT')
DOLPHIN_EMBED_URL = os.getenv('DOLPHIN_EMBED_URL')
DOLPHIN_EMBED_IMG = os.getenv('DOLPHIN_EMBED_IMG')
DOLPHIN_CMD_SCOPE = int(os.getenv('DOLPHIN_CMD_SCOPE',str(1156064224225808488)))
DOLPHIN_CMD_CHANNEL = int(os.getenv('DOLPHIN_CMD_CHANNEL',str(1189670522653511740)))
DOLPHIN_MAX_REQ = int(os.getenv('DOLPHIN_MAX_REQ', str(1)))

class CommandsDolphin(Extension):
    """
    This class contains the CommandsDolphin.
    """

    def __init__(self, bot) -> None:
        self.bot = bot
        self.concurrency = 0
        self.conversations = {}
        self.chat_store = RedisChatStore(redis_url=f"redis://{DOLPHIN_REDIS}:6379", ttl=300)
        models_strings = DOLPHIN_MODELS.split(",")
        self.models = []
        for model_str in models_strings:
            name, path = model_str.split(":")
            self.models.append({
                "name": name,
                "file": f"{DOLPHIN_PATH}/{path}"
            })
        self.add_ext_check(self.a_check)

    async def a_check(self, ctx: SlashContext) -> bool:
        """
        This function contains the check validation.
        """
        print(f"Check Status:\nChannel:{ctx.channel.id == DOLPHIN_CMD_CHANNEL}")
        return bool(ctx.channel.id == DOLPHIN_CMD_CHANNEL)

    # def drop(self):
    #     super().drop()

    @max_concurrency(bucket=Buckets.CHANNEL, concurrent=DOLPHIN_MAX_REQ)
    @slash_command(
        name="dolphin",
        description="Cognitive Computations: Large Language Model Text Generation Inference Bot.",
        dm_permission=False,
        scopes=[DOLPHIN_CMD_SCOPE]
    )
    @slash_option(
        name="prompt",
        description="Write your SAFE question to Cognitive Computations llm models",
        opt_type=OptionType.STRING,
        required=True,
        min_length=1,
        max_length=2048
    )
    @slash_option(
        name="model",
        description="Select your Dolphin model. Default: dolphin 2.8 experiment26 7B",
        required=False,
        opt_type=OptionType.INTEGER,
        choices=[
            SlashCommandChoice(name="dolphin 2.9 llama 8B", value=0),
            SlashCommandChoice(name="dolphin 2.8 mistral 7B v02", value=1),
            SlashCommandChoice(name="dolphin 2.8 experiment26 7B", value=2),
            SlashCommandChoice(name="dolphin 2.6 mistral 7B DPO", value=3),
            SlashCommandChoice(name="laserxtral", value=4)
        ]
    )
    @slash_option(
        name="max_new_tokens",
        description="The maximum length of the sequence to be generated. Default: 2048",
        required=False,
        opt_type=OptionType.INTEGER,
        min_value=32,
        max_value=2048
    )
    @slash_option(
        name="temperature",
        description="The sampling temperature. Default: 0.1",
        required=False,
        opt_type=OptionType.NUMBER,
        min_value=0.05,
        max_value=2
    )
    @slash_option(
        name="repeat_penalty",
        description="The penalty for repetition. Default: 1.3",
        required=False,
        opt_type=OptionType.NUMBER,
        min_value=0.05,
        max_value=2
    )
    @slash_option(
        name="top_k",
        description="The number of highest probability vocabulary tokens to keep. Default: 50",
        required=False,
        opt_type=OptionType.INTEGER,
        min_value=0,
        max_value=100
    )
    @slash_option(
        name="top_p",
        description="The cumulative probability for top-p filtering. Default: 0.95",
        required=False,
        opt_type=OptionType.NUMBER,
        min_value=0.05,
        max_value=2
    )

    async def command(
        self,
        ctx: SlashContext,
        prompt: str,
        model: int = 0,
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
        repeat_penalty: float = 1.3,
        top_k: int = 50,
        top_p: float = 0.95
    ) -> None:
        """
        This function executes the command.
        
        Args:
            ctx (SlashContext): The context of the command.
            prompt (str): The prompt for the model.
            model (int, optional): The model to use (default is 0).
            max_new_tokens (int, optional): The maximum number of tokens
            to generate (default is 2048).
            temperature (float, optional): The temperature parameter for
            text generation (default is 0.1).
            repeat_penalty (float, optional): The repeat penalty parameter (default is 1.3).
            top_k (int, optional): The top k parameter for text generation (default is 50).
            top_p (float, optional): The top p parameter for text generation (default is 0.95).
        """
        print("Command start")
        conversation_id = uuid.uuid4()
        try:
            print(model)
            model_selected=self.models[model]
            print(model_selected["name"])
            self.concurrency += 1
            self.conversations[f"{ctx.author.id}_{conversation_id}_cancel"] = False
            history = ChatMemoryBuffer.from_defaults(
                token_limit=4096,
                chat_store=self.chat_store,
                chat_store_key=f"{ctx.author.id}",
            )
            response = ""
            chat_template = self.get_chat_template(prompt=prompt, messages=history.get_all())
            embeds = self.get_chat_embeds(ctx=ctx, prompt=prompt, model_name=model_selected["name"])
            cancel = Button(
                custom_id=f"button_cancel_{ctx.author.id}_{conversation_id}",
                style=ButtonStyle.RED,
                label="Cancel",
            )
            components: list[ActionRow] = [
                ActionRow(
                    Button(
                        custom_id=f"button_regenerate_{ctx.author.id}_{model}",
                        style=ButtonStyle.PRIMARY,
                        label="Regenerate",
                    ),
                    Button(
                        custom_id=f"button_show_{ctx.author.id}",
                        style=ButtonStyle.GREEN,
                        label="Show Chat",
                    ),
                    Button(
                        custom_id=f"button_send_{ctx.author.id}",
                        style=ButtonStyle.GREY,
                        label="Send Chat",
                    ),
                    Button(
                        custom_id=f"button_clear_{ctx.author.id}",
                        style=ButtonStyle.RED,
                        label="Clear Chat",
                    )
                )
            ]
            await ctx.defer()

            print("llama start")

            llm = LlamaCPP(
                model_path=model_selected["file"],
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                context_window=8192,
                generate_kwargs={
                    "top_k": top_k,
                    "top_p": top_p,
                    "repeat_penalty": repeat_penalty
                },
                model_kwargs={
                    "n_threads": int(DOLPHIN_NTHREADS),
                    "n_gpu_layers": int(DOLPHIN_GPU_LAYERS)
                },
                messages_to_prompt=messages_to_prompt,
                completion_to_prompt=completion_to_prompt,
                verbose=True,
            )

            ## Stream (give me multiple chunks to form the response)

            update_interval = 0.6
            last_update_time = time.time()

            for chunk in llm.stream_chat(chat_template):
                if self.conversations[f"{ctx.author.id}_{conversation_id}_cancel"] is True:
                    llm.stop # noqa: W0104
                    break
                response += str(chunk.delta)
                current_time = time.time()
                if current_time - last_update_time >= update_interval:
                    last_update_time = current_time
                    if len(response) <= 4094:
                        embeds[2].description = f"{response[:4094]}"
                        await ctx.edit(embeds=embeds[:3], components=[cancel])
                    elif len(response) > 4094:
                        embeds[2].description = f"{response[:4094]}"
                        embeds[2].footer = ""
                        embeds[3].description = f"{response[4095:5700]}"
                        await ctx.edit(embeds=embeds, components=[cancel])
                    print(f"Embed updated at {time.strftime('%X')}", end="", flush=True)

            response = response.replace("<|im_end|>","")
            await asyncio.sleep(0.6)

            if self.conversations[f"{ctx.author.id}_{conversation_id}_cancel"] is True:
                embeds[1].description = f"~~{embeds[1].description[:500]}~~"
                embeds[2].description = f"~~{response[:4094]}~~"
                if len(response) <= 4094:
                    await ctx.edit(embeds=embeds[:3], components=[])
                elif len(response) > 4094:
                    embeds[3].description = f"~~{response[4095:5700]}~~"
                    await ctx.edit(embeds=embeds, components=[])
            else:
                if len(response) <= 4094:
                    embeds[2].description = f"{response[:4094]}"
                    await ctx.edit(embeds=embeds[:3], components=components)
                elif len(response) > 4094:
                    await ctx.edit(embeds=embeds, components=components)
                history.put(chat_template[-1])
                history.put(ChatMessage(role=MessageRole.ASSISTANT,content=f"{response}"))

        except ImportError:
            print(f"Error occurred in command: {ImportError}")

        finally:
            print("llama end")
            print(ctx.resolved)
            self.concurrency -= 1
            del self.conversations[f"{ctx.author.id}_{conversation_id}_cancel"]

    @listen()
    async def an_event_handler(self, event: Component):
        """
        Listen an_event_handler function
        """
        author_id = f"{event.ctx.author.id}"
        ctx = event.ctx
        component_split = ctx.custom_id.split("_")
        if (component_split and len(component_split) > 1 and component_split[0] == "button"):
            ####
            # Handle Cancel Button
            ####
            if (component_split[1] == "cancel" and author_id == component_split[-2]):
                print("\n\ncancel press button\n\n")
                if f"{author_id}_{component_split[-1]}_cancel" in self.conversations:
                    self.conversations[f"{author_id}_{component_split[-1]}_cancel"] = True
            ####
            # Handle Show Button
            ####
            elif component_split[1] == "show":
                history = ChatMemoryBuffer.from_defaults(
                    token_limit=4096,
                    chat_store=self.chat_store,
                    chat_store_key=f"{component_split[-1]}",
                )
                messages = history.get_all()
                if len(messages) == 0:
                    await ctx.send("You don't have nothing on chat!")
                else:
                    chat_embeds = []
                    user = await self.client.fetch_user(int(component_split[-1]))
                    for message in messages:
                        if message.role == "user":
                            chat_embeds.append(Embed(
                                description=f"{message.content}",
                                author = EmbedAuthor(
                                    name=f"{user.display_name}",
                                    icon_url=f"{user.avatar_url}",
                                    url=f"{DOLPHIN_EMBED_URL}"
                                    )
                                )
                            )
                        elif message.role == "assistant":
                            if len(message.content) <= 4094:
                                chat_embeds.append(Embed(
                                    description=f"{message.content[:4094]}",
                                    author = EmbedAuthor(
                                        name=f"{self.client.app.name}",
                                        icon_url=f"{self.client.user.avatar_url}",
                                        url=f"{DOLPHIN_EMBED_URL}"
                                    )
                                ))
                            elif len(message.content) > 4094:
                                chat_embeds.append(Embed(
                                    description=f"{message.content[:4094]}",
                                    author = EmbedAuthor(
                                        name=f"{self.client.app.name}",
                                        icon_url=f"{self.client.user.avatar_url}",
                                        url=f"{DOLPHIN_EMBED_URL}"
                                    )
                                ))
                                chat_embeds.append(Embed(
                                    description=f"{message.content[4095:6000]}",
                                ))
                    paginator = Paginator.create_from_embeds(self.client, *chat_embeds)
                    await paginator.send(ctx)
            ####
            # Handle Send Chat Button
            ####
            elif component_split[1] == "send" and author_id == component_split[-1]:
                history = ChatMemoryBuffer.from_defaults(
                    token_limit=4096,
                    chat_store=self.chat_store,
                    chat_store_key=f"{component_split[-1]}",
                )
                user_dm = self.client.get_user(event.ctx.author.id)
                messages = history.get_all()
                formatted_messages = [
                    f"{ctx.author.display_name}:{message.content}\n"
                    if message.role == "user"
                    else f"{self.client.app.name}:{message.content}\n"
                    for message in messages
                ]
                result_text = ''.join(formatted_messages)
                virtual_file = io.StringIO(result_text)
                if len(messages) == 0:
                    await ctx.send("You don't have nothing on chat!")
                else:
                    await user_dm.send(
                        file=File(
                            file=virtual_file,
                            file_name="chat.txt",
                            content_type="text/plain"
                        )
                    )
                    await ctx.send("Chat send to DM!")
            ####
            # Handle Cleat Chat Button
            ####
            elif (component_split[1] == "clear" and author_id == component_split[-1]):
                print("\n\nclear press button\n\n")
                history = ChatMemoryBuffer.from_defaults(
                    token_limit=4096,
                    chat_store=self.chat_store,
                    chat_store_key=f"{component_split[-1]}",
                )
                history.reset()
                await ctx.send("Chat clear!")
            ####
            # Handle Regenerate Button
            ####
            elif (component_split[1] == "regenerate" and author_id == component_split[-2]):
                print("\n\nregenerate\n\n")
                history = ChatMemoryBuffer.from_defaults(
                    token_limit=4096,
                    chat_store=self.chat_store,
                    chat_store_key=f"{component_split[-1]}",
                )
                history_messages = history.get_all()
                if len(history_messages) > 0:
                    model_selected=self.models[int(component_split[-1])]
                    messages = history_messages[:-2]
                    response = ""
                    chat_template = self.get_chat_template(
                        prompt=history_messages[-2].content,
                        messages=messages
                    )
                    embeds = self.get_chat_embeds(
                        ctx=ctx, prompt=history_messages[-2].content,
                        model_name=model_selected["name"]
                    )
                    print(chat_template)
                    llm = LlamaCPP(
                        model_path=model_selected["file"],
                        temperature=0.1,
                        max_new_tokens=4096,
                        context_window=8192,
                        generate_kwargs={
                            "top_k": 50,
                            "top_p": 0.95,
                            "repeat_penalty": 1.3
                        },
                        model_kwargs={
                            "n_threads": int(DOLPHIN_NTHREADS),
                            "n_gpu_layers": int(DOLPHIN_GPU_LAYERS)
                        },
                        messages_to_prompt=messages_to_prompt,
                        completion_to_prompt=completion_to_prompt,
                        verbose=True,
                    )
                    await ctx.defer()
                    response = await llm.chat(chat_template)
                    print(response)
                    if len(response) <= 4094:
                        embeds[2].description = f"{response[:4094]}"
                        await ctx.send(embeds=embeds[:3])
                    elif len(response) > 4094:
                        await ctx.send(embeds=embeds)
                else:
                    await ctx.send("Your chat conversation is empty to `regenerate` last question.")
                # await ctx.edit_origin(content="we are working on regenerate!")

    @command.error
    async def command_error(self, e, *args, **kwargs):
        """
        Command error function handle error event
        """
        print(f"Command hit error with {args=}, {kwargs=}")
        print(e)

    @command.pre_run
    async def command_pre_run(self, *args, **kwargs):
        """
        Command pre-run function event
        """
        print(f"I ran before the command did! {args=}, {kwargs=}")

    @command.post_run
    async def command_post_run(self, *args, **kwargs):
        """
        Command post-run function event
        """
        print(f"I ran after the command did! {args=}, {kwargs=}")

    def get_chat_template(self, prompt: str, messages: List[ChatMessage]):
        """
        function get_chat_template
        """
        chat_template = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=f"<|im_start|>system\n{DOLPHIN_SYSTEM_PROMPT}<|im_end|>\n"
            )
        ]
        chat_template.extend(messages)
        chat_template.extend([ChatMessage(role=MessageRole.USER,content=f"{prompt}")])
        return chat_template

    def get_chat_embeds(self, ctx: SlashContext, prompt: str, model_name: str) -> List[Embed]:
        """
        function get_chat_embeds
        """
        chat_embeds = [
            Embed(
                description=f"**System Prompt**\n{DOLPHIN_SYSTEM_PROMPT}",
                author = EmbedAuthor(
                    name=f"{self.client.app.name}",
                    icon_url=f"{self.client.user.avatar_url}",
                    url=f"{DOLPHIN_EMBED_URL}"
                )
            ),
            Embed(
                description=f"{prompt[:500]}",
                author = EmbedAuthor(
                    name=f"{ctx.author.display_name}",
                    icon_url=f"{ctx.author.avatar_url}",
                    url=f"{DOLPHIN_EMBED_URL}"
                )
            ),
            Embed(
                description="",
                author = EmbedAuthor(
                    name=f"{self.client.app.name}",
                    icon_url=f"{self.client.user.avatar_url}",
                    url=f"{DOLPHIN_EMBED_URL}"
                ),
                footer = EmbedFooter(
                    text=f"{model_name}",
                    icon_url=f"{DOLPHIN_EMBED_IMG}"
                )
            ),
            Embed(
                description="",
                footer = EmbedFooter(text=f"{model_name}", icon_url=f"{DOLPHIN_EMBED_IMG}")
            )
        ]

        return chat_embeds


def setup(bot):
    """
    Setup def for CommandsDolphin
    """
    CommandsDolphin(bot)
