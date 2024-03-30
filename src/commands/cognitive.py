import asyncio
import os
import time
import re
import io
import uuid

from dotenv import load_dotenv
from typing import List
from interactions import slash_command, slash_option, SlashContext, max_concurrency, Buckets, Button, ActionRow, ButtonStyle, Embed, EmbedAuthor, EmbedFooter, Extension, OptionType, listen, File
from interactions.ext.paginators import Paginator
from interactions.api.events import Component
from langchain_openai import ChatOpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ChatMessage
from langchain_community.chat_message_histories import RedisChatMessageHistory
from utils.chat import Message, chat_messages_otemplate

load_dotenv()
MODEL_FILE = "cognitivecomputations_laserxtral-exl2_6.5"
MODEL_NAME = "cognitive-computations/laserxtral"
MODEL_PATH = os.getenv('MODEL_PATH')
DOLPHIN_GPU_LAYERS = os.getenv('DOLPHIN_GPU_LAYERS')
DOLPHIN_NTHREADS = os.getenv('DOLPHIN_NTHREADS')
DOLPHIN_SYSTEM_PROMPT = os.getenv('DOLPHIN_SYSTEM_PROMPT')
DOLPHIN_EMBED_URL = os.getenv('DOLPHIN_EMBED_URL')
DOLPHIN_EMBED_IMG = os.getenv('DOLPHIN_EMBED_IMG')
DOLPHIN_CMD_SCOPE = int(os.getenv('DOLPHIN_CMD_SCOPE',1156064224225808488))
DOLPHIN_CMD_CHANNEL = int(os.getenv('DOLPHIN_CMD_CHANNEL',1189670522653511740))
DOLPHIN_MAX_REQ = 1

os.environ["OPENAI_API_KEY"] = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # can be anything
os.environ["OPENAI_API_BASE"] = "http://localhost:8080/v1"
os.environ["OPENAI_API_HOST"] = "http://localhost:8080/"

regex_regenerate = re.compile(r"button_regenerate_([0-9]+)")

class CommandsCoginitive(Extension):

    def __init__(self, bot) -> None:
        self.concurrency = 0
        self.conversations = {}
        self.add_ext_check(self.a_check)

    async def a_check(self, ctx: SlashContext) -> bool:
        print(f"Check Status:\nChannel:{ctx.channel.id == DOLPHIN_CMD_CHANNEL}\nDOLPHIN_MAX_REQ: {DOLPHIN_MAX_REQ}")
        return bool(ctx.channel.id == DOLPHIN_CMD_CHANNEL)

    def drop(self):
        super().drop()

    @max_concurrency(bucket=Buckets.CHANNEL, concurrent=DOLPHIN_MAX_REQ)
    @slash_command(
        name="cognitive",
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
        name="max_new_tokens",
        description="The maximum length of the sequence to be generated. Default: 2048",
        required=False,
        opt_type=OptionType.INTEGER,
        min_value=32,
        max_value=32768
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
        description="The number of highest probability vocabulary tokens to keep for top-k filtering. Default: 50",
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
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
        repeat_penalty: float = 1.3,
        top_k: int = 50,
        top_p: float = 0.95
    ):
        conversation_id = uuid.uuid4()
        try:
            self.concurrency += 1
            self.conversations[f"{ctx.author.id}_{conversation_id}_cancel"] = False
            history = RedisChatMessageHistory(f"{ctx.author.id}", url="redis://127.0.0.1:6379")
            response = ""
            chat_template = self.get_chat_template(prompt=prompt, messages=history.messages)
            embeds = self.get_chat_embeds(ctx=ctx, prompt=prompt)
            cancel = Button(
                custom_id=f"cbutton_cancel_{ctx.author.id}_{conversation_id}",
                style=ButtonStyle.RED,
                label="Cancel",
            )
            components: list[ActionRow] = [
                ActionRow(
                    Button(
                        custom_id=f"cbutton_regenerate_{ctx.author.id}",
                        style=ButtonStyle.PRIMARY,
                        label="Regenerate",
                    ),
                    Button(
                        custom_id=f"cbutton_show_{ctx.author.id}",
                        style=ButtonStyle.GREEN,
                        label="Show Chat",
                    ),
                    Button(
                        custom_id=f"cbutton_send_{ctx.author.id}",
                        style=ButtonStyle.GREY,
                        label="Send Chat",
                    ),
                    Button(
                        custom_id=f"cbutton_clear_{ctx.author.id}",
                        style=ButtonStyle.RED,
                        label="Clear Chat",
                    )
                )
            ]

            await ctx.defer()

            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            llm = ChatOpenAI(
                model_name=f"{MODEL_FILE}",
                openai_api_base="http://localhost:8080/v1",
                temperature=temperature,
                max_tokens=max_new_tokens,
                callback_manager=callback_manager,
                streaming=True,
                verbose=True
            )

            update_interval = 0.6
            last_update_time = time.time()
            print(chat_template)
            async for chunk in llm.astream(chat_template):
                if self.conversations[f"{ctx.author.id}_{conversation_id}_cancel"] == True:
                    llm.stop
                    break
                response += str(chunk.content)
                current_time = time.time()
                if current_time - last_update_time >= update_interval:
                    last_update_time = current_time
                    if len(response) <= 4094:
                        embeds[2].description = f"{response[:4094]}"
                        # re = embeds + embed
                        await ctx.edit(embeds=embeds[:3], components=[cancel])
                    elif len(response) > 4094:
                        embeds[2].description = f"{response[:4094]}"
                        embeds[2].footer = f""
                        embeds[3].description = f"{response[4095:5700]}"
                        await ctx.edit(embeds=embeds, components=[cancel])
                    print(f"Embed updated at {time.strftime('%X')}", end="", flush=True)

            response = response.replace("<|im_end|>","")
            await asyncio.sleep(0.6)

            if self.conversations[f"{ctx.author.id}_{conversation_id}_cancel"] == True:
                embeds[1].description = f"~~{embeds[1].description[:500]}~~"
                embeds[2].description = f"~~{response[:4094]}~~"
                if len(response) <= 4094:
                    await ctx.edit(embeds=embeds[:3], components=components)
                elif len(response) > 4094:
                    embeds[3].description = f"~~{response[4095:5700]}~~"
                    await ctx.edit(embeds=embeds, components=components)
            else:
                if len(response) <= 4094:
                    embeds[2].description = f"{response[:4094]}"
                    await ctx.edit(embeds=embeds[:3], components=components)
                elif len(response) > 4094:
                    await ctx.edit(embeds=embeds, components=components)
                history.add_user_message(f"{prompt}")
                history.add_ai_message(f"{response}")

        except Exception as e:
            print(f"Error occurred in command: {e}")

        finally:
            print(ctx.resolved)
            self.concurrency -= 1
            del self.conversations[f"{ctx.author.id}_{conversation_id}_cancel"]

    @listen()
    async def an_event_handler(self, event: Component):
        author_id = f"{event.ctx.author.id}"
        ctx = event.ctx
        component_split = ctx.custom_id.split("_")
        if (component_split and len(component_split) > 1 and component_split[0] == "cbutton"):
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
            elif (component_split[1] == "show"):
                history = RedisChatMessageHistory(component_split[-1], url="redis://localhost:6379")
                messages = history.messages
                if (len(messages) == 0):
                    await ctx.send("You don't have nothing on chat!")
                else:
                    chat_embeds = []
                    user = await self.client.fetch_user(int(component_split[-1]))
                    for message in messages:
                        if type(message) == HumanMessage:
                            chat_embeds.append(Embed(
                                description=f"{message.content}",
                                author = EmbedAuthor(name=f"{user.display_name}",icon_url=f"{user.avatar_url}", url=f"{DOLPHIN_EMBED_URL}"),
                            ))
                        elif type(message) == AIMessage:  
                            if len(message.content) <= 4094:
                                chat_embeds.append(Embed(
                                    description=f"{message.content[:4094]}",
                                    author = EmbedAuthor(name=f"{self.client.app.name}",icon_url=f"{self.client.user.avatar_url}", url=f"{DOLPHIN_EMBED_URL}"),
                                ))
                            elif len(message.content) > 4094:
                                chat_embeds.append(Embed(
                                    description=f"{message.content[:4094]}",
                                    author = EmbedAuthor(name=f"{self.client.app.name}",icon_url=f"{self.client.user.avatar_url}", url=f"{DOLPHIN_EMBED_URL}"),
                                ))
                                chat_embeds.append(Embed(
                                    description=f"{message.content[4095:6000]}",
                                ))
                    paginator = Paginator.create_from_embeds(self.client, *chat_embeds)
                    await paginator.send(ctx)
            ####
            # Handle Send Chat Button
            ####
            elif (component_split[1] == "send" and author_id == component_split[-1]):
                history = RedisChatMessageHistory(component_split[-1], url="redis://localhost:6379")
                user_dm = self.client.get_user(event.ctx.author.id)
                messages = history.messages
                formatted_messages = [
                    f"{ctx.author.display_name}:{message.content}\n" if isinstance(message, HumanMessage) else f"{self.client.app.name}:{message.content}\n"
                    for message in messages
                ]
                result_text = ''.join(formatted_messages)
                virtual_file = io.StringIO(result_text)
                if (len(messages) == 0):
                    await ctx.send("You don't have nothing on chat!")
                else:
                    await user_dm.send(file=File(file=virtual_file, file_name="chat.txt", content_type="text/plain"))
                    await ctx.send("Chat send to DM!")
            ####
            # Handle Cleat Chat Button
            ####
            elif (component_split[1] == "clear" and author_id == component_split[-1]):
                print("\n\nclear press button\n\n")
                history = RedisChatMessageHistory(component_split[-1], url="redis://localhost:6379")
                history.clear()
                await ctx.send("Chat clear!")
            ####
            # Handle Regenerate Button
            ####
            elif (component_split[1] == "regenerate" and author_id == component_split[-1]):
                print("\n\nregenerate\n\n")
                history = RedisChatMessageHistory(component_split[-1], url="redis://localhost:6379")
                if (len(history.messages) > 0):
                    messages = history.messages[:-2]
                    response = ""
                    chat_template = self.get_chat_template(prompt=history.messages[-2].content, messages=messages)
                    embeds = self.get_chat_embeds(ctx=ctx, prompt=history.messages[-2].content)
                    print(chat_template)
                    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
                    llm = ChatOpenAI(
                        model_name=f"{MODEL_FILE}",
                        openai_api_base="https://localhost:8080/v1",
                        temperature=0.1,
                        max_tokens=2048,
                        callback_manager=callback_manager,
                        streaming=True,
                        verbose=True
                    )
                    await ctx.defer()
                    response = await llm.ainvoke(chat_template)
                    print(response)
                    if len(response.content) <= 4094:
                        embeds[2].description = f"{response.content[:4094]}"
                        await ctx.edit_origin(embeds=embeds[:3])
                    elif len(response.content) > 4094:
                        await ctx.edit_origin(embeds=embeds)
                else:
                    await ctx.send("Your chat conversation is empty to `regenerate` last question.")

    @command.error
    async def command_error(self, e, *args, **kwargs):
        print(f"Command hit error with {args=}, {kwargs=}")
        print(e)

    @command.pre_run
    async def command_pre_run(self, context, *args, **kwargs):
        print("I ran before the command did!")

    @command.post_run
    async def command_post_run(self, context, *args, **kwargs):
        print("I ran after the command did!")
        print(f"kwargs {kwargs.get('prompt')=}, {kwargs=}")
    
    def get_chat_template(self, prompt: str, messages: List[ChatMessage]):
        chat_template = [
            SystemMessage(
                content=f"<|im_start|>system\n{DOLPHIN_SYSTEM_PROMPT}<|im_end|>\n"
            )
        ]
        chat_template.extend(chat_messages_otemplate(messages))
        chat_template.extend(chat_messages_otemplate([HumanMessage(content=f"{prompt}")]))
        chat_template.append(AIMessage(content=f"<|im_start|>assistant:"))
        return chat_template
    
    def get_chat_embeds(self, ctx: SlashContext, prompt: str) -> List[Embed]:
        chat_embeds = [
            Embed(
                description=f"**System Prompt**\n{DOLPHIN_SYSTEM_PROMPT}",
                author = EmbedAuthor(name=f"{self.client.app.name}",icon_url=f"{self.client.user.avatar_url}", url=f"{DOLPHIN_EMBED_URL}"),
            ),
            Embed(
                description=f"{prompt[:500]}",
                author = EmbedAuthor(name=f"{ctx.author.display_name}",icon_url=f"{ctx.author.avatar_url}", url=f"{DOLPHIN_EMBED_URL}"),
            ),
            Embed(
                description=f"",
                author = EmbedAuthor(name=f"{self.client.app.name}",icon_url=f"{self.client.user.avatar_url}", url=f"{DOLPHIN_EMBED_URL}"),
                footer = EmbedFooter(text=f"{MODEL_NAME}", icon_url=f"{DOLPHIN_EMBED_IMG}")
            ),
            Embed(
                description=f"",
                footer = EmbedFooter(text=f"{MODEL_NAME}", icon_url=f"{DOLPHIN_EMBED_IMG}")
            )
        ]

        return chat_embeds

def setup(bot):
    CommandsCoginitive(bot)