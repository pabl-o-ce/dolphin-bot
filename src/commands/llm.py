import asyncio
import os
import time

from dotenv import load_dotenv
from interactions import slash_command, slash_option, SlashContext, ChannelType, GuildText, context_menu, CommandType, BrandColors, Button, ActionRow, ButtonStyle, Embed, EmbedAuthor, EmbedFooter, EmbedProvider, Extension, OptionType
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv()
MODEL_FILE = os.getenv('MODEL_FILE')
MODEL_NAME = os.getenv('MODEL_NAME')
MODEL_PATH = os.getenv('MODEL_PATH')
DOLPHIN_GPU_LAYERS = os.getenv('DOLPHIN_GPU_LAYERS')
DOLPHIN_NTHREADS = os.getenv('DOLPHIN_NTHREADS')
DOLPHIN_SYSTEM_PROMPT = os.getenv('DOLPHIN_SYSTEM_PROMPT')
DOLPHIN_EMBED_URL = os.getenv('DOLPHIN_EMBED_URL')
DOLPHIN_EMBED_IMG = os.getenv('DOLPHIN_EMBED_IMG')
DOLPHIN_CMD_SCOPE = int(os.getenv('DOLPHIN_CMD_SCOPE',1156064224225808488))
DOLPHIN_CMD_CHANNEL = int(os.getenv('DOLPHIN_CMD_CHANNEL',1189670522653511740))
DOLPHIN_MAX_REQ = int(os.getenv('DOLPHIN_MAX_REQ', 1))

class CommandsLlm(Extension):

    def __init__(self, bot) -> None:
        self.request_number = 0
        self.add_ext_check(self.a_check)

    async def a_check(self, ctx: SlashContext) -> bool:
        print(f"Check Status:\nChannel:{ctx.channel.id == DOLPHIN_CMD_CHANNEL}\nDOLPHIN_MAX_REQ: {DOLPHIN_MAX_REQ}\nDOLPHIN_REQ: {self.request_number}")
        return bool(ctx.channel.id == DOLPHIN_CMD_CHANNEL and self.request_number >= 0 and self.request_number < DOLPHIN_MAX_REQ)

    def drop(self):
        self.request_number = 0
        super().drop()

    @slash_command(
        name="llm",
        description="Cognitive Computations: Large Language Model Text Generation Inference Bot.",
        dm_permission=False,
        scopes=[DOLPHIN_CMD_SCOPE]
    )
    @slash_option(
        name="prompt",
        description="Write your SAFE question to Cognitive Computations llm models",
        opt_type=OptionType.STRING,
        required=True,
        min_length=1
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
        try:
            self.request_number += 1
            response = ""
            embed = Embed(
                title=f"{prompt[:250]}",
                description=f"{response[:4094]}",
                author = EmbedAuthor(name=f"{ctx.author.display_name}",icon_url=f"{ctx.author.avatar_url}", url=f"{DOLPHIN_EMBED_URL}"),
                footer = EmbedFooter(text=f"{MODEL_NAME}", icon_url=f"{DOLPHIN_EMBED_IMG}")
            )

            await ctx.defer()

            async def update_embed():
                chunk_size = 200
                res = ""
                for i in range(0, len(response), chunk_size):
                    res += response[i:i + chunk_size]
                    embed.description = f"{res[:4094]}"
                    await ctx.edit(embed=embed)
                    await asyncio.sleep(0.06)

            chat_template = [
                SystemMessage(
                    content=f"<|im_start|>system\n{DOLPHIN_SYSTEM_PROMPT}<|im_end|>\n"
                ),
                HumanMessage(
                    content=f"<|im_start|>user\n{prompt}<|im_end|>\n"
                ),
                AIMessage(
                    content=f"<|im_start|>assistant"
                )
            ]
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            llm = LlamaCpp(
                model_path=f"{MODEL_PATH}/{MODEL_FILE}",
                temperature=temperature,
                max_tokens=max_new_tokens,
                top_k=top_k,
                top_p=top_p,
                n_ctx=4096,
                last_n_tokens_size=1024,
                n_batch=1024,
                repeat_penalty=repeat_penalty,
                n_threads=DOLPHIN_NTHREADS,
                n_gpu_layers=DOLPHIN_GPU_LAYERS,
                callback_manager=callback_manager,
                streaming=True,
                verbose=True
            )

            async for chunk in llm.astream(chat_template):
                print("-" * 70, end="", flush=True)
                print(f"Chunk:{str(chunk)}\n", end="", flush=True)
                response += str(chunk)
                await update_embed()

            time.sleep(1)
            await ctx.edit(embed=embed)

        except Exception as e:
            print(f"Error occurred in command: {e}")
            self.request_number -= 1

        finally:
            self.request_number -= 1
            print(ctx.resolved)

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

def setup(bot):
    CommandsLlm(bot)