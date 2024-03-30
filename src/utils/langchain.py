import asyncio
import os
import time
from dotenv import load_dotenv

import reactivex as rx
from reactivex import operators as ops
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv()
MODEL_PATH = os.getenv('MODEL_PATH')

def llama (
    prompt: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.1,
    repeat_penalty: float = 1.3,
    top_k: int = 50,
    top_p: float = 0.95,
    context_window = 3900,
    n_threads = 8,
    n_gpu_layers = 0
):
    chat_template = [
        SystemMessage(
            content=f"<|im_start|>system\nYou are Dolphin, a friendly and ethical and helpful and law abiding AI assistant<|im_end|>\n"
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
        model_path=f"{MODEL_PATH}/dolphin-2.6-mistral-7b-dpo-laser-f16.gguf",
        temperature=temperature,
        max_tokens=max_new_tokens,
        top_k=top_k,
        top_p=top_p,
        n_ctx=3900,
        repeat_penalty=repeat_penalty,
        # n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        callback_manager=callback_manager,
        streaming=True,
        verbose=True
    )

    # async def llm_async_iterator(observer):
    #     try:
    #         async for chunk in llm.astream(chat_template):
    #             observer.on_next(chunk.dict()['content'])
    #     except Exception as e:
    #         observer.on_error(e)
    #     else:
    #         observer.on_completed()

    # def on_subscribe(observer, _):
    #     try:
    #         for chunk in llm.stream(chat_template):
    #             observer.on_next(chunk)
    #     except Exception as e:
    #         observer.on_error(e)

    # return rx.create(on_subscribe)

    for chunk in llm.astream(chat_template):
        yield chunk
