import os
from groq import AsyncGroq
from typing import AsyncGenerator
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = AsyncGroq(api_key=GROQ_API_KEY)


async def call_llm(messages: list[dict], model="llama-3.1-8b-instant", max_tokens=1024) -> str:
    response = await client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
    message_text = response.choices[0].message.content

    return message_text


async def call_llm_streaming(messages: list[dict], model="llama-3.1-8b-instant") -> AsyncGenerator[str, None]:
    """
    Call and yield the text content of the response as it comes in.
    """
    response = await client.chat.completions.create(model=model, messages=messages, stream=True)
    async for chunk in response:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content
