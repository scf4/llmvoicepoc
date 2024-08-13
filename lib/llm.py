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