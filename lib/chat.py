from lib.llm import call_llm


async def call_chat_llm(messages: list[dict], username: str, ai_name: str):
    system_message = {
        "role": "system",
        "content": f"You are a human-like “person”. Your name is {ai_name}. Keep responses to 1 or 2 sentences or phrases at most. Do not start messages with '{ai_name}'. Do not overuse {username}'s name.",
    }
    messages.insert(0, system_message)

    return await call_llm(messages, model="llama-3.1-8b-instant")
