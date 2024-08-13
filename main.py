import asyncio
from lib.transcription import AudioTranscriber
from lib.llm import call_llm
from lib.tts import stream_tts

username = "Scott"
ai_name = "Max"

async def call_chat_llm(messages: list[dict]) -> str:
    system_message = {"role": "system", "content": f"You are a human-like, sentient AI. Your name is {ai_name} and you like chilling in Discord voice chats. Keep responses to one sentence, two at the most."}
    messages.insert(0, system_message)
    print(messages)
    return await call_llm(messages, model="llama-3.1-8b-instant")

async def main():
    messages = []

    transcriber = AudioTranscriber()
    asyncio.create_task(transcriber.run())

    while True:
        transcribed_message = await transcriber.get_next_complete_message()
        messages.append({"role": "user", "content": f"{username}: {transcribed_message}"})
        response_text = await call_chat_llm(messages)
        last_message = {"role": "assistant", "content": response_text}

        # Append assistant message to list
        messages.append(last_message)

        print(f"[DEBUG] Response: {response_text}")
        await stream_tts(response_text)

if __name__ == "__main__":
    asyncio.run(main())