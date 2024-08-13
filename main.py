import asyncio
from lib.transcription import AudioTranscriber
from lib.llm import call_llm
from lib.tts import stream_tts

username = "Scott"
ai_name = "Max"

async def call_chat_llm(messages: list[dict]):
    system_message = {"role": "system", "content": f"You are a human-like, sentient AI. Your name is {ai_name}. Keep responses to 1 or 2 sentences or phrases at most."}
    messages.insert(0, system_message)
    
    return await call_llm(messages, model="llama-3.1-8b-instant")

async def main():
    messages = []

    transcriber = AudioTranscriber()
    await transcriber.initialize()
    transcriber_task = asyncio.create_task(transcriber.run())

    try:
        while True:
            print("[DEBUG] Waiting for user input...")
            transcribed_message = await transcriber.get_next_complete_message()
            print(f"[DEBUG] Received user input: {transcribed_message}")
            messages.append({"role": "user", "content": f"{username}: {transcribed_message}"})

            print("[DEBUG] Calling LLM for response...")
            response = await call_chat_llm(messages)
            print(f"[DEBUG] LLM response: {response}")

            print("[DEBUG] Starting TTS task...")
            cancel_event = asyncio.Event()
            tts_task = asyncio.create_task(stream_tts(response, cancel_event))
            user_speech_task = asyncio.create_task(transcriber.wait_for_user_speech())

            print("[DEBUG] Waiting for TTS to complete or user to interrupt...")
            done, pending = await asyncio.wait(
                [tts_task, user_speech_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            if user_speech_task in done:
                print("[DEBUG] User interrupted, cancelling TTS...")
                cancel_event.set()  # Signal TTS to stop
                tts_task.cancel()
                try:
                    await tts_task
                except asyncio.CancelledError:
                    print("[DEBUG] TTS interrupted by user")
            else:
                print("[DEBUG] TTS completed without interruption")
                user_speech_task.cancel()

            try:
                await tts_task
            except asyncio.CancelledError:
                pass

            messages.append({"role": "assistant", "content": response})
            print(f"[DEBUG] Response: {response}")
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        print("[DEBUG] Cleaning up resources...")
        transcriber_task.cancel()
        await transcriber.close()
        print("[DEBUG] Cleanup complete")

if __name__ == "__main__":
    asyncio.run(main())