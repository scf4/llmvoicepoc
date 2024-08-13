import os
import asyncio
import subprocess
from typing import AsyncGenerator, Optional, Dict, Any
from cartesia import AsyncCartesia

CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")


async def __local_tts__(text: str):
    """
    Use macOS built-in TTS to convert text to speech.
    """
    try:
        process = subprocess.Popen(
            ['say', text],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"TTS Error: {stderr.decode('utf-8')}")
            return None
        
        return stdout
    except Exception as e:
        print(f"TTS Exception: {str(e)}")
        return None


async def stream_tts(
    text_stream: AsyncGenerator[str, None],
    model_id: str = "sonic-english",
    voice_id: Optional[str] = None,
    output_format: Optional[Dict[str, Any]] = None,
    language: Optional[str] = None,
) -> AsyncGenerator[bytes, None]:
    async with AsyncCartesia(api_key=CARTESIA_API_KEY) as client:
        ws = await client.tts.websocket()
        context = ws.context()

        if not output_format:
            output_format = client.tts.get_output_format("raw_mp3_44100")

        async def send_chunks():
            try:
                async for chunk in text_stream:
                    await context.send(
                        model_id=model_id,
                        transcript=chunk,
                        output_format=output_format,
                        voice_id=voice_id,
                        language=language,
                        continue_=True,
                    )
            finally:
                await context.no_more_inputs()

        send_task = asyncio.create_task(send_chunks())

        try:
            async for response in context.receive():
                if "audio" in response:
                    yield response["audio"]
        finally:
            await send_task
            await ws.close()