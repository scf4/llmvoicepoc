import asyncio
import pyaudio
import os
from typing import AsyncGenerator
from cartesia import AsyncCartesia
from dotenv import load_dotenv

load_dotenv()


async def stream_tts(text: str, cancel_event):
    """
    Streams text to the TTS service and receives audio
    """
    print(f"[DEBUG] Starting TTS for text: {text}")
    client = AsyncCartesia(api_key=os.environ.get("CARTESIA_API_KEY"))
    ws = await client.tts.websocket()
    ctx = ws.context()

    async def send_transcripts():
        try:
            print("[DEBUG] Sending transcript to TTS service")
            await ctx.send(
                model_id="sonic-english",
                transcript=text,
                voice_id="1001d611-b1a8-46bd-a5ca-551b23505334",
                output_format={"container": "raw", "sample_rate": 16000, "encoding": "pcm_s16le"},
                continue_=True,
            )
            print("[DEBUG] Transcript sent successfully")
            await ctx.no_more_inputs()
        except Exception as e:
            print(f"Error in send_transcripts: {e}")

    try:
        send_task = asyncio.create_task(send_transcripts())
        listen_task = asyncio.create_task(receive_and_play_audio(ctx, cancel_event))

        await asyncio.gather(send_task, listen_task)
    except asyncio.CancelledError:
        print("[DEBUG] TTS stream was cancelled")
    except Exception as e:
        print(f"Error in stream_tts: {e}")
    finally:
        await ws.close()
        await client.close()
        print("[DEBUG] TTS stream completed")


async def receive_and_play_audio(ctx, cancel_event):
    """
    Receives audio from the TTS service and plays it.
    """
    p = pyaudio.PyAudio()
    stream = None
    rate = 16000
    buffer_count = 0

    try:
        async for output in ctx.receive():
            if cancel_event.is_set():
                print("[DEBUG] TTS playback cancelled")
                break

            buffer = output.get("audio")

            if buffer is None:
                print("[DEBUG] Received empty buffer from TTS")
                continue

            buffer_count += 1
            print(f"[DEBUG] Received audio buffer {buffer_count} with length {len(buffer)}")

            if not stream:
                stream = p.open(format=pyaudio.paInt16, frames_per_buffer=1024, channels=1, rate=rate, output=True)

            stream.write(buffer)
    except Exception as e:
        print(f"Error in receive_and_play_audio: {e}")
    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        p.terminate()
        print(f"[DEBUG] Played {buffer_count} audio buffers in total")
