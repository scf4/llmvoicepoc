import asyncio
import io
import numpy as np
import os
import pyaudio
import queue
import threading

from dotenv import load_dotenv
from groq import AsyncGroq
from pydub import AudioSegment
from lib.llm import call_llm
from lib.chat import call_chat_llm

load_dotenv()

CHUNK = 1024
FORMAT = pyaudio.paInt8
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 4
SILENCE_THRESHOLD = 18.0
MINIMUM_SILENCE_SECONDS = 0.5

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

check_message_did_end_system_prompt = open("prompts/check_message_did_end.txt").read()


async def generate_response(transcription, username, ai_name):
    messages = [{"role": "user", "content": f"{username}: {transcription}"}]
    response = await call_chat_llm(messages, username, ai_name)
    return response


class AudioTranscriber:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.groq_client = None
        self.is_speaking = False
        self.silence_counter = 0
        self.current_transcription: str = ""
        self.finished_message_queue = asyncio.Queue()
        self.baseline_energy = 0.0
        self.user_speaking_event = asyncio.Event()

    async def initialize(self):
        self.groq_client = AsyncGroq(api_key=GROQ_API_KEY)

    async def close(self):
        if self.groq_client:
            await self.groq_client.close()

    def audio_capture_thread(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

        print("Listening...")

        while True:
            data = stream.read(CHUNK)
            self.audio_queue.put(data)

    async def check_message_did_end(self, text):
        messages = [
            {"role": "system", "content": check_message_did_end_system_prompt},
            {"role": "user", "content": f"Live transcription: {text}"},
        ]

        result = await call_llm(messages, max_tokens=2)

        print(f"[DEBUG] Message did end?: {result}")

        did_end = result.strip().upper() == "YES"
        possibly_did_end = result.strip().upper().startswith("UN")

        return did_end or possibly_did_end

    async def transcribe_with_groq(self, audio_data):
        if not self.groq_client:
            await self.initialize()

        try:
            audio_segment = AudioSegment(
                audio_data.tobytes(), frame_rate=RATE, sample_width=audio_data.dtype.itemsize, channels=CHANNELS
            )

            buffer = io.BytesIO()
            audio_segment.export(buffer, format="mp3")
            buffer.seek(0)
            buffer.name = "audio.mp3"

            print("[DEBUG] Calling Groq Whisper...")
            response = await self.groq_client.audio.transcriptions.create(model="whisper-large-v3", file=buffer, language="en")

            return response.text
        except Exception as e:
            print(f"Error during transcription: {e}")
            return ""

    async def get_next_complete_message(self):
        return await self.finished_message_queue.get()

    async def wait_for_user_speech(self):
        await self.user_speaking_event.wait()

    async def run(self, username, ai_name):
        audio_thread = threading.Thread(target=self.audio_capture_thread)
        audio_thread.start()

        asyncio.create_task(self.process_audio_thread(username, ai_name))

    async def process_audio_thread(self, username, ai_name):
        audio_buffer = np.array([], dtype=np.int8)
        silence_start_time = None
        speech_detected = False
        speech_start_index = None
        SPEECH_BUFFER_SECONDS = 1
        SPEECH_BUFFER_SAMPLES = int(RATE * SPEECH_BUFFER_SECONDS)

        while True:
            if not self.audio_queue.empty():
                audio_chunk = self.audio_queue.get()
                audio_data = np.frombuffer(audio_chunk, dtype=np.int8)
                audio_buffer = np.concatenate((audio_buffer, audio_data))

                window_size = RATE // 4  # 0.25 second window
                step_size = RATE // 10  # 0.1 second step
                energy_levels = []

                for i in range(0, len(audio_buffer) - window_size, step_size):
                    window = audio_buffer[i : i + window_size]
                    avg_energy = np.mean(np.abs(window))
                    peak_energy = np.max(np.abs(window))
                    # Combine average and peak energy
                    energy = (avg_energy + peak_energy) / 2
                    energy_levels.append(energy)

                if energy_levels:
                    current_energy = max(energy_levels[-3:])  # Max energy of last 3 windows
                    print(f"[DEBUG] Current energy: {current_energy}")

                    if current_energy > SILENCE_THRESHOLD:
                        if not speech_detected:
                            print("[DEBUG] Speech started")
                            speech_detected = True
                            speech_start_index = max(0, len(audio_buffer) - SPEECH_BUFFER_SAMPLES)
                            self.user_speaking_event.set()  # User has started speaking
                        silence_start_time = None
                    else:
                        if speech_detected:
                            if silence_start_time is None:
                                silence_start_time = asyncio.get_event_loop().time()
                            elif asyncio.get_event_loop().time() - silence_start_time >= MINIMUM_SILENCE_SECONDS:
                                print("[DEBUG] Speech ended. Transcribing...")
                                speech_end_index = len(audio_buffer)
                                speech_audio = audio_buffer[speech_start_index:speech_end_index]

                                transcription = await self.transcribe_with_groq(speech_audio)
                                if transcription:
                                    self.current_transcription += " " + transcription.strip()
                                    print(f"[DEBUG] Current transcription: {self.current_transcription}")

                                    sentence_complete_task = asyncio.create_task(
                                        self.check_message_did_end(self.current_transcription)
                                    )
                                    response_task = asyncio.create_task(
                                        generate_response(self.current_transcription, username, ai_name)
                                    )

                                    sentence_complete, response = await asyncio.gather(sentence_complete_task, response_task)

                                    if sentence_complete:
                                        print("[DEBUG] Message complete, adding to queue")
                                        await self.finished_message_queue.put((self.current_transcription.strip(), response))
                                        self.current_transcription = ""
                                    else:
                                        print("[DEBUG] Message incomplete, continuing to listen")

                                # Reset for next speech segment
                                audio_buffer = np.array([], dtype=np.int8)
                                silence_start_time = None
                                speech_detected = False
                                speech_start_index = None
                                self.user_speaking_event.clear()

                # Trim audio buffer to prevent unbounded growth
                max_buffer_size = RATE * 30  # 30 seconds of audio
                if len(audio_buffer) > max_buffer_size:
                    audio_buffer = audio_buffer[-max_buffer_size:]

            else:
                await asyncio.sleep(0.1)
