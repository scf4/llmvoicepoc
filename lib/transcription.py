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

check_message_end_prompt = open("prompts/check_message_end.txt", "r").read()

load_dotenv()

CHUNK = 1024
FORMAT = pyaudio.paInt8
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 4
SILENCE_THRESHOLD = 15.0
MINIMUM_SILENCE_SECONDS = 0.8

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

class AudioTranscriber:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.groq_client = AsyncGroq(api_key=GROQ_API_KEY)
        self.is_speaking = False
        self.silence_counter = 0
        self.current_transcription: str = ""
        self.finished_message_queue = asyncio.Queue()

    def audio_capture_thread(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("Listening...")

        while True:
            data = stream.read(CHUNK)
            self.audio_queue.put(data)

    async def check_message_did_end(self, text):
        print("[DEBUG] Checking speaking completion...")
        system_message = check_message_end_prompt

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Live transcription: {text}"}
        ]

        result = await call_llm(messages, model="llama-3.1-8b-instant", max_tokens=1)
        
        print(f"[DEBUG] Completion result: {result}")
        
        return result.strip().upper() == "YES"

    async def process_audio_thread(self):
        audio_buffer = np.array([], dtype=np.int8)
        silence_start_time = None
        speech_detected = False
        speech_start_index = None
        SPEECH_BUFFER_SECONDS = 0.5
        SPEECH_BUFFER_SAMPLES = int(RATE * SPEECH_BUFFER_SECONDS)

        while True:
            if not self.audio_queue.empty():
                audio_chunk = self.audio_queue.get()
                audio_data = np.frombuffer(audio_chunk, dtype=np.int8)
                audio_buffer = np.concatenate((audio_buffer, audio_data))

                window_size = RATE // 4  # 0.25 second window
                step_size = RATE // 10   # 0.1 second step
                energy_levels = []

                for i in range(0, len(audio_buffer) - window_size, step_size):
                    window = audio_buffer[i:i+window_size]
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
                                    
                                    sentence_complete = await self.check_message_did_end(self.current_transcription)
                                    
                                    if sentence_complete:
                                        print("[DEBUG] Message complete, adding to queue")
                                        await self.finished_message_queue.put(self.current_transcription.strip())
                                        self.current_transcription = ""
                                    else:
                                        print("[DEBUG] Message incomplete, continuing to listen")
                                
                                # Reset for next speech segment
                                audio_buffer = np.array([], dtype=np.int8)
                                silence_start_time = None
                                speech_detected = False
                                speech_start_index = None

                # Trim audio buffer to prevent unbounded growth
                max_buffer_size = RATE * 10  # 10 seconds of audio
                if len(audio_buffer) > max_buffer_size:
                    audio_buffer = audio_buffer[-max_buffer_size:]

            else:
                await asyncio.sleep(0.1)

    async def transcribe_with_groq(self, audio_data):
        try:
            audio_segment = AudioSegment(
                audio_data.tobytes(), 
                frame_rate=RATE,
                sample_width=audio_data.dtype.itemsize, 
                channels=CHANNELS
            )

            buffer = io.BytesIO()
            audio_segment.export(buffer, format="mp3")
            buffer.seek(0)
            buffer.name = 'audio.mp3'

            print("[DEBUG] Calling Groq Whisper...")
            response = await self.groq_client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=buffer,
                language="en"
            )
            
            return response.text
        except Exception as e:
            print(f"Error during transcription: {e}")
            return ""

    async def run(self):
        audio_thread = threading.Thread(target=self.audio_capture_thread)
        audio_thread.start()

        asyncio.create_task(self.process_audio_thread())

    async def get_next_complete_message(self):
        return await self.finished_message_queue.get()