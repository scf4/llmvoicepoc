# Whisper/LLM Voice Chat
A quick/messy proof of concept as an attempt at a low-latency, natural-feeling (and interruptible) voice-to-voice chat using LLMs.

## How to run
1. Clone the repo `https://github.com/scf4/shapevoice.git`
2. Install required packages `pip install -r requirements.txt`
3. Rename `.env.example` to `.env` and add your [Cartesia](https://cartesia.ai) and [Groq](https://groq.com) API keys.
4. Run `main.py`
5. **Wear headphones** (see below for limitations)

> **Note:** You will likely need to adjust some of the constants such as silence threshold etc.

## Recommended next steps
- [ ] Improve voice activity detection (VAD). Currently, it's very sensitive to background noise which Whisper infers as phrases such as "Thank you" or "Thanks for watching", etc. 
- [ ] Echo cancellation to prevent the assistant from hearing itself and responding. 
- [ ] Stream LLM output to TTS
- [ ] When the user interrupts, replace the assistant's response in the message history with a cut-off version at the same point
- [ ] Don't queue up multiple user messages/responses at once. Replace with messages concatenated into a single message.
- [ ] Remove noise from the audio input, adjust thresholds based on user automatically etc.
- [ ] General refactor/cleanup
