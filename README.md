# Shapes Voice
A proof of concept for Shapes voice chat.

**Important:** Wear headphones.

## Next steps
- [ ] Improve voice activity detection (VAD). Currently, it's very sensitive to background noise which Whisper infers as phrases such as "Thank you" or "Thanks for watching", etc. 
- [ ] Echo cancellation to prevent the assistant from hearing itself and responding. 
- [ ] Stream LLM output to TTS
- [ ] When the user interrupts, replace the assistant's response in the message history with a cut-off version at the same point
- [ ] Don't queue up multiple user messages/responses at once. Replace with messages concatenated into a single message.
- [ ] Remove noise from the audio input, adjust thresholds based on user automatically etc.
- [ ] General refactor/cleanup