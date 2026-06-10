class VoiceMode:
    def __init__(self):
        self.asr_model = None
        self._init_asr()
        self.tts_voice = "en-US-GuyNeural"
        self.sample_rate = 16000

    def _init_asr(self):
        try:
            from faster_whisper import WhisperModel
            self.asr_model = WhisperModel("tiny", device="cuda", compute_type="int8")
            print("[voice] ASR model loaded (tiny, int8)")
        except Exception as e:
            print(f"[voice] ASR init failed: {e}")

    def listen(self, duration=5):
        import sounddevice as sd
        import numpy as np
        print(f"[voice] Recording for {duration}s... (speak now)")
        audio = sd.rec(int(duration * self.sample_rate), samplerate=self.sample_rate,
                       channels=1, dtype="float32")
        sd.wait()
        audio = np.squeeze(audio)
        print(f"[voice] Recorded {len(audio)/self.sample_rate:.1f}s")
        return audio

    def transcribe(self, audio):
        if self.asr_model is None:
            return None
        segments, info = self.asr_model.transcribe(audio, beam_size=5, language="en")
        text = " ".join(seg.text.strip() for seg in segments)
        return text if text.strip() else None

    def speak(self, text):
        import edge_tts
        import asyncio
        import miniaudio
        import sounddevice as sd
        import numpy as np
        async def _speak():
            tts = edge_tts.Communicate(text, voice=self.tts_voice)
            stream = tts.stream()
            chunks = []
            async for chunk in stream:
                if chunk["type"] == "audio":
                    chunks.append(chunk["data"])
            mp3_data = b"".join(chunks)
            decoded = miniaudio.decode(mp3_data, output_format=miniaudio.SampleFormat.SIGNED16)
            samples = np.frombuffer(decoded.samples, dtype=np.int16).astype(np.float32) / 32768.0
            sd.play(samples, samplerate=decoded.sample_rate)
            sd.wait()
        asyncio.run(_speak())

    def chat_voice_loop(self, agent):
        print("[voice] Voice mode ready. Speak after each prompt.")
        print("[voice] Ctrl+C to exit.")
        while True:
            try:
                audio = self.listen(duration=5)
                text = self.transcribe(audio)
                if not text:
                    print("[voice] No speech detected")
                    continue
                print(f"[voice] You: {text}")
                reply = agent.chat(text)
                print(f"[voice] Adam: {reply}")
                self.speak(reply)
            except KeyboardInterrupt:
                print("\n[voice] Exiting voice mode.")
                break
            except Exception as e:
                print(f"[voice] Error: {e}")
