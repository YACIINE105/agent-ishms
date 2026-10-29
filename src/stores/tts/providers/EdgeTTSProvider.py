import logging
import edge_tts
import livekit.agents as la
from livekit.agents import tts
from ..TTSInterface import TTS_Interface

class EdgeChunkedStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        arabic_chars = sum(1 for c in self.input_text if "\u0600" <= c <= "\u06FF")
        latin_chars = sum(1 for c in self.input_text if "a" <= c.lower() <= "z")
        target_voice = "ar-SA-HamedNeural" if arabic_chars > latin_chars else self._tts._voice

        communicate = edge_tts.Communicate(self.input_text, target_voice)
        output_emitter.initialize(
            request_id="",
            sample_rate=24000,
            num_channels=1,
            mime_type="audio/mp3",
        )
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                output_emitter.push(chunk["data"])

class EdgeLiveKitTTS(tts.TTS):
    def __init__(self, voice: str = "en-US-ChristopherNeural"):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=24000,
            num_channels=1,
        )
        self._voice = voice

    def set_voice(self, voice: str):
        self._voice = voice

    def synthesize(
        self, text: str, *, conn_options=None
    ) -> tts.ChunkedStream:
        return EdgeChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options or la.DEFAULT_API_CONNECT_OPTIONS,
        )

class EdgeTTSProvider(TTS_Interface):
    """100% Free, unlimited, bilingual Microsoft Edge Neural TTS Provider."""

    def __init__(self, voice: str = "en-US-ChristopherNeural", model_id: str = "edge-tts"):
        self.voice = voice
        self.model_id = model_id
        self.logger = logging.getLogger("uvicorn.error")
        self._livekit_tts = EdgeLiveKitTTS(voice=self.voice)

    def set_tts_model(self, model_id: str):
        self.model_id = model_id

    def set_voice(self, voice: str):
        self.voice = voice
        self._livekit_tts.set_voice(voice)

    async def synthesize(self, text: str, voice: str = None) -> bytes:
        arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
        latin_chars = sum(1 for c in text if "a" <= c.lower() <= "z")
        target_voice = voice or ("ar-SA-HamedNeural" if arabic_chars > latin_chars else self.voice)
        communicate = edge_tts.Communicate(text, target_voice)
        audio_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])
        return b"".join(audio_chunks)

    def get_livekit_tts(self):
        return self._livekit_tts
