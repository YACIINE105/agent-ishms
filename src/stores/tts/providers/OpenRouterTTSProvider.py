import logging
import httpx
from ..TTSInterface import TTS_Interface
from ..TTSEnums import OpenRouterTTSModels, HeyGenVoices

class OpenRouterTTSProvider(TTS_Interface):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        model_id: str = OpenRouterTTSModels.HEYGEN_VOICE_TURBO.value,
        voice: str = HeyGenVoices.GARETH.value,
        response_format: str = "pcm",
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.voice = voice
        self.response_format = response_format
        self.logger = logging.getLogger("uvicorn.error")

    def set_tts_model(self, model_id: str):
        """Switch the TTS model dynamically."""
        self.logger.info(f"Switching TTS model to: {model_id}")
        self.model_id = model_id

    def set_voice(self, voice: str):
        """Switch the TTS voice dynamically."""
        self.logger.info(f"Switching TTS voice to: {voice}")
        self.voice = voice

    async def synthesize(self, text: str, voice: str = None) -> bytes:
        """Directly synthesize speech to raw audio bytes via OpenRouter audio/speech API."""
        target_voice = voice or self.voice
        url = f"{self.base_url}/audio/speech"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_id,
            "input": text,
            "voice": target_voice,
            "response_format": self.response_format,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code != 200:
                self.logger.error(
                    f"OpenRouter TTS error {response.status_code}: {response.text}"
                )
                raise RuntimeError(
                    f"OpenRouter TTS failed ({response.status_code}): {response.text}"
                )
            return response.content

    def get_livekit_tts(self):
        """Return LiveKit-compatible TTS object calibrated for 44.1 kHz HeyGen PCM output."""
        from livekit.plugins.openai import tts as openai_tts
        from livekit.agents import tts

        class HeyGenChunkedStream(openai_tts.ChunkedStream):
            async def _run(self, output_emitter: tts.AudioEmitter) -> None:
                oai_stream = self._tts._client.audio.speech.with_streaming_response.create(
                    input=self.input_text,
                    model=self._opts.model,
                    voice=self._opts.voice,
                    response_format=self._opts.response_format,
                    speed=self._opts.speed,
                    stream_format="audio",
                )
                async with oai_stream as stream:
                    output_emitter.initialize(
                        request_id=stream.request_id or "",
                        sample_rate=44100,
                        num_channels=1,
                        mime_type="audio/pcm",
                    )
                    async for chunk in stream.iter_bytes():
                        output_emitter.push(chunk)

        class HeyGenLiveKitTTS(openai_tts.TTS):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self._sample_rate = 44100

            def synthesize(
                self, text: str, *, conn_options=openai_tts.DEFAULT_API_CONNECT_OPTIONS
            ) -> tts.ChunkedStream:
                return HeyGenChunkedStream(
                    tts=self, input_text=text, conn_options=conn_options
                )

        return HeyGenLiveKitTTS(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model_id,
            voice=self.voice,
            response_format=self.response_format,
        )
