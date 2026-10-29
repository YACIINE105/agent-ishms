import json
import os
from .TTSEnums import TTS_Enums, OpenRouterTTSModels, HeyGenVoices
from .providers import OpenRouterTTSProvider, EdgeTTSProvider

class TTSProviderFactory:
    def __init__(self, config: dict = None):
        self.config = config or {}

    def _get_val(self, key: str, default=None):
        val = None
        if isinstance(self.config, dict):
            val = self.config.get(key)
        else:
            val = getattr(self.config, key, None)

        if val is None:
            val = os.getenv(key) or os.getenv(key.lower()) or default
        return val

    def create(self, provider: str = None):
        target_provider = provider or self._get_val("TTS_PROVIDER", TTS_Enums.EDGE.value)
        if isinstance(target_provider, TTS_Enums):
            target_provider = target_provider.value

        if target_provider.upper() in [TTS_Enums.EDGE.value, "EDGE", "EDGE_TTS"]:
            voice = self._get_val("TTS_VOICE", "en-US-ChristopherNeural")
            return EdgeTTSProvider(voice=voice)

        if target_provider.upper() in [TTS_Enums.OPENROUTER.value, "OPENROUTER", "HEYGEN"]:
            api_key = self._get_val("OPENROUTER_API_KEY")
            if api_key and isinstance(api_key, str) and api_key.startswith("["):
                try:
                    api_key = json.loads(api_key)[0]
                except Exception:
                    pass

            base_url = self._get_val("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            model_id = self._get_val("TTS_MODEL_ID", OpenRouterTTSModels.HEYGEN_VOICE_TURBO.value)
            voice = self._get_val("TTS_VOICE", HeyGenVoices.GARETH.value)
            response_format = self._get_val("TTS_RESPONSE_FORMAT", "pcm")

            return OpenRouterTTSProvider(
                api_key=api_key,
                base_url=base_url,
                model_id=model_id,
                voice=voice,
                response_format=response_format,
            )

        raise ValueError(f"Unsupported TTS provider: {target_provider}")
