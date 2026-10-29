from abc import ABC, abstractmethod

class TTS_Interface(ABC):
    @abstractmethod
    def set_tts_model(self, model_id: str):
        """Configure or change the TTS model."""
        pass

    @abstractmethod
    def set_voice(self, voice: str):
        """Configure or change the voice."""
        pass

    @abstractmethod
    async def synthesize(self, text: str, voice: str = None) -> bytes:
        """Synthesize text to audio bytes."""
        pass

    @abstractmethod
    def get_livekit_tts(self):
        """Return a LiveKit-compatible TTS plugin instance."""
        pass
