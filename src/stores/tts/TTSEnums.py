from enum import Enum

class TTS_Enums(Enum):
    EDGE = "EDGE"
    OPENROUTER = "OPENROUTER"
    OPENAI = "OPENAI"
    ELEVENLABS = "ELEVENLABS"

class OpenRouterTTSModels(Enum):
    HEYGEN_VOICE_TURBO = "heygen/heygen-voice-1-turbo"
    HEYGEN_VOICE_TURBO_FREE = "heygen/heygen-voice-1-turbo:free"
    FISH_AUDIO_S2_1_FREE = "fishaudio/s2.1-pro:free"

class HeyGenVoices(Enum):
    GARETH = "Gareth"
    ANN = "Ann"
    CHARLIE = "Charlie"
    DONNA = "Donna"
    EDWARD = "Edward"
    FAITH = "Faith"
    GARY = "Gary"
    HELEN = "Helen"
