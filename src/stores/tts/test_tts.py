import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

src_path = Path(__file__).resolve().parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

root_dir = src_path.parent
load_dotenv(root_dir / ".env.voice")
load_dotenv(src_path / ".env")

from stores.tts import TTSProviderFactory, OpenRouterTTSModels, HeyGenVoices

async def run_tests():
    print("--- 1. Testing TTSProviderFactory & Configuration ---")
    factory = TTSProviderFactory()
    tts_provider = factory.create("OPENROUTER")
    print(f"✓ Initialized TTS Provider: {tts_provider.__class__.__name__}")
    print(f"  Model ID: {tts_provider.model_id}")
    print(f"  Voice: {tts_provider.voice}")
    assert tts_provider.model_id == OpenRouterTTSModels.HEYGEN_VOICE_TURBO.value
    assert tts_provider.voice == HeyGenVoices.GARETH.value

    print("\n--- 2. Testing Dynamic Model & Voice Switching ---")
    tts_provider.set_tts_model("fishaudio/s2.1-pro:free")
    assert tts_provider.model_id == "fishaudio/s2.1-pro:free"
    print("✓ Successfully switched model to: fishaudio/s2.1-pro:free")

    tts_provider.set_tts_model(OpenRouterTTSModels.HEYGEN_VOICE_TURBO.value)
    tts_provider.set_voice(HeyGenVoices.GARETH.value)
    assert tts_provider.model_id == "heygen/heygen-voice-1-turbo"
    assert tts_provider.voice == "Gareth"
    print("✓ Successfully switched back to HeyGen Gareth voice!")

    print("\n--- 3. Testing Direct Audio Synthesis via OpenRouter ---")
    audio_bytes = await tts_provider.synthesize("مرحباً، أنا غاريث عبر نموذج هيجن.")
    print(f"✓ Audio bytes received: {len(audio_bytes)} bytes")
    assert len(audio_bytes) > 1000

    print("\n--- 4. Testing LiveKit TTS Integration (Streaming Frames) ---")
    livekit_tts = tts_provider.get_livekit_tts()
    stream = livekit_tts.synthesize("Hello Gareth, this is a LiveKit audio stream test.")
    frames = []
    async for frame in stream:
        frames.append(frame)
    print(f"✓ LiveKit audio stream produced {len(frames)} PCM frames!")
    assert len(frames) > 0

    print("\n✓ ALL TTS PROVIDER TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    asyncio.run(run_tests())
