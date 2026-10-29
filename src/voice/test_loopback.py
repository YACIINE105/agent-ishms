import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

src_path = Path(__file__).resolve().parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

root_dir = src_path.parent
env_voice_path = root_dir / ".env.voice"
if env_voice_path.exists():
    load_dotenv(dotenv_path=env_voice_path)

load_dotenv(dotenv_path=src_path / ".env")

from livekit.agents.voice import AgentSession
from livekit.plugins import openai, silero
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli

from stores.tts import TTSProviderFactory

async def entrypoint(ctx: JobContext):
    print(f"\n[LiveKit Worker] Connecting to room: {ctx.room.name}...")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_api_key and openrouter_api_key.startswith("["):
        try:
            openrouter_api_key = json.loads(openrouter_api_key)[0]
        except Exception:
            pass

    if not openrouter_api_key:
        print("⚠ Warning: OPENROUTER_API_KEY is not set in .env.voice or src/.env")

    llm = openai.LLM(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
        model=os.getenv("GENERATION_MODEL_ID", "google/gemini-2.5-flash"),
    )

    tts_factory = TTSProviderFactory()
    tts_provider = tts_factory.create()
    print(f"✓ Using TTS Provider: {tts_provider.model_id} (Voice: {tts_provider.voice})")
    livekit_tts = tts_provider.get_livekit_tts()

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=openai.STT(),
        llm=llm,
        tts=livekit_tts,
    )

    await session.start(ctx.room)

    await session.say(
        "Hello! I am speaking with HeyGen Voice Turbo using the Gareth voice. Speak into your microphone to test.",
        allow_interruptions=True
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
