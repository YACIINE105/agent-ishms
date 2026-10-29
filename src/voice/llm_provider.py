import os
import json
import logging
from livekit.agents import llm
from livekit.plugins import openai

logger = logging.getLogger("livekit-agent")

def build_generation_llm():
    """Build multi-provider LLM failover chain: a2agent -> google -> openrouter."""
    providers = []

    a2_key = os.getenv("A2AGENT_API_KEY")
    if a2_key:
        a2_url = os.getenv("A2AGENT_BASE_URL", "https://api.a2agent.me/v1")
        a2_model = os.getenv("A2AGENT_MODEL_ID", "minimax-m3.1-flash")
        providers.append(
            openai.LLM(
                base_url=a2_url.strip("\"'"),
                api_key=a2_key.strip("\"'"),
                model=a2_model.strip("\"'"),
                max_completion_tokens=300,
            )
        )
        logger.info(f"Registered primary LLM: A2Agent ({a2_model})")

    g_key = os.getenv("GOOGLE_AI_API_KEY")
    if g_key:
        g_url = (os.getenv("GOOGLE_AI_BASE_URL") or "").strip("\"'") or "https://generativelanguage.googleapis.com/v1beta/openai/"
        g_model = os.getenv("GOOGLE_AI_MODEL_ID", "gemini-2.5-flash")
        providers.append(
            openai.LLM(
                base_url=g_url,
                api_key=g_key.strip("\"'"),
                model=g_model.strip("\"'"),
                max_completion_tokens=300,
            )
        )
        logger.info(f"Registered secondary LLM: Google AI ({g_model})")

    or_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("openrouter_api_key") or os.getenv("OPENAI_API_KEY")
    if or_key:
        if isinstance(or_key, str) and or_key.startswith("["):
            try:
                or_key = json.loads(or_key)[0]
            except Exception:
                pass
        or_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        or_model = os.getenv("GENERATION_MODEL_ID", "google/gemini-2.5-flash")
        providers.append(
            openai.LLM(
                base_url=or_url.strip("\"'"),
                api_key=or_key.strip("\"'"),
                model=or_model.strip("\"'"),
                max_completion_tokens=300,
            )
        )
        logger.info(f"Registered fallback LLM: OpenRouter ({or_model})")

    if not providers:
        raise ValueError("No generation LLM providers found in environment!")

    if len(providers) == 1:
        return providers[0]

    return llm.FallbackAdapter(
        providers,
        attempt_timeout=8.0,
        max_retry_per_llm=0,
    )
