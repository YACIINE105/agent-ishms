import os
import sys
import json
import asyncio
import logging
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

from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    AgentServer,
    cli,
)
from livekit.agents.voice import AgentSession, Agent
from livekit.plugins import groq, silero
from stores.tts import TTSProviderFactory

from voice.llm_provider import build_generation_llm
from voice.tools import create_knowledge_base_rag_tool
from helpers.persona import load_project_persona, build_system_instructions
from models import ConversationModel

logger = logging.getLogger("livekit-agent")

_shared_nlp_controller = None
_shared_db_client = None

def set_shared_nlp_controller(nlp_controller, db_client=None):
    """Inject shared FastAPI NLPController and database client."""
    global _shared_nlp_controller, _shared_db_client
    _shared_nlp_controller = nlp_controller
    if db_client:
        _shared_db_client = db_client
    elif nlp_controller and hasattr(nlp_controller, "vector_db_client") and hasattr(nlp_controller.vector_db_client, "db_client"):
        _shared_db_client = nlp_controller.vector_db_client.db_client

def get_shared_nlp_controller():
    return _shared_nlp_controller

def get_shared_db_client():
    global _shared_db_client
    if _shared_db_client:
        return _shared_db_client
    if _shared_nlp_controller and hasattr(_shared_nlp_controller, "vector_db_client"):
        return getattr(_shared_nlp_controller.vector_db_client, "db_client", None)
    return None

def get_langfuse_client():
    """Initialize Langfuse client from credentials."""
    try:
        from langfuse import Langfuse
        pk = os.getenv("LANGFUSE_PUBLIC_KEY")
        sk = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
        if pk and sk:
            return Langfuse(public_key=pk, secret_key=sk, host=host)
    except Exception as e:
        logger.warning(f"Langfuse setup skipped: {e}")
    return None

def resolve_project_context(room_metadata_str: str | None) -> dict:
    """Resolve project id and persona: reads cached project persona or falls back to defaults."""
    project_id = int(os.getenv("DEFAULT_PROJECT_ID", "1"))
    if room_metadata_str:
        try:
            meta = json.loads(room_metadata_str)
            if "project_id" in meta:
                project_id = int(meta["project_id"])
        except Exception:
            pass

    persona = load_project_persona(project_id)
    if not persona:
        persona = {
            "project_id": project_id,
            "name": os.getenv("PROJECT_NAME", f"Project {project_id} Assistant"),
            "domain": os.getenv("PROJECT_DOMAIN", "Official documentation, regulations, and FAQs"),
            "summary": "Official project knowledge base.",
        }
    return persona

def _bootstrap_service_fallback():
    """Fallback when running agent standalone via CLI without FastAPI."""
    try:
        from controllers.NLPController import NLPController
        from helpers.config import get_settings
        from stores.VectorDB import VectorDBPRoviderFactory
        from stores.llm import LLMProviderFactory
        from stores.llm.templates import TemplateParser
        from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
        from sqlalchemy.orm import sessionmaker

        settings = get_settings()
        host = "localhost" if settings.POSTGRES_HOST == "pgvector" else settings.POSTGRES_HOST
        postgres_conn = (
            f"postgresql+asyncpg://{settings.POSTGRES_USERNAME}:{settings.POSTGRES_PASSWORD}@"
            f"{host}:{settings.POSTGRES_PORT}/{settings.POSTGRES_MAIN_DATABASE}"
        )
        db_engine = create_async_engine(postgres_conn)
        db_client = sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

        llm_factory = LLMProviderFactory(settings)
        vector_db_factory = VectorDBPRoviderFactory(settings)

        vector_db_client = vector_db_factory.create(settings.VECTOR_DB_BACKEND)
        vector_db_client.db_client = db_client

        embedding_client = llm_factory.create(provider=settings.EMBEDDING_PROVIDER)
        embedding_client.set_embedding_model(
            model_id=settings.EMBEDDING_MODEL_ID,
            embedding_size=settings.EMBEDDING_MODEL_SIZE,
        )

        generation_client = llm_factory.create(provider=settings.GENERATION_PROVIDER)
        generation_client.set_generation_model(model_id=settings.GENERATION_MODEL_ID)

        template_parser = TemplateParser(
            language=settings.PRIMARY_LANG,
            default_language=settings.DEFAULT_LANG,
        )

        controller = NLPController(
            generation_client=generation_client,
            embedding_client=embedding_client,
            vector_db_client=vector_db_client,
            template_parser=template_parser,
        )
        set_shared_nlp_controller(controller, db_client=db_client)
        return controller

    except Exception as e:
        logger.warning(f"Fallback bootstrap skipped: {e}")
        return None

async def entrypoint(ctx: JobContext):
    logger.info(f"Connecting to room: {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    persona = resolve_project_context(ctx.room.metadata)
    project_id = persona["project_id"]
    logger.info(f"Serving Project #{project_id}: '{persona['name']}' (Domain: {persona['domain']})")

    langfuse = get_langfuse_client()
    session_id = ctx.room.name

    nlp_controller = (
        get_shared_nlp_controller()
        or (ctx.proc.userdata.get("nlp") if hasattr(ctx, "proc") and ctx.proc else None)
        or _bootstrap_service_fallback()
    )

    db_client = (
        get_shared_db_client()
        or (ctx.proc.userdata.get("db") if hasattr(ctx, "proc") and ctx.proc else None)
    )

    conversation_model = await ConversationModel.create_instance(db_client=db_client) if db_client else None
    if conversation_model:
        try:
            await conversation_model.get_or_create_session(session_id=session_id, project_id=project_id)
            logger.info(f"Initialized PostgreSQL ConversationSession: {session_id}")
        except Exception as e:
            logger.warning(f"Failed to initialize ConversationSession in DB: {e}")

    turn_context = {
        "last_user_query": "",
        "last_rag_query": "",
        "last_rag_answer": "",
        "last_sources_cited": [],
        "interrupted": False,
    }

    bilingual_whisper_prompt = (
        "مرحباً، أهلاً وسهلاً بك. أنا هنا لمساعدتك. "
        "Hello, how can I help you today? Inquiries about university admission, registration, courses, and fees."
    )
    stt_engine = groq.STT(
        model="whisper-large-v3-turbo",
        api_key=os.getenv("GROQ_API_KEY"),
        detect_language=True,
        prompt=bilingual_whisper_prompt,
    )
    tts_provider = TTSProviderFactory().create()
    livekit_tts = tts_provider.get_livekit_tts()

    llm_engine = build_generation_llm()

    rag_tool = create_knowledge_base_rag_tool(
        project_id=project_id,
        project_name=persona["name"],
        project_domain=persona["domain"],
        nlp_controller=nlp_controller,
        langfuse=langfuse,
        session_id=session_id,
        turn_context=turn_context,
    )

    agent = Agent(
        instructions=build_system_instructions(persona),
        tools=[rag_tool],
    )

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=stt_engine,
        llm=llm_engine,
        tts=livekit_tts,
        max_tool_steps=1,
        turn_handling={
            "endpointing": {"min_delay": 0.15, "max_delay": 0.4},
            "preemptive_generation": {"enabled": True},
        },
    )

    @session.on("user_input_transcribed")
    def on_user_speech(ev):
        if ev.is_final and ev.transcript:
            turn_context["last_user_query"] = ev.transcript
            turn_context["interrupted"] = False
        if langfuse and ev.transcript:
            langfuse.create_event(
                name="user_speech_transcript",
                input={"transcript": ev.transcript, "session_id": session_id},
                metadata={"is_final": ev.is_final},
            )

    @session.on("agent_state_changed")
    def on_agent_state(ev):
        if getattr(ev, "interrupted", False):
            turn_context["interrupted"] = True

    @session.on("conversation_item_added")
    def on_item_added(ev):

        item = getattr(ev, "item", None)
        if item and getattr(item, "role", None) == "assistant":
            spoken_text = item.text_content

            if spoken_text and not spoken_text.startswith("Hello! I am your AI assistant") and not spoken_text.startswith("أهلاً بك"):
                user_q = turn_context.get("last_user_query") or turn_context.get("last_rag_query") or "(voice query)"
                sources = turn_context.get("last_sources_cited") or []
                interrupted = turn_context.get("interrupted", False)

                if conversation_model:
                    async def save_turn_task():
                        try:
                            await conversation_model.record_turn(
                                session_id=session_id,
                                project_id=project_id,
                                user_query=user_q,
                                spoken_answer=spoken_text,
                                sources_cited=sources,
                                interrupted=interrupted,
                            )
                            logger.info(f"Persisted turn to PostgreSQL for session {session_id} (sources: {sources})")
                        except Exception as err:
                            logger.error(f"Failed to persist turn to PostgreSQL: {err}")

                    asyncio.create_task(save_turn_task())

    @session.on("close")
    def on_close():
        if langfuse:
            langfuse.flush()

    await session.start(agent, room=ctx.room)

    greeting = f"Hello! I am your AI assistant for {persona['name']}. How can I assist you today? أهلاً بك، كيف يمكنني مساعدتك؟"
    await session.say(greeting, allow_interruptions=False)

def prewarm(proc: JobProcess):
    """Pre-warm NLP controller and database client at process startup."""
    logger.info("Pre-warming NLPController service and database client...")
    controller = get_shared_nlp_controller() or _bootstrap_service_fallback()
    proc.userdata["nlp"] = controller
    if controller and hasattr(controller, "vector_db_client"):
        proc.userdata["db"] = getattr(controller.vector_db_client, "db_client", None)

async def create_voice_server(nlp_controller=None, db_client=None) -> AgentServer:
    if nlp_controller:
        set_shared_nlp_controller(nlp_controller, db_client=db_client)
    return AgentServer(options=WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

if __name__ == "__main__":
    agent_name = os.getenv("LIVEKIT_AGENT_NAME", "agent-ishms")
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm, agent_name=agent_name))
