from contextlib import asynccontextmanager
from fastapi import FastAPI
from routes import base, data, nlp, voice
from controllers import NLPController
from motor.motor_asyncio import AsyncIOMotorClient
from helpers.config import get_settings
from stores.llm import LLMProviderFactory
from stores.VectorDB import VectorDBPRoviderFactory
from stores.llm.templates import TemplateParser
from sqlalchemy.ext.asyncio import create_async_engine , AsyncSession
from sqlalchemy.orm  import sessionmaker
from utils.metrics import setup_metrics

@asynccontextmanager

async def lifespan(app: FastAPI):
    settings = get_settings()
    host = "localhost" if settings.POSTGRES_HOST == "pgvector" else settings.POSTGRES_HOST
    postgres_conn = f"postgresql+asyncpg://{settings.POSTGRES_USERNAME}:{settings.POSTGRES_PASSWORD}@{host}:{settings.POSTGRES_PORT}/{settings.POSTGRES_MAIN_DATABASE}"
    app.db_engine = create_async_engine(postgres_conn)

    app.db_client = sessionmaker(app.db_engine,
                                 class_=AsyncSession,
                                 expire_on_commit=False)

    llm_provider_factory = LLMProviderFactory(settings)

    vector_db_provider_factory = VectorDBPRoviderFactory(settings)

    app.generation_client = llm_provider_factory.create(provider=settings.GENERATION_PROVIDER)
    app.generation_client.set_generation_model(model_id=settings.GENERATION_MODEL_ID)

    app.embedding_client = llm_provider_factory.create(provider=settings.EMBEDDING_PROVIDER, for_embeddings=True)
    app.embedding_client.set_embedding_model(
        model_id=settings.EMBEDDING_MODEL_ID,
        embedding_size=settings.EMBEDDING_MODEL_SIZE
    )

    app.vector_db_client = vector_db_provider_factory.create(settings.VECTOR_DB_BACKEND)

    app.vector_db_client.db_client = app.db_client

    app.reranker_client = None

    await app.vector_db_client.connect()

    app.template_parser = TemplateParser(language =settings.PRIMARY_LANG,
                                         default_language = settings.DEFAULT_LANG)

    app.state.nlp_controller = NLPController(
        vector_db_client=app.vector_db_client,
        generation_client=app.generation_client,
        embedding_client=app.embedding_client,
        reranker_client=app.reranker_client,
        template_parser=app.template_parser
    )

    yield

    await app.db_engine.dispose()
    await app.vector_db_client.disconnect()

app = FastAPI(lifespan=lifespan)

setup_metrics(app=app)

app.include_router(base.base_router)
app.include_router(data.data_router)
app.include_router(nlp.nlp_router)
app.include_router(voice.voice_router)
