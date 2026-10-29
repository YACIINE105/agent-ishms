from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
import os

class Settings(BaseSettings):

    APP_NAME : str
    APP_VERSION : str

    FILE_ALLOWED_TYPES : list
    FILE_MAX_SIZE : int
    FILE_CHHUNK_SIZE : int

    DEFAULT_LANG : str = "en"
    PRIMARY_LANG : str = "en"

    MONGO_URL: str | None = None
    MONGO_DATABASE: str | None = None

    POSTGRES_USERNAME : str
    POSTGRES_PASSWORD : str
    POSTGRES_HOST : str
    POSTGRES_PORT : int
    POSTGRES_MAIN_DATABASE : str

    GENERATION_PROVIDER : str
    EMBEDDING_PROVIDER : str

    OPENAI_BASE_URL : str = None
    OPENAI_API_KEY : str = None

    COHERE_API_KEY : str = None

    GOOGLE_AI_API_KEY : str = None
    GOOGLE_AI_BASE_URL: str | None = "https://generativelanguage.googleapis.com/v1beta/openai/"
    GOOGLE_AI_MODEL_ID: str | None = "gemini-2.5-flash"

    A2AGENT_API_KEY: str | None = None
    A2AGENT_BASE_URL: str | None = "https://api.a2agent.me/v1"
    A2AGENT_MODEL_ID: str | None = "minimax-m3.1-flash"

    OCR_SPACE_API_KEY: str | None = None

    LIVEKIT_URL: str | None = None
    LIVEKIT_API_KEY: str | None = None
    LIVEKIT_API_SECRET: str | None = None

    NARA_API_KEY : str = None
    NARA_BASE_URL:str

    OPENROUTER_API_KEY : str = None 
    OPENROUTER_BASE_URL : str = "https://openrouter.ai/api/v1" 

    GENERATION_MODEL_ID_LITTERAL: List[str] | None = None
    GENERATION_MODEL_ID : str | None = None
    GENERATION_MODEL_PATH: str | None = None
    GENERATION_MODEL_TEMPERATURE : float | None = None

    EMBEDDING_MODEL_ID_LITTERAL: List[str] | None = None
    EMBEDDING_MODEL_ID : str | None = None
    EMBEDDING_MODEL_PATH : str | None = None
    EMBEDDING_MODEL_SIZE : int | None = None

    INPUT_MAX_CHARACTERS : int | None = None
    MAX_OUTPUT_TOKENS : int | None = None

    LLAMA_CPP_N_GPU_LAYERS : int = 0
    LLAMA_CPP_N_CTX : int = 2048
    LLama_CPP_API_KEY: str | None = None
    LLAMA_CPP_URL : str | None = None

    VECTOR_DB_BACKEND_LITTERAL: List[str] = None
    VECTOR_DB_BACKEND : str
    VECTOR_DB_DISTANCE_METRIC : str
    VECTOR_DB_PATH : str
    VECTOR_DB_PGVEC_INDEX_THRESHOLD:int

    class Config:
        extra = "ignore"
        env_file = (
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env.voice"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
            ".env",
            ".env.voice",
        )    

def get_settings():
    return Settings()
