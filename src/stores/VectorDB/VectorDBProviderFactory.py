from .Providers import PGVectorProvider
from .VectorDBEnums import VectorDBEnums
from controllers.BaseController import BaseController

from sqlalchemy.orm import sessionmaker

class VectorDBPRoviderFactory:
    def __init__(self, config:str, db_client:sessionmaker = None):
        self.config = config
        self.base_controller = BaseController()
        self.db_client = db_client

    def create(self, provider:str):
        # PostgreSQL + pgvector is the single authoritative vector database
        if provider == VectorDBEnums.PGVector.value or provider == VectorDBEnums.QDRANT.value:
            return PGVectorProvider(
                db_client=self.db_client,
                distance_method=self.config.VECTOR_DB_DISTANCE_METRIC,
                default_vector_size=self.config.EMBEDDING_MODEL_SIZE,
                index_threshold=self.config.VECTOR_DB_PGVEC_INDEX_THRESHOLD,
            )
        return None
