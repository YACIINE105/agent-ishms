import uuid
import logging
from .BaseDataModel import BaseDataModel
from .db_schems.agent_ishms.schemes import ConversationSession, ConversationTurn
from sqlalchemy.future import select

logger = logging.getLogger("uvicorn.error")

class ConversationModel(BaseDataModel):
    def __init__(self, db_client: object):
        super().__init__(db_client=db_client)
        self.db_client = db_client

    @classmethod
    async def create_instance(cls, db_client: object):
        return cls(db_client)

    async def get_or_create_session(self, session_id: str | uuid.UUID, project_id: int) -> ConversationSession:
        """Fetch or create a ConversationSession by UUID."""
        if isinstance(session_id, str):
            try:
                session_uuid = uuid.UUID(session_id)
            except ValueError:
                session_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, session_id)
        else:
            session_uuid = session_id

        async with self.db_client() as session:
            async with session.begin():
                query = select(ConversationSession).where(ConversationSession.session_id == session_uuid)
                result = await session.execute(query)
                conv_session = result.scalar_one_or_none()

                if conv_session is None:
                    conv_session = ConversationSession(
                        session_id=session_uuid,
                        project_id=project_id,
                    )
                    session.add(conv_session)
                    await session.flush()
                    await session.refresh(conv_session)

                return conv_session

    async def record_turn(
        self,
        session_id: str | uuid.UUID,
        project_id: int,
        user_query: str,
        spoken_answer: str,
        sources_cited: list | dict | None = None,
        interrupted: bool = False,
    ) -> ConversationTurn:
        """Insert a conversation turn record into PostgreSQL."""
        conv_session = await self.get_or_create_session(session_id=session_id, project_id=project_id)

        async with self.db_client() as session:
            async with session.begin():
                turn = ConversationTurn(
                    session_id=conv_session.session_id,
                    user_query=user_query,
                    spoken_answer=spoken_answer,
                    sources_cited=sources_cited,
                    interrupted=interrupted,
                )
                session.add(turn)
                await session.flush()
                await session.refresh(turn)
                return turn
