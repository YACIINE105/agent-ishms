from .agent_ishms_base import SQLAlchemyBase
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
import uuid

class ConversationSession(SQLAlchemyBase):
    __tablename__ = "sessions" 

    session_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(Integer, ForeignKey("projects.project_id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    project = relationship("Project", back_populates="sessions")
    turns = relationship("ConversationTurn", back_populates="session", cascade="all, delete-orphan")

class ConversationTurn(SQLAlchemyBase):
    __tablename__ = "turns"

    turn_id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.session_id"), nullable=False)

    user_query = Column(String, nullable=False)
    spoken_answer = Column(String, nullable=True)
    sources_cited = Column(JSONB, nullable=True)
    interrupted = Column(Boolean, default=False, nullable=False)  
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    session = relationship("ConversationSession", back_populates="turns")
