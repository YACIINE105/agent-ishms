from pydantic import BaseModel, Field
from typing import Optional

class VoiceTokenRequest(BaseModel):
    room_name: Optional[str] = Field(default=None, description="Optional custom room name to join")
    participant_name: Optional[str] = Field(default="student_user", description="Name/Identity of the joining participant")

class VoiceTokenResponse(BaseModel):
    token: str = Field(description="Signed LiveKit JWT access token")
    url: Optional[str] = Field(default=None, description="LiveKit WebSocket server URL")
    room_name: str = Field(description="Name of the room")
    participant_name: str = Field(description="Assigned participant identity")
