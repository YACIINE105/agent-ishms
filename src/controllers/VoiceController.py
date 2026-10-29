from .BaseController import BaseController
from livekit import api
from datetime import timedelta
import os
import logging

logger = logging.getLogger("uvicorn.error")

class VoiceController(BaseController):
    def __init__(self):
        super().__init__()
        self.livekit_url = self.app_settings.LIVEKIT_URL or os.getenv("LIVEKIT_URL")
        self.api_key = self.app_settings.LIVEKIT_API_KEY or os.getenv("LIVEKIT_API_KEY")
        self.api_secret = self.app_settings.LIVEKIT_API_SECRET or os.getenv("LIVEKIT_API_SECRET")

    def generate_token(self, room_name: str | None = None, participant_name: str = "student_user") -> dict:
        if not self.api_key or not self.api_secret:
            raise ValueError(
                "LIVEKIT_API_KEY or LIVEKIT_API_SECRET is missing. "
                "Please configure them in .env or .env.voice."
            )

        if not room_name or not room_name.strip():
            room_name = f"room_{self.generate_random_strings(lenght=8)}"

        clean_identity = participant_name.strip() if participant_name else "student_user"

        token = (
            api.AccessToken(self.api_key.strip("\"'"), self.api_secret.strip("\"'"))
            .with_identity(clean_identity)
            .with_name(clean_identity)
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_publish=True,
                    can_subscribe=True,
                )
            )
            .with_ttl(timedelta(hours=2))
        )

        jwt_token = token.to_jwt()

        return {
            "token": jwt_token,
            "url": self.livekit_url.strip("\"'") if self.livekit_url else None,
            "room_name": room_name,
            "participant_name": clean_identity,
        }
