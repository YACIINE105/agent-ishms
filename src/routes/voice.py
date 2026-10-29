from fastapi import APIRouter, status, Request
from fastapi.responses import JSONResponse
from .schemes.voice import VoiceTokenRequest, VoiceTokenResponse
from controllers import VoiceController
import logging

logger = logging.getLogger("uvicorn.error")

voice_router = APIRouter(
    prefix="/api/v1/voice",
    tags=["api_v1", "voice"],
)

@voice_router.post("/token", response_model=VoiceTokenResponse)
async def get_voice_token(request: Request, body: VoiceTokenRequest = VoiceTokenRequest()):
    try:
        controller = VoiceController()
        data = controller.generate_token(
            room_name=body.room_name,
            participant_name=body.participant_name,
        )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=data,
        )
    except ValueError as ve:
        logger.warning(f"Voice token configuration error: {ve}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": str(ve)},
        )
    except Exception as exc:
        logger.error(f"Failed to generate voice token: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Internal server error generating voice token"},
        )
