from fastapi import APIRouter, Depends, HTTPException, status

from app.core.admin_auth import authenticate_admin, create_admin_token, require_admin
from app.core.config import Settings, get_settings
from app.models.schemas import AdminLoginRequest, AdminLoginResponse

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/login", response_model=AdminLoginResponse)
def login_admin(
    payload: AdminLoginRequest,
    settings: Settings = Depends(get_settings),
) -> AdminLoginResponse:
    if not authenticate_admin(payload.username, payload.password, settings):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin credentials.")

    return AdminLoginResponse(
        access_token=create_admin_token(settings),
        expires_in=settings.admin_session_ttl_minutes * 60,
    )


@router.get("/session")
def get_admin_session(_claims=Depends(require_admin)) -> dict[str, str]:
    return {"status": "ok"}
