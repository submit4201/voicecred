from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import jwt, JWTError

# NOTE: In prod, keep keys in secure secret store
JWT_SECRET = "dev-secret-key-please-change"
JWT_ALG = "HS256"


def create_session_token(session_id: str, ttl_sec: int = 60 * 60) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": session_id,
        "exp": now + timedelta(seconds=ttl_sec),
        "iat": now,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def verify_session_token(token: str) -> Optional[str]:
    try:
        data = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        return data.get("sub")
    except JWTError:
        return None
