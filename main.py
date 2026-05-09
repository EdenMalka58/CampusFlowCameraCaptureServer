import base64
import os
import logging
from contextlib import asynccontextmanager
from urllib.parse import urlparse, urlunparse

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from detector import PeopleDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

detector: PeopleDetector | None = None


# ------------------------------------------------------------------
# AES-256-CBC helpers
# ------------------------------------------------------------------

def _decrypt_password(encrypted_b64: str) -> str:
    """
    Decrypt a password that was AES-256-CBC encrypted by the C# client.

    The caller must base64-encode the 16-byte IV prepended to the cipher bytes,
    i.e.  base64( IV[16] + ciphertext ).

    Requires env var: CAMERA_ENCRYPT_KEY  (base64-encoded 32-byte key)
    """
    raw_key = os.environ.get("CAMERA_ENCRYPT_KEY", "")
    if not raw_key:
        raise RuntimeError("CAMERA_ENCRYPT_KEY environment variable is not set.")

    key = base64.b64decode(raw_key)
    if len(key) != 32:
        raise RuntimeError("CAMERA_ENCRYPT_KEY must decode to exactly 32 bytes (AES-256).")

    data = base64.b64decode(encrypted_b64)
    if len(data) < 17:
        raise ValueError("Encrypted payload is too short to contain an IV.")

    iv, ciphertext = data[:16], data[16:]

    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    padded = decryptor.update(ciphertext) + decryptor.finalize()

    # Strip PKCS7 padding
    pad_len = padded[-1]
    if pad_len < 1 or pad_len > 16:
        raise ValueError("Invalid PKCS7 padding — wrong key or corrupted data.")

    return padded[:-pad_len].decode("utf-8")


def _build_camera_url(camera_url: str, encrypted_password: str | None) -> str:
    """
    Return the RTSP URL with the decrypted password injected.
    If *encrypted_password* is None the original URL is returned unchanged
    (supports cameras that need no authentication).
    """
    if not encrypted_password:
        return camera_url

    plain_password = _decrypt_password(encrypted_password)
    parsed = urlparse(camera_url)

    host = parsed.hostname or ""
    port_part = f":{parsed.port}" if parsed.port else ""
    user_part = f"{parsed.username}:{plain_password}" if parsed.username else plain_password
    netloc = f"{user_part}@{host}{port_part}"

    return urlunparse(parsed._replace(netloc=netloc))


# ------------------------------------------------------------------
# FastAPI setup
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    logger.info("Loading YOLO model...")
    detector = PeopleDetector()
    logger.info("YOLO model loaded successfully.")
    yield
    detector = None


app = FastAPI(
    title="CampusFlow Camera Capture Server",
    description="Detects and counts people in an IP camera feed using YOLOv8.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# Request / response models
# ------------------------------------------------------------------

class CameraRequest(BaseModel):
    camera_url: str
    """
    RTSP URL **without** the password, e.g.
    ``rtsp://username@192.168.1.100:554/stream``
    The password is supplied separately via *encrypted_password*.
    For cameras that require no authentication, omit *encrypted_password*
    and embed the full URL here as before.
    """
    encrypted_password: str | None = None
    """
    AES-256-CBC encrypted password, base64-encoded.
    Format: base64( IV[16 bytes] + ciphertext )
    Encrypted with the shared CAMERA_ENCRYPT_KEY.
    """
    confidence: float = 0.4

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "camera_url": "rtsp://username@192.168.1.100:554/stream",
                    "encrypted_password": "<base64( IV + AES-256-CBC(password) )>",
                    "confidence": 0.4,
                }
            ]
        }
    }


class PeopleCountResponse(BaseModel):
    camera_url: str
    people_count: int
    confidence_threshold: float
    detections: list[dict]


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": detector is not None}


@app.post("/count-people", response_model=PeopleCountResponse)
def count_people(request: CameraRequest):
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # Log the URL without credentials
    logger.info("Received request for camera: %s", request.camera_url)

    try:
        full_url = _build_camera_url(request.camera_url, request.encrypted_password)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Password decryption failed: {exc}") from exc

    try:
        count, detections = detector.count_people(
            camera_url=full_url,
            confidence=request.confidence,
        )
    except ConnectionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error processing camera %s", request.camera_url)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PeopleCountResponse(
        camera_url=request.camera_url,   # never expose the password in the response
        people_count=count,
        confidence_threshold=request.confidence,
        detections=detections,
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
