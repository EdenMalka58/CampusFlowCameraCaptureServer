import os
import logging
from contextlib import asynccontextmanager
from urllib.parse import urlparse, urlunparse

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from detector import PeopleDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

detector: PeopleDetector | None = None


def _camera_url_without_password(url: str) -> str:
    """Strip password from URL for logs and API responses (keep username if any)."""
    p = urlparse(url)
    host = p.hostname or ""
    port = f":{p.port}" if p.port else ""
    if p.username:
        netloc = f"{p.username}@{host}{port}"
    else:
        netloc = f"{host}{port}"
    return urlunparse(p._replace(netloc=netloc))


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
    """Full RTSP URL, including credentials if required, e.g. ``rtsp://user:pass@host:554/stream``."""
    confidence: float = 0.4

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "camera_url": "rtsp://username:password@192.168.1.100:554/stream",
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

    safe_url = _camera_url_without_password(request.camera_url)
    logger.info("Received request for camera: %s", safe_url)

    try:
        count, detections = detector.count_people(
            camera_url=request.camera_url,
            confidence=request.confidence,
        )
    except ConnectionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error processing camera %s", safe_url)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PeopleCountResponse(
        camera_url=safe_url,
        people_count=count,
        confidence_threshold=request.confidence,
        detections=detections,
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
