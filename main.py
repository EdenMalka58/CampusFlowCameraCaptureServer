import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from detector import PeopleDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

detector: PeopleDetector | None = None


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


class CameraRequest(BaseModel):
    camera_url: str
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


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": detector is not None}


@app.post("/count-people", response_model=PeopleCountResponse)
def count_people(request: CameraRequest):
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    logger.info("Received request for camera: %s", request.camera_url)

    try:
        count, detections = detector.count_people(
            camera_url=request.camera_url,
            confidence=request.confidence,
        )
    except ConnectionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error processing camera %s", request.camera_url)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PeopleCountResponse(
        camera_url=request.camera_url,
        people_count=count,
        confidence_threshold=request.confidence,
        detections=detections,
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
