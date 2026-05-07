import logging
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# YOLO class index for "person" in the COCO dataset
PERSON_CLASS_ID = 0

# How long (seconds) to try reading frames before giving up
FRAME_TIMEOUT = 10

# Path to cache the downloaded model weights alongside this file
MODEL_WEIGHTS = Path(__file__).parent / "yolov8n.pt"


class PeopleDetector:
    """Wraps a YOLOv8-nano model and provides camera-frame people counting."""

    def __init__(self, weights: str | Path = MODEL_WEIGHTS) -> None:
        # Ultralytics downloads the weights automatically on first use.
        self.model = YOLO(str(weights))
        # Warm-up pass so the first real request isn't slow.
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)
        logger.info("PeopleDetector ready (weights=%s)", weights)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def count_people(
        self,
        camera_url: str,
        confidence: float = 0.4,
    ) -> tuple[int, list[dict]]:
        """
        Connect to *camera_url*, grab one frame, run inference, and return
        the number of people detected together with their bounding boxes.

        Returns
        -------
        count : int
            Number of people found in the frame.
        detections : list[dict]
            Each dict contains ``bbox`` (x1, y1, x2, y2) and ``confidence``.

        Raises
        ------
        ConnectionError
            When the camera stream cannot be opened or no frame is received
            within ``FRAME_TIMEOUT`` seconds.
        """
        frame = self._grab_frame(camera_url)
        return self._detect(frame, confidence)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _grab_frame(self, camera_url: str) -> np.ndarray:
        # Allow numeric strings ("0", "1", …) to address local webcam devices.
        source: str | int = int(camera_url) if camera_url.strip().lstrip("-").isdigit() else camera_url
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise ConnectionError(
                f"Could not open camera stream: {source!r}. "
                "Check the URL, credentials, and network connectivity."
            )

        deadline = time.monotonic() + FRAME_TIMEOUT
        frame = None

        try:
            while time.monotonic() < deadline:
                ret, f = cap.read()
                if ret and f is not None:
                    frame = f
                    break
                time.sleep(0.1)
        finally:
            cap.release()

        if frame is None:
            raise ConnectionError(
                f"No frame received from {source!r} within {FRAME_TIMEOUT}s. "
                "The stream may be offline or unreachable."
            )

        logger.info("Frame captured from %s (%dx%d)", source, frame.shape[1], frame.shape[0])
        return frame

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect(
        self, frame: np.ndarray, confidence: float
    ) -> tuple[int, list[dict]]:
        results = self.model(frame, conf=confidence, verbose=False)

        detections: list[dict] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls = int(box.cls[0])
                if cls != PERSON_CLASS_ID:
                    continue
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                conf = float(box.conf[0])
                detections.append(
                    {
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        "confidence": round(conf, 4),
                    }
                )

        logger.info("Detected %d person(s) in frame.", len(detections))
        return len(detections), detections
