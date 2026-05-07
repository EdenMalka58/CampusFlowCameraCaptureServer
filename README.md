# CampusFlow Camera Capture Server

A FastAPI microservice that connects to an IP camera, captures a single frame, and returns the number of people detected using **YOLOv8-nano** (Ultralytics).

---

## Endpoints

### `GET /health`
Returns model load status.

```json
{ "status": "ok", "model_loaded": true }
```

---

### `POST /count-people`

**Request body**

| Field | Type | Default | Description |
|---|---|---|---|
| `camera_url` | string | required | RTSP / HTTP stream URL |
| `confidence` | float | `0.4` | Minimum detection confidence (0–1) |

**Example request**

```bash
curl -X POST http://localhost:8000/count-people \
  -H "Content-Type: application/json" \
  -d '{"camera_url": "rtsp://admin:pass@192.168.1.100:554/stream"}'
```

**Example response**

```json
{
  "camera_url": "rtsp://admin:pass@192.168.1.100:554/stream",
  "people_count": 3,
  "confidence_threshold": 0.4,
  "detections": [
    { "bbox": { "x1": 120, "y1": 45, "x2": 210, "y2": 380 }, "confidence": 0.91 },
    { "bbox": { "x1": 300, "y1": 60, "x2": 390, "y2": 400 }, "confidence": 0.87 },
    { "bbox": { "x1": 510, "y1": 80, "x2": 600, "y2": 420 }, "confidence": 0.76 }
  ]
}
```

Interactive docs are available at `/docs` (Swagger UI) and `/redoc`.

---

## Local development

### Prerequisites

- Python 3.10+
- A reachable IP camera (RTSP or HTTP stream)

### Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the server
python main.py
```

The server listens on `http://localhost:8000` by default.

On first startup, Ultralytics automatically downloads the `yolov8n.pt` weights (~6 MB) and caches them locally.

---

## Deployment on Render.com

The repository ships with a `render.yaml` Blueprint so deployment is a single click.

### Steps

1. Push this folder (or the whole monorepo) to a GitHub / GitLab repository.
2. Go to **[Render Dashboard](https://dashboard.render.com/)** → **New** → **Blueprint**.
3. Connect your repository. Render detects `render.yaml` automatically.
4. Review the service settings and click **Apply**.

### Key deployment notes

| Topic | Detail |
|---|---|
| **Plan** | Use the **Standard** plan (2 GB RAM). The free tier (512 MB) is not enough for PyTorch + YOLOv8. |
| **Build time** | The first build takes ~5–8 minutes while pip installs PyTorch. Subsequent builds are faster thanks to Render's build cache. |
| **Model weights** | A 1 GB persistent disk is mounted at `/opt/render/project/src/.ultralytics`. The weights are downloaded once and reused across restarts. |
| **Camera access** | Your Render instance must be able to reach the camera URL. For cameras behind a private LAN, expose the stream through a VPN or a reverse proxy with a public URL. |
| **Region** | Change the `region` field in `render.yaml` to the region closest to your cameras to reduce latency. |

### Environment variables set by render.yaml

| Variable | Value | Purpose |
|---|---|---|
| `MPLBACKEND` | `Agg` | Prevents matplotlib from trying to open a display |
| `YOLO_CONFIG_DIR` | `/opt/render/project/src/.ultralytics` | Persistent model weight cache |

---

## Project structure

```
CampusFlowCameraCaptureServer/
├── main.py          # FastAPI app, lifespan, endpoints
├── detector.py      # PeopleDetector — YOLO inference + frame capture
├── requirements.txt
├── render.yaml      # Render.com Blueprint
└── README.md
```

---

## How it works

```
POST /count-people
        │
        ▼
  PeopleDetector.count_people(camera_url)
        │
        ├─► cv2.VideoCapture(camera_url)   # open RTSP / HTTP stream
        ├─► cap.read()                     # grab one frame
        └─► YOLO(frame, conf=threshold)    # run inference
                │
                └─► filter class == "person"
                        │
                        └─► return count + bounding boxes
```
