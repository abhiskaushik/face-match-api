# Face Match API

A FastAPI service that checks whether the person in a reference photo appears in a video. Uses **InsightFace** (SCRFD detection + ArcFace embeddings) for state-of-the-art face recognition.

## Features

- **SCRFD** face detection (fast, accurate)
- **ArcFace** 512-dim embeddings for recognition
- Cosine similarity matching with configurable threshold
- Frame sampling control (process every Nth frame)
- Handles multiple faces per video frame
- Dark-themed drag-and-drop test UI
- Docker support for easy deployment

## Quick Start

### Local (Python)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The first run downloads the `buffalo_l` model (~326 MB). Open http://localhost:8000 for the test UI.

### Docker

```bash
docker build -t face-match-api .
docker run -p 8000:8000 face-match-api
```

### GPU Support

For NVIDIA GPU acceleration, install `onnxruntime-gpu` instead of `onnxruntime`:

```bash
pip install onnxruntime-gpu
```

## API

### `POST /api/match`

Upload a reference image and a video. Returns whether the person appears in the video.

**Form fields:**
| Field   | Type | Description                          |
|---------|------|--------------------------------------|
| `image` | file | Reference photo (JPEG/PNG) — one face |
| `video` | file | Video to search (MP4/AVI/MOV/WebM)   |

**Query params:**
| Param            | Type  | Default | Description                         |
|------------------|-------|---------|-------------------------------------|
| `threshold`      | float | 0.45    | Cosine similarity threshold (0–1)   |
| `sample_every_n` | int   | 1       | Process every Nth frame (1 = all)   |

**Response:**
```json
{
  "match": true,
  "best_similarity": 0.6823,
  "threshold_used": 0.45,
  "best_frame_index": 142,
  "best_timestamp_sec": 4.73,
  "video_fps": 30.0,
  "video_duration_sec": 12.5,
  "total_frames": 375,
  "frames_processed": 375,
  "faces_detected_total": 1024,
  "sample_every_n_frames": 1,
  "processing_time_sec": 8.42
}
```

### `GET /health`

Returns `{"status": "ok", "model_loaded": true}`.

## Configuration Tips

- **Threshold**: 0.45 is conservative. Raise to 0.5–0.6 if you get false positives; lower to 0.3–0.4 if missing real matches.
- **Frame sampling**: Set `sample_every_n=3` for ~3x speedup on long videos with minimal accuracy loss.
- **Video length**: For very long videos (>5 min), consider sampling every 2–5 frames.

## Project Structure

```
face-match-api/
├── app.py              # FastAPI application
├── static/
│   └── index.html      # Test UI
├── requirements.txt
├── Dockerfile
└── README.md
```
