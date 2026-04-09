"""
Face Match API
==============
Takes a reference image (photo of one person) and a video.
Returns whether that person appears in any frame of the video.

Uses InsightFace (SCRFD detection + ArcFace embeddings) for SOTA face matching.
"""

import io
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
face_app: FaceAnalysis | None = None

SIMILARITY_THRESHOLD = 0.45  # conservative default; tune per use-case
MAX_VIDEO_FRAMES = 5000      # safety cap


# ---------------------------------------------------------------------------
# Lifespan – load model once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global face_app
    print("⏳ Loading InsightFace model (buffalo_l)…")
    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    print("✅ Model loaded.")
    yield
    face_app = None


app = FastAPI(
    title="Face Match API",
    version="1.0.0",
    description="Checks whether the person in a reference photo appears in a video.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (the test UI)
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_image_from_bytes(data: bytes) -> np.ndarray:
    """Decode image bytes to a BGR numpy array (OpenCV format)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def get_reference_embedding(img: np.ndarray) -> np.ndarray:
    """Extract the single face embedding from a reference photo."""
    faces = face_app.get(img)
    if len(faces) == 0:
        raise ValueError("No face detected in the reference image")
    if len(faces) > 1:
        # Pick the largest face (by bounding-box area)
        faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    return faces[0].embedding


def match_face_in_video(
    video_path: str,
    ref_embedding: np.ndarray,
    threshold: float,
    sample_every_n: int = 1,
) -> dict:
    """
    Scan a video frame-by-frame and check if the reference face appears.

    Returns a dict with match result, best similarity, and diagnostics.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    best_similarity = -1.0
    best_frame_idx = -1
    best_timestamp = 0.0
    frames_processed = 0
    faces_detected_total = 0
    matched = False

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx >= MAX_VIDEO_FRAMES:
            break

        # Optionally skip frames for performance
        if frame_idx % sample_every_n != 0:
            frame_idx += 1
            continue

        faces = face_app.get(frame)
        faces_detected_total += len(faces)
        frames_processed += 1

        for face in faces:
            sim = float(cosine_similarity(
                ref_embedding.reshape(1, -1),
                face.embedding.reshape(1, -1),
            )[0][0])

            if sim > best_similarity:
                best_similarity = sim
                best_frame_idx = frame_idx
                best_timestamp = frame_idx / fps if fps > 0 else 0

            if sim >= threshold:
                matched = True

        frame_idx += 1

    cap.release()

    return {
        "match": matched,
        "best_similarity": round(best_similarity, 4),
        "threshold_used": threshold,
        "best_frame_index": best_frame_idx,
        "best_timestamp_sec": round(best_timestamp, 2),
        "video_fps": round(fps, 2),
        "video_duration_sec": round(duration_sec, 2),
        "total_frames": total_frames,
        "frames_processed": frames_processed,
        "faces_detected_total": faces_detected_total,
        "sample_every_n_frames": sample_every_n,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the test UI."""
    ui_path = static_dir / "index.html"
    if ui_path.exists():
        return HTMLResponse(ui_path.read_text())
    return HTMLResponse("<h1>Face Match API</h1><p>Upload UI not found. Place index.html in /static.</p>")


@app.post("/api/match")
async def match_face(
    image: UploadFile = File(..., description="Reference photo (JPEG/PNG) of the person to find"),
    video: UploadFile = File(..., description="Video file (MP4/AVI/MOV) to search"),
    threshold: float = Query(SIMILARITY_THRESHOLD, ge=0.0, le=1.0, description="Cosine similarity threshold"),
    sample_every_n: int = Query(1, ge=1, le=100, description="Process every Nth frame (1 = all frames)"),
):
    """
    Upload a reference image and a video.
    Returns whether the person in the image appears in the video.
    """
    t0 = time.time()

    # ---- Read reference image ----
    image_bytes = await image.read()
    try:
        img = read_image_from_bytes(image_bytes)
    except ValueError:
        raise HTTPException(status_code=400, detail="Could not decode reference image. Send a valid JPEG/PNG.")

    try:
        ref_embedding = get_reference_embedding(img)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # ---- Save video to temp file (OpenCV needs a file path) ----
    video_bytes = await video.read()
    suffix = Path(video.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    # ---- Run matching ----
    try:
        result = match_face_in_video(tmp_path, ref_embedding, threshold, sample_every_n)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    elapsed = round(time.time() - t0, 2)
    result["processing_time_sec"] = elapsed

    return JSONResponse(content=result)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": face_app is not None}
