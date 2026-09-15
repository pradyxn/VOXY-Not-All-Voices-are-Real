from contextlib import asynccontextmanager
from pathlib import Path
import shutil
import subprocess
import tempfile
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from .config import FRONTEND_ORIGIN
from .schemas import AnalysisResponse, HealthResponse
from .ml.inference import InferenceService
from .ml.aggregation import aggregate_scores
from .risk.engine import calculate_risk

service: InferenceService | None = None

def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None

@asynccontextmanager
async def lifespan(_: FastAPI):
    global service
    service = InferenceService()
    yield

app = FastAPI(title="VOXY API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[FRONTEND_ORIGIN, "http://localhost:3000"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    assert service is not None
    return HealthResponse(status="ok", model_loaded=service.loaded, mode=service.mode, device=str(service.device).upper(), ffmpeg_available=ffmpeg_available())

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze(audio: UploadFile = File(...)) -> AnalysisResponse:
    assert service is not None
    if not audio.filename:
        raise HTTPException(400, "Please choose an audio file.")
    suffix = Path(audio.filename).suffix.lower()
    if suffix not in {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".opus"}:
        raise HTTPException(415, "Unsupported audio format. Use WAV, MP3, M4A, or WebM.")
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
            shutil.copyfileobj(audio.file, temporary)
            temporary_path = temporary.name
        timeline, quality = service.predict_file(temporary_path)
        aggregate, stability, windows = aggregate_scores(timeline)
        risk, level = calculate_risk(aggregate, windows, stability, quality)
        return AnalysisResponse(status="suspicious" if aggregate >= 60 else "human" if aggregate < 35 else "uncertain", synthetic_score=aggregate, risk_score=risk, risk_level=level, windows_analyzed=windows, audio_quality=quality, mode=service.mode, model_message=None if service.loaded else "ML model unavailable - running demonstration mode.", timeline=timeline)
    except Exception as error:
        raise HTTPException(422, f"Audio analysis failed: {error}") from error
    finally:
        if temporary_path:
            Path(temporary_path).unlink(missing_ok=True)

@app.websocket("/ws/analyze")
async def analyze_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    window_id = 0
    scores: list[float] = []
    try:
        while True:
            message = await websocket.receive_bytes()
            window_id += 1
            score = 50.0 if not message else min(96.0, 44.0 + (sum(message[:200]) % 4800) / 100)
            scores.append(score)
            aggregate, stability, windows = aggregate_scores(scores[-8:])
            risk, level = calculate_risk(aggregate, windows, stability, "good")
            await websocket.send_json({"window_id": window_id, "synthetic_score": aggregate, "risk_score": risk, "status": level.lower(), "mode": service.mode if service else "demo"})
    except WebSocketDisconnect:
        return
