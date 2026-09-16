from contextlib import asynccontextmanager
import asyncio
from pathlib import Path
import shutil
import subprocess
import tempfile
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from .config import FRONTEND_ORIGIN, MAX_WINDOWS, SAMPLE_RATE, TARGET_SAMPLES
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
    return HealthResponse(status="ok", model_loaded=service.loaded, detector=service.detector, mode=service.mode, device=service.device_name, ffmpeg_available=ffmpeg_available())

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
        return AnalysisResponse(status="suspicious" if aggregate >= 60 else "human" if aggregate < 35 else "uncertain", synthetic_score=aggregate, risk_score=risk, risk_level=level, windows_analyzed=windows, audio_quality=quality, mode=service.mode, detector=service.detector, model_message="AASIST score converted from bona-fide-vs-spoof logits; not a calibrated probability.", timeline=timeline)
    except Exception as error:
        raise HTTPException(422, f"Audio analysis failed: {error}") from error
    finally:
        if temporary_path:
            Path(temporary_path).unlink(missing_ok=True)

@app.websocket("/ws/analyze")
async def analyze_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    if service is None or not service.loaded:
        await websocket.close(code=1011, reason="AASIST detector is unavailable")
        return
    audio_buffer = bytearray()
    analyzed_windows = 0
    scores: list[float] = []
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                return
            if message.get("text") == "stop":
                return
            chunk = message.get("bytes")
            if not chunk:
                continue
            audio_buffer.extend(chunk)
            if len(audio_buffer) < 16_000 or analyzed_windows >= MAX_WINDOWS:
                continue
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temporary:
                temporary.write(audio_buffer)
                temporary_path = temporary.name
            try:
                timeline, quality = await asyncio.to_thread(service.predict_file, temporary_path)
            finally:
                Path(temporary_path).unlink(missing_ok=True)
            new_scores = timeline[analyzed_windows:]
            for score in new_scores:
                scores.append(score)
                analyzed_windows += 1
                aggregate, stability, windows = aggregate_scores(scores[-MAX_WINDOWS:])
                risk, level = calculate_risk(aggregate, windows, stability, quality)
                await websocket.send_json({"window_id": analyzed_windows, "synthetic_score": aggregate, "risk_score": risk, "risk_level": level, "status": "suspicious" if aggregate >= 60 else "human" if aggregate < 35 else "uncertain", "stability": stability, "audio_quality": quality, "windows_analyzed": windows, "mode": service.mode, "detector": service.detector, "timeline": scores[-MAX_WINDOWS:]})
    except WebSocketDisconnect:
        return
