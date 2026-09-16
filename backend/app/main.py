from contextlib import asynccontextmanager
import asyncio
from pathlib import Path
import shutil
import tempfile
import time
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from .config import FRONTEND_ORIGIN, MAX_WINDOWS
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

    # MediaRecorder sends approximately one WebM chunk per second. Decode only
    # every two chunks and run AASIST only for genuinely new complete windows.
    # This avoids re-running the model over the entire growing recording.
    audio_buffer = bytearray()
    received_chunks = 0
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
            received_chunks += 1

            # The frontend uses MediaRecorder(...).start(1000), so checking
            # every second chunk gives us roughly a two-second cadence. This
            # is only a decode check; AASIST does not run until a full window
            # is actually available.
            if received_chunks % 2 != 0 or analyzed_windows >= MAX_WINDOWS:
                continue

            temporary_path: str | None = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temporary:
                    temporary.write(audio_buffer)
                    temporary_path = temporary.name

                started = time.perf_counter()
                new_scores, quality, available_windows = await asyncio.to_thread(
                    service.predict_new_file_windows,
                    temporary_path,
                    analyzed_windows,
                )
                elapsed = time.perf_counter() - started

                if new_scores:
                    for score in new_scores:
                        scores.append(score)
                        analyzed_windows += 1
                        aggregate, stability, windows = aggregate_scores(scores[-MAX_WINDOWS:])
                        risk, level = calculate_risk(aggregate, windows, stability, quality)
                        await websocket.send_json({
                            "window_id": analyzed_windows,
                            "synthetic_score": aggregate,
                            "risk_score": risk,
                            "risk_level": level,
                            "status": "suspicious" if aggregate >= 60 else "human" if aggregate < 35 else "uncertain",
                            "stability": stability,
                            "audio_quality": quality,
                            "windows_analyzed": windows,
                            "mode": service.mode,
                            "detector": service.detector,
                            "timeline": scores[-MAX_WINDOWS:],
                        })
                    print(f"VOXY live: analyzed {len(new_scores)} new window(s) in {elapsed:.2f}s; total={analyzed_windows}")
                elif available_windows == 0:
                    print("VOXY live: waiting for first complete 4-second window")
            except Exception as error:
                print(f"VOXY live analysis error: {error}")
                await websocket.close(code=1011, reason="Live AASIST analysis failed")
                return
            finally:
                if temporary_path:
                    Path(temporary_path).unlink(missing_ok=True)
    except WebSocketDisconnect:
        return
