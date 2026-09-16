from contextlib import asynccontextmanager
import asyncio
from pathlib import Path
import shutil
import tempfile

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


app = FastAPI(title="VOXY API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    assert service is not None
    return HealthResponse(
        status="ok",
        model_loaded=service.loaded,
        detector=service.detector,
        mode=service.mode,
        device=service.device_name,
        ffmpeg_available=ffmpeg_available(),
    )


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

        timeline, quality = await asyncio.to_thread(service.predict_file, temporary_path)
        aggregate, stability, windows = aggregate_scores(timeline)
        risk, level = calculate_risk(aggregate, windows, stability, quality)

        return AnalysisResponse(
            status="suspicious" if aggregate >= 60 else "human" if aggregate < 35 else "uncertain",
            synthetic_score=aggregate,
            risk_score=risk,
            risk_level=level,
            windows_analyzed=windows,
            audio_quality=quality,
            mode=service.mode,
            detector=service.detector,
            model_message="AASIST score converted from bona-fide-vs-spoof logits; not a calibrated probability.",
            timeline=timeline,
        )
    except Exception as error:
        raise HTTPException(422, f"Audio analysis failed: {error}") from error
    finally:
        if temporary_path:
            Path(temporary_path).unlink(missing_ok=True)


@app.websocket("/ws/analyze")
async def analyze_stream(websocket: WebSocket) -> None:
    """
    Live microphone protocol.

    The frontend sends one complete, independently decodable WebM/Opus blob
    approximately every four seconds. Each blob is decoded exactly once and
    produces one AASIST score. Keeping segment boundaries explicit avoids trying
    to decode incomplete MediaRecorder WebM fragments.
    """
    await websocket.accept()

    if service is None or not service.loaded:
        await websocket.close(code=1011, reason="AASIST detector is unavailable")
        return

    scores: list[float] = []
    window_id = 0

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

            temporary_path: str | None = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temporary:
                    temporary.write(chunk)
                    temporary_path = temporary.name

                # Inference is CPU-bound on Render, so keep it off the
                # asyncio event loop while the WebSocket remains responsive.
                timeline, quality = await asyncio.to_thread(
                    service.predict_file,
                    temporary_path,
                )

                if not timeline:
                    print("VOXY live: segment contained insufficient usable audio")
                    continue

                # A four-second segment should normally yield exactly one
                # complete model window. If decoding produces more, only use
                # the first score so each live segment corresponds to one point
                # on the real-time graph.
                score = float(timeline[0])
                scores.append(score)
                scores = scores[-MAX_WINDOWS:]
                window_id += 1

                aggregate, stability, windows = aggregate_scores(scores)
                risk, level = calculate_risk(aggregate, windows, stability, quality)

                payload = {
                    "window_id": window_id,
                    "synthetic_score": aggregate,
                    "risk_score": risk,
                    "risk_level": level,
                    "status": (
                        "suspicious"
                        if aggregate >= 60
                        else "human"
                        if aggregate < 35
                        else "uncertain"
                    ),
                    "stability": stability,
                    "audio_quality": quality,
                    "windows_analyzed": windows,
                    "mode": service.mode,
                    "detector": service.detector,
                    "timeline": scores,
                }

                await websocket.send_json(payload)
                print(
                    f"VOXY live: window={window_id} "
                    f"score={score:.1f} aggregate={aggregate:.1f} "
                    f"risk={risk} level={level}"
                )

            except Exception as error:
                # A bad/empty browser segment must not tear down the whole
                # live connection. Log it and wait for the next segment.
                print(f"VOXY live segment error: {error}")
                continue
            finally:
                if temporary_path:
                    Path(temporary_path).unlink(missing_ok=True)

    except WebSocketDisconnect:
        return
