from contextlib import asynccontextmanager
import asyncio
from io import BytesIO
from pathlib import Path
import shutil
import tempfile
import wave
from time import perf_counter

import numpy as np
from fastapi import Body, FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import FRONTEND_ORIGIN, MAX_WINDOWS
from .schemas import AnalysisResponse, HealthResponse
from .ml.inference import InferenceService
from .ml.aggregation import aggregate_scores
from .ml.preprocessing import quality_check
from .risk.engine import calculate_risk

service: InferenceService | None = None


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global service
    service = InferenceService()
    print(
        f"VOXY startup: detector_loaded={service.loaded} "
        f"detector={service.detector} device={service.device_name}"
    )
    yield


app = FastAPI(title="VOXY API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    # Public hackathon demo: no cookies/authentication are used.
    allow_origins=["*"],
    allow_credentials=False,
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

        started = perf_counter()
        timeline, quality = await asyncio.to_thread(service.predict_file, temporary_path)
        elapsed_ms = round((perf_counter() - started) * 1000.0)
        print(f"VOXY upload: AASIST analysis completed in {elapsed_ms} ms")
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
            model_message=f"AASIST spoof probability derived from the official 2-class output; higher means more synthetic/spoof signal. Inference: {elapsed_ms} ms.",
            timeline=timeline,
        )
    except Exception as error:
        raise HTTPException(422, f"Audio analysis failed: {error}") from error
    finally:
        if temporary_path:
            Path(temporary_path).unlink(missing_ok=True)


def decode_live_wav(payload: bytes) -> np.ndarray:
    """Decode a browser-generated fixed 16 kHz mono PCM snapshot."""
    with wave.open(BytesIO(payload), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if channels != 1:
        raise ValueError(f"Expected mono live audio, received {channels} channels")
    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM live audio, received {sample_width * 8}-bit")
    if sample_rate != 16000:
        raise ValueError(f"Expected 16000 Hz live audio, received {sample_rate} Hz")

    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if audio.size == 0:
        raise ValueError("Live audio snapshot was empty")
    return audio


@app.post("/api/live")
async def analyze_live_snapshot(audio: bytes = Body(...)) -> dict:
    """Run one fixed live microphone window over ordinary HTTPS."""
    assert service is not None
    if not service.loaded:
        raise HTTPException(503, "AASIST detector is unavailable")
    if not audio:
        raise HTTPException(400, "Live audio snapshot was empty.")

    try:
        waveform = decode_live_wav(audio)
        quality = quality_check(waveform)
        if quality == "insufficient":
            raise HTTPException(422, "Live audio contains insufficient speech signal.")

        started = perf_counter()
        score = await asyncio.to_thread(service.predict_window, waveform)
        elapsed_ms = round((perf_counter() - started) * 1000.0)
        value = round(float(score), 1)
        risk, level = calculate_risk(value, 1, 100.0, quality)
        print(f"VOXY live HTTP: AASIST inference completed in {elapsed_ms} ms score={value:.1f}")

        return {
            "window_id": 1,
            "synthetic_score": value,
            "risk_score": risk,
            "risk_level": level,
            "status": "suspicious" if value >= 60 else "human" if value < 35 else "uncertain",
            "stability": 100.0,
            "audio_quality": quality,
            "windows_analyzed": 1,
            "mode": service.mode,
            "detector": service.detector,
            "timeline": [value],
            "inference_ms": elapsed_ms,
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(422, f"Live analysis failed: {error}") from error


@app.websocket("/ws/analyze")
async def analyze_stream(websocket: WebSocket) -> None:
    """Analyze complete fixed 16 kHz mono microphone snapshots with AASIST."""
    await websocket.accept()

    if service is None or not service.loaded:
        await websocket.send_json({"error": "AASIST detector is unavailable"})
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

            try:
                audio = decode_live_wav(chunk)
                quality = quality_check(audio)
                if quality == "insufficient":
                    print("VOXY live: snapshot contained insufficient usable audio")
                    continue

                started = perf_counter()
                score = await asyncio.to_thread(service.predict_window, audio)
                elapsed_ms = round((perf_counter() - started) * 1000.0)
                print(f"VOXY live: AASIST inference completed in {elapsed_ms} ms")

                scores.append(float(score))
                scores = scores[-MAX_WINDOWS:]
                window_id += 1

                _aggregate, stability, windows = aggregate_scores(scores)
                risk, level = calculate_risk(float(score), windows, stability, quality)

                payload = {
                    "window_id": window_id,
                    "synthetic_score": round(float(score), 1),
                    "risk_score": risk,
                    "risk_level": level,
                    "status": (
                        "suspicious"
                        if score >= 60
                        else "human"
                        if score < 35
                        else "uncertain"
                    ),
                    "stability": stability,
                    "audio_quality": quality,
                    "windows_analyzed": windows,
                    "mode": service.mode,
                    "detector": service.detector,
                    "timeline": scores,
                    "inference_ms": elapsed_ms,
                }

                await websocket.send_json(payload)
                print(
                    f"VOXY live: window={window_id} "
                    f"score={score:.1f} risk={risk} level={level}"
                )

            except Exception as error:
                print(f"VOXY live snapshot error: {error}")
                await websocket.send_json({"error": f"Live analysis failed: {error}"})
                continue

    except WebSocketDisconnect:
        return
