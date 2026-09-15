# VOXY

## Not All Voices Are Real.

VOXY is a Smart India Hackathon 2026 prototype for detecting patterns associated with AI-generated or cloned speech. It produces a synthetic-likelihood model score, aggregates multiple analysis windows, converts the evidence into a risk level, and recommends verification when the signal is elevated.

## Architecture

- `frontend/`: Next.js, TypeScript, Framer Motion-ready UI, Recharts timeline, responsive VOXY workspace.
- `backend/`: FastAPI REST and WebSocket service, librosa preprocessing, PyTorch CNN, aggregation, and risk engine.
- `models/model_final.pth`: optional compatible model file. It is not included in this empty scaffold.

Pipeline: audio -> 16 kHz mono -> 4-second windows -> 128-band Mel spectrogram -> CNN -> synthetic class score -> temporal aggregation -> risk engine.

## Installation

Install Python, Node.js, and **ffmpeg on PATH**. Browser MediaRecorder commonly produces WebM/Opus, which requires ffmpeg for server-side decoding.

```bat
cd /d "C:\Users\prady\Documents\Omniroute\Projects\Voxy-Not All Voices are Real"
call "backend\setup.bat"
cd /d "frontend"
npm install
```

The setup script stops with a clear message if ffmpeg is missing.

## Running locally

From the quoted project directory, run:

```bat
start_voxy.bat
```

Or run each service manually:

```bat
cd /d "C:\Users\prady\Documents\Omniroute\Projects\Voxy-Not All Voices are Real\backend"
call ".venv\Scripts\activate.bat"
uvicorn app.main:app --reload --port 8000
```

```bat
cd /d "C:\Users\prady\Documents\Omniroute\Projects\Voxy-Not All Voices are Real\frontend"
npm run dev
```

Open `http://localhost:3000`. Frontend API configuration lives in `frontend/.env.local` as `NEXT_PUBLIC_API_URL`.

## Model and demo mode

The CNN architecture is `1 -> 16 -> 32 -> 64 -> 128 -> Linear(128, 2)` with adaptive average pooling and dropout. The backend loads `models/model_final.pth` once and fails clearly if a present checkpoint is incompatible. When the file is absent, the application remains usable in labelled **DEMO MODE**; deterministic simulation values are never presented as genuine predictions. Class order must be verified against the training metadata before using a supplied checkpoint.

Training dataset names and accuracy are intentionally not claimed because none are supplied in this workspace.

## Risk engine

Prototype thresholds: LOW 0-29, MODERATE 30-59, HIGH 60-85, CRITICAL 86-100. Aggregation uses a configurable EMA alpha of 0.35 and rolling evidence. These are prototype parameters, not scientifically optimal values.

## Realtime limitation

The WebSocket and call screen support microphone analysis and a clearly labelled **Realtime Call Simulation** only. A normal browser cannot intercept arbitrary cellular call audio. Future Native Call Integration may feed permitted native audio into the same streaming pipeline.

## Testing

```bat
cd /d "C:\Users\prady\Documents\Omniroute\Projects\Voxy-Not All Voices are Real"
python scripts\test_model.py
python scripts\test_audio.py
```

`test_model.py` verifies the CNN tensor shape and two-class output. Add a compatible model and sample audio to extend it to real checkpoint/audio validation.

## Privacy and limitations

Audio is processed for analysis and is not intentionally retained by the prototype. VOXY does not identify callers, guarantee authenticity, or replace trusted verification. Poor, short, silent, or clipped audio can be unreliable and should result in an insufficient-quality state in production.

## Future scope

MFCC, spectral, and prosodic feature branches; stronger calibration; native mobile audio integration where OS permissions allow; and a richer verification workflow can be added without merging the model, signal processing, aggregation, risk, and UI layers.
