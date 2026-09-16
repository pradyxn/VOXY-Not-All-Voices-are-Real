# VOXY

## Not All Voices Are Real.

VOXY is an AI voice-security prototype for analyzing acoustic signals associated with synthetic or cloned speech. It accepts an audio recording, normalizes it to 16 kHz mono, evaluates overlapping raw-waveform windows with the pretrained AASIST anti-spoofing model, aggregates evidence across windows, and presents a 0-100 risk signal with practical verification guidance.

VOXY is decision support, not an infallible detector. It does not guarantee that a voice is real or fake, identify the caller, or replace independent verification.

## Project Status

The repository currently contains:

- A working Next.js frontend with the VOXY product experience and analysis workspace.
- A FastAPI backend with REST and WebSocket endpoints.
- Shared preprocessing, model inference, temporal aggregation, and risk-engine layers.
- A pretrained AASIST ONNX detector downloaded by `setup_model.py`.
- Real upload and browser microphone WebSocket inference paths.
- A metadata file at `models/model_metadata.json` describing score direction and preprocessing.

`models/model_final.pth` is retained only as an archived smoke-test artifact and is no longer loaded by production inference. It must not be treated as a production-quality accuracy result.

## Repository Layout

```text
.
|-- backend/
|   |-- app/
|   |   |-- main.py                 FastAPI app and API/WebSocket routes
|   |   |-- config.py               Shared sample/window/model configuration
|   |   |-- schemas.py              API response models
|   |   |-- ml/
|   |   |   |-- model.py             Archived smoke-test architecture (unused)
|   |   |   |-- preprocessing.py     Raw waveform/audio decoding pipeline
|   |   |   |-- inference.py         File inference service
|   |   |   |-- aggregation.py       Rolling score aggregation
|   |   |-- risk/
|   |       |-- engine.py            Risk-level calculation
|   |-- requirements.txt
|   |-- setup.bat                    Creates the backend venv and starts Uvicorn
|-- data/
|   |-- LA/LA/                       ASVspoof 2019 LA dataset layout
|-- frontend/
|   |-- app/
|   |   |-- page.tsx                 VOXY homepage and analysis workspace
|   |   |-- globals.css              Global visual system and responsive styles
|   |   |-- layout.tsx               Next.js metadata and root layout
|   |-- package.json
|   |-- .env.local                   Frontend API URL
|-- models/
|   |-- AASIST.onnx                 Downloaded pretrained AASIST artifact (not committed)
|   |-- model_final.pth              Archived smoke-test checkpoint, unused in production
|   |-- model_metadata.json          AASIST checkpoint and score metadata
|-- scripts/
|   |-- train.py                     ASVspoof LA training and smoke-test script
|   |-- test_model.py                Model tensor/output smoke check
|   |-- test_audio.py                Risk and audio-quality checks
|-- sample_audio/                    Optional local audio fixtures
|-- start_voxy.bat                   Starts backend, frontend, and browser
|-- README.md
```

## Architecture

VOXY is organized into a shared analysis path:

```text
Audio file or microphone bytes
        |
        v
Temporary server-side audio file
        |
        v
16 kHz mono conversion
        |
        v
4-second / 64,000-sample windows
        |
        v
Raw float32 waveform, padded/truncated to 64,600 samples
        |
        v
AASIST pretrained anti-spoofing inference
        |
        v
Window scores -> rolling/EMA aggregation
        |
        v
0-100 synthetic-likelihood signal
        |
        v
Risk score, risk level, timeline, and verification guidance
```

The production inference service uses the self-contained AASIST ONNX artifact. The legacy VoiceCNN training path remains only as historical smoke-test code and is not used by the API.

## Frontend

The frontend is a Next.js App Router application using React and TypeScript. The homepage includes:

- VOXY editorial hero with an animated CSS acoustic/Saturn visual.
- Responsive navigation with Services, About, Insights, Contact, Login, and Get Started actions.
- Dark VOXY detection-story section with four feature cards.
- Interactive detection pipeline steps.
- Existing audio analysis workspace.
- Human, Synthetic, and Uncertain labelled demo states.
- Risk result card with score, level, windows, timeline, and verification guidance.
- Minimal Contact modal with a `mailto:` link.

The current visual system is implemented with plain global CSS in `frontend/app/globals.css`. No new frontend dependency is required for the acoustic visual or the motion system. The project has Framer Motion listed in `package.json`, but the current page uses CSS animations and `IntersectionObserver` rather than importing Framer Motion.

### Frontend configuration

`frontend/.env.local` contains:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

Change this value when the backend is hosted at a different origin. The backend CORS configuration must allow the matching frontend origin.

## Backend API

The backend is a FastAPI application served by Uvicorn.

### Health check

```http
GET /api/health
```

Example response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "mode": "ml",
  "device": "CUDA",
  "ffmpeg_available": true
}
```

### File analysis

```http
POST /api/analyze
Content-Type: multipart/form-data

audio=<supported audio file>
```

Supported formats:

- `.wav`
- `.mp3`
- `.m4a`
- `.webm`
- `.ogg`
- `.opus`

The response contains:

```json
{
  "status": "human | uncertain | suspicious",
  "synthetic_score": 0,
  "risk_score": 0,
  "risk_level": "LOW | MODERATE | HIGH | CRITICAL",
  "windows_analyzed": 0,
  "audio_quality": "good | clipped | insufficient",
        "mode": "pretrained | demo",
        "detector": "AASIST pretrained anti-spoofing model",
  "model_message": null,
  "timeline": []
}
```

Uploaded files are written to a temporary path for analysis and removed in the backend `finally` block after processing.

### Streaming analysis

```text
WebSocket /ws/analyze
```

The browser microphone simulation sends WebM/Opus chunks over the WebSocket. The backend buffers and decodes them, runs AASIST after enough audio is available, and returns incremental real model scores. A normal browser cannot intercept arbitrary cellular-call audio.

## Backend and ML Configuration

The shared configuration in `backend/app/config.py` defines:

| Setting | Value | Purpose |
|---|---:|---|
| Sample rate | 16,000 Hz | Target audio rate |
| Target samples | 64,600 | AASIST model input |
| Window hop | 32,300 | 50% overlap |
| Maximum windows | 8 | Upload/live workload bound |
| Aggregation | Median | Robust temporal score |

The AASIST ONNX model returns two logits:

```text
bonafide -> logit 0
spoof    -> logit 1

The published AASIST convention is higher bona fide. VOXY explicitly converts the logits with `sigmoid(spoof_logit - bonafide_logit) * 100`, so higher VOXY scores indicate more model-indicated spoof risk. This is a score, not a calibrated probability.
```

The inference service loads `models/AASIST.onnx` once when FastAPI starts. If it is missing or cannot load, startup fails clearly; the upload endpoint never substitutes fabricated scores. Demo values remain frontend-only and are explicitly labelled.

Prototype risk bands are:

```text
LOW       0-29
MODERATE 30-59
HIGH     60-85
CRITICAL 86-100
```

These are product prototype thresholds, not calibrated scientific confidence intervals.

## Dataset Layout

The training script expects ASVspoof 2019 LA under:

```text
data/LA/LA/
|-- ASVspoof2019_LA_train/flac/*.flac
|-- ASVspoof2019_LA_dev/flac/*.flac
|-- ASVspoof2019_LA_cm_protocols/
|   |-- ASVspoof2019.LA.cm.protocols.txt
|   |-- ASVspoof2019.LA.cm.train.trn.txt
|   |-- ASVspoof2019.LA.cm.dev.trl.txt
```

`scripts/train.py` reads the train and dev protocol files, validates that every referenced FLAC file exists, maps `bonafide` and `spoof` labels, and uses the same `load_audio()` and `create_model_tensor()` functions as backend inference.

The dataset is not redistributed by this repository. Place the dataset in the expected local path and comply with its license and access terms.

## Installation on Windows

Prerequisites:

- Python 3.11 or a compatible Python version supported by the pinned packages.
- Node.js and npm.
- FFmpeg available on `PATH`.
- NVIDIA CUDA support is optional. The detector selects ONNX Runtime CUDA when available and otherwise uses CPU.

FFmpeg is important because browser `MediaRecorder` commonly produces WebM/Opus audio that needs server-side decoding.

### Backend setup

From the repository root:

```bat
cd /d "C:\Users\prady\Documents\Omniroute\Projects\Voxy-Not All Voices are Real"
call "backend\setup.bat"
```

The setup script creates `backend/.venv`, installs `backend/requirements.txt`, checks for FFmpeg, and starts Uvicorn on port 8000. To only install dependencies, run the commands manually:

```bat
cd /d "C:\Users\prady\Documents\Omniroute\Projects\Voxy-Not All Voices are Real\backend"
python -m venv .venv
call ".venv\Scripts\activate.bat"
python -m pip install -r requirements.txt

cd ..
python setup_model.py
```

`setup_model.py` downloads the official maintained AASIST checkpoint from Hugging Face and verifies its `[batch, 64600] -> [batch, 2]` inference contract. The model is ignored by Git and must be downloaded on each new machine.

### Frontend setup

```bat
cd /d "C:\Users\prady\Documents\Omniroute\Projects\Voxy-Not All Voices are Real\frontend"
npm install
```

## Running Locally

### One-command startup

From the repository root:

```bat
start_voxy.bat
```

This checks FFmpeg, initializes the backend environment if needed, opens the backend on port 8000, starts the Next.js frontend on port 3000, and opens the browser.

### Manual startup

Start the backend in one terminal:

```bat
cd /d "C:\Users\prady\Documents\Omniroute\Projects\Voxy-Not All Voices are Real\backend"
call ".venv\Scripts\activate.bat"
uvicorn app.main:app --reload --port 8000
```

Start the frontend in another terminal:

```bat
cd /d "C:\Users\prady\Documents\Omniroute\Projects\Voxy-Not All Voices are Real\frontend"
npm run dev
```

Open:

- Frontend: `http://localhost:3000`
- Backend health: `http://localhost:8000/api/health`
- FastAPI documentation: `http://localhost:8000/docs`

For a production-style frontend preview:

```bat
cd /d "C:\Users\prady\Documents\Omniroute\Projects\Voxy-Not All Voices are Real\frontend"
npm run build
npm run start
```

## Legacy Training

The old training script uses the ASVspoof LA train and dev protocols and writes the archived smoke-test VoiceCNN files. It is not part of production inference and should not be used to claim detector quality.

- `models/model_final.pth`
- `models/model_metadata.json`

Run the legacy integration smoke test only when maintaining that historical path:

```bat
cd /d "C:\Users\prady\Documents\Omniroute\Projects\Voxy-Not All Voices are Real"
call ".venv\Scripts\activate.bat"
python scripts\train.py --smoke-test
```

The smoke test selects a small balanced subset, performs a limited train/dev pass, and validates checkpoint creation. It is an integration test, not a meaningful accuracy evaluation.

For a full train/dev run:

```bat
python scripts\train.py --epochs 20 --batch-size 8 --learning-rate 0.0001
```

Useful options:

```text
--epochs          Number of training epochs; default 20
--batch-size      Batch size; default 8
--learning-rate   Adam learning rate; default 1e-4
--num-workers     DataLoader workers; default 0, recommended on Windows
--seed            Reproducibility seed; default 42
--smoke-test      Run the small balanced integration pass
--smoke-batches   Maximum batches per smoke-test phase; default 2
--smoke-samples   Number of train samples in smoke mode; default 16
```

The best checkpoint is selected by dev F1. Always review the full dev metrics and test on a separate evaluation set before making performance claims.

## Verification and Tests

Pretrained AASIST detector check:

```bat
python backend\scripts\test_detector.py
```

This prints real model-derived scores for any audio files placed under `sample_audio\human` and `sample_audio\synthetic`, plus a zero-waveform load/inference check. No expected score is hardcoded.

Audio-quality and risk-threshold check:

```bat
python scripts\test_audio.py
```

Frontend production validation:

```bat
cd /d "C:\Users\prady\Documents\Omniroute\Projects\Voxy-Not All Voices are Real\frontend"
npm run build
```

Manual integration checks should include:

1. `GET /api/health` returns `status: ok`, `model_loaded: true`, `detector: AASIST`, and `device: CPU` or `CUDA`.
2. The homepage opens on port 3000.
3. A labelled demo changes the result card.
4. A supported audio upload returns real AASIST-derived scores and `mode: pretrained`.
5. The `Analyze a voice` CTA reaches the existing workspace.
6. The Contact modal uses `mailto:pradyun.marukala@gmail.com`.
7. Open microphone simulation, grant permission, and confirm live windows update the result card.

## Troubleshooting

### Frontend shows a Next.js runtime or missing-chunk error

The development server and production build must not run against the same `.next` directory simultaneously. Stop stale Node processes, clear generated artifacts, and start one frontend server:

```powershell
Get-Process -Name node -ErrorAction SilentlyContinue | Stop-Process -Force
Set-Location "C:\Users\prady\Documents\Omniroute\Projects\Voxy-Not All Voices are Real\frontend"
Remove-Item -Recurse -Force .next
npm run dev -- --port 3000
```

### Backend reports a missing detector

Run `python setup_model.py` from the repository root. The API fails clearly when the AASIST checkpoint cannot be loaded; it does not substitute fake scores.

### Browser audio upload fails

Check that:

- FFmpeg is installed and available on `PATH`.
- The file uses a supported extension.
- The backend is running on port 8000.
- `frontend/.env.local` points to the correct backend origin.
- The backend CORS configuration allows the frontend origin.

### Verify CUDA selection

Run:

```bat
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
curl http://localhost:8000/api/health
```

Health reports `CUDA` only when `CUDAExecutionProvider` is available and selected. Otherwise it honestly reports `CPU`; install a compatible NVIDIA driver/CUDA/cuDNN runtime and the `onnxruntime-gpu` requirement before expecting GPU execution.

### Audio is marked insufficient

The quality gate rejects empty, very short, silent, or heavily clipped audio. Use a clear recording with at least approximately one second of audible content; the model still analyzes fixed four-second windows after preprocessing.

## Privacy and Limitations

Audio is processed for analysis and not intentionally retained by the prototype. Uploaded files are stored temporarily on the backend while decoding and inference run, then removed by the request cleanup path.

VOXY currently cannot:

- Intercept arbitrary cellular calls from a normal browser.
- Guarantee the authenticity or falsity of a voice.
- Identify who is speaking.
- Replace trusted caller verification.
- Support performance claims based only on the smoke-test checkpoint.

When a signal is elevated, users should independently contact the person through a trusted channel and avoid sharing OTPs, credentials, or transferring money based on the call alone.

## Future Work

Potential next steps include:

- Full ASVspoof LA training and independent evaluation.
- Calibration of synthetic-likelihood and risk thresholds.
- Separate test-set reporting with reproducible metrics.
- Richer spectral, prosodic, and temporal feature branches.
- Native mobile audio integration with explicit operating-system permissions.
- A production authentication and user-session model.
- Stronger streaming inference around the existing WebSocket contract.
- Privacy-preserving deployment options after the threat model is defined.

## License and Dataset Notice

This repository is a project prototype. Dataset files are not redistributed here. Use ASVspoof data according to its own terms, and review all third-party dependency licenses before deploying VOXY beyond local development.
