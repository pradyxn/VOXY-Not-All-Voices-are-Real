# VOXY

## Not All Voices Are Real.

VOXY is an AI voice-security prototype for analyzing acoustic signals associated with synthetic or cloned speech. It accepts an audio recording, processes it through a consistent signal pipeline, evaluates the resulting Mel-spectrogram with a PyTorch CNN, aggregates evidence across multiple windows, and presents a 0-100 risk signal with practical verification guidance.

VOXY is decision support, not an infallible detector. It does not guarantee that a voice is real or fake, identify the caller, or replace independent verification.

## Project Status

The repository currently contains:

- A working Next.js frontend with the VOXY product experience and analysis workspace.
- A FastAPI backend with REST and WebSocket endpoints.
- Shared preprocessing, model inference, temporal aggregation, and risk-engine layers.
- An ASVspoof 2019 LA training script that uses the same model and preprocessing path as inference.
- A generated compatible checkpoint at `models/model_final.pth`.
- A metadata file at `models/model_metadata.json`.

The checked-in checkpoint was produced by the deterministic smoke-test path. It proves that dataset parsing, audio loading, spectrogram creation, model forward/backward passes, checkpoint saving, and inference loading work end to end. It must not be treated as a production-quality accuracy result. The current metadata records 16 train samples, 16 dev samples, one epoch, and validation F1 of `0.0`.

## Repository Layout

```text
.
|-- backend/
|   |-- app/
|   |   |-- main.py                 FastAPI app and API/WebSocket routes
|   |   |-- config.py               Shared sample/window/model configuration
|   |   |-- schemas.py              API response models
|   |   |-- ml/
|   |   |   |-- model.py             VoiceCNN architecture
|   |   |   |-- preprocessing.py     Audio and Mel-spectrogram pipeline
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
|   |-- model_final.pth              Compatible VoiceCNN checkpoint
|   |-- model_metadata.json          Checkpoint and training metadata
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
128-band Mel-spectrogram -> power-to-dB conversion
        |
        v
VoiceCNN neural-network inference
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

The same `VoiceCNN` and preprocessing functions are imported by both the training script and the backend inference service. This keeps the training and inference tensor contracts aligned.

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
  "mode": "ml | demo",
  "model_message": null,
  "timeline": []
}
```

Uploaded files are written to a temporary path for analysis and removed in the backend `finally` block after processing.

### Streaming analysis

```text
WebSocket /ws/analyze
```

The client can send audio bytes over the WebSocket. The backend returns incremental window, synthetic-score, risk-score, status, and mode values. In the browser product, realtime analysis is presented honestly as a simulation because a normal browser cannot intercept arbitrary cellular-call audio.

## Backend and ML Configuration

The shared configuration in `backend/app/config.py` defines:

| Setting | Value | Purpose |
|---|---:|---|
| Sample rate | 16,000 Hz | Target audio rate |
| Target samples | 64,000 | Four seconds at 16 kHz |
| Window length | 4 seconds | Per-window analysis duration |
| Minimum windows | 3 | Aggregation configuration |
| EMA alpha | 0.35 | Rolling score aggregation |

The `VoiceCNN` uses two output classes:

```text
bonafide -> 0
spoof    -> 1
```

The inference service loads `models/model_final.pth` when it exists and matches the architecture exactly. If no compatible checkpoint is available, the backend remains usable in clearly labelled demo mode. Demo values must not be presented as real model predictions.

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
- NVIDIA CUDA support is optional. PyTorch automatically selects CUDA when available and otherwise uses CPU.

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
```

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

## Training

The training script uses the real ASVspoof LA train and dev protocols and writes:

- `models/model_final.pth`
- `models/model_metadata.json`

Run a deterministic smoke test first:

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

Model tensor and output check:

```bat
python scripts\test_model.py
```

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

1. `GET /api/health` returns `status: ok`.
2. The homepage opens on port 3000.
3. A labelled demo changes the result card.
4. A supported audio upload returns the documented response fields.
5. The `Analyze a voice` CTA reaches the existing workspace.
6. The Contact modal uses `mailto:pradyun.marukala@gmail.com`.

## Troubleshooting

### Frontend shows a Next.js runtime or missing-chunk error

The development server and production build must not run against the same `.next` directory simultaneously. Stop stale Node processes, clear generated artifacts, and start one frontend server:

```powershell
Get-Process -Name node -ErrorAction SilentlyContinue | Stop-Process -Force
Set-Location "C:\Users\prady\Documents\Omniroute\Projects\Voxy-Not All Voices are Real\frontend"
Remove-Item -Recurse -Force .next
npm run dev -- --port 3000
```

### Backend reports demo mode

Check that `models/model_final.pth` exists and matches the current `VoiceCNN` architecture. The API intentionally reports `mode: demo` when the model is unavailable or no compatible checkpoint is loaded.

### Browser audio upload fails

Check that:

- FFmpeg is installed and available on `PATH`.
- The file uses a supported extension.
- The backend is running on port 8000.
- `frontend/.env.local` points to the correct backend origin.
- The backend CORS configuration allows the frontend origin.

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
