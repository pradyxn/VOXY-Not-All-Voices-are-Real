from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "models" / "model_final.pth"
SAMPLE_RATE = 16_000
TARGET_SAMPLES = 64_000
WINDOW_SECONDS = 4
MIN_WINDOWS = 3
EMA_ALPHA = 0.35
FRONTEND_ORIGIN = os.getenv("VOXY_FRONTEND_ORIGIN", "http://localhost:3000")
