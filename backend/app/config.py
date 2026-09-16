from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "models" / "AASIST.onnx"
SAMPLE_RATE = 16_000
TARGET_SAMPLES = 64_600
WINDOW_SECONDS = 4
WINDOW_HOP_SAMPLES = 32_300
MAX_WINDOWS = 8
MIN_WINDOWS = 1
RISK_THRESHOLDS = {"moderate": 30, "high": 60, "critical": 86}
FRONTEND_ORIGIN = os.getenv("VOXY_FRONTEND_ORIGIN", "http://localhost:3000")
