from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.ml.inference import InferenceService
from app.ml.preprocessing import load_audio, quality_check, split_windows


def find_sample(folder: Path) -> Path | None:
    supported = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".opus"}
    return next((path for path in folder.iterdir() if path.suffix.lower() in supported), None) if folder.exists() else None


def report(detector: InferenceService, label: str, path: Path | None) -> None:
    if path is None:
        print(f"{label}: no sample found; add an audio file under sample_audio/{label.lower()}/")
        return
    started = time.perf_counter()
    audio, _ = load_audio(path)
    quality = quality_check(audio)
    windows = split_windows(audio)
    started = time.perf_counter()
    details = [detector.predict_window_details(window) for window in windows]
    elapsed = (time.perf_counter() - started) * 1000
    scores = [detail[2] for detail in details]
    print(f"{label} SAMPLE")
    print(f"raw AASIST logits (bonafide, spoof): {[(round(item[0], 5), round(item[1], 5)) for item in details]}")
    print(f"raw model direction: higher bona fide; windows: {len(scores)}; quality: {quality}")
    print(f"VOXY spoof scores: {scores}")
    print(f"latency: {elapsed:.1f} ms")


def main() -> None:
    detector = InferenceService()
    print(f"detector: {detector.detector}")
    print(f"provider: {detector.provider}")
    report(detector, "HUMAN", find_sample(ROOT / "sample_audio" / "human"))
    report(detector, "SYNTHETIC", find_sample(ROOT / "sample_audio" / "synthetic"))
    zero_score = detector.predict_window(np.zeros(64_600, dtype=np.float32))
    print(f"zero-waveform smoke score: {zero_score:.1f}")


if __name__ == "__main__":
    main()
