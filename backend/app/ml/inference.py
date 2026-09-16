from pathlib import Path
import numpy as np
import torch
from .preprocessing import load_audio, quality_check, split_windows
from ..config import MAX_WINDOWS, MODEL_PATH, TARGET_SAMPLES, WINDOW_HOP_SAMPLES


class InferenceService:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.session = None
        self.providers: list[str] = []
        self.loaded = False
        self.mode = "pretrained"
        self.detector = "AASIST pretrained anti-spoofing model"
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"AASIST checkpoint missing: {MODEL_PATH}. Run setup_model.py first.")
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
            preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.providers = [provider for provider in preferred if provider in available]
            if not self.providers:
                raise RuntimeError("ONNX Runtime has no usable execution provider")

            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            # Render's free instance is CPU-constrained; keep a single worker
            # thread so multiple inference thread pools do not fight each other.
            session_options.intra_op_num_threads = 1
            session_options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(
                str(MODEL_PATH),
                sess_options=session_options,
                providers=self.providers,
            )
            input_shape = self.session.get_inputs()[0].shape
            if input_shape[-1] != TARGET_SAMPLES:
                raise RuntimeError(f"Unexpected AASIST input shape: {input_shape}")
            self.loaded = True
        except ImportError as error:
            raise RuntimeError("onnxruntime is required for AASIST inference") from error

    def predict_file(self, path: str | Path) -> tuple[list[float], str]:
        audio, _ = load_audio(path)
        return self.predict_audio(audio)

    def predict_audio(self, audio: np.ndarray) -> tuple[list[float], str]:
        quality = quality_check(audio)
        if quality == "insufficient":
            return [], quality
        windows = split_windows(audio)[:MAX_WINDOWS]
        return [self.predict_window(window) for window in windows], quality

    def predict_new_file_windows(
        self,
        path: str | Path,
        start_index: int,
    ) -> tuple[list[float], str, int]:
        """Decode the live recording once, but infer only windows not seen before."""
        audio, _ = load_audio(path)
        quality = quality_check(audio)
        if quality == "insufficient":
            return [], quality, 0

        if len(audio) < TARGET_SAMPLES:
            return [], quality, 0

        available = 1 + (len(audio) - TARGET_SAMPLES) // WINDOW_HOP_SAMPLES
        available = min(available, MAX_WINDOWS)
        if available <= start_index:
            return [], quality, available

        windows = split_windows(audio)[:available]
        new_windows = windows[start_index:available]
        return [self.predict_window(window) for window in new_windows], quality, available

    def predict_window(self, audio: np.ndarray) -> float:
        _, _, spoof_score = self.predict_window_details(audio)
        return spoof_score

    def predict_window_details(self, audio: np.ndarray) -> tuple[float, float, float]:
        if not self.loaded or self.session is None:
            raise RuntimeError("AASIST detector is not loaded")

        waveform = np.asarray(pad_waveform(audio), dtype=np.float32)[None, :]
        with torch.inference_mode():
            logits = self.session.run(None, {self.session.get_inputs()[0].name: waveform})[0]

            # Official AASIST class convention: 0 = spoof, 1 = bona fide.
            logits = np.asarray(logits, dtype=np.float32)
            if logits.shape != (1, 2):
                raise RuntimeError(f"Unexpected AASIST output shape: {logits.shape}")
            shifted = logits[0] - float(np.max(logits[0]))
            probabilities = np.exp(shifted) / np.sum(np.exp(shifted))
            spoof_score = float(probabilities[0] * 100.0)
            bona_fide_logit = float(logits[0, 1])
            spoof_logit = float(logits[0, 0])

        return bona_fide_logit, spoof_logit, round(spoof_score, 1)

    @property
    def provider(self) -> str:
        return self.providers[0] if self.providers else "unavailable"

    @property
    def device_name(self) -> str:
        return "CUDA" if self.provider == "CUDAExecutionProvider" else "CPU"


def pad_waveform(audio: np.ndarray) -> np.ndarray:
    """Match the official AASIST eval padding: repeat, then take 64600 samples."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return np.zeros(TARGET_SAMPLES, dtype=np.float32)
    if audio.size >= TARGET_SAMPLES:
        return audio[:TARGET_SAMPLES]
    repeats = (TARGET_SAMPLES + audio.size - 1) // audio.size
    return np.tile(audio, repeats)[:TARGET_SAMPLES]
