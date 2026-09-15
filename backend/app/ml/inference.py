from pathlib import Path
import torch
from .model import VoiceCNN
from .preprocessing import create_model_tensor, load_audio, quality_check
from .aggregation import aggregate_scores
from ..config import MODEL_PATH

class InferenceService:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VoiceCNN().to(self.device)
        self.loaded = False
        self.mode = "demo"
        if MODEL_PATH.exists():
            checkpoint = torch.load(MODEL_PATH, map_location=self.device)
            state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            self.model.load_state_dict(state_dict, strict=True)
            self.model.eval()
            self.loaded = True
            self.mode = "ml"

    def predict_file(self, path: str | Path) -> tuple[list[float], str]:
        audio, _ = load_audio(path)
        quality = quality_check(audio)
        if quality == "insufficient":
            return [], quality
        windows = [audio[start:start + 64_000] for start in range(0, max(1, len(audio) - 1), 64_000)]
        scores = []
        for window in windows[:8]:
            if self.loaded:
                with torch.inference_mode():
                    logits = self.model(create_model_tensor(window).to(self.device))
                    scores.append(float(torch.softmax(logits, dim=1)[0, 1].item() * 100))
            else:
                # Deterministic fallback is deliberately marked as demo mode by the API.
                energy = float(torch.tensor(window).abs().mean())
                scores.append(round(38 + min(54, energy * 1000), 1))
        return scores or [50.0], quality

    def predict_tensor(self, tensor: torch.Tensor) -> float:
        if not self.loaded:
            return 50.0
        with torch.inference_mode():
            return float(torch.softmax(self.model(tensor.to(self.device)), dim=1)[0, 1].item() * 100)
