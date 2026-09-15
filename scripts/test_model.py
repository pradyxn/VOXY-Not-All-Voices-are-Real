from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.ml.model import VoiceCNN
from app.ml.preprocessing import create_model_tensor
import numpy as np
import torch

model = VoiceCNN().eval()
tensor = create_model_tensor(np.zeros(64_000, dtype=np.float32))
with torch.inference_mode():
    output = model(tensor)
assert tuple(tensor.shape[:2]) == (1, 1)
assert output.shape == (1, 2)
print("Model architecture and tensor pipeline OK")
print(f"Input tensor: {tuple(tensor.shape)}")
print(f"Output tensor: {tuple(output.shape)}")
model_path = ROOT / "models" / "model_final.pth"
print("model_final.pth: available" if model_path.exists() else "model_final.pth: missing; DEMO MODE is expected")
