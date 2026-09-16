from pathlib import Path
from urllib.request import urlopen

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "AASIST.onnx"
MODEL_URL = "https://huggingface.co/SpeechAntiSpoofingBenchmarks/AASIST/resolve/main/aasist.onnx?download=true"


def main() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        print(f"Downloading official AASIST checkpoint to {MODEL_PATH}")
        with urlopen(MODEL_URL) as response, MODEL_PATH.open("wb") as destination:
            while chunk := response.read(1024 * 1024):
                destination.write(chunk)
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    output = session.run(None, {input_info.name: np.zeros((1, 64600), dtype=np.float32)})[0]
    if list(input_info.shape) != ["batch", 64600] or output.shape != (1, 2):
        raise RuntimeError(f"Unexpected AASIST contract: input={input_info.shape}, output={output.shape}")
    print(f"AASIST checkpoint verified: {MODEL_PATH}")
    print(f"Input: {input_info.shape}; output: {output.shape}; providers: {session.get_providers()}")


if __name__ == "__main__":
    main()
