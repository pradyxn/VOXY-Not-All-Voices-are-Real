from pathlib import Path
import subprocess
import librosa
import numpy as np
import torch
from ..config import SAMPLE_RATE, TARGET_SAMPLES, WINDOW_HOP_SAMPLES

def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    try:
        audio, rate = librosa.load(str(path), sr=SAMPLE_RATE, mono=False)
    except Exception:
        command = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"]
        decoded = subprocess.run(command, check=True, capture_output=True).stdout
        audio = np.frombuffer(decoded, dtype=np.float32)
        rate = SAMPLE_RATE
    return convert_to_mono(np.asarray(audio, dtype=np.float32)), rate

def convert_to_mono(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=0) if audio.ndim > 1 else audio

def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int = SAMPLE_RATE) -> np.ndarray:
    if source_rate == target_rate:
        return audio
    return librosa.resample(audio, orig_sr=source_rate, target_sr=target_rate)

def pad_or_trim(audio: np.ndarray, target_samples: int = TARGET_SAMPLES) -> np.ndarray:
    if len(audio) >= target_samples:
        return audio[:target_samples]
    return np.pad(audio, (0, target_samples - len(audio)))

def split_windows(audio: np.ndarray) -> list[np.ndarray]:
    if len(audio) <= TARGET_SAMPLES:
        return [pad_or_trim(audio)]
    starts = list(range(0, len(audio) - TARGET_SAMPLES + 1, WINDOW_HOP_SAMPLES))
    windows = [audio[start:start + TARGET_SAMPLES] for start in starts]
    if starts[-1] + TARGET_SAMPLES < len(audio):
        windows.append(audio[-TARGET_SAMPLES:])
    return windows

def create_mel_spectrogram(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    return librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128, power=2.0)

def convert_to_db(spectrogram: np.ndarray) -> np.ndarray:
    return librosa.power_to_db(spectrogram, ref=np.max)

def create_model_tensor(audio: np.ndarray) -> torch.Tensor:
    normalized = pad_or_trim(audio)
    mel = convert_to_db(create_mel_spectrogram(normalized))
    return torch.from_numpy(mel).float().unsqueeze(0).unsqueeze(0)

def quality_check(audio: np.ndarray) -> str:
    if audio.size == 0 or len(audio) < SAMPLE_RATE:
        return "insufficient"
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio))))
    silence_ratio = float(np.mean(np.abs(audio) < 0.005))
    if peak < 0.01 or rms < 0.003 or silence_ratio > 0.97:
        return "insufficient"
    if peak > 0.995:
        return "clipped"
    return "good"
