from pathlib import Path
import librosa
import numpy as np
import torch
from ..config import SAMPLE_RATE, TARGET_SAMPLES

def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    audio, rate = librosa.load(str(path), sr=SAMPLE_RATE, mono=False)
    return convert_to_mono(audio), rate

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
