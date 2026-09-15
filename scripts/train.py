"""Train VOXY's existing VoiceCNN on the ASVspoof2019 LA train/dev splits."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import SAMPLE_RATE, TARGET_SAMPLES  # noqa: E402
from app.ml.model import VoiceCNN  # noqa: E402
from app.ml.preprocessing import create_model_tensor, load_audio  # noqa: E402

DATA_ROOT = PROJECT_ROOT / "data" / "LA" / "LA"
TRAIN_AUDIO_DIR = DATA_ROOT / "ASVspoof2019_LA_train" / "flac"
TRAIN_PROTOCOL = DATA_ROOT / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt"
DEV_AUDIO_DIR = DATA_ROOT / "ASVspoof2019_LA_dev" / "flac"
DEV_PROTOCOL = DATA_ROOT / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.dev.trl.txt"
MODEL_OUTPUT = PROJECT_ROOT / "models" / "model_final.pth"
METADATA_OUTPUT = PROJECT_ROOT / "models" / "model_metadata.json"
CLASS_MAPPING = {"bonafide": 0, "spoof": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0, help="Keep at 0 on Windows unless multiprocessing is tested.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true", help="Run a tiny balanced train/dev pass and save a checkpoint.")
    parser.add_argument("--smoke-batches", type=int, default=2)
    parser.add_argument("--smoke-samples", type=int, default=16)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_protocol(protocol_path: Path, audio_dir: Path) -> list[tuple[Path, int]]:
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Protocol file not found: {protocol_path}")
    records: list[tuple[Path, int]] = []
    for line_number, line in enumerate(protocol_path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Expected 5 protocol fields at {protocol_path}:{line_number}, got {len(fields)}")
        _, utterance_id, _, _, label = fields
        if label not in CLASS_MAPPING:
            raise ValueError(f"Unknown label {label!r} at {protocol_path}:{line_number}")
        audio_path = audio_dir / f"{utterance_id}.flac"
        if not audio_path.is_file():
            raise FileNotFoundError(f"Protocol audio file not found: {audio_path}")
        records.append((audio_path, CLASS_MAPPING[label]))
    if not records:
        raise ValueError(f"Protocol file is empty: {protocol_path}")
    return records


def balanced_smoke_records(records: list[tuple[Path, int]], limit: int) -> list[tuple[Path, int]]:
    by_class = {label: [record for record in records if record[1] == label] for label in CLASS_MAPPING.values()}
    selected: list[tuple[Path, int]] = []
    while len(selected) < limit and any(by_class.values()):
        for label in sorted(by_class):
            if by_class[label] and len(selected) < limit:
                selected.append(by_class[label].pop(0))
    return selected


class ASVspoofDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, records: list[tuple[Path, int]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        audio_path, label = self.records[index]
        audio, _ = load_audio(audio_path)
        tensor = create_model_tensor(audio).squeeze(0)
        return tensor, torch.tensor(label, dtype=torch.long)


def make_loader(records: list[tuple[Path, int]], batch_size: int, num_workers: int, shuffle: bool, seed: int, device: torch.device) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ASVspoofDataset(records),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def class_weights(records: Iterable[tuple[Path, int]]) -> torch.Tensor:
    counts = Counter(label for _, label in records)
    total = sum(counts.values())
    return torch.tensor([total / (len(CLASS_MAPPING) * counts[label]) for label in range(len(CLASS_MAPPING))], dtype=torch.float32)


def metrics(loss_total: float, correct: int, total: int, true_positive: int, false_positive: int, false_negative: int) -> dict[str, float]:
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"loss": loss_total / max(1, total), "accuracy": correct / max(1, total), "precision": precision, "recall": recall, "f1": f1}


def train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, device: torch.device, use_amp: bool, max_batches: int | None = None) -> dict[str, float]:
    model.train()
    loss_total = correct = total = true_positive = false_positive = false_negative = 0
    for batch_index, (inputs, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        predictions = outputs.argmax(dim=1)
        loss_total += loss.item() * targets.size(0)
        correct += int((predictions == targets).sum())
        total += targets.size(0)
        true_positive += int(((predictions == 1) & (targets == 1)).sum())
        false_positive += int(((predictions == 1) & (targets == 0)).sum())
        false_negative += int(((predictions == 0) & (targets == 1)).sum())
    return metrics(loss_total, correct, total, true_positive, false_positive, false_negative)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, max_batches: int | None = None) -> dict[str, float]:
    model.eval()
    loss_total = correct = total = true_positive = false_positive = false_negative = 0
    with torch.inference_mode():
        for batch_index, (inputs, targets) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            predictions = outputs.argmax(dim=1)
            loss_total += loss.item() * targets.size(0)
            correct += int((predictions == targets).sum())
            total += targets.size(0)
            true_positive += int(((predictions == 1) & (targets == 1)).sum())
            false_positive += int(((predictions == 1) & (targets == 0)).sum())
            false_negative += int(((predictions == 0) & (targets == 1)).sum())
    return metrics(loss_total, correct, total, true_positive, false_positive, false_negative)


def save_checkpoint(model: nn.Module, metadata: dict) -> None:
    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_OUTPUT)
    METADATA_OUTPUT.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("epochs and batch-size must be positive; num-workers cannot be negative")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and hasattr(torch, "amp")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Device: {'CUDA' if device.type == 'cuda' else 'CPU'}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("GPU: unavailable; PyTorch selected CPU")
    print(f"Mixed precision: {'enabled' if use_amp else 'disabled'}")

    train_records = parse_protocol(TRAIN_PROTOCOL, TRAIN_AUDIO_DIR)
    dev_records = parse_protocol(DEV_PROTOCOL, DEV_AUDIO_DIR)
    if args.smoke_test:
        train_records = balanced_smoke_records(train_records, args.smoke_samples)
        dev_records = balanced_smoke_records(dev_records, max(2, min(args.smoke_samples, args.batch_size * 2)))
        epochs = 1
        max_batches = args.smoke_batches
    else:
        epochs = args.epochs
        max_batches = None
    print(f"Train: {len(train_records)}")
    print(f"Dev: {len(dev_records)}")
    counts = Counter(label for _, label in train_records)
    weights = class_weights(train_records)
    print(f"Class counts: bonafide={counts[0]}, spoof={counts[1]}")
    print(f"Class weights: {[round(value, 6) for value in weights.tolist()]}")

    train_loader = make_loader(train_records, args.batch_size, args.num_workers, True, args.seed, device)
    dev_loader = make_loader(dev_records, args.batch_size, args.num_workers, False, args.seed, device)
    model = VoiceCNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_f1 = -1.0
    best_metrics: dict[str, float] = {}

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp, max_batches)
        dev_metrics = evaluate(model, dev_loader, criterion, device, max_batches)
        print(f"Train loss: {train_metrics['loss']:.4f} | accuracy: {train_metrics['accuracy']:.4f}")
        print(f"Dev loss: {dev_metrics['loss']:.4f} | Accuracy: {dev_metrics['accuracy']:.4f} | Precision: {dev_metrics['precision']:.4f} | Recall: {dev_metrics['recall']:.4f} | F1: {dev_metrics['f1']:.4f}")
        if dev_metrics["f1"] > best_f1:
            best_f1 = dev_metrics["f1"]
            best_metrics = dev_metrics
            metadata = {
                "model_architecture": "VoiceCNN",
                "class_mapping": CLASS_MAPPING,
                "sample_rate": SAMPLE_RATE,
                "target_samples": TARGET_SAMPLES,
                "mel_bins": 128,
                "train_sample_counts": {"bonafide": counts[0], "spoof": counts[1]},
                "dev_sample_counts": {"bonafide": sum(label == 0 for _, label in dev_records), "spoof": sum(label == 1 for _, label in dev_records)},
                "epochs": epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "best_validation_metrics": best_metrics,
                "device": str(device),
                "smoke_test": args.smoke_test,
                "seed": args.seed,
            }
            save_checkpoint(model, metadata)
            print(f"Saved best checkpoint: {MODEL_OUTPUT}")
    print(f"Best validation F1: {best_f1:.4f}")
    print(f"Metadata: {METADATA_OUTPUT}")


if __name__ == "__main__":
    main()
