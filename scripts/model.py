"""Shared model definition and artifact loading for SER training and inference."""

from pathlib import Path
import json

import joblib
import numpy as np
import torch
import torch.nn as nn


class EmotionClassifier(nn.Module):
    """MLP classifier used by training, evaluation, and the Flask app."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)


def load_artifacts(model_dir: Path, device: torch.device):
    """Load the classifier and preprocessing artifacts saved by training."""
    model_dir = Path(model_dir)
    model_path = model_dir / "emotion_classifier.pth"
    labels_path = model_dir / "label_names.npy"
    info_path = model_dir / "model_info.json"
    scaler_path = model_dir / "feature_scaler.pkl"

    required = [model_path, labels_path, info_path, scaler_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model artifacts: " + ", ".join(missing)
        )

    try:
        label_names = np.load(labels_path, allow_pickle=False)
    except ValueError:
        # Older training runs saved sklearn labels as an object array. Keep
        # backward compatibility for local artifacts, while new training
        # writes a plain Unicode array below.
        label_names = np.load(labels_path, allow_pickle=True)
    with info_path.open("r", encoding="utf-8") as handle:
        model_info = json.load(handle)

    input_dim = int(model_info.get("input_dim", 768))
    model = EmotionClassifier(input_dim, len(label_names)).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    scaler = joblib.load(scaler_path)
    return model, label_names, model_info, scaler, input_dim
