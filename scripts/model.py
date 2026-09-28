"""Classifier and loader matching train_emotion_classifier.py."""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class EmotionClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, features):
        return self.fc(features)


def load_artifacts(model_dir: Path, device: torch.device):
    """Load existing weights; recover labels using the training CSV's row order.

    The original trainer saves only weights and does not scale features.
    Keep its feature CSV unchanged: factorize assigns labels by first appearance.
    """
    model_dir = Path(model_dir)
    checkpoint = torch.load(
        model_dir / "emotion_classifier.pth", map_location=device, weights_only=True
    )
    input_dim = checkpoint["fc.0.weight"].shape[1]
    num_classes = checkpoint["fc.6.weight"].shape[0]
    labels_path = model_dir / "label_names.npy"
    if labels_path.exists():
        label_names = np.load(labels_path, allow_pickle=False)
    else:
        features_path = model_dir.parent.parent / "Extracted Features" / "combined_wav2vec_features.csv"
        if not features_path.exists():
            raise FileNotFoundError(
                "Label mapping unavailable. Restore the original training feature CSV "
                "or provide label_names.npy in training class order."
            )
        frame = pd.read_csv(features_path).drop(columns=["Dataset"], errors="ignore").dropna()
        if frame.shape[1] - 1 != input_dim:
            raise ValueError("Training feature CSV does not match checkpoint input size.")
        label_names = np.asarray(pd.factorize(frame["label"])[1], dtype=str)
    if label_names.ndim != 1 or len(label_names) != num_classes:
        raise ValueError("Label mapping does not match checkpoint class count.")
    model = EmotionClassifier(input_dim, num_classes).to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    model_info = {
        "input_dim": input_dim,
        "num_classes": num_classes,
        "labels": label_names.tolist(),
        "architecture": "768-feature MLP (128, 64 hidden units)",
        "feature_scaling": "none",
    }
    return model, label_names, model_info, None, input_dim
