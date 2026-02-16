# train_emotion_classifier.py
# Two-phase training optimized for CPU:
#   Phase 1: Extract Wav2Vec2 features ONCE (frozen backbone, cached)
#   Phase 2: Train attention pooling + classifier on cached features
#
# This avoids running Wav2Vec2 forward pass every epoch (the slow part).
# On CPU, this reduces training from hours to ~10-15 minutes.

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import joblib
import librosa
from pathlib import Path
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import warnings
import random

warnings.filterwarnings("ignore", category=UserWarning)

# ==================== Paths ====================
BASE_DIR = Path(__file__).parent.parent
PREPROCESSED_CSV = BASE_DIR / "Preprocessed Data" / "combined_data.csv"
OUTPUT_MODEL_DIR = BASE_DIR / "models" / "emotion_classifier"
RESULTS_DIR = BASE_DIR / "results"
CACHE_DIR = BASE_DIR / "models" / "feature_cache"

os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ==================== Hyperparameters ====================
EPOCHS = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 64
VALIDATION_FRACTION = 0.15
PATIENCE = 15
WEIGHT_DECAY = 1e-4
TARGET_SR = 16000
MAX_AUDIO_LEN = 5 * TARGET_SR  # 5 seconds
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ==================== Attention Pooling ====================
class AttentionPooling(nn.Module):
    """Learns which time frames are most important for classification."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x: [batch, time, hidden]
        attn_weights = self.attention(x).squeeze(-1)  # [batch, time]
        attn_weights = F.softmax(attn_weights, dim=-1)
        pooled = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # [batch, hidden]
        return pooled


# ==================== Classifier Head ====================
class EmotionClassifier(nn.Module):
    """
    Attention pooling + classifier trained on cached Wav2Vec2 hidden states.
    Input: [batch, time_steps, 768] hidden states
    Output: [batch, num_classes] logits
    """
    def __init__(self, hidden_dim=768, num_classes=8):
        super().__init__()
        self.attention_pool = AttentionPooling(hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: [batch, time, 768]
        pooled = self.attention_pool(x)  # [batch, 768]
        pooled = self.layer_norm(pooled)
        return self.classifier(pooled)


# ==================== Feature Extraction (one-time) ====================
def extract_all_features(file_paths, labels, cache_path):
    """
    Extract Wav2Vec2 hidden states for all audio files ONCE and cache them.
    Returns list of (hidden_states, label) tuples.
    """
    if cache_path.exists():
        print(f"Loading cached features from {cache_path}")
        cached = torch.load(cache_path, weights_only=False)
        print(f"Loaded {len(cached['features'])} cached features")
        return cached['features'], cached['labels']

    print("Extracting Wav2Vec2 features (one-time, will be cached)...")

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    wav2vec.eval()

    # Freeze everything
    for param in wav2vec.parameters():
        param.requires_grad = False

    all_features = []
    valid_labels = []
    failed = 0

    for i, (fp, lbl) in enumerate(tqdm(zip(file_paths, labels),
                                        total=len(file_paths),
                                        desc="Extracting features")):
        try:
            audio, sr = librosa.load(fp, sr=TARGET_SR)
            if len(audio) < TARGET_SR * 0.1:
                failed += 1
                continue

            audio = librosa.util.normalize(audio)
            audio, _ = librosa.effects.trim(audio, top_db=25)

            if len(audio) < TARGET_SR * 0.1:
                failed += 1
                continue

            # Truncate to max length
            if len(audio) > MAX_AUDIO_LEN:
                audio = audio[:MAX_AUDIO_LEN]

            audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                hidden_states = wav2vec(audio_tensor).last_hidden_state  # [1, T, 768]

            all_features.append(hidden_states.squeeze(0))  # [T, 768]
            valid_labels.append(lbl)

        except Exception as e:
            failed += 1
            continue

    print(f"Extracted: {len(all_features)}/{len(file_paths)} | Failed: {failed}")

    # Cache to disk
    torch.save({'features': all_features, 'labels': valid_labels}, cache_path)
    print(f"Cached features to {cache_path}")

    # Free memory
    del wav2vec, processor
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return all_features, valid_labels


# ==================== Dataset ====================
class HiddenStateDataset(Dataset):
    """Dataset of cached Wav2Vec2 hidden states."""
    def __init__(self, features, labels, max_time_steps=None, augment=False):
        self.features = features
        self.labels = labels
        self.max_time_steps = max_time_steps
        self.augment = augment

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx]  # [T, 768]
        label = self.labels[idx]

        # Pad/truncate to fixed time steps
        if self.max_time_steps:
            T = feat.size(0)
            if T > self.max_time_steps:
                if self.augment:
                    start = random.randint(0, T - self.max_time_steps)
                else:
                    start = 0
                feat = feat[start:start + self.max_time_steps]
            elif T < self.max_time_steps:
                pad = torch.zeros(self.max_time_steps - T, feat.size(1))
                feat = torch.cat([feat, pad], dim=0)

        # Simple augmentation: add small noise to features
        if self.augment and random.random() < 0.3:
            noise = torch.randn_like(feat) * 0.01
            feat = feat + noise

        return feat, label


def collate_fn(batch):
    """Pad hidden states to same length within batch."""
    feats, labels = zip(*batch)
    max_len = max(f.size(0) for f in feats)
    padded = torch.zeros(len(feats), max_len, feats[0].size(1))
    for i, f in enumerate(feats):
        padded[i, :f.size(0)] = f
    return padded, torch.tensor(labels, dtype=torch.long)


# ==================== Main ====================
def main():
    print("=" * 60)
    print("SPEECH EMOTION RECOGNITION - CPU-OPTIMIZED TRAINING")
    print("=" * 60)

    if not PREPROCESSED_CSV.exists():
        raise FileNotFoundError(
            f"Not found: {PREPROCESSED_CSV}\nRun preprocess.py first."
        )

    df = pd.read_csv(PREPROCESSED_CSV)
    df = df[df['Path'].apply(lambda p: os.path.exists(p))].reset_index(drop=True)
    print(f"Total valid samples: {len(df)}")

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df['Emotions'].values)
    label_names = le.classes_
    file_paths = df['Path'].values.tolist()

    print(f"Classes: {len(label_names)} -> {list(label_names)}")

    # ---- Phase 1: Extract features (one-time) ----
    cache_path = CACHE_DIR / "wav2vec_hidden_states.pt"
    all_features, all_labels = extract_all_features(file_paths, y, cache_path)

    # Split indices
    indices = list(range(len(all_features)))
    all_labels_np = np.array(all_labels)

    idx_train_val, idx_test = train_test_split(
        indices, test_size=0.15, random_state=SEED, stratify=all_labels_np
    )
    idx_train, idx_val = train_test_split(
        idx_train_val, test_size=VALIDATION_FRACTION / (1 - 0.15),
        random_state=SEED, stratify=all_labels_np[idx_train_val]
    )

    train_feats = [all_features[i] for i in idx_train]
    train_labels = [all_labels[i] for i in idx_train]
    val_feats = [all_features[i] for i in idx_val]
    val_labels = [all_labels[i] for i in idx_val]
    test_feats = [all_features[i] for i in idx_test]
    test_labels = [all_labels[i] for i in idx_test]

    print(f"Train: {len(train_feats)} | Val: {len(val_feats)} | Test: {len(test_feats)}")

    # Save test data for evaluation script
    np.save(OUTPUT_MODEL_DIR / "test_data.npy", {
        'features': test_feats,
        'labels': test_labels,
        'label_names': label_names
    }, allow_pickle=True)

    # Save label artifacts
    joblib.dump(le, OUTPUT_MODEL_DIR / "label_encoder.pkl")
    np.save(OUTPUT_MODEL_DIR / "label_names.npy", label_names)

    # Compute max time steps across dataset for padding
    max_time = max(f.size(0) for f in all_features)
    print(f"Max time steps: {max_time}")

    # DataLoaders
    train_dataset = HiddenStateDataset(train_feats, train_labels,
                                        max_time_steps=max_time, augment=True)
    val_dataset = HiddenStateDataset(val_feats, val_labels,
                                      max_time_steps=max_time, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn, num_workers=0)

    # Class weights
    train_labels_np = np.array(train_labels)
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels_np),
                                         y=train_labels_np)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print(f"Class weights: {dict(zip(label_names, class_weights.round(3)))}")

    # ---- Phase 2: Train classifier ----
    device = torch.device("cpu")
    model = EmotionClassifier(hidden_dim=768, num_classes=len(label_names)).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Classifier parameters: {trainable:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                   weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))

    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    train_losses, val_losses, val_accs = [], [], []

    print(f"\nStarting training for up to {EPOCHS} epochs...")
    print("-" * 60)

    for epoch in range(EPOCHS):
        # Train
        model.train()
        epoch_loss = 0
        num_batches = 0

        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(feats)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        train_losses.append(epoch_loss / num_batches)

        # Validate
        model.eval()
        val_loss = 0
        preds, trues = [], []
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                outputs = model(feats)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                trues.extend(labels.cpu().numpy())

        val_losses.append(val_loss / len(val_loader))
        acc = accuracy_score(trues, preds)
        val_accs.append(acc)
        scheduler.step()

        if acc > best_val_acc:
            best_val_acc = acc
            patience_counter = 0
            torch.save(model.state_dict(), OUTPUT_MODEL_DIR / "emotion_classifier.pth")
            print(f"  ** New best: Val Acc = {acc:.4f}")
        else:
            patience_counter += 1

        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train Loss: {train_losses[-1]:.4f} | "
              f"Val Loss: {val_losses[-1]:.4f} | "
              f"Val Acc: {acc:.4f} | "
              f"LR: {lr:.6f} | "
              f"Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}.")
            break

    # Save final
    torch.save(model.state_dict(), OUTPUT_MODEL_DIR / "emotion_classifier_final.pth")

    # Save model info
    model_info = {
        "input_type": "wav2vec2_hidden_states",
        "target_sr": TARGET_SR,
        "max_audio_len": MAX_AUDIO_LEN,
        "num_classes": len(label_names),
        "class_names": list(label_names),
        "architecture": "AttentionPooling + Classifier (on frozen Wav2Vec2 features)",
        "feature_extractor": "facebook/wav2vec2-base (frozen)",
        "best_val_accuracy": round(best_val_acc, 4),
        "test_accuracy": None,
        "classifier_params": trainable
    }
    with open(OUTPUT_MODEL_DIR / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=4)

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(train_losses, label="Train Loss", linewidth=2)
    axes[0].plot(val_losses, label="Val Loss", linewidth=2)
    axes[0].set_title("Loss Curves", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(val_accs, label="Val Accuracy", color="green", linewidth=2)
    axes[1].axhline(y=best_val_acc, color='red', linestyle='--',
                    label=f"Best: {best_val_acc:.4f}")
    axes[1].set_title(f"Validation Accuracy (Best: {best_val_acc:.4f})",
                      fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if len(train_losses) == len(val_losses):
        gap = [v - t for t, v in zip(train_losses, val_losses)]
        axes[2].plot(gap, label="Val - Train Loss", color="orange", linewidth=2)
        axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[2].set_title("Overfitting Indicator", fontsize=14, fontweight='bold')
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Loss Gap")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_curves.png", dpi=200)
    plt.close()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Model saved -> {OUTPUT_MODEL_DIR / 'emotion_classifier.pth'}")
    print(f"\nNext: Run test_emotion_classifier.py then app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
