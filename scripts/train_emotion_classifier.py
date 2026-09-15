# train_emotion_classifier.py
# Improved: feature normalization, LabelEncoder, LR scheduler,
#           early stopping, class weighting, proper validation split
import os
import random
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import numpy as np
import json
import joblib
from pathlib import Path

from model import EmotionClassifier

# ==================== Paths ====================
BASE_DIR = Path(__file__).parent.parent
FEATURES_FILE = BASE_DIR / "Extracted Features" / "combined_wav2vec_features.csv"
OUTPUT_MODEL_DIR = BASE_DIR / "models" / "emotion_classifier"
RESULTS_DIR = BASE_DIR / "results"

os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==================== Hyperparameters ====================
EPOCHS = 150
LEARNING_RATE = 0.001
BATCH_SIZE = 32
VALIDATION_FRACTION = 0.15          # increased from 0.1
PATIENCE = 20                       # early stopping patience
WEIGHT_DECAY = 1e-4                 # L2 regularization

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ==================== Dataset Class ====================
class EmotionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# ==================== Load & Prepare Data ====================
print("Loading features...")
df = pd.read_csv(FEATURES_FILE)
df = df.drop(columns=['Dataset'], errors='ignore')
df.dropna(inplace=True)

X = df.iloc[:, :-1].values.astype(np.float32)
input_dim = X.shape[1]

# --- FIX: Use LabelEncoder with sorted classes (deterministic) ---
le = LabelEncoder()
y = le.fit_transform(df['label'].values)
label_names = le.classes_                     # alphabetically sorted, always consistent

print(f"Total samples: {len(X)} | Classes: {len(label_names)} -> {list(label_names)}")

# --- Held-out 30 % test set (never used in training) ---
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --- Validation from training ---
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=VALIDATION_FRACTION, random_state=42, stratify=y_train_val
)

# --- FIX: Feature normalization (StandardScaler) ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# Save scaler for inference (critical for app.py)
joblib.dump(scaler, OUTPUT_MODEL_DIR / "feature_scaler.pkl")
print("Feature scaler saved.")

# Save test set + labels for final evaluation
np.save(OUTPUT_MODEL_DIR / "test_data.npy", {
    'X_test': X_test,
    'y_test': y_test,
    'label_names': label_names
})

# Save LabelEncoder for Flask app (replaces label_names.npy)
joblib.dump(le, OUTPUT_MODEL_DIR / "label_encoder.pkl")
np.save(OUTPUT_MODEL_DIR / "label_names.npy", np.asarray(label_names, dtype=str))

# Save model info
model_info = {
    "input_dim": input_dim,
    "num_classes": len(label_names),
    "class_names": list(label_names),
    "architecture": f"MLP ({input_dim} -> 256 -> 128 -> 64 -> N) with BatchNorm",
    "feature_extractor": "facebook/wav2vec2-base (frozen, mean-pooled)",
    "feature_scaler": "StandardScaler (saved as feature_scaler.pkl)",
    "test_accuracy": None
}
with open(OUTPUT_MODEL_DIR / "model_info.json", "w") as f:
    json.dump(model_info, f, indent=4)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ==================== DataLoaders ====================
train_loader = DataLoader(EmotionDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(EmotionDataset(X_val, y_val),     batch_size=BATCH_SIZE, shuffle=False)

# ==================== Class Weights (handle imbalance) ====================
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
print(f"Class weights: {dict(zip(label_names, class_weights.round(3)))}")

# ==================== Training Setup ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionClassifier(input_dim=input_dim, num_classes=len(label_names)).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# --- FIX: Learning rate scheduler ---
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=7, min_lr=1e-6
)

# ==================== Training Loop with Early Stopping ====================
best_val_acc = 0.0
patience_counter = 0
train_losses, val_losses, val_accs = [], [], []

print("Starting training...")
for epoch in range(EPOCHS):
    # --- Train ---
    model.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
        optimizer.step()
        epoch_loss += loss.item()
    train_losses.append(epoch_loss / len(train_loader))

    # --- Validate ---
    model.eval()
    val_loss = 0
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            loss = criterion(outputs, yb)
            val_loss += loss.item()
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            trues.extend(yb.cpu().numpy())
    val_losses.append(val_loss / len(val_loader))
    acc = accuracy_score(trues, preds)
    val_accs.append(acc)

    # Step LR scheduler
    scheduler.step(acc)

    # Save best model
    if acc > best_val_acc:
        best_val_acc = acc
        patience_counter = 0
        torch.save(model.state_dict(), OUTPUT_MODEL_DIR / "emotion_classifier.pth")
        print(f"  -> New best model saved! Val Acc: {acc:.4f}")
    else:
        patience_counter += 1

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
          f"Train Loss: {train_losses[-1]:.4f} | "
          f"Val Loss: {val_losses[-1]:.4f} | "
          f"Val Acc: {acc:.4f} | "
          f"LR: {current_lr:.6f} | "
          f"Patience: {patience_counter}/{PATIENCE}")

    # --- Early stopping ---
    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping triggered at epoch {epoch+1}.")
        break

# ==================== Final Save & Plot ====================
torch.save(model.state_dict(), OUTPUT_MODEL_DIR / "emotion_classifier_final.pth")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss curves
axes[0].plot(train_losses, label="Train Loss")
axes[0].plot(val_losses, label="Val Loss")
axes[0].set_title("Loss Curves")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Validation accuracy
axes[1].plot(val_accs, label="Val Accuracy", color="green")
axes[1].axhline(y=best_val_acc, color='red', linestyle='--', label=f"Best: {best_val_acc:.4f}")
axes[1].set_title(f"Validation Accuracy (Best: {best_val_acc:.4f})")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Train vs Val loss gap (overfitting indicator)
if len(train_losses) == len(val_losses):
    gap = [v - t for t, v in zip(train_losses, val_losses)]
    axes[2].plot(gap, label="Val Loss - Train Loss", color="orange")
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[2].set_title("Overfitting Indicator")
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
print(f"Model saved        -> {OUTPUT_MODEL_DIR / 'emotion_classifier.pth'}")
print(f"Label encoder saved -> {OUTPUT_MODEL_DIR / 'label_encoder.pkl'}")
print(f"Feature scaler saved-> {OUTPUT_MODEL_DIR / 'feature_scaler.pkl'}")
print(f"Label names saved   -> {OUTPUT_MODEL_DIR / 'label_names.npy'}")
print(f"Test set saved      -> {OUTPUT_MODEL_DIR / 'test_data.npy'}")
print(f"Ready for deployment with app.py")
print("=" * 60)
