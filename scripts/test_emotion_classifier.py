# test_emotion_classifier.py
# Evaluates the trained emotion classifier on the held-out test set
# Works with cached Wav2Vec2 features (matching the training approach)
# Generates: confusion matrices, per-class metrics, confidence analysis

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
import json
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==================== Paths ====================
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models" / "emotion_classifier"
MODEL_PATH = MODEL_DIR / "emotion_classifier.pth"
TEST_DATA_NPY = MODEL_DIR / "test_data.npy"
MODEL_INFO_PATH = MODEL_DIR / "model_info.json"
RESULTS_DIR = BASE_DIR / "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ==================== Model Architecture (must match training) ====================
class TransformerSERHead(nn.Module):
    """
    Must match train_emotion_classifier.py exactly.
    Architecture params are loaded from model_info.json.
    """

    def __init__(self, input_dim=768, d_model=256, num_heads=8,
                 num_layers=2, dim_ff=512, num_classes=8, dropout=0.25):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_drop = nn.Dropout(dropout)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model),
        )
        self.cls_drop = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x, src_key_padding_mask=None):
        B = x.size(0)
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.input_drop(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        if src_key_padding_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            src_key_padding_mask = torch.cat([cls_mask, src_key_padding_mask], dim=1)

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        cls_out = x[:, 0]
        cls_out = self.cls_drop(cls_out)
        return self.classifier(cls_out)


# ==================== Load Test Data ====================
logging.info("Loading test data...")
test_data = np.load(TEST_DATA_NPY, allow_pickle=True).item()
test_features = test_data['features']  # Cached Wav2Vec2 features
test_labels = test_data['labels']
label_names = test_data['label_names']

num_classes = len(label_names)
logging.info(f"Test samples: {len(test_features)} | Classes: {num_classes} -> {list(label_names)}")

# Load model info
with open(MODEL_INFO_PATH, "r") as f:
    model_info = json.load(f)

# Pad variable-length features and build padding masks
logging.info("Padding features and building masks...")
max_len = max(f.shape[0] for f in test_features)
logging.info(f"Max sequence length: {max_len}")

padded_features, padding_masks = [], []
for f in test_features:
    t = torch.tensor(f, dtype=torch.float32)
    T = t.shape[0]
    mask = torch.ones(max_len, dtype=torch.bool)    # True = padding
    if T < max_len:
        pad = torch.zeros(max_len - T, t.shape[1])
        t = torch.cat([t, pad], dim=0)
    mask[:T] = False    # real data
    padded_features.append(t)
    padding_masks.append(mask)

test_features_tensor = torch.stack(padded_features)
test_masks_tensor    = torch.stack(padding_masks)
test_labels_tensor   = torch.tensor(test_labels, dtype=torch.long)

from torch.utils.data import TensorDataset
test_dataset = TensorDataset(test_features_tensor, test_labels_tensor, test_masks_tensor)
test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False)

# ==================== Load Model ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerSERHead(
    input_dim=768,
    d_model=model_info.get("d_model", 256),
    num_heads=model_info.get("num_heads", 8),
    num_layers=model_info.get("num_layers", 2),
    dim_ff=model_info.get("dim_ff", 512),
    num_classes=num_classes,
    dropout=model_info.get("dropout", 0.25),
).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
model.eval()
logging.info(f"Model loaded from: {MODEL_PATH} (device: {device})")

# ==================== Testing ====================
y_true, y_pred, all_probs = [], [], []
logging.info("Running evaluation...")

with torch.no_grad():
    for features, labels, masks in tqdm(test_loader, desc="Evaluating"):
        features = features.to(device)
        labels   = labels.to(device)
        masks    = masks.to(device)
        outputs  = model(features, src_key_padding_mask=masks)
        probs    = torch.softmax(outputs, dim=1)
        preds    = torch.argmax(outputs, dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
all_probs = np.array(all_probs)

# ==================== Results ====================
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

report = classification_report(y_true, y_pred, target_names=label_names, digits=4)
logging.info("\n" + "=" * 50)
logging.info("FINAL TEST RESULTS (Held-out 15%)")
logging.info(f"Accuracy:  {accuracy:.4f}")
logging.info(f"Precision: {precision:.4f}")
logging.info(f"Recall:    {recall:.4f}")
logging.info(f"F1-Score:  {f1:.4f}")
logging.info("\nDetailed Report:\n" + report)

# Save metrics
metrics = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1}
pd.DataFrame([metrics]).to_csv(RESULTS_DIR / "final_test_metrics.csv", index=False)

# Update model_info.json
model_info["test_accuracy"] = round(accuracy, 4)
model_info["test_f1"] = round(f1, 4)
with open(MODEL_INFO_PATH, "w") as f:
    json.dump(model_info, f, indent=4)
logging.info("Updated model_info.json with test metrics.")

# ==================== Confusion Matrix ====================
cm = confusion_matrix(y_true, y_pred)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Raw counts
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_names, yticklabels=label_names, ax=axes[0])
axes[0].set_title(f"Confusion Matrix (Acc: {accuracy:.2%})", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Predicted", fontsize=12)
axes[0].set_ylabel("True", fontsize=12)

# Normalized
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
            xticklabels=label_names, yticklabels=label_names, ax=axes[1])
axes[1].set_title("Normalized Confusion Matrix", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Predicted", fontsize=12)
axes[1].set_ylabel("True", fontsize=12)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "final_confusion_matrix.png", dpi=200)
plt.close()
logging.info(f"Confusion matrix saved to {RESULTS_DIR / 'final_confusion_matrix.png'}")

# ==================== Per-class Metrics Bar Chart ====================
report_dict = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)
class_names = [name for name in report_dict if name not in ('accuracy', 'macro avg', 'weighted avg')]
class_metrics = pd.DataFrame({
    name: {
        'precision': report_dict[name]['precision'],
        'recall': report_dict[name]['recall'],
        'f1-score': report_dict[name]['f1-score']
    }
    for name in class_names
}).T

fig, ax = plt.subplots(figsize=(14, 6))
class_metrics.plot(kind='bar', ax=ax, width=0.8, color=['#42a5f5', '#66bb6a', '#ef5350'])
ax.set_title("Per-class Precision, Recall, F1-Score", fontsize=14, fontweight='bold')
ax.set_ylabel("Score", fontsize=12)
ax.set_xlabel("Emotion", fontsize=12)
ax.set_ylim(0, 1.05)
ax.legend(title="Metric", loc="lower right")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "final_class_metrics.png", dpi=200)
plt.close()
logging.info(f"Per-class metrics saved to {RESULTS_DIR / 'final_class_metrics.png'}")

# ==================== Confidence Analysis ====================
correct_mask = y_true == y_pred
correct_conf = all_probs[np.arange(len(y_pred)), y_pred][correct_mask]
wrong_conf = all_probs[np.arange(len(y_pred)), y_pred][~correct_mask]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Histogram
axes[0].hist(correct_conf, bins=30, alpha=0.7,
             label=f"Correct ({len(correct_conf)})", color='#66bb6a')
axes[0].hist(wrong_conf, bins=30, alpha=0.7,
             label=f"Wrong ({len(wrong_conf)})", color='#ef5350')
axes[0].set_title("Confidence Distribution", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Confidence")
axes[0].set_ylabel("Count")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Per-class accuracy
per_class_acc = []
for i, name in enumerate(label_names):
    mask = y_true == i
    if mask.sum() > 0:
        per_class_acc.append(accuracy_score(y_true[mask], y_pred[mask]))
    else:
        per_class_acc.append(0)

bars = axes[1].bar(label_names, per_class_acc, color='#42a5f5')
axes[1].set_title("Per-class Accuracy", fontsize=14, fontweight='bold')
axes[1].set_ylabel("Accuracy")
axes[1].set_ylim(0, 1.05)
axes[1].set_xticklabels(label_names, rotation=45, ha='right')
axes[1].grid(axis='y', alpha=0.3)
for bar, acc in zip(bars, per_class_acc):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{acc:.1%}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "confidence_distribution.png", dpi=200)
plt.close()
logging.info(f"Confidence analysis saved to {RESULTS_DIR / 'confidence_distribution.png'}")

# ==================== Most Confused Pairs ====================
logging.info("\nMost confused emotion pairs:")
for i in range(num_classes):
    for j in range(num_classes):
        if i != j and cm[i, j] > 0:
            pct = cm[i, j] / cm[i].sum() * 100
            if pct > 5:
                logging.info(f"  {label_names[i]} -> {label_names[j]}: "
                             f"{cm[i, j]} samples ({pct:.1f}%)")

logging.info(f"\nAll results saved to {RESULTS_DIR}")
logging.info("=" * 50)
