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

    def forward(self, hidden_states):
        # hidden_states: [batch, time, 768]
        pooled = self.attention_pool(hidden_states)  # [batch, 768]
        pooled = self.layer_norm(pooled)
        logits = self.classifier(pooled)
        return logits


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

# Convert to tensors and pad to same length
logging.info("Converting features to tensors...")
# Find max sequence length
max_len = max(f.shape[0] for f in test_features)
logging.info(f"Max sequence length: {max_len}")

# Pad all features to max_len
padded_features = []
for f in test_features:
    feature_tensor = torch.tensor(f, dtype=torch.float32)
    if feature_tensor.shape[0] < max_len:
        # Pad with zeros
        padding = torch.zeros(max_len - feature_tensor.shape[0], feature_tensor.shape[1])
        feature_tensor = torch.cat([feature_tensor, padding], dim=0)
    padded_features.append(feature_tensor)

test_features_tensor = torch.stack(padded_features)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

# Create dataset and loader using cached features
test_dataset = TensorDataset(test_features_tensor, test_labels_tensor)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# ==================== Load Model ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionClassifier(hidden_dim=768, num_classes=num_classes).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
model.eval()
logging.info(f"Model loaded from: {MODEL_PATH} (device: {device})")

# ==================== Testing ====================
y_true, y_pred, all_probs = [], [], []
logging.info("Running evaluation...")

with torch.no_grad():
    for features, labels in tqdm(test_loader, desc="Evaluating"):
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
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
