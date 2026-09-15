import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import logging
import json
from pathlib import Path

from model import EmotionClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==================== Paths ====================
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "emotion_classifier" / "emotion_classifier.pth"
TEST_DATA_NPY = BASE_DIR / "models" / "emotion_classifier" / "test_data.npy"
MODEL_INFO_PATH = BASE_DIR / "models" / "emotion_classifier" / "model_info.json"
RESULTS_DIR = BASE_DIR / "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ==================== Model & Dataset ====================
class EmotionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ==================== Load Test Data ====================
logging.info("Loading test data...")
test_data = np.load(TEST_DATA_NPY, allow_pickle=True).item()
X_test = test_data['X_test']
y_test = test_data['y_test']
label_names = test_data['label_names']

input_dim = X_test.shape[1]
num_classes = len(label_names)

logging.info(f"Test samples: {len(X_test)} | Classes: {num_classes} -> {list(label_names)}")

test_dataset = EmotionDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ==================== Load Model ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()
logging.info(f"Model loaded from: {MODEL_PATH} (device: {device})")

# ==================== Testing ====================
y_true, y_pred, all_probs = [], [], []
with torch.no_grad():
    for features, labels in test_loader:
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

class_ids = np.arange(num_classes)
report = classification_report(
    y_true, y_pred, labels=class_ids, target_names=label_names, digits=4, zero_division=0
)
logging.info("\n" + "=" * 50)
logging.info("FINAL TEST RESULTS (Held-out 30%)")
logging.info(f"Accuracy:  {accuracy:.4f}")
logging.info(f"Precision: {precision:.4f}")
logging.info(f"Recall:    {recall:.4f}")
logging.info(f"F1-Score:  {f1:.4f}")
logging.info("\nDetailed Report:\n" + report)

# Save metrics
metrics = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1}
pd.DataFrame([metrics]).to_csv(RESULTS_DIR / "final_test_metrics.csv", index=False)

# Update model_info.json with test accuracy
if MODEL_INFO_PATH.exists():
    with open(MODEL_INFO_PATH, "r") as f:
        model_info = json.load(f)
    model_info["test_accuracy"] = round(accuracy, 4)
    with open(MODEL_INFO_PATH, "w") as f:
        json.dump(model_info, f, indent=4)
    logging.info("Updated model_info.json with test accuracy.")

# ==================== Confusion Matrix ====================
cm = confusion_matrix(y_true, y_pred, labels=class_ids)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=label_names, yticklabels=label_names
)
plt.title(f"Confusion Matrix - Test Set (Acc: {accuracy:.2%})")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "final_confusion_matrix.png", dpi=200)
plt.close()

# ==================== Normalized Confusion Matrix ====================
cm_normalized = np.divide(
    cm.astype(float),
    cm.sum(axis=1, keepdims=True),
    out=np.zeros_like(cm, dtype=float),
    where=cm.sum(axis=1, keepdims=True) != 0,
)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm_normalized, annot=True, fmt=".2f", cmap="YlOrRd",
    xticklabels=label_names, yticklabels=label_names
)
plt.title("Normalized Confusion Matrix (row-wise)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "final_confusion_matrix_normalized.png", dpi=200)
plt.close()

# ==================== Class-wise Bar Plot ====================
# FIX: Original had .iloc[:-3, :1:3] which only selects 1 column (precision).
# Correct approach: explicitly extract precision, recall, f1-score per class.
report_dict = classification_report(
    y_true, y_pred, labels=class_ids, target_names=label_names,
    output_dict=True, zero_division=0
)
class_names = [name for name in report_dict if name not in ('accuracy', 'macro avg', 'weighted avg')]
class_metrics = pd.DataFrame({
    name: {
        'precision': report_dict[name]['precision'],
        'recall': report_dict[name]['recall'],
        'f1-score': report_dict[name]['f1-score']
    }
    for name in class_names
}).T

class_metrics.plot(kind='bar', figsize=(14, 6), width=0.8)
plt.title("Per-class Precision, Recall, F1-Score (Test Set)")
plt.ylabel("Score")
plt.xlabel("Emotion")
plt.ylim(0, 1.05)
plt.legend(title="Metric", loc="lower right")
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "final_class_metrics.png", dpi=200)
plt.close()

# ==================== Confidence Analysis ====================
correct_mask = y_true == y_pred
correct_confidences = all_probs[np.arange(len(y_pred)), y_pred][correct_mask]
wrong_confidences = all_probs[np.arange(len(y_pred)), y_pred][~correct_mask]

plt.figure(figsize=(10, 6))
plt.hist(correct_confidences, bins=30, alpha=0.7, label=f"Correct ({len(correct_confidences)})", color='green')
plt.hist(wrong_confidences, bins=30, alpha=0.7, label=f"Wrong ({len(wrong_confidences)})", color='red')
plt.title("Prediction Confidence Distribution")
plt.xlabel("Confidence")
plt.ylabel("Count")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "confidence_distribution.png", dpi=200)
plt.close()

# ==================== Most Confused Pairs ====================
logging.info("\nMost confused emotion pairs:")
for i in range(num_classes):
    for j in range(num_classes):
        if i != j and cm[i, j] > 0:
            pct = cm[i, j] / cm[i].sum() * 100
            if pct > 5:  # only show significant confusions
                logging.info(f"  {label_names[i]} -> {label_names[j]}: "
                           f"{cm[i, j]} samples ({pct:.1f}%)")

logging.info(f"\nAll final results saved to {RESULTS_DIR}")
