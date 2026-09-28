import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split  # Add this import
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
FEATURES_FILE = "../Extracted Features/combined_wav2vec_features.csv"
MODEL_PATH = "../models/emotion_classifier/emotion_classifier.pth"
RESULTS_DIR = "../results/"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Dataset Class
class EmotionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Model Definition
class EmotionClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(EmotionClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# Load Features
logging.info("Loading features for testing...")
df = pd.read_csv(FEATURES_FILE)
df = df.drop(columns=['Dataset'], errors='ignore')  # Drop dataset column if present
df.dropna(inplace=True)  # Remove NaN values

X = df.iloc[:, :-1].values  # Features
y = pd.factorize(df['label'])[0]  # Convert labels to integer encoding
label_names = pd.factorize(df['label'])[1]  # Original labels for confusion matrix

# Split Data for Testing (already split in training script)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
test_dataset = EmotionDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionClassifier(input_dim=X.shape[1], num_classes=len(set(y))).to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Testing Loop
y_true, y_pred = [], []
logging.info("Testing the model...")
with torch.no_grad():
    for features, labels in test_loader:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        preds = torch.argmax(outputs, dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Compute Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
report = classification_report(y_true, y_pred, target_names=label_names)

logging.info("\n--- Testing Results ---")
logging.info(f"Accuracy: {accuracy * 100:.2f}%")
logging.info(report)

# Save Metrics
metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1
}
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv(os.path.join(RESULTS_DIR, "test_metrics.csv"), index=False)

# Save Predictions
predictions_df = pd.DataFrame({
    "True Label": [label_names[i] for i in y_true],
    "Predicted Label": [label_names[i] for i in y_pred]
})
predictions_df.to_csv(os.path.join(RESULTS_DIR, "test_predictions.csv"), index=False)

# Confusion Matrix Plot
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
plt.show()

# Bar Plot for Precision, Recall, and F1-Score
report_dict = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)
metrics_df = pd.DataFrame(report_dict).transpose().iloc[:-3, :]  # Exclude "accuracy", "macro avg", "weighted avg"
metrics_df[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(12, 6))
plt.title("Class-wise Precision, Recall, and F1-Score")
plt.ylabel("Score")
plt.xlabel("Classes")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "class_metrics.png"))
plt.show()

# Pie Chart for Accuracy
plt.figure(figsize=(6, 6))
plt.pie([accuracy, 1 - accuracy], labels=["Correct Predictions", "Incorrect Predictions"], autopct='%1.1f%%', startangle=140)
plt.title("Overall Accuracy")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "accuracy_pie_chart.png"))
plt.show()

logging.info(f"Test metrics, predictions, and plots saved to {RESULTS_DIR}")