import os
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Paths are anchored to the project, regardless of the working directory.
BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_FILE = BASE_DIR / "extracted_features" / "combined_wav2vec_features.csv"
OUTPUT_MODEL_DIR = BASE_DIR / "models" / "emotion_classifier"
RESULTS_DIR = BASE_DIR / "results"
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Hyperparameters
EPOCHS = 50
LEARNING_RATE = 0.0001
BATCH_SIZE = 32

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

# Load Data
df = pd.read_csv(FEATURES_FILE)
df = df.drop(columns=['Dataset'], errors='ignore')  # Drop dataset column if present
df.dropna(inplace=True)  # Remove NaN values

X = df.iloc[:, :-1].values  # Features
y = pd.factorize(df['label'])[0]  # Encode labels as integers

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
train_dataset = EmotionDataset(X_train, y_train)
val_dataset = EmotionDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionClassifier(input_dim=X.shape[1], num_classes=len(set(y))).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
train_losses, val_losses, accuracies = [], [], []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    y_train_true, y_train_pred = [], []

    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        y_train_true.extend(labels.cpu().numpy())
        y_train_pred.extend(preds.cpu().numpy())

    train_losses.append(epoch_loss / len(train_loader))

    # Validation
    model.eval()
    val_loss, y_val_true, y_val_pred = 0, [], []
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            val_loss += criterion(outputs, labels).item()
            preds = torch.argmax(outputs, dim=1)
            y_val_true.extend(labels.cpu().numpy())
            y_val_pred.extend(preds.cpu().numpy())

    val_losses.append(val_loss / len(val_loader))
    accuracies.append(accuracy_score(y_val_true, y_val_pred))
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Accuracy: {accuracies[-1]:.4f}")

# Save Model
torch.save(model.state_dict(), os.path.join(OUTPUT_MODEL_DIR, "emotion_classifier.pth"))
print("Model saved successfully.")

# Save Metrics & Plot
plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss")
plt.plot(range(1, EPOCHS + 1), val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid()
plt.savefig(os.path.join(RESULTS_DIR, "training_loss_plot.png"))
plt.show()
