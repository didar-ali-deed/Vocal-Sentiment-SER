# app.py - Speech Emotion Recognition Web App
# Uses classifier trained on cached Wav2Vec2 features

from flask import Flask, request, render_template, jsonify
from pathlib import Path
from werkzeug.utils import secure_filename
import os
import uuid
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import json
import time
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== Configuration ====================
CONFIDENCE_THRESHOLD = 0.40
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}
MAX_AUDIO_DURATION = 5   # seconds — must match MAX_AUDIO_LEN in training (5s)
TARGET_SR = 16000

# ==================== Paths ====================
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models" / "emotion_classifier"
MODEL_PATH = MODEL_DIR / "emotion_classifier.pth"
LABEL_NAMES_PATH = MODEL_DIR / "label_names.npy"
MODEL_INFO_PATH = MODEL_DIR / "model_info.json"

UPLOAD_FOLDER = BASE_DIR / "deployment" / "uploads"
RESULTS_DIR = BASE_DIR / "results"
PREDICTIONS_FILE = RESULTS_DIR / "inference_predictions.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB

# ==================== Check Required Files ====================
required = [MODEL_PATH, LABEL_NAMES_PATH, MODEL_INFO_PATH]
missing = [str(f) for f in required if not f.exists()]
if missing:
    raise FileNotFoundError(f"Missing required files: {missing}")

# ==================== Load Model Info & Labels ====================
label_names = np.load(LABEL_NAMES_PATH, allow_pickle=True)
with open(MODEL_INFO_PATH, "r") as f:
    model_info = json.load(f)

num_classes = len(label_names)
logger.info(f"Loaded {num_classes} emotion classes: {list(label_names)}")


# ==================== Model Architecture (must match training) ====================
class TransformerSERHead(nn.Module):
    """
    Must exactly match train_emotion_classifier.py.
    Architecture hyper-params are read from model_info.json at startup.
    Uses LayerNorm only — no BatchNorm — so it works correctly at batch size 1.
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


# ==================== Load Models ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Wav2Vec2 for feature extraction (frozen backbone)
logger.info("Loading Wav2Vec2 feature extractor...")
wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec2_model     = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
wav2vec2_model.eval()

# Load trained TransformerSERHead — read arch params from model_info.json
logger.info("Loading emotion classifier...")
emotion_classifier = TransformerSERHead(
    input_dim=768,
    d_model=model_info.get("d_model", 256),
    num_heads=model_info.get("num_heads", 8),
    num_layers=model_info.get("num_layers", 2),
    dim_ff=model_info.get("dim_ff", 512),
    num_classes=num_classes,
    dropout=model_info.get("dropout", 0.25),
).to(device)
emotion_classifier.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=False)
)
emotion_classifier.eval()

logger.info(f"Models loaded on {device}")
logger.info(f"Emotions: {list(label_names)}")


# ==================== Audio Preprocessing ====================
def preprocess_audio(file_path, target_sr=TARGET_SR):
    """Load, normalize, and trim audio. Must match training pipeline."""
    audio, sr = librosa.load(file_path, sr=target_sr)

    if len(audio) < target_sr * 0.1:
        raise ValueError("Audio is too short (< 0.1s).")

    duration = len(audio) / target_sr
    if duration > MAX_AUDIO_DURATION:
        logger.warning(f"Audio too long ({duration:.1f}s), truncating to {MAX_AUDIO_DURATION}s")
        audio = audio[:target_sr * MAX_AUDIO_DURATION]

    audio = librosa.util.normalize(audio)
    audio, _ = librosa.effects.trim(audio, top_db=25)

    if len(audio) < target_sr * 0.1:
        raise ValueError("Audio is mostly silence.")

    return audio


# ==================== Prediction ====================
def predict_emotion(file_path):
    """
    Two-stage pipeline matching the training pipeline exactly:
      1. Preprocess audio → Wav2Vec2Processor (zero-mean / unit-var normalisation)
      2. Extract frozen Wav2Vec2 hidden states
      3. Classify via TransformerSERHead (no padding mask needed for single sample)
    """
    audio = preprocess_audio(file_path)

    # Use the processor for the same normalisation applied during training
    inputs = wav2vec2_processor(
        audio, sampling_rate=TARGET_SR, return_tensors="pt"
    )
    input_values = inputs.input_values.to(device)   # [1, T]

    with torch.no_grad():
        # Stage 1: frozen Wav2Vec2 feature extraction
        hidden_states = wav2vec2_model(input_values).last_hidden_state  # [1, T, 768]

        # Stage 2: transformer classifier — single sample, no padding needed
        logits = emotion_classifier(hidden_states, src_key_padding_mask=None)
        probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_idx = int(torch.argmax(logits, dim=1).cpu().numpy()[0])

    predicted_emotion = str(label_names[predicted_idx])
    confidence = float(probabilities[predicted_idx])
    prob_dict = {str(label_names[i]): float(probabilities[i]) for i in range(len(label_names))}
    is_uncertain = confidence < CONFIDENCE_THRESHOLD

    return predicted_emotion, prob_dict, confidence, is_uncertain


# ==================== Save Prediction History ====================
def save_prediction(filename, emotion, probabilities, confidence, is_uncertain=False):
    entry = {
        "filename": filename,
        "predicted_emotion": emotion,
        "confidence": round(confidence, 4),
        "is_uncertain": is_uncertain,
        "probabilities": {k: round(v, 4) for k, v in probabilities.items()},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    history = []
    if PREDICTIONS_FILE.exists():
        try:
            with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except (json.JSONDecodeError, IOError):
            history = []

    history.append(entry)
    history = history[-50:]

    with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


# ==================== Routes ====================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "No file selected"}), 400

    safe_name = secure_filename(file.filename)
    if not safe_name:
        return jsonify({"status": "error", "message": "Invalid filename"}), 400

    ext = os.path.splitext(safe_name)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return jsonify({
            "status": "error",
            "message": f"Invalid format. Allowed: {', '.join(SUPPORTED_EXTENSIONS)}"
        }), 400

    # Unique name prevents path-traversal and concurrent-upload collisions
    temp_path = UPLOAD_FOLDER / f"{uuid.uuid4().hex}_{safe_name}"
    file.save(temp_path)

    try:
        emotion, probs, confidence, is_uncertain = predict_emotion(temp_path)
        save_prediction(safe_name, emotion, probs, confidence, is_uncertain)

        logger.info(f"{'[UNCERTAIN] ' if is_uncertain else ''}"
                    f"{safe_name} -> {emotion} ({confidence:.2%})")

        return jsonify({
            "status": "success",
            "data": {
                "predicted_emotion": emotion,
                "confidence": round(confidence, 4),
                "is_uncertain": is_uncertain,
                "probabilities": {k: round(v, 4) for k, v in probs.items()}
            }
        })

    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

    finally:
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except OSError:
                pass


@app.route("/predictions")
def get_predictions():
    if PREDICTIONS_FILE.exists():
        try:
            with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return jsonify({"status": "success", "data": data[::-1]})
        except (json.JSONDecodeError, IOError):
            pass
    return jsonify({"status": "success", "data": []})


@app.route("/model-info")
def get_model_info():
    return jsonify({
        "status": "success",
        "data": {
            "model_info": model_info,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "supported_extensions": list(SUPPORTED_EXTENSIONS),
            "device": str(device)
        }
    })


if __name__ == "__main__":
    print("=" * 60)
    print("Speech Emotion Recognition App")
    print(f"Architecture: {model_info.get('architecture', 'Wav2Vec2 + Classifier')}")
    print(f"Emotions: {list(label_names)}")
    print(f"Device: {device}")
    print(f"URL: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=False, host="0.0.0.0", port=5000)
