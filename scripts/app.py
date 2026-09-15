"""Flask inference app for the paper-linked speech emotion model."""

from functools import lru_cache
from flask import Flask, request, render_template, jsonify
from pathlib import Path
import json
import logging
import os
import tempfile
import threading
import time
from uuid import uuid4

from werkzeug.utils import secure_filename
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2Processor, Wav2Vec2Model

try:
    from model import load_artifacts
except ImportError:
    from scripts.model import load_artifacts

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "emotion_classifier"
UPLOAD_FOLDER = BASE_DIR / "deployment" / "uploads"
RESULTS_DIR = BASE_DIR / "results"
PREDICTIONS_FILE = RESULTS_DIR / "inference_predictions.json"

CONFIDENCE_THRESHOLD = 0.40
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}
TARGET_SAMPLE_RATE = 16_000
MAX_AUDIO_DURATION = 30
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
history_lock = threading.Lock()

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


@lru_cache(maxsize=1)
def get_runtime():
    """Load large model assets once, on the first request that needs them."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading Wav2Vec2 and classifier on %s", device)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
    wav2vec_model.eval()

    classifier, label_names, model_info, scaler, input_dim = load_artifacts(
        MODEL_DIR, device
    )
    if input_dim != scaler.n_features_in_:
        raise ValueError(
            f"Artifact mismatch: model expects {input_dim} features, "
            f"but scaler expects {scaler.n_features_in_}."
        )
    logger.info("Loaded classes: %s", list(label_names))
    return {
        "device": device,
        "processor": processor,
        "wav2vec_model": wav2vec_model,
        "classifier": classifier,
        "label_names": label_names,
        "model_info": model_info,
        "scaler": scaler,
    }


# ==================== Audio Preprocessing ====================
def preprocess_audio(file_path, target_sr=TARGET_SAMPLE_RATE):
    """Load, normalize, trim, and validate an audio file."""
    try:
        audio, _ = librosa.load(file_path, sr=target_sr, mono=True)
    except Exception as error:
        logger.warning("Unable to decode uploaded audio %s: %s", file_path, error)
        raise ValueError(
            "This audio file could not be decoded. Try exporting it again as a "
            "standard WAV, MP3, FLAC, or OGG file."
        ) from error
    if len(audio) < target_sr * 0.1:
        raise ValueError("Audio is too short; upload a file longer than 0.1 seconds.")

    duration = len(audio) / target_sr
    if duration > MAX_AUDIO_DURATION:
        raise ValueError(
            f"Audio is too long ({duration:.1f}s). Maximum duration is "
            f"{MAX_AUDIO_DURATION} seconds."
        )

    audio = librosa.util.normalize(audio)
    audio, _ = librosa.effects.trim(audio, top_db=25)
    if len(audio) < target_sr * 0.1:
        raise ValueError("Audio is mostly silence; upload a recording with audible speech.")
    return audio


# ==================== Feature Extraction ====================
def extract_wav2vec_features(file_path):
    """Extract Wav2Vec2 features with preprocessing matching training pipeline."""
    runtime = get_runtime()
    audio = preprocess_audio(file_path)
    inputs = runtime["processor"](
        audio, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt", padding=True
    )
    inputs = {key: value.to(runtime["device"]) for key, value in inputs.items()}

    with torch.inference_mode():
        hidden = runtime["wav2vec_model"](**inputs).last_hidden_state
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            feature_mask = runtime["wav2vec_model"]._get_feature_vector_attention_mask(
                hidden.shape[1], attention_mask
            ).unsqueeze(-1).to(hidden.dtype)
            hidden = (hidden * feature_mask).sum(dim=1) / feature_mask.sum(dim=1).clamp_min(1)
        else:
            hidden = hidden.mean(dim=1)

    features = hidden.squeeze(0).cpu().numpy()
    return runtime["scaler"].transform(features.reshape(1, -1)).squeeze(0)


# ==================== Prediction ====================
def predict_emotion(features):
    """Predict emotion from features, with confidence threshold."""
    runtime = get_runtime()
    tensor = torch.as_tensor(features, dtype=torch.float32)
    tensor = tensor.unsqueeze(0).to(runtime["device"])
    with torch.inference_mode():
        probabilities = F.softmax(runtime["classifier"](tensor), dim=1).cpu().numpy()[0]

    label_names = runtime["label_names"]
    predicted_idx = int(np.argmax(probabilities))
    predicted_emotion = str(label_names[predicted_idx])
    confidence = float(probabilities[predicted_idx])
    prob_dict = {
        str(label_names[index]): float(probabilities[index])
        for index in range(len(label_names))
    }
    return predicted_emotion, prob_dict, confidence, confidence < CONFIDENCE_THRESHOLD


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

    with history_lock:
        history = []
        if PREDICTIONS_FILE.exists():
            try:
                history = json.loads(PREDICTIONS_FILE.read_text(encoding="utf-8"))
                if not isinstance(history, list):
                    history = []
            except (json.JSONDecodeError, OSError):
                history = []

        history = (history + [entry])[-50:]
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=RESULTS_DIR,
                prefix="predictions-", suffix=".tmp", delete=False
            ) as handle:
                temporary_path = Path(handle.name)
                json.dump(history, handle, indent=2)
            os.replace(temporary_path, PREDICTIONS_FILE)
        finally:
            if temporary_path and temporary_path.exists():
                temporary_path.unlink(missing_ok=True)


# ==================== Routes ====================
@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"status": "error", "message": "Maximum upload size is 10 MB."}), 413


@app.errorhandler(FileNotFoundError)
def model_not_ready(error):
    logger.error("Model artifacts are not ready: %s", error)
    return jsonify({
        "status": "error",
        "message": "Model artifacts are missing. Run the training script first.",
    }), 503


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "speech-emotion-recognition"})


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify({"status": "error", "message": "Please choose an audio file."}), 400

    original_filename = secure_filename(file.filename)
    ext = Path(original_filename).suffix.lower()
    if not original_filename or ext not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        return jsonify({"status": "error", "message": f"Allowed formats: {allowed}"}), 400

    # Never use the browser-provided filename as a filesystem path.
    temp_path = UPLOAD_FOLDER / f"{uuid4().hex}{ext}"
    file.save(temp_path)

    try:
        # Extract features -> predict
        features = extract_wav2vec_features(temp_path)
        emotion, probs, confidence, is_uncertain = predict_emotion(features)

        # Save to history
        save_prediction(original_filename, emotion, probs, confidence, is_uncertain)

        logger.info("%s -> %s (%.2f%%)", original_filename, emotion, confidence * 100)

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

    except FileNotFoundError:
        raise

    except Exception:
        logger.exception("Processing failed for %s", original_filename)
        return jsonify({
            "status": "error",
            "message": "Prediction failed. Check the server logs."
        }), 500

    finally:
        temp_path.unlink(missing_ok=True)


@app.route("/predictions")
def get_predictions():
    try:
        data = json.loads(PREDICTIONS_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            data = []
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        data = []
    return jsonify({"status": "success", "data": data[::-1]})


@app.route("/model-info")
def get_model_info():
    """Return model metadata for the frontend."""
    runtime = get_runtime()
    return jsonify({
        "status": "success",
        "data": {
            "model_info": runtime["model_info"],
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
            "device": str(runtime["device"]),
            "has_scaler": True
        }
    })


if __name__ == "__main__":
    logger.info("Speech Emotion Recognition app: http://127.0.0.1:5000")
    app.run(debug=False, host="0.0.0.0", port=5000)
