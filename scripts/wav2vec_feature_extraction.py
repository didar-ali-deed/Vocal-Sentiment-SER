import os
import pandas as pd
import librosa
import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from pathlib import Path
import logging
import argparse
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_NAME = "facebook/wav2vec2-base"


def load_encoder():
    """Load the encoder only when the CLI is actually executed."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Loading Wav2Vec2 on %s", device)
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    return processor, model, device


def preprocess_audio(file_path, target_sr=16000):
    """
    Load and preprocess audio: resample, normalize volume, trim silence.

    This preprocessing is CRITICAL and must match what app.py does at inference
    time. Any change here must be mirrored in app.py's extract_wav2vec_features().

    Args:
        file_path (str): Path to the audio file.
        target_sr (int): Target sample rate.

    Returns:
        np.ndarray: Preprocessed audio signal, or None on error.
    """
    try:
        # Load audio and resample
        audio, _ = librosa.load(file_path, sr=target_sr, mono=True)

        # Skip very short audio (< 0.1 seconds)
        if len(audio) < target_sr * 0.1:
            logging.warning(f"Skipping too-short file: {file_path} ({len(audio)} samples)")
            return None

        # --- FIX: Normalize volume (consistent amplitude across files) ---
        audio = librosa.util.normalize(audio)

        # --- FIX: Trim silence from start and end ---
        audio, _ = librosa.effects.trim(audio, top_db=25)

        # Re-check length after trimming
        if len(audio) < target_sr * 0.1:
            logging.warning(f"File too short after trimming: {file_path}")
            return None

        return audio

    except Exception as e:
        logging.error(f"Error preprocessing {file_path}: {e}")
        return None


def extract_batch_features(audios, processor, model, device, target_sr):
    """Extract masked mean-pooled embeddings for a batch of variable-length audio."""
    inputs = processor(
        audios,
        return_tensors="pt",
        sampling_rate=target_sr,
        padding=True,
        return_attention_mask=True,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        hidden = model(**inputs).last_hidden_state
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            feature_mask = model._get_feature_vector_attention_mask(
                hidden.shape[1], attention_mask
            ).unsqueeze(-1).to(hidden.dtype)
            hidden = (hidden * feature_mask).sum(dim=1) / feature_mask.sum(dim=1).clamp_min(1)
        else:
            hidden = hidden.mean(dim=1)
    return hidden.cpu().numpy()


def process_dataset(input_path, output_path, batch_size, target_sr):
    """
    Process the entire dataset to extract Wav2Vec2 features.

    Args:
        input_path (Path): Path to the input CSV file.
        output_path (Path): Path to save the extracted features.
        batch_size (int): Number of files to process in each batch.
        target_sr (int): Audio sample rate passed to Wav2Vec2.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    processor, model, device = load_encoder()
    df = pd.read_csv(input_path)
    file_paths = df['Path'].tolist()
    labels = df['Emotions'].tolist()
    datasets = df['Dataset'].tolist()

    all_features = []
    valid_labels = []
    valid_datasets = []
    failed_count = 0

    for i in tqdm(range(0, len(file_paths), batch_size), desc="Processing batches"):
        batch_paths = file_paths[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]
        batch_datasets = datasets[i:i + batch_size]

        valid_audios = []
        valid_labels_batch = []
        valid_datasets_batch = []
        for fp, lbl, ds in zip(batch_paths, batch_labels, batch_datasets):
            audio = preprocess_audio(fp, target_sr=target_sr)
            if audio is None:
                failed_count += 1
                continue

            valid_audios.append(audio)
            valid_labels_batch.append(lbl)
            valid_datasets_batch.append(ds)

        if valid_audios:
            try:
                batch_features = extract_batch_features(
                    valid_audios, processor, model, device, target_sr
                )
                all_features.extend(batch_features)
                valid_labels.extend(valid_labels_batch)
                valid_datasets.extend(valid_datasets_batch)
            except Exception as error:
                logging.error("Batch feature extraction failed: %s", error)
                failed_count += len(valid_audios)

        # Clear GPU memory periodically
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    logging.info(f"Successfully processed: {len(all_features)} / {len(file_paths)} files")
    if failed_count > 0:
        logging.warning(f"Failed to process: {failed_count} files")

    # Save extracted features
    features_df = pd.DataFrame(all_features)
    features_df['label'] = valid_labels
    features_df['Dataset'] = valid_datasets
    features_df.to_csv(output_path, index=False)
    logging.info(f"Features saved to: {output_path}")

    return features_df


def plot_feature_distribution(features_df, output_path):
    """
    Plot the distribution of Wav2Vec2 features.

    Args:
        features_df (pd.DataFrame): DataFrame containing the extracted features.
    """
    plt.figure(figsize=(12, 6))
    feature_columns = [
        column for column in features_df.columns if column not in {"label", "Dataset"}
    ]
    feature_means = features_df[feature_columns].mean(axis=1)
    plt.hist(feature_means, bins=50, color='blue', alpha=0.7)
    plt.title("Wav2Vec2 Feature Distribution (after audio normalization + trimming)")
    plt.xlabel("Mean Feature Value")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig(Path(output_path).parent / "feature_distribution.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Wav2Vec2 features from audio files.")
    parser.add_argument("--input_path", type=str, default="Preprocessed Data/combined_data.csv")
    parser.add_argument("--output_path", type=str, default="Extracted Features/combined_wav2vec_features.csv")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--target_sr", type=int, default=16000)
    args = parser.parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Starting feature extraction with audio normalization + silence trimming...")
    features_df = process_dataset(input_path, output_path, args.batch_size, args.target_sr)

    # Plot feature distribution
    plot_feature_distribution(features_df, output_path)
    logging.info("Feature distribution plot saved.")
    logging.info("Done.")
