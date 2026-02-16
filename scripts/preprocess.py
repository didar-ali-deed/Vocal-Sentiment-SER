# preprocess.py
# Data preprocessing pipeline for TESS and RAVDESS datasets
# Extracts metadata, validates audio files, generates distribution plots

import os
import pandas as pd
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Emotion mappings
TESS_EMOTION_MAP = {
    'neutral': 'neutral',
    'happy': 'happy',
    'sad': 'sad',
    'angry': 'angry',
    'fear': 'fear',
    'disgust': 'disgust',
    'pleasant_surprise': 'surprise'
}

RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise"
}

AUDIO_FORMATS = [".wav", ".mp3", ".flac"]


def validate_dataset_path(path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")
    if not any(path.iterdir()):
        raise ValueError(f"Dataset path is empty: {path}")


def process_file(args):
    """Process a single audio file. Takes a tuple for compatibility with pool.map."""
    file, dataset_name, emotion_map = args
    try:
        if dataset_name == "TESS":
            emotion_raw = file.parent.name.split('_')[-1].lower()
            emotion = emotion_map.get(emotion_raw, emotion_raw)
        else:
            parts = file.stem.split("-")
            if len(parts) < 3:
                return None
            emotion = emotion_map.get(parts[2], "unknown")

        if emotion == "unknown":
            return None

        y, sr = librosa.load(str(file), sr=None)
        duration = librosa.get_duration(y=y, sr=sr)

        return (emotion, str(file), duration, dataset_name, sr, len(y))
    except Exception as e:
        logging.error(f"Error processing {file}: {e}")
        return None


def process_dataset(path, dataset_name, emotion_map):
    """Process all audio files in a dataset using multiprocessing."""
    files = [f for fmt in AUDIO_FORMATS for f in path.rglob(f"*{fmt}")]
    logging.info(f"Found {len(files)} audio files in {dataset_name}")

    args_list = [(f, dataset_name, emotion_map) for f in files]

    with Pool(min(cpu_count(), 8)) as pool:
        results = list(tqdm(
            pool.imap(process_file, args_list),
            desc=f"Processing {dataset_name}",
            total=len(files)
        ))

    valid = [r for r in results if r is not None]
    logging.info(f"{dataset_name}: {len(valid)}/{len(files)} files processed")

    return pd.DataFrame(
        valid,
        columns=['Emotions', 'Path', 'Duration', 'Dataset', 'SampleRate', 'NumSamples']
    )


def generate_graphs(df, output_dir):
    """Generate distribution visualizations."""
    plt.style.use('seaborn-v0_8-whitegrid')

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Emotion distribution
    sns.countplot(
        data=df, x="Emotions", hue="Dataset",
        order=df["Emotions"].value_counts().index,
        palette="Set2", ax=axes[0]
    )
    axes[0].set_title("Emotion Distribution", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Emotion", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].legend(title="Dataset")

    # Duration distribution
    sns.histplot(
        data=df, x="Duration", bins=30, kde=True,
        hue="Dataset", element="step", palette="Set2", ax=axes[1]
    )
    axes[1].set_title("Audio Duration Distribution", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Duration (seconds)", fontsize=12)
    axes[1].set_ylabel("Count", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / "dataset_distributions.png", dpi=200)
    plt.close()


def print_stats(df):
    print("\n" + "=" * 50)
    print("DATASET STATISTICS")
    print("=" * 50)
    print(f"Total audio files: {len(df)}")
    print(f"\nEmotion distribution:")
    print(df['Emotions'].value_counts().to_string())
    print(f"\nBy dataset:")
    print(df['Dataset'].value_counts().to_string())
    print(f"\nDuration: min={df['Duration'].min():.2f}s, "
          f"max={df['Duration'].max():.2f}s, "
          f"mean={df['Duration'].mean():.2f}s")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Preprocess TESS and RAVDESS datasets")
    parser.add_argument("--tess_path", type=str,
                        default="../data/tess/TESS Toronto emotional speech set data/")
    parser.add_argument("--ravdess_path", type=str, default="../data/ravdess/")
    parser.add_argument("--output_dir", type=str, default="../Preprocessed Data/")
    args = parser.parse_args()

    tess_path = Path(args.tess_path)
    ravdess_path = Path(args.ravdess_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    validate_dataset_path(tess_path)
    validate_dataset_path(ravdess_path)

    logging.info("Processing TESS...")
    tess_df = process_dataset(tess_path, "TESS", TESS_EMOTION_MAP)

    logging.info("Processing RAVDESS...")
    ravdess_df = process_dataset(ravdess_path, "RAVDESS", RAVDESS_EMOTION_MAP)

    combined = pd.concat([tess_df, ravdess_df], ignore_index=True)
    output_file = output_dir / "combined_data.csv"
    combined.to_csv(output_file, index=False)
    logging.info(f"Saved: {output_file}")

    print_stats(combined)
    generate_graphs(combined, output_dir)
    logging.info("Done.")


if __name__ == "__main__":
    main()
