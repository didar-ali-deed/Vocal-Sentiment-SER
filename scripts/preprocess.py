import os
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import unittest
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add argument parsing
parser = argparse.ArgumentParser(description="Preprocess TESS and RAVDESS datasets.")
parser.add_argument("--tess_path", type=str, default="../data/tess/TESS Toronto emotional speech set data/", help="Path to TESS dataset")
parser.add_argument("--ravdess_path", type=str, default="../data/ravdess/", help="Path to RAVDESS dataset")
parser.add_argument("--output_dir", type=str, default="../Preprocessed Data/", help="Output directory for preprocessed data")
parser.add_argument("--add_noise", action="store_true", help="Add noise to audio files during preprocessing")
parser.add_argument("--noise_level", type=float, default=0.005, help="Level of noise to add (default: 0.005)")
args = parser.parse_args()

# Use the arguments
TESS_PATH = Path(args.tess_path)
RAVDESS_PATH = Path(args.ravdess_path)
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_FILE = OUTPUT_DIR / "combined_data.csv"
ADD_NOISE = args.add_noise
NOISE_LEVEL = args.noise_level

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Emotion mapping for TESS
tess_emotion_map = {
    'neutral': 'neutral',
    'happy': 'happy',
    'sad': 'sad',
    'angry': 'angry',
    'fear': 'fear',
    'disgust': 'disgust',
    'pleasant_surprise': 'surprise'
}

# Emotion mapping for RAVDESS (from filename structure)
ravdess_emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise"
}

# Supported audio formats
AUDIO_FORMATS = [".wav", ".mp3", ".flac"]

def validate_dataset_path(path):
    """
    Validate that the dataset path exists and is not empty.
    
    Args:
        path (Path): Path to the dataset.
    
    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the path is empty.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")
    if not any(path.iterdir()):
        raise ValueError(f"Dataset path is empty: {path}")

def add_noise(audio, noise_level=0.005):
    """
    Add random noise to the audio signal.
    
    Args:
        audio (np.array): The audio signal.
        noise_level (float): The level of noise to add.
    
    Returns:
        np.array: The noisy audio signal.
    """
    noise = np.random.randn(len(audio)) * noise_level
    return audio + noise

def extract_metadata(file_path):
    """
    Extract metadata (sample rate and number of samples) from an audio file.
    
    Args:
        file_path (str): Path to the audio file.
    
    Returns:
        tuple: (sample_rate, num_samples) or (None, None) if an error occurs.
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
        return sr, len(y)
    except Exception as e:
        logging.error(f"Error extracting metadata from {file_path}: {e}")
        return None, None

def process_file(file, dataset_name, emotion_map, add_noise_flag=False, noise_level=0.005):
    """
    Process a single audio file to extract emotion, path, duration, and metadata.
    
    Args:
        file (Path): Path to the audio file.
        dataset_name (str): Name of the dataset ("TESS" or "RAVDESS").
        emotion_map (dict): Mapping of emotion codes to emotion labels.
        add_noise_flag (bool): Whether to add noise to the audio.
        noise_level (float): The level of noise to add.
    
    Returns:
        tuple: (emotion, file_path, duration, dataset_name, sample_rate, num_samples) or None if an error occurs.
    """
    try:
        if dataset_name == "TESS":
            emotion = file.parent.name.split('_')[-1].lower()
        else:
            filename_parts = file.stem.split("-")
            if len(filename_parts) < 3:
                return None
            emotion_code = filename_parts[2]
            emotion = emotion_map.get(emotion_code, "unknown")

        y, sr = librosa.load(str(file), sr=None)
        
        if add_noise_flag:
            y = add_noise(y, noise_level)
        
        duration = librosa.get_duration(y=y, sr=sr)
        sample_rate, num_samples = sr, len(y)

        return emotion, str(file), duration, dataset_name, sample_rate, num_samples
    except Exception as e:
        logging.error(f"Error processing file {file}: {e}")
        return None

def process_dataset(path, dataset_name, emotion_map, add_noise_flag=False, noise_level=0.005):
    """
    Process all audio files in a dataset using parallel processing.
    
    Args:
        path (Path): Path to the dataset.
        dataset_name (str): Name of the dataset ("TESS" or "RAVDESS").
        emotion_map (dict): Mapping of emotion codes to emotion labels.
        add_noise_flag (bool): Whether to add noise to the audio.
        noise_level (float): The level of noise to add.
    
    Returns:
        pd.DataFrame: DataFrame containing processed data.
    """
    files = [file for format in AUDIO_FORMATS for file in path.rglob(f"*{format}")]
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.starmap(process_file, [(file, dataset_name, emotion_map, add_noise_flag, noise_level) for file in files]), desc=f"Processing {dataset_name} files", total=len(files)))
    return pd.DataFrame([result for result in results if result is not None], columns=['Emotions', 'Path', 'Duration', 'Dataset', 'SampleRate', 'NumSamples'])

def generate_graphs(df):
    """
    Generate and save visualizations for emotion and duration distributions.
    
    Args:
        df (pd.DataFrame): DataFrame containing the processed data.
    """
    # Emotion Distribution
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x="Emotions", hue="Dataset", order=df["Emotions"].value_counts().index, palette="pastel")
    plt.title("Distribution of Emotions in TESS & RAVDESS Datasets")
    plt.xlabel("Emotions")
    plt.ylabel("Number of Files")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle="--", alpha=0.7)
    plt.legend(title="Dataset")
    plt.savefig(OUTPUT_DIR / "emotion_distribution.png")
    plt.show()

    # Duration Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="Duration", bins=30, kde=True, hue="Dataset", element="step", palette="coolwarm")
    plt.title("Audio File Duration Distribution (TESS & RAVDESS)")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Number of Files")
    plt.grid(axis='y', linestyle="--", alpha=0.7)
    plt.savefig(OUTPUT_DIR / "audio_duration_distribution.png")
    plt.show()

class TestPreprocessing(unittest.TestCase):
    """
    Unit tests for the preprocessing script.
    """
    def test_process_tess(self):
        df = process_dataset(TESS_PATH, "TESS", tess_emotion_map)
        self.assertGreater(len(df), 0)
        self.assertTrue(all(col in df.columns for col in ['Emotions', 'Path', 'Duration', 'Dataset', 'SampleRate', 'NumSamples']))

    def test_process_ravdess(self):
        df = process_dataset(RAVDESS_PATH, "RAVDESS", ravdess_emotion_map)
        self.assertGreater(len(df), 0)
        self.assertTrue(all(col in df.columns for col in ['Emotions', 'Path', 'Duration', 'Dataset', 'SampleRate', 'NumSamples']))

if __name__ == "__main__":
    # Validate dataset paths
    validate_dataset_path(TESS_PATH)
    validate_dataset_path(RAVDESS_PATH)

    # Process datasets
    logging.info("Processing TESS dataset...")
    tess_df = process_dataset(TESS_PATH, "TESS", tess_emotion_map, add_noise_flag=ADD_NOISE, noise_level=NOISE_LEVEL)
    logging.info("Processing RAVDESS dataset...")
    ravdess_df = process_dataset(RAVDESS_PATH, "RAVDESS", ravdess_emotion_map, add_noise_flag=ADD_NOISE, noise_level=NOISE_LEVEL)

    # Combine datasets
    combined_df = pd.concat([tess_df, ravdess_df], ignore_index=True)
    combined_df.to_csv(OUTPUT_FILE, index=False)
    logging.info(f"Combined dataset saved to: {OUTPUT_FILE}")

    # Generate graphs
    generate_graphs(combined_df)
    logging.info("Graphs saved for presentation.")

    # Run unit tests
    unittest.main(argv=[''], exit=False)