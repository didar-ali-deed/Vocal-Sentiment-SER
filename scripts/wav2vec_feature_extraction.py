import os
import pandas as pd
import librosa
import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from pathlib import Path
import logging
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add argument parsing
parser = argparse.ArgumentParser(description="Extract Wav2Vec2 features from audio files.")
parser.add_argument("--input_path", type=str, default="../Preprocessed Data/combined_data.csv", help="Path to the input CSV file")
parser.add_argument("--output_path", type=str, default="../Extracted Features/combined_wav2vec_features.csv", help="Path to save the extracted features")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for feature extraction")
args = parser.parse_args()

# Use the arguments
INPUT_PATH = Path(args.input_path)
OUTPUT_PATH = Path(args.output_path)
BATCH_SIZE = args.batch_size

# Ensure output directory exists
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Load pre-trained Wav2Vec2 processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

def extract_wav2vec_features(file_path):
    """
    Extract Wav2Vec2 features from an audio file.
    
    Args:
        file_path (str): Path to the audio file.
    
    Returns:
        np.ndarray: Extracted features (mean-pooled) or None if an error occurs.
    """
    try:
        # Load audio file and resample to 16kHz
        audio, sr = librosa.load(file_path, sr=16000)
        
        # Process audio with Wav2Vec2
        inputs = processor(audio, return_tensors="pt", sampling_rate=16000, padding=True)
        with torch.no_grad():
            features = model(**inputs).last_hidden_state
        
        # Apply mean pooling to reduce dimensionality
        return features.mean(dim=1).squeeze().numpy()
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return None

def process_batch(file_paths):
    """
    Process a batch of audio files to extract Wav2Vec2 features.
    
    Args:
        file_paths (list): List of file paths to process.
    
    Returns:
        list: List of extracted features (or None for failed files).
    """
    features = []
    for file_path in file_paths:
        feature = extract_wav2vec_features(file_path)
        features.append(feature)
    return features

def process_dataset(input_path, output_path, batch_size):
    """
    Process the entire dataset to extract Wav2Vec2 features.
    
    Args:
        input_path (Path): Path to the input CSV file.
        output_path (Path): Path to save the extracted features.
        batch_size (int): Number of files to process in each batch.
    """
    # Load the dataset
    df = pd.read_csv(input_path)
    file_paths = df['Path'].tolist()
    labels = df['Emotions'].tolist()
    datasets = df['Dataset'].tolist()

    # Process files in batches
    features = []
    for i in tqdm(range(0, len(file_paths), batch_size), desc="Processing batches"):
        batch_paths = file_paths[i:i + batch_size]
        batch_features = process_batch(batch_paths)
        features.extend(batch_features)

    # Save extracted features
    features_df = pd.DataFrame(features)
    features_df['label'] = labels
    features_df['Dataset'] = datasets
    features_df.to_csv(output_path, index=False)
    logging.info(f"Features saved to: {output_path}")

def plot_feature_distribution(features_df):
    """
    Plot the distribution of Wav2Vec2 features.
    
    Args:
        features_df (pd.DataFrame): DataFrame containing the extracted features.
    """
    plt.figure(figsize=(12, 6))
    feature_means = features_df.iloc[:, :-2].mean(axis=1)  # Ignore 'label' and 'Dataset' columns
    plt.hist(feature_means, bins=50, color='blue', alpha=0.7)
    plt.title("Wav2Vec2 Feature Distribution")
    plt.xlabel("Mean Feature Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(OUTPUT_PATH.parent / "feature_distribution.png")
    plt.show()

if __name__ == "__main__":
    # Process the dataset
    logging.info("Starting feature extraction...")
    process_dataset(INPUT_PATH, OUTPUT_PATH, BATCH_SIZE)

    # Load the extracted features and plot their distribution
    features_df = pd.read_csv(OUTPUT_PATH)
    plot_feature_distribution(features_df)
    logging.info("Feature distribution plot saved.")