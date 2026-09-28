# VocalSense: Speech Emotion Recognition

VocalSense classifies uploaded speech into eight emotions: angry, calm, disgust,
fear, happy, neutral, sad, and surprise. It uses a pretrained Wav2Vec2 feature
extractor and a PyTorch MLP classifier trained on RAVDESS and TESS, with a Flask
web interface for audio preview, predictions, probabilities, and recent history.

## Setup

Use Python 3.11 for the pinned dependencies (the locally tested Python version).
From the project root, run these commands in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS/Linux, activate with `source .venv/bin/activate` instead.
PyTorch selects CUDA when available and otherwise uses the CPU.

## Run the web app

With the environment activated, run from the project root:

```powershell
python scripts/app.py
```

Or, if already inside `scripts/`, run `python app.py`.
Open http://127.0.0.1:5000. Stop the server with Ctrl+C and restart it after code
changes; automatic reloading is disabled.

The app requires these local files:

- `models/emotion_classifier/emotion_classifier.pth`
- `Extracted Features/combined_wav2vec_features.csv`

The original trainer saves only the model weights. The app recovers class names
in their original first-appearance order from the training feature CSV, using
the same missing-row filtering as training. Keep that CSV unchanged and paired
with its checkpoint. Alternatively, supply `label_names.npy` in the model folder
as a one-dimensional Unicode array in the exact training class order.
Do not alphabetically sort the labels.

The classifier uses 768 input features and hidden layers of 128 and 64 units.
It does not use feature scaling, so no `feature_scaler.pkl` or `model_info.json`
is required. Model metadata is constructed by the loader.

The first prediction loads `facebook/wav2vec2-base`; internet access is required
if the Hugging Face model is not already cached. Later requests reuse the loaded
models. The page also loads Chart.js and fonts from external services.

Supported uploads are WAV, MP3, FLAC, and OGG. Audio must be at least 0.1 seconds
and at most 30 seconds. The request limit is 10 MB, including multipart overhead.
Audio is resampled to 16 kHz mono on the server and temporary uploads are removed
after processing. The last 50 predictions are stored in
`results/inference_predictions.json`; the interface shows the newest 10.
Confidence below 40% is marked uncertain.

## Rebuild the training pipeline

Download and extract the datasets manually:

- [RAVDESS dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
- [TESS dataset](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)

Use this directory layout:

```text
data/
  ravdess/
    Actor_01/
    ...
  tess/
    TESS Toronto emotional speech set data/
      OAF_angry/
      ...
```

Run all pipeline commands from `scripts/`: their default paths and the saved
audio paths are relative to that working directory. Folder names contain spaces.

```powershell
cd scripts
python preprocess.py
python wav2vec_feature_extraction.py
python train_emotion_classifier.py
python test_emotion_classifier.py
```

Run each command after the previous command finishes. Plot windows may need to
be closed before the script continues. Training overwrites the existing classifier
checkpoint; keep its matching feature CSV if preserving an older model.

| Script | Purpose |
| --- | --- |
| `preprocess.py` | Index audio labels, paths, durations, and sample rates; create distribution plots. |
| `wav2vec_feature_extraction.py` | Extract mean-pooled Wav2Vec2 features into a CSV. |
| `train_emotion_classifier.py` | Train the MLP for 50 epochs and save weights and a loss plot. |
| `test_emotion_classifier.py` | Calculate metrics and save predictions and evaluation plots. |
| `model.py` | Load the existing classifier and recover its class labels for inference. |
| `app.py` | Serve the web interface and prediction endpoints. |

### Current evaluation limitations

The evaluation script recreates the same 30% split used for validation during
training. Its results are validation metrics, not an independent held-out test.
Splitting is by recording rather than speaker, so it does not establish performance
on unseen speakers. The saved accuracy is approximately 78.93% on that split.

The preprocessing `--add_noise` option currently does not persist augmented audio;
feature extraction still reads the original recordings. It should not be treated
as training augmentation.

## Project structure

```text
SER_FYP/
  scripts/
    app.py
    model.py
    preprocess.py
    wav2vec_feature_extraction.py
    train_emotion_classifier.py
    test_emotion_classifier.py
    templates/index.html
    static/
      professional.js
      professional.css
      styles.css
      favicon.svg
  data/
  Preprocessed Data/
  Extracted Features/
  models/emotion_classifier/
  results/
  deployment/uploads/
  requirements.txt
  README.md
  .gitignore
```

Both CSS files are currently loaded by the template. `professional.js` is the
active frontend script.

## Git and local artifacts

`.gitignore` excludes virtual environments, Python caches, datasets, generated
features/models/results, temporary uploads, logs, and local configuration.
These files remain on disk but are not included in new commits. A fresh clone
must restore the matching model and features or run the pipeline before inference.
Ignore rules do not remove files already tracked by Git.

## Troubleshooting

- **Missing model or labels (503):** restore the checkpoint and its original feature
  CSV at the paths above, or provide the matching `label_names.npy`.
- **Pipeline cannot find files:** run the pipeline from `scripts/` and preserve the
  folder names with spaces.
- **Audio cannot be decoded:** export a standard WAV file and try again.
- **Wav2Vec2 unused-weight messages:** the app loads the feature extractor rather
  than the pretraining heads. These messages alone do not mean prediction failed.

The bundled Flask server is intended for local development. It currently listens
on all network interfaces at port 5000, so other machines may reach it when the
firewall permits. Prediction history is shared by users of that server.
