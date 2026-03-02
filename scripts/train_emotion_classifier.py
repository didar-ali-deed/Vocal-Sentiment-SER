# train_emotion_classifier.py
# Transformer-based Speech Emotion Recognition
#
# Architecture: Frozen Wav2Vec2 → TransformerSERHead
#   - Learnable CLS token aggregation (no pooling approximation)
#   - Pre-norm multi-head self-attention blocks
#   - Variable-length input via padding mask (no information leakage from pad tokens)
#   - LayerNorm only — no BatchNorm (works correctly at batch-size 1 in inference)
#
# Training improvements over the baseline:
#   - Wav2Vec2Processor used for correct input normalization
#   - SpecAugment on hidden states (time + channel masking)
#   - Mixup augmentation
#   - Label smoothing
#   - Warmup + cosine LR schedule
#   - Early stopping on weighted F1 (not accuracy)
#   - Versioned feature cache (old cache is not silently reused)

import os
import math
import random
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model

warnings.filterwarnings("ignore", category=UserWarning)

# ==================== Paths ====================
BASE_DIR = Path(__file__).parent.parent
PREPROCESSED_CSV = BASE_DIR / "Preprocessed Data" / "combined_data.csv"
OUTPUT_MODEL_DIR = BASE_DIR / "models" / "emotion_classifier"
RESULTS_DIR = BASE_DIR / "results"
CACHE_DIR = BASE_DIR / "models" / "feature_cache"

os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ==================== Hyperparameters ====================
EPOCHS = 60
LEARNING_RATE = 3e-4
BATCH_SIZE = 32
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
PATIENCE = 12           # early stopping on best val F1
WEIGHT_DECAY = 1e-3
TARGET_SR = 16000
MAX_AUDIO_LEN = 5 * TARGET_SR   # 5 s — must match app.py MAX_AUDIO_DURATION
WARMUP_FRACTION = 0.10  # first 10 % of steps are linear warmup
MIXUP_ALPHA = 0.2       # beta distribution parameter for mixup
LABEL_SMOOTHING = 0.1   # prevents overconfidence
SEED = 42

# Transformer head architecture
D_MODEL = 256
NUM_HEADS = 8           # 256 / 8 = 32 dim per head
NUM_LAYERS = 2
DIM_FF = 512
DROPOUT = 0.25

# SpecAugment parameters (applied on hidden states)
NUM_TIME_MASKS = 2
TIME_MASK_PARAM = 30    # max consecutive time steps to mask
NUM_FREQ_MASKS = 2
FREQ_MASK_PARAM = 64    # max consecutive channels to mask (out of 768)

# Cache version — bump this string to force feature re-extraction
CACHE_VERSION = "v2_processor"

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==================== Architecture ====================
class TransformerSERHead(nn.Module):
    """
    Transformer-based classifier head on frozen Wav2Vec2 hidden states.

    Key design choices:
      - Learnable CLS token prepended to the sequence; its output at the last
        layer is used for classification (same as BERT).
      - Pre-norm (norm_first=True) for more stable gradient flow.
      - LayerNorm only — works correctly at any batch size including 1.
      - src_key_padding_mask prevents attention to zero-padded positions.

    Input:  hidden_states [B, T, 768] + pad_mask [B, T] (True = padding)
    Output: logits [B, num_classes]
    """

    def __init__(self, input_dim=768, d_model=D_MODEL, num_heads=NUM_HEADS,
                 num_layers=NUM_LAYERS, dim_ff=DIM_FF,
                 num_classes=8, dropout=DROPOUT):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Project from wav2vec2 dim to transformer dim
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_drop = nn.Dropout(dropout)

        # Learnable CLS token (initialised near zero, learned during training)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer encoder with pre-norm for stable training
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,    # pre-norm: more stable than post-norm
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Classification head — two-layer MLP on the CLS token
        self.cls_drop = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.input_proj.weight, std=0.02)
        nn.init.zeros_(self.input_proj.bias)
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, src_key_padding_mask=None):
        """
        x:                  [B, T, 768]
        src_key_padding_mask: [B, T] bool — True = padding (ignored in attention)
        """
        B = x.size(0)

        # Project and normalise
        x = self.input_proj(x)      # [B, T, d_model]
        x = self.input_norm(x)
        x = self.input_drop(x)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        x = torch.cat([cls, x], dim=1)          # [B, T+1, d_model]

        # Extend mask: CLS is never masked
        if src_key_padding_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            src_key_padding_mask = torch.cat(
                [cls_mask, src_key_padding_mask], dim=1
            )  # [B, T+1]

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        cls_out = x[:, 0]           # [B, d_model] — CLS token output
        cls_out = self.cls_drop(cls_out)
        return self.classifier(cls_out)


# ==================== Feature Extraction (one-time, cached) ====================
def extract_all_features(file_paths, labels, cache_dir):
    """
    Extract Wav2Vec2 hidden states for every audio file and cache them.
    Uses Wav2Vec2Processor for correct per-sample normalisation.
    The cache filename includes CACHE_VERSION so changing the pipeline
    automatically triggers re-extraction rather than silently reusing stale data.
    """
    cache_path = cache_dir / f"wav2vec_hidden_states_{CACHE_VERSION}.pt"

    if cache_path.exists():
        print(f"Loading cached features from {cache_path}")
        cached = torch.load(cache_path, weights_only=False)
        print(f"  Loaded {len(cached['features'])} samples")
        return cached['features'], cached['labels']

    print("Extracting Wav2Vec2 features (one-time, will be cached)...")
    print(f"  Cache version: {CACHE_VERSION}")

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    wav2vec.eval()
    for param in wav2vec.parameters():
        param.requires_grad = False

    all_features, valid_labels, failed = [], [], 0

    for fp, lbl in tqdm(zip(file_paths, labels),
                        total=len(file_paths), desc="Extracting features"):
        try:
            audio, _ = librosa.load(fp, sr=TARGET_SR)

            if len(audio) < TARGET_SR * 0.1:
                failed += 1
                continue

            audio = librosa.util.normalize(audio)
            audio, _ = librosa.effects.trim(audio, top_db=25)

            if len(audio) < TARGET_SR * 0.1:
                failed += 1
                continue

            if len(audio) > MAX_AUDIO_LEN:
                audio = audio[:MAX_AUDIO_LEN]

            # Wav2Vec2Processor performs per-sample zero-mean / unit-var
            # normalisation — this is what the model was pretrained to expect.
            inputs = processor(
                audio, sampling_rate=TARGET_SR, return_tensors="pt"
            )

            with torch.no_grad():
                hidden = wav2vec(inputs.input_values).last_hidden_state  # [1, T, 768]

            all_features.append(hidden.squeeze(0).cpu())   # [T, 768]
            valid_labels.append(lbl)

        except Exception:
            failed += 1
            continue

    print(f"  Extracted: {len(all_features)} / {len(file_paths)} | Failed: {failed}")

    torch.save({'features': all_features, 'labels': valid_labels}, cache_path)
    print(f"  Cached to {cache_path}")

    del wav2vec, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_features, valid_labels


# ==================== Dataset with SpecAugment ====================
class HiddenStateDataset(Dataset):
    """
    Dataset of cached Wav2Vec2 hidden states.

    SpecAugment (Park et al. 2019) is applied directly on the [T, 768] tensors:
      - Time masking:    zero out up to TIME_MASK_PARAM consecutive time steps
      - Channel masking: zero out up to FREQ_MASK_PARAM consecutive hidden dims

    This is much faster than re-running audio augmentation and re-extracting
    features, while still giving the model exposure to partially corrupted inputs
    — improving generalisation to real-world audio artefacts.
    """

    def __init__(self, features, labels, augment=False):
        self.features = features
        self.labels = labels
        self.augment = augment

    def _spec_augment(self, feat):
        """Apply SpecAugment-style masking on a [T, F] hidden-state tensor."""
        feat = feat.clone()
        T, F = feat.shape

        for _ in range(NUM_TIME_MASKS):
            t = random.randint(0, min(TIME_MASK_PARAM, T))
            t0 = random.randint(0, max(0, T - t))
            feat[t0:t0 + t, :] = 0.0

        for _ in range(NUM_FREQ_MASKS):
            f = random.randint(0, min(FREQ_MASK_PARAM, F))
            f0 = random.randint(0, max(0, F - f))
            feat[:, f0:f0 + f] = 0.0

        return feat

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx].clone()   # [T, 768]
        label = self.labels[idx]

        if self.augment:
            # Small Gaussian noise
            if random.random() < 0.5:
                feat = feat + torch.randn_like(feat) * 0.005
            # SpecAugment
            feat = self._spec_augment(feat)

        return feat, label


def collate_fn(batch):
    """
    Pad variable-length hidden states within a batch and build a padding mask.

    Returns:
      padded    [B, max_T, 768]   — zero-padded hidden states
      labels    [B]               — class indices
      pad_mask  [B, max_T] bool   — True = padding position (ignored by transformer)
    """
    feats, labels = zip(*batch)
    max_T = max(f.size(0) for f in feats)
    B = len(feats)
    F = feats[0].size(1)

    padded = torch.zeros(B, max_T, F)
    pad_mask = torch.ones(B, max_T, dtype=torch.bool)   # default: all padding

    for i, f in enumerate(feats):
        T = f.size(0)
        padded[i, :T] = f
        pad_mask[i, :T] = False     # real data = not padding

    return padded, torch.tensor(labels, dtype=torch.long), pad_mask


# ==================== Mixup ====================
def mixup_batch(x, y, mask, alpha=MIXUP_ALPHA):
    """
    Mixup (Zhang et al. 2018): linearly interpolate two random samples.

    The padding mask is merged with logical-AND so that a position is
    considered real data only when *both* contributing samples have real data.
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    B = x.size(0)
    idx = torch.randperm(B, device=x.device)

    x_mix = lam * x + (1 - lam) * x[idx]
    mask_mix = mask & mask[idx]

    return x_mix, y, y[idx], lam, mask_mix


def mixup_loss(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ==================== LR Scheduler ====================
def warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, min_ratio=0.05):
    """Linear warmup followed by cosine decay."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(min_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ==================== Main ====================
def main():
    print("=" * 65)
    print("TRANSFORMER-BASED SPEECH EMOTION RECOGNITION — TRAINING")
    print("  Architecture : Wav2Vec2 (frozen) → TransformerSERHead")
    print(f"  Transformer  : {NUM_LAYERS} layers, d={D_MODEL}, heads={NUM_HEADS}, ff={DIM_FF}")
    print(f"  Augmentation : SpecAugment + Mixup(α={MIXUP_ALPHA}) + Gaussian noise")
    print(f"  Regularise   : Dropout={DROPOUT}, LabelSmoothing={LABEL_SMOOTHING}")
    print(f"  Schedule     : Warmup {int(WARMUP_FRACTION*100)}% → cosine decay")
    print("=" * 65)

    if not PREPROCESSED_CSV.exists():
        raise FileNotFoundError(
            f"Not found: {PREPROCESSED_CSV}\nRun preprocess.py first."
        )

    df = pd.read_csv(PREPROCESSED_CSV)
    df = df[df['Path'].apply(lambda p: os.path.exists(p))].reset_index(drop=True)
    print(f"Valid samples: {len(df)}")

    le = LabelEncoder()
    y = le.fit_transform(df['Emotions'].values)
    label_names = le.classes_
    file_paths = df['Path'].values.tolist()
    print(f"Classes ({len(label_names)}): {list(label_names)}")

    # ---- Phase 1: Extract / load features ----
    all_features, all_labels = extract_all_features(file_paths, y, CACHE_DIR)

    # ---- Split: 70 / 15 / 15 ----
    indices = list(range(len(all_features)))
    all_labels_np = np.array(all_labels)

    idx_trainval, idx_test = train_test_split(
        indices, test_size=TEST_FRACTION,
        random_state=SEED, stratify=all_labels_np
    )
    val_frac_adj = VALIDATION_FRACTION / (1 - TEST_FRACTION)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_frac_adj,
        random_state=SEED, stratify=all_labels_np[idx_trainval]
    )

    train_feats  = [all_features[i] for i in idx_train]
    train_labels = [all_labels[i]   for i in idx_train]
    val_feats    = [all_features[i] for i in idx_val]
    val_labels   = [all_labels[i]   for i in idx_val]
    test_feats   = [all_features[i] for i in idx_test]
    test_labels  = [all_labels[i]   for i in idx_test]

    print(f"Split  — Train: {len(train_feats)} | Val: {len(val_feats)} | Test: {len(test_feats)}")

    # Save artefacts
    joblib.dump(le, OUTPUT_MODEL_DIR / "label_encoder.pkl")
    np.save(OUTPUT_MODEL_DIR / "label_names.npy", label_names)
    np.save(OUTPUT_MODEL_DIR / "test_data.npy",
            {'features': test_feats, 'labels': test_labels, 'label_names': label_names},
            allow_pickle=True)

    # ---- Datasets & loaders ----
    train_ds = HiddenStateDataset(train_feats, train_labels, augment=True)
    val_ds   = HiddenStateDataset(val_feats,   val_labels,   augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    # ---- Class weights (handle imbalance) ----
    train_labels_np = np.array(train_labels)
    class_weights = compute_class_weight(
        'balanced', classes=np.unique(train_labels_np), y=train_labels_np
    )
    class_w_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print(f"Class weights: {dict(zip(label_names, class_weights.round(3)))}")

    # ---- Phase 2: Train ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = TransformerSERHead(
        input_dim=768, d_model=D_MODEL, num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS, dim_ff=DIM_FF,
        num_classes=len(label_names), dropout=DROPOUT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    total_steps   = EPOCHS * len(train_loader)
    warmup_steps  = int(WARMUP_FRACTION * total_steps)
    scheduler     = warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)

    # Label smoothing prevents overconfidence; class weights handle imbalance
    criterion = nn.CrossEntropyLoss(
        weight=class_w_tensor.to(device),
        label_smoothing=LABEL_SMOOTHING,
    )

    best_val_f1 = 0.0
    patience_ctr = 0
    train_losses, val_losses, val_accs, val_f1s = [], [], [], []

    print(f"Training up to {EPOCHS} epochs (early stop on val F1, patience={PATIENCE})...")
    print("-" * 65)

    for epoch in range(EPOCHS):
        # ---- Train ----
        model.train()
        epoch_loss = 0.0

        for feats, labels, pad_mask in train_loader:
            feats    = feats.to(device)
            labels   = labels.to(device)
            pad_mask = pad_mask.to(device)

            # Mixup in the hidden-state space
            feats_m, y_a, y_b, lam, mask_m = mixup_batch(feats, labels, pad_mask)

            optimizer.zero_grad()
            logits = model(feats_m, src_key_padding_mask=mask_m)
            loss   = mixup_loss(criterion, logits, y_a, y_b, lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        preds, trues = [], []

        with torch.no_grad():
            for feats, labels, pad_mask in val_loader:
                feats    = feats.to(device)
                labels   = labels.to(device)
                pad_mask = pad_mask.to(device)

                logits = model(feats, src_key_padding_mask=pad_mask)
                loss   = criterion(logits, labels)
                val_loss += loss.item()

                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                trues.extend(labels.cpu().numpy())

        val_losses.append(val_loss / len(val_loader))
        acc = accuracy_score(trues, preds)
        f1  = f1_score(trues, preds, average='weighted', zero_division=0)
        val_accs.append(acc)
        val_f1s.append(f1)

        lr_now = optimizer.param_groups[0]['lr']
        improved = f1 > best_val_f1

        if improved:
            best_val_f1 = f1
            patience_ctr = 0
            torch.save(model.state_dict(), OUTPUT_MODEL_DIR / "emotion_classifier.pth")
            tag = "** BEST **"
        else:
            patience_ctr += 1
            tag = f"patience {patience_ctr}/{PATIENCE}"

        print(
            f"Epoch {epoch+1:3d}/{EPOCHS} | "
            f"Loss {train_losses[-1]:.4f}/{val_losses[-1]:.4f} | "
            f"Acc {acc:.4f} | F1 {f1:.4f} | "
            f"LR {lr_now:.2e} | {tag}"
        )

        if patience_ctr >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}.")
            break

    # Save final (not necessarily best) model
    torch.save(model.state_dict(), OUTPUT_MODEL_DIR / "emotion_classifier_final.pth")

    # ---- Save model info ----
    model_info = {
        "architecture": "Wav2Vec2 (frozen) + TransformerSERHead (CLS, pre-norm)",
        "feature_extractor": "facebook/wav2vec2-base (frozen, Wav2Vec2Processor)",
        "input_type": "wav2vec2_hidden_states",
        "cache_version": CACHE_VERSION,
        "target_sr": TARGET_SR,
        "max_audio_len": MAX_AUDIO_LEN,
        "num_classes": int(len(label_names)),
        "class_names": list(label_names),
        "d_model": D_MODEL,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "dim_ff": DIM_FF,
        "dropout": DROPOUT,
        "classifier_params": n_params,
        "best_val_f1": round(best_val_f1, 4),
        "best_val_accuracy": round(max(val_accs), 4),
        "test_accuracy": None,
        "test_f1": None,
    }
    with open(OUTPUT_MODEL_DIR / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=4)

    # ---- Plots ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(train_losses, label="Train", linewidth=2)
    axes[0].plot(val_losses,   label="Val",   linewidth=2)
    axes[0].set_title("Loss Curves", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(val_accs, label="Val Accuracy",    color='steelblue',  linewidth=2)
    axes[1].plot(val_f1s,  label="Val F1 (weighted)", color='darkorange', linewidth=2)
    axes[1].axhline(y=best_val_f1, color='red', linestyle='--',
                    label=f"Best F1: {best_val_f1:.4f}")
    axes[1].set_title("Validation Metrics", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    gap = [v - t for t, v in zip(train_losses, val_losses)]
    axes[2].plot(gap, color='darkorange', linewidth=2)
    axes[2].axhline(y=0, color='black', linewidth=0.8)
    axes[2].set_title("Generalisation Gap (Val − Train Loss)",
                      fontsize=14, fontweight='bold')
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss Gap")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_curves.png", dpi=200)
    plt.close()

    print("\n" + "=" * 65)
    print("TRAINING COMPLETE")
    print(f"  Best Val F1       : {best_val_f1:.4f}")
    print(f"  Best Val Accuracy : {max(val_accs):.4f}")
    print(f"  Model saved       : {OUTPUT_MODEL_DIR / 'emotion_classifier.pth'}")
    print(f"\nNext steps:")
    print("  python test_emotion_classifier.py")
    print("  python app.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
