"""
comment_toxicity_detection_pytorch.py
=======================================
Full pipeline for Comment Toxicity Detection using PyTorch.
This is a complete replacement for the TensorFlow version.

What changed vs the TensorFlow version:
  - tensorflow / keras  →  torch, torch.nn, torch.utils.data
  - Sequential model    →  Custom nn.Module class (BiLSTMClassifier)
  - Tokenizer (Keras)   →  Manual vocabulary builder using collections.Counter
  - pad_sequences       →  Manual padding using torch.nn.utils.rnn.pad_sequence
  - model.fit()         →  Manual training loop (for epoch / for batch)
  - ModelCheckpoint     →  torch.save() / torch.load()
  - EarlyStopping       →  Manual best-loss tracking with patience counter

Everything else (EDA, preprocessing, evaluation, Streamlit artifacts) is the same.

Run this script with:
    python comment_toxicity_detection_pytorch.py

Make sure train.csv and test.csv are in the same folder.
"""

# =============================================================================
# STEP 1 — Import All Required Libraries
# =============================================================================

import os
import re
import string
import pickle
import warnings
import numpy as np
import pandas as pd
import collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    hamming_loss, roc_auc_score
)

# ── PyTorch imports (replaces all of tensorflow.keras) ──────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

warnings.filterwarnings('ignore')

# Fix random seeds so results are reproducible
np.random.seed(42)
torch.manual_seed(42)

# Use GPU if available, otherwise CPU
# PyTorch requires you to explicitly move data and model to the device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 60)
print("   COMMENT TOXICITY DETECTION — PyTorch PIPELINE")
print("=" * 60)
print(f"PyTorch version : {torch.__version__}")
print(f"Device in use   : {DEVICE}")
print("All libraries imported successfully!\n")


# =============================================================================
# CONSTANTS
# =============================================================================

TRAIN_FILE     = "train.csv"
TEST_FILE      = "test.csv"
MAX_VOCAB      = 30000       # Keep top 30,000 most common words
MAX_SEQ_LEN    = 200         # Pad / truncate comments to 200 words
EMBEDDING_DIM  = 64          # Each word → 64-dimension vector
HIDDEN_DIM     = 64          # Number of LSTM hidden units (each direction)
BATCH_SIZE     = 128
EPOCHS         = 5
VAL_SPLIT      = 0.2
THRESHOLD      = 0.5
LABEL_COLS     = [
    'toxic', 'severe_toxic', 'obscene',
    'threat', 'insult', 'identity_hate'
]
PLOTS_DIR      = "plots"
MODEL_PATH     = "toxicity_model_pytorch.pth"   # .pth is the PyTorch convention
VOCAB_PATH     = "vocab_pytorch.pkl"            # We save the vocabulary separately
OUTPUT_CSV     = "test_predictions_pytorch.csv"

# Special tokens used in the vocabulary
PAD_TOKEN = "<PAD>"   # Used to pad short sequences to MAX_SEQ_LEN
OOV_TOKEN = "<OOV>"   # Used for words not seen during training


# =============================================================================
# STEP 2 — Load the Dataset
# =============================================================================
import os

os.chdir(r"D:\Python\Project\Project 5")
print("Current Directory:", os.getcwd())

def load_data(train_path: str, test_path: str):
    """Load training and test CSV files."""
    print("─" * 50)
    print("STEP 2 — Loading Dataset")
    print("─" * 50)

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    print(f"Train shape : {train_df.shape}")
    print(f"Test shape  : {test_df.shape}")
    print(f"\nColumns (train): {train_df.columns.tolist()}")
    print(f"Columns (test) : {test_df.columns.tolist()}")

    return train_df, test_df


# =============================================================================
# STEP 3 — Exploratory Data Analysis (EDA)
# =============================================================================

def run_eda(train_df: pd.DataFrame, plots_dir: str):
    """
    Perform EDA and save 4 visualisation plots.
    This step is identical to the TensorFlow version — EDA does not depend
    on which deep learning framework we use.
    """
    print("\n" + "─" * 50)
    print("STEP 3 — Exploratory Data Analysis (EDA)")
    print("─" * 50)

    os.makedirs(plots_dir, exist_ok=True)

    print(f"Missing values : {train_df.isnull().sum().sum()}")
    print(f"Duplicate rows : {train_df.duplicated().sum()}")

    label_counts = train_df[LABEL_COLS].sum().sort_values(ascending=False)
    print("\nComments per toxicity label:")
    for lbl, cnt in label_counts.items():
        print(f"  {lbl:<16}: {cnt:,}")

    clean_n = (train_df[LABEL_COLS].sum(axis=1) == 0).sum()
    toxic_n = (train_df[LABEL_COLS].sum(axis=1) >  0).sum()
    print(f"\nClean comments : {clean_n:,}  ({clean_n/len(train_df)*100:.1f}%)")
    print(f"Toxic comments : {toxic_n:,}  ({toxic_n/len(train_df)*100:.1f}%)")

    # Plot 1 — Label distribution
    plt.figure(figsize=(10, 5))
    label_counts.plot(kind='bar', color='tomato', edgecolor='black', alpha=0.85)
    plt.title('Number of Comments per Toxicity Label', fontsize=14, fontweight='bold')
    plt.xlabel('Toxicity Label')
    plt.ylabel('Count')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    p1 = os.path.join(plots_dir, 'label_distribution.png')
    plt.savefig(p1, dpi=150); plt.close()
    print(f"\nSaved plot : {p1}")

    # Plot 2 — Class imbalance
    plt.figure(figsize=(6, 5))
    bars = plt.bar(['Clean Comments', 'Toxic Comments'], [clean_n, toxic_n],
                   color=['steelblue', 'tomato'], edgecolor='black')
    plt.title('Class Imbalance: Clean vs Toxic', fontsize=13, fontweight='bold')
    plt.ylabel('Number of Comments')
    for bar, val in zip(bars, [clean_n, toxic_n]):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 500, f'{val:,}',
                 ha='center', fontweight='bold')
    plt.tight_layout()
    p2 = os.path.join(plots_dir, 'class_imbalance.png')
    plt.savefig(p2, dpi=150); plt.close()
    print(f"Saved plot : {p2}")

    # Plot 3 — Comment length distribution
    train_df['comment_length'] = train_df['comment_text'].str.len()
    mean_len = train_df['comment_length'].mean()
    plt.figure(figsize=(10, 4))
    plt.hist(train_df['comment_length'], bins=50,
             color='steelblue', edgecolor='black', alpha=0.8)
    plt.axvline(mean_len, color='red', linestyle='--',
                label=f'Mean: {mean_len:.0f} chars')
    plt.title('Distribution of Comment Lengths', fontsize=13, fontweight='bold')
    plt.xlabel('Number of Characters'); plt.ylabel('Count')
    plt.legend(); plt.tight_layout()
    p3 = os.path.join(plots_dir, 'comment_lengths.png')
    plt.savefig(p3, dpi=150); plt.close()
    print(f"Saved plot : {p3}")

    # Plot 4 — Label co-occurrence heatmap
    plt.figure(figsize=(8, 6))
    cooccur = train_df[LABEL_COLS].T.dot(train_df[LABEL_COLS])
    sns.heatmap(cooccur, annot=True, fmt='d', cmap='Reds', linewidths=0.5)
    plt.title('Label Co-occurrence Heatmap', fontsize=13, fontweight='bold')
    plt.tight_layout()
    p4 = os.path.join(plots_dir, 'label_cooccurrence.png')
    plt.savefig(p4, dpi=150); plt.close()
    print(f"Saved plot : {p4}")


# =============================================================================
# STEP 4 — Text Preprocessing
# =============================================================================

def clean_text(text: str) -> str:
    """
    Clean a raw comment string.
    Same logic as the TensorFlow version — preprocessing is framework-agnostic.
    """
    text = text.lower()
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_texts(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Apply clean_text() to all comments."""
    print("\n" + "─" * 50)
    print("STEP 4 — Text Preprocessing")
    print("─" * 50)

    print("Cleaning training comments...")
    train_df['clean_text'] = train_df['comment_text'].apply(clean_text)
    print("Cleaning test comments...")
    test_df['clean_text']  = test_df['comment_text'].apply(clean_text)

    sample_raw = train_df['comment_text'].iloc[3]
    sample_cln = train_df['clean_text'].iloc[3]
    print(f"\nSample raw     : {sample_raw[:100]}")
    print(f"Sample cleaned : {sample_cln[:100]}")
    print("Text preprocessing done!")

    return train_df, test_df


# =============================================================================
# STEP 5 — Build Vocabulary and Encode Sequences (replaces Keras Tokenizer)
# =============================================================================

def build_vocabulary(texts: list, max_vocab: int = MAX_VOCAB) -> dict:
    """
    Build a word → integer ID mapping from a list of cleaned comment strings.

    In TensorFlow we used Keras Tokenizer().
    In PyTorch we build the vocabulary manually using collections.Counter.

    Steps:
      1. Count how often every word appears across all comments.
      2. Keep only the top max_vocab most common words.
      3. Add special tokens PAD (id=0) and OOV (id=1).
      4. Return a dictionary: { word: integer_id }
    """
    print("\n" + "─" * 50)
    print("STEP 5 — Building Vocabulary")
    print("─" * 50)

    # Count every word in every comment
    word_counts = collections.Counter()
    for text in texts:
        word_counts.update(text.split())

    # Keep only the top max_vocab words
    most_common = word_counts.most_common(max_vocab)

    # Assign IDs: 0 = PAD, 1 = OOV, 2+ = real words
    vocab = {PAD_TOKEN: 0, OOV_TOKEN: 1}
    for idx, (word, _) in enumerate(most_common, start=2):
        vocab[word] = idx

    print(f"Total unique words found : {len(word_counts):,}")
    print(f"Vocabulary size kept     : {len(vocab):,}  (including PAD + OOV)")

    return vocab


def encode_and_pad(texts: list, vocab: dict,
                   max_seq_len: int = MAX_SEQ_LEN) -> np.ndarray:
    """
    Convert a list of cleaned comment strings into a 2D NumPy array
    of shape (num_comments, max_seq_len).

    Each word is replaced by its integer ID from vocab.
    Unknown words get OOV id. Short comments are padded with 0 (PAD).
    Long comments are truncated to max_seq_len words.

    In TensorFlow we used:
        tokenizer.texts_to_sequences() + pad_sequences()
    In PyTorch we do this manually.
    """
    oov_id = vocab[OOV_TOKEN]
    pad_id = vocab[PAD_TOKEN]

    encoded = []
    for text in texts:
        # Convert each word to its ID (use OOV if not in vocab)
        ids = [vocab.get(word, oov_id) for word in text.split()]
        # Truncate if too long
        ids = ids[:max_seq_len]
        # Pad with zeros if too short
        ids += [pad_id] * (max_seq_len - len(ids))
        encoded.append(ids)

    return np.array(encoded, dtype=np.int64)


def prepare_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Build vocabulary, encode all texts, and split into train/validation sets.
    Returns NumPy arrays (later converted to PyTorch tensors inside the Dataset).
    """
    print("\n" + "─" * 50)
    print("STEP 5 — Tokenization and Encoding")
    print("─" * 50)

    # Build vocabulary ONLY from training data (never look at test data)
    vocab = build_vocabulary(train_df['clean_text'].tolist(), MAX_VOCAB)

    # Encode all comments to integer sequences + pad to same length
    print("\nEncoding and padding training comments...")
    X_all = encode_and_pad(train_df['clean_text'].tolist(), vocab, MAX_SEQ_LEN)

    print("Encoding and padding test comments...")
    X_test = encode_and_pad(test_df['clean_text'].tolist(), vocab, MAX_SEQ_LEN)

    # Labels
    y_all = train_df[LABEL_COLS].values.astype(np.float32)

    # Train / validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=VAL_SPLIT, random_state=42
    )

    print(f"\nX_train shape : {X_train.shape}")
    print(f"X_val shape   : {X_val.shape}")
    print(f"X_test shape  : {X_test.shape}")
    print(f"y_train shape : {y_train.shape}")

    return X_train, X_val, y_train, y_val, X_test, vocab


# =============================================================================
# STEP 5b — PyTorch Dataset and DataLoader
# =============================================================================

class ToxicCommentDataset(Dataset):
    """
    PyTorch Dataset class.

    In TensorFlow, model.fit() accepts raw NumPy arrays directly.
    In PyTorch, we wrap our data in a Dataset + DataLoader.

    A Dataset needs two methods:
      __len__  — returns number of samples
      __getitem__ — returns one (input, label) pair as PyTorch tensors
    """

    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        # Convert NumPy arrays to PyTorch tensors
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def create_dataloaders(X_train, y_train, X_val, y_val):
    """
    Wrap data in ToxicCommentDataset and create DataLoaders.

    DataLoader handles:
      - Batching (groups BATCH_SIZE samples together)
      - Shuffling (shuffle=True for training data)
      - Automatic conversion to tensors (handled by Dataset)
    """
    train_dataset = ToxicCommentDataset(X_train, y_train)
    val_dataset   = ToxicCommentDataset(X_val,   y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=256,         shuffle=False)

    print(f"\nTrain batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}")

    return train_loader, val_loader


# =============================================================================
# STEP 6 — Build the Bidirectional LSTM Model (PyTorch nn.Module)
# =============================================================================

class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM model for multi-label toxicity classification.

    In TensorFlow we used keras.Sequential([...]).
    In PyTorch we define a class that inherits from nn.Module.

    The __init__ method defines all the layers.
    The forward method defines how data flows through those layers.

    Architecture (same as TensorFlow version):
        Embedding → BiLSTM → Global Max Pool → Dense(relu) → Dropout → Dense(sigmoid)
    """

    def __init__(self, vocab_size: int, embedding_dim: int,
                 hidden_dim: int, num_labels: int,
                 max_seq_len: int, dropout: float = 0.3):
        super(BiLSTMClassifier, self).__init__()

        # Layer 1: Embedding
        # Converts word IDs (integers) into dense vectors of size embedding_dim
        # padding_idx=0 tells the model to ignore PAD tokens (ID=0)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        # Layer 2: Bidirectional LSTM
        # batch_first=True means input shape is (batch, seq_len, features)
        # bidirectional=True means it reads left-to-right AND right-to-left
        # Output will have size hidden_dim * 2 (both directions concatenated)
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Layer 3: Dense layer (fully connected)
        # Input: hidden_dim * 2 (because BiLSTM doubles the size)
        # Output: 64 features
        self.fc1 = nn.Linear(hidden_dim * 2, 64)

        # Layer 4: Dropout — randomly zeros 30% of neurons during training
        self.dropout = nn.Dropout(dropout)

        # Layer 5: Output layer
        # 64 inputs → num_labels outputs (one per toxicity category)
        self.fc2 = nn.Linear(64, num_labels)

        # Activation functions
        self.relu    = nn.ReLU()
        self.sigmoid = nn.Sigmoid()   # sigmoid outputs probability 0–1 per label

    def forward(self, x):
        """
        Define the forward pass — how input flows through each layer.
        In TensorFlow this is done automatically by Sequential.
        In PyTorch we write this explicitly.
        """
        # x shape: (batch_size, max_seq_len)

        # Step 1: Embedding lookup
        embedded = self.embedding(x)
        # embedded shape: (batch_size, max_seq_len, embedding_dim)

        # Step 2: BiLSTM
        # lstm_out contains the output at every time step
        lstm_out, _ = self.bilstm(embedded)
        # lstm_out shape: (batch_size, max_seq_len, hidden_dim * 2)

        # Step 3: Global Max Pooling
        # In TensorFlow: GlobalMaxPooling1D()
        # In PyTorch: torch.max() across the sequence dimension (dim=1)
        pooled, _ = torch.max(lstm_out, dim=1)
        # pooled shape: (batch_size, hidden_dim * 2)

        # Step 4: Dense layer with ReLU activation
        out = self.relu(self.fc1(pooled))
        # out shape: (batch_size, 64)

        # Step 5: Dropout
        out = self.dropout(out)

        # Step 6: Output layer with Sigmoid activation
        out = self.sigmoid(self.fc2(out))
        # out shape: (batch_size, num_labels)  — probabilities for each label

        return out


def build_model(vocab_size: int) -> BiLSTMClassifier:
    """Build the BiLSTM model and move it to the correct device (CPU/GPU)."""
    print("\n" + "─" * 50)
    print("STEP 6 — Building the PyTorch Model")
    print("─" * 50)

    model = BiLSTMClassifier(
        vocab_size    = vocab_size,
        embedding_dim = EMBEDDING_DIM,
        hidden_dim    = HIDDEN_DIM,
        num_labels    = len(LABEL_COLS),
        max_seq_len   = MAX_SEQ_LEN,
        dropout       = 0.3
    )

    # Move model to GPU if available
    model = model.to(DEVICE)

    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"\nTotal trainable parameters : {total_params:,}")

    return model


# =============================================================================
# STEP 7 — Train the Model (Manual Training Loop)
# =============================================================================

def train_model(model: BiLSTMClassifier,
                train_loader: DataLoader,
                val_loader:   DataLoader):
    """
    Train the model using a manual epoch loop.

    In TensorFlow this was one line:  model.fit(X_train, y_train, ...)
    In PyTorch we write the training loop ourselves. This gives more control.

    Each epoch:
      1. Loop through training batches → compute loss → backpropagate → update weights
      2. Evaluate on validation batches → compute val loss
      3. If val loss is the best so far → save the model
      4. If val loss hasn't improved for `patience` epochs → stop early

    Returns the history dictionary (loss per epoch) for plotting.
    """
    print("\n" + "─" * 50)
    print("STEP 7 — Training the Model")
    print("─" * 50)
    print(f"Epochs     : {EPOCHS}")
    print(f"Batch size : {BATCH_SIZE}")
    print(f"Device     : {DEVICE}")

    # Optimizer: Adam (same as TensorFlow version)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Loss function: BCELoss = Binary Cross-Entropy
    # (same as binary_crossentropy in TensorFlow)
    # We use BCELoss because model already applies sigmoid in forward()
    criterion = nn.BCELoss()

    # Track training history for plotting
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc':  [], 'val_acc':  []
    }

    # Early stopping variables
    best_val_loss = float('inf')
    patience      = 2
    patience_counter = 0

    print("\nStarting training...\n")

    for epoch in range(1, EPOCHS + 1):

        # ── Training phase ────────────────────────────────────
        model.train()    # Tell PyTorch we are in training mode (enables Dropout)
        train_loss = 0.0
        train_correct = 0
        train_total   = 0

        for batch_X, batch_y in train_loader:
            # Move batch to GPU/CPU
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            # Zero the gradients from the previous batch
            # (PyTorch accumulates gradients by default — we reset them each batch)
            optimizer.zero_grad()

            # Forward pass — get predictions
            predictions = model(batch_X)

            # Calculate loss
            loss = criterion(predictions, batch_y)

            # Backward pass — compute gradients
            loss.backward()

            # Update model weights
            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)

            # Calculate accuracy (predicted label matches true label)
            pred_labels  = (predictions >= THRESHOLD).float()
            train_correct += (pred_labels == batch_y).all(dim=1).sum().item()
            train_total   += batch_X.size(0)

        # ── Validation phase ──────────────────────────────────
        model.eval()     # Tell PyTorch we are evaluating (disables Dropout)
        val_loss    = 0.0
        val_correct = 0
        val_total   = 0

        with torch.no_grad():   # No gradient calculation needed for validation
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)

                val_loss += loss.item() * batch_X.size(0)
                pred_labels = (predictions >= THRESHOLD).float()
                val_correct += (pred_labels == batch_y).all(dim=1).sum().item()
                val_total   += batch_X.size(0)

        # ── Calculate average losses and accuracies ───────────
        avg_train_loss = train_loss / train_total
        avg_val_loss   = val_loss   / val_total
        avg_train_acc  = train_correct / train_total
        avg_val_acc    = val_correct   / val_total

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_acc'].append(avg_val_acc)

        print(f"Epoch {epoch}/{EPOCHS}  |  "
              f"Train Loss: {avg_train_loss:.4f}  Train Acc: {avg_train_acc:.4f}  |  "
              f"Val Loss: {avg_val_loss:.4f}  Val Acc: {avg_val_acc:.4f}")

        # ── Save best model (replaces ModelCheckpoint callback) ──
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # torch.save saves the model weights (state_dict)
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✅ Best model saved → {MODEL_PATH}  (val_loss improved to {best_val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  ⚠️  No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"\n  Early stopping triggered after epoch {epoch}!")
                break

    # Load the best weights back into the model
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("\nTraining complete! Best model weights restored.")

    return history


# =============================================================================
# STEP 8 — Plot Training History
# =============================================================================

def plot_training_history(history: dict, plots_dir: str):
    """Save training loss and accuracy curves."""
    print("\n" + "─" * 50)
    print("STEP 8 — Plotting Training History")
    print("─" * 50)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    axes[0].plot(history['train_loss'], label='Train Loss',      color='steelblue', marker='o')
    axes[0].plot(history['val_loss'],   label='Validation Loss', color='tomato',    marker='s')
    axes[0].set_title('Model Loss over Epochs', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Binary Cross-Entropy Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curve
    axes[1].plot(history['train_acc'], label='Train Accuracy',      color='steelblue', marker='o')
    axes[1].plot(history['val_acc'],   label='Validation Accuracy', color='tomato',    marker='s')
    axes[1].set_title('Model Accuracy over Epochs', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Exact Match Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plots_dir, 'training_history.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved plot : {path}")

    last = len(history['train_loss']) - 1
    print(f"Final train loss     : {history['train_loss'][last]:.4f}")
    print(f"Final val loss       : {history['val_loss'][last]:.4f}")
    print(f"Final train accuracy : {history['train_acc'][last]:.4f}")
    print(f"Final val accuracy   : {history['val_acc'][last]:.4f}")


# =============================================================================
# STEP 9 — Evaluate the Model
# =============================================================================

def evaluate_model(model: BiLSTMClassifier,
                   X_val: np.ndarray,
                   y_val: np.ndarray,
                   plots_dir: str):
    """
    Evaluate on the validation set.
    In TensorFlow model.predict() works directly on NumPy arrays.
    In PyTorch we use a DataLoader and call model() inside torch.no_grad().
    """
    print("\n" + "─" * 50)
    print("STEP 9 — Evaluating the Model")
    print("─" * 50)

    model.eval()

    val_dataset = ToxicCommentDataset(X_val)
    val_loader  = DataLoader(val_dataset, batch_size=256, shuffle=False)

    all_probs = []
    with torch.no_grad():
        for batch_X in val_loader:
            batch_X = batch_X.to(DEVICE)
            probs   = model(batch_X)
            all_probs.append(probs.cpu().numpy())   # move back to CPU + NumPy

    y_pred_prob = np.vstack(all_probs)
    y_pred      = (y_pred_prob >= THRESHOLD).astype(int)

    # Per-label metrics
    print(f"\n{'Label':<16}  {'Precision':>9}  {'Recall':>7}  {'F1':>6}")
    print("-" * 44)
    auc_scores = []
    for i, label in enumerate(LABEL_COLS):
        f1  = f1_score(y_val[:, i],  y_pred[:, i],  zero_division=0)
        pre = precision_score(y_val[:, i], y_pred[:, i], zero_division=0)
        rec = recall_score(y_val[:, i],    y_pred[:, i], zero_division=0)
        auc = roc_auc_score(y_val[:, i], y_pred_prob[:, i])
        auc_scores.append(auc)
        print(f"{label:<16}  {pre:>9.3f}  {rec:>7.3f}  {f1:>6.3f}")

    h_loss  = hamming_loss(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_prob, average='macro')
    print(f"\nHamming Loss    : {h_loss:.4f}  (lower is better)")
    print(f"ROC-AUC (macro) : {roc_auc:.4f}  (higher is better, max=1)")

    # ROC-AUC bar chart
    colors = ['#e74c3c','#e67e22','#f1c40f','#2ecc71','#3498db','#9b59b6']
    plt.figure(figsize=(9, 5))
    bars = plt.bar([l.replace('_', ' ').title() for l in LABEL_COLS],
                   auc_scores, color=colors, edgecolor='black', alpha=0.85)
    plt.ylim(0, 1)
    plt.title('ROC-AUC Score per Toxicity Label', fontsize=13, fontweight='bold')
    plt.ylabel('ROC-AUC Score')
    plt.xticks(rotation=20, ha='right')
    for bar, score in zip(bars, auc_scores):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01,
                 f'{score:.3f}', ha='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(plots_dir, 'roc_auc_scores.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved plot : {path}")

    return h_loss, roc_auc


# =============================================================================
# STEP 10 — Predict on a Single Comment
# =============================================================================

def predict_toxicity(comment: str,
                     model:     BiLSTMClassifier,
                     vocab:     dict,
                     max_seq_len: int   = MAX_SEQ_LEN,
                     threshold:  float  = THRESHOLD) -> dict:
    """
    Predict toxicity for one raw comment string.
    Used both here and by the Streamlit app (app.py).
    """
    # Clean and encode the comment
    cleaned = clean_text(comment)
    encoded = encode_and_pad([cleaned], vocab, max_seq_len)  # shape: (1, max_seq_len)

    # Convert to PyTorch tensor and move to device
    tensor = torch.tensor(encoded, dtype=torch.long).to(DEVICE)

    # Run inference
    model.eval()
    with torch.no_grad():
        probs = model(tensor)[0].cpu().numpy()   # shape: (6,)

    return {label: float(prob) for label, prob in zip(LABEL_COLS, probs)}


def test_sample_comments(model: BiLSTMClassifier, vocab: dict):
    """Test predict_toxicity() on a few hard-coded examples."""
    print("\n" + "─" * 50)
    print("STEP 10 — Testing on Sample Comments")
    print("─" * 50)

    sample_comments = [
        "I love your work! This is really amazing and helpful.",
        "You are so stupid and worthless, nobody likes you!",
        "I will find you and make you regret this, just wait.",
        "Great article, very informative and well written.",
        "This is complete garbage written by an idiot.",
    ]

    for comment in sample_comments:
        results = predict_toxicity(comment, model, vocab)
        display = comment[:70] + "..." if len(comment) > 70 else comment
        print(f"\nComment: \"{display}\"")
        print("-" * 55)
        for label, prob in results.items():
            flag = "🚨 DETECTED" if prob >= THRESHOLD else "✅ Clean"
            print(f"  {label:<16}: {prob:.3f}  {flag}")


# =============================================================================
# STEP 11 — Generate Predictions on Test Set
# =============================================================================

def generate_test_predictions(model:      BiLSTMClassifier,
                               X_test:     np.ndarray,
                               test_df:    pd.DataFrame,
                               output_path: str):
    """Run model on the full test set and save predictions as CSV."""
    print("\n" + "─" * 50)
    print("STEP 11 — Generating Test Set Predictions")
    print("─" * 50)

    model.eval()
    test_dataset = ToxicCommentDataset(X_test)
    test_loader  = DataLoader(test_dataset, batch_size=256, shuffle=False)

    all_probs = []
    print(f"Predicting on {len(test_df):,} test comments...")

    with torch.no_grad():
        for batch_X in test_loader:
            batch_X = batch_X.to(DEVICE)
            probs   = model(batch_X)
            all_probs.append(probs.cpu().numpy())

    preds = np.vstack(all_probs)

    submission = pd.DataFrame(preds, columns=LABEL_COLS)
    submission.insert(0, 'id', test_df['id'])
    submission.to_csv(output_path, index=False)

    print(f"Saved : {output_path}")
    print(submission.head())


# =============================================================================
# STEP 12 — Save Model and Vocabulary
# =============================================================================

def save_artifacts(model: BiLSTMClassifier, vocab: dict,
                   model_path: str = MODEL_PATH,
                   vocab_path:  str = VOCAB_PATH):
    """
    Save the trained PyTorch model weights and vocabulary dictionary.

    In TensorFlow:
        model.save('model.h5')                   → saves full model
        pickle.dump(tokenizer, f)                → saves tokenizer

    In PyTorch:
        torch.save(model.state_dict(), path)     → saves weights only
        pickle.dump(vocab, f)                    → saves vocabulary dict

    NOTE: To reload a PyTorch model you must:
      1. Re-create the model architecture (BiLSTMClassifier(...))
      2. Load the saved weights:  model.load_state_dict(torch.load(path))
    """
    print("\n" + "─" * 50)
    print("STEP 12 — Saving Model and Vocabulary")
    print("─" * 50)

    # Save model weights (already saved best during training, saving again explicitly)
    torch.save(model.state_dict(), model_path)
    print(f"Model weights saved : {model_path}")

    # Save vocabulary
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Vocabulary saved    : {vocab_path}")

    # Also save model config so we can rebuild the architecture on load
    model_config = {
        'vocab_size':    len(vocab),
        'embedding_dim': EMBEDDING_DIM,
        'hidden_dim':    HIDDEN_DIM,
        'num_labels':    len(LABEL_COLS),
        'max_seq_len':   MAX_SEQ_LEN,
    }
    config_path = 'model_config_pytorch.pkl'
    with open(config_path, 'wb') as f:
        pickle.dump(model_config, f)
    print(f"Model config saved  : {config_path}")
    print("Artifacts ready for Streamlit deployment!")


# =============================================================================
# STEP 13 — Print Final Summary
# =============================================================================

def print_summary(train_df, history, h_loss, roc_auc):
    """Print final pipeline summary."""
    print("\n" + "=" * 60)
    print("   COMMENT TOXICITY DETECTION (PyTorch) — FINAL SUMMARY")
    print("=" * 60)
    print(f"Framework           : PyTorch {torch.__version__}")
    print(f"Training comments   : {len(train_df):,}")
    print(f"Toxicity labels     : {len(LABEL_COLS)}  (multi-label)")
    print(f"Vocabulary size     : {MAX_VOCAB:,} words")
    print(f"Sequence length     : {MAX_SEQ_LEN} tokens")
    print(f"Epochs trained      : {len(history['train_loss'])}")
    print(f"Device used         : {DEVICE}")
    print(f"Hamming Loss        : {h_loss:.4f}")
    print(f"ROC-AUC (macro)     : {roc_auc:.4f}")
    print()
    print("Files generated:")
    print(f"  -> {MODEL_PATH}           (PyTorch model weights)")
    print(f"  -> {VOCAB_PATH}           (vocabulary dictionary)")
    print(f"  -> model_config_pytorch.pkl  (architecture config)")
    print(f"  -> {OUTPUT_CSV}")
    print(f"  -> plots/  (6 visualisation images)")
    print()
    print("To launch the Streamlit web app, run:")
    print("    streamlit run app_pytorch.py")
    print("=" * 60)


# =============================================================================
# MAIN — Run the Full Pipeline
# =============================================================================

if __name__ == "__main__":

    # Step 2 — Load data
    train_df, test_df = load_data(TRAIN_FILE, TEST_FILE)

    # Step 3 — EDA
    run_eda(train_df, PLOTS_DIR)

    # Step 4 — Clean text
    train_df, test_df = preprocess_texts(train_df, test_df)

    # Step 5 — Build vocab, encode, pad, split
    X_train, X_val, y_train, y_val, X_test, vocab = prepare_data(train_df, test_df)

    # Step 5b — Create DataLoaders
    train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val)

    # Step 6 — Build model
    model = build_model(vocab_size=len(vocab))

    # Step 7 — Train
    history = train_model(model, train_loader, val_loader)

    # Step 8 — Plot training history
    plot_training_history(history, PLOTS_DIR)

    # Step 9 — Evaluate
    h_loss, roc_auc = evaluate_model(model, X_val, y_val, PLOTS_DIR)

    # Step 10 — Sample predictions
    test_sample_comments(model, vocab)

    # Step 11 — Test set predictions
    generate_test_predictions(model, X_test, test_df, OUTPUT_CSV)

    # Step 12 — Save artifacts
    save_artifacts(model, vocab)

    # Step 13 — Summary
    print_summary(train_df, history, h_loss, roc_auc)
