"""
app_pytorch.py — Streamlit Web App for Comment Toxicity Detection (PyTorch)
=============================================================================
This is the PyTorch version of the Streamlit app.
It replaces:
  - load_model('toxicity_model.h5')         → BiLSTMClassifier + torch.load()
  - pickle.load(tokenizer.pkl)              → pickle.load(vocab_pytorch.pkl)
  - tokenizer.texts_to_sequences()          → manual encode_and_pad()
  - model.predict(padded)                   → model(tensor) inside torch.no_grad()

Run this app with:
    streamlit run app_pytorch.py

Required files in the same folder:
    toxicity_model_pytorch.pth
    vocab_pytorch.pkl
    model_config_pytorch.pkl
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import string
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Comment Toxicity Detector (PyTorch)",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Constants ──────────────────────────────────────────────────────────────────
MAX_SEQ_LEN  = 200
THRESHOLD    = 0.5
LABEL_COLS   = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
LABEL_COLORS = {
    'toxic':         '#e74c3c',
    'severe_toxic':  '#c0392b',
    'obscene':       '#e67e22',
    'threat':        '#8e44ad',
    'insult':        '#2980b9',
    'identity_hate': '#16a085'
}
LABEL_ICONS = {
    'toxic':         '🔴',
    'severe_toxic':  '🚨',
    'obscene':       '🟠',
    'threat':        '🟣',
    'insult':        '🔵',
    'identity_hate': '🟢'
}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PAD_TOKEN = "<PAD>"
OOV_TOKEN = "<OOV>"


# ── PyTorch Model Definition (must match training script exactly) ─────────────

class BiLSTMClassifier(nn.Module):
    """Same architecture as in comment_toxicity_detection_pytorch.py"""

    def __init__(self, vocab_size, embedding_dim, hidden_dim,
                 num_labels, max_seq_len, dropout=0.3):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bilstm    = nn.LSTM(embedding_dim, hidden_dim,
                                 batch_first=True, bidirectional=True)
        self.fc1       = nn.Linear(hidden_dim * 2, 64)
        self.dropout   = nn.Dropout(dropout)
        self.fc2       = nn.Linear(64, num_labels)
        self.relu      = nn.ReLU()
        self.sigmoid   = nn.Sigmoid()

    def forward(self, x):
        embedded      = self.embedding(x)
        lstm_out, _   = self.bilstm(embedded)
        pooled, _     = torch.max(lstm_out, dim=1)
        out           = self.relu(self.fc1(pooled))
        out           = self.dropout(out)
        return self.sigmoid(self.fc2(out))


# ── Helper Functions ──────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = text.lower()
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def encode_and_pad(texts: list, vocab: dict, max_seq_len: int = MAX_SEQ_LEN) -> np.ndarray:
    oov_id = vocab.get(OOV_TOKEN, 1)
    pad_id = vocab.get(PAD_TOKEN, 0)
    encoded = []
    for text in texts:
        ids = [vocab.get(w, oov_id) for w in text.split()]
        ids = ids[:max_seq_len]
        ids += [pad_id] * (max_seq_len - len(ids))
        encoded.append(ids)
    return np.array(encoded, dtype=np.int64)


@st.cache_resource
def load_artifacts():
    """Load the trained PyTorch model, vocabulary, and config (cached)."""
    with open('vocab_pytorch.pkl', 'rb') as f:
        vocab = pickle.load(f)
    with open('model_config_pytorch.pkl', 'rb') as f:
        config = pickle.load(f)

    model = BiLSTMClassifier(
        vocab_size    = config['vocab_size'],
        embedding_dim = config['embedding_dim'],
        hidden_dim    = config['hidden_dim'],
        num_labels    = config['num_labels'],
        max_seq_len   = config['max_seq_len'],
    )
    model.load_state_dict(torch.load('toxicity_model_pytorch.pth', map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return model, vocab


def predict_single(comment: str, model, vocab) -> dict:
    cleaned = clean_text(comment)
    encoded = encode_and_pad([cleaned], vocab, MAX_SEQ_LEN)
    tensor  = torch.tensor(encoded, dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        probs = model(tensor)[0].cpu().numpy()
    return {label: float(p) for label, p in zip(LABEL_COLS, probs)}


def predict_batch(texts: list, model, vocab) -> pd.DataFrame:
    cleaned = [clean_text(t) for t in texts]
    encoded = encode_and_pad(cleaned, vocab, MAX_SEQ_LEN)
    dataset = torch.utils.data.TensorDataset(torch.tensor(encoded, dtype=torch.long))
    loader  = DataLoader(dataset, batch_size=64, shuffle=False)
    all_probs = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            probs = model(batch).cpu().numpy()
            all_probs.append(probs)
    preds = np.vstack(all_probs)
    df = pd.DataFrame(preds, columns=LABEL_COLS)
    df.insert(0, 'comment', texts)
    return df


# ── Load Model ────────────────────────────────────────────────────────────────
try:
    model, vocab = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error   = str(e)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚨 Toxicity Detector")
    st.markdown("**Framework: PyTorch**")
    st.markdown("---")
    st.markdown("### About")
    st.info(
        "This app uses a **Bidirectional LSTM** built in **PyTorch** "
        "to detect toxic content in online comments across **6 toxicity labels**."
    )
    st.markdown("### Labels Detected")
    for label, icon in LABEL_ICONS.items():
        st.markdown(f"{icon} **{label.replace('_', ' ').title()}**")
    st.markdown("---")
    threshold = st.slider("Detection threshold", 0.1, 0.9, 0.5, 0.05,
                          help="Labels above this probability are flagged as toxic.")
    st.markdown("---")
    st.caption("Built with PyTorch + Streamlit")


# ── Main Page ─────────────────────────────────────────────────────────────────
st.title("🚨 Comment Toxicity Detection System")
st.markdown(
    "Detect **toxic, obscene, threatening, insulting, or hateful** content "
    "in online comments — powered by a **PyTorch Bidirectional LSTM** model."
)
st.markdown("---")

if not model_loaded:
    st.error(
        f"❌ Could not load model: {load_error}\n\n"
        "Make sure **toxicity_model_pytorch.pth**, **vocab_pytorch.pkl**, "
        "and **model_config_pytorch.pkl** are in the same folder as app_pytorch.py."
    )
    st.stop()

tab1, tab2, tab3 = st.tabs(["💬 Single Comment", "📂 Bulk CSV Upload", "📊 Dataset Insights"])


# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Analyze a Single Comment")
    comment_input = st.text_area("Enter your comment here:",
                                 placeholder="Type a comment to check for toxicity...",
                                 height=120)
    analyze_btn = st.button("🔍 Analyze", type="primary")

    if analyze_btn:
        if not comment_input.strip():
            st.warning("⚠️ Please enter a comment first.")
        else:
            with st.spinner("Running PyTorch inference..."):
                results = predict_single(comment_input, model, vocab)

            flagged   = [k for k, v in results.items() if v >= threshold]
            is_toxic  = len(flagged) > 0

            if is_toxic:
                st.error(f"🚨 **TOXIC CONTENT DETECTED** — Flagged as: "
                         f"**{', '.join(f.replace('_', ' ').title() for f in flagged)}**")
            else:
                st.success("✅ **This comment appears to be CLEAN.**")

            st.markdown("---")
            st.markdown("### Probability Scores per Label")
            cols = st.columns(3)
            for i, (label, score) in enumerate(results.items()):
                with cols[i % 3]:
                    flagged_label = score >= threshold
                    st.markdown(f"**{LABEL_ICONS[label]} {label.replace('_', ' ').title()}**")
                    st.progress(float(score))
                    badge = "🚨 DETECTED" if flagged_label else "✅ Clean"
                    st.markdown(
                        f"<span style='background-color:{'#fde8e8' if flagged_label else '#e8fdf0'};"
                        f"padding:3px 8px;border-radius:5px;font-size:0.85em'>"
                        f"Score: **{score:.3f}** — {badge}</span>",
                        unsafe_allow_html=True
                    )
                    st.markdown("")

            st.markdown("---")
            st.markdown("### Visual Summary")
            fig, ax = plt.subplots(figsize=(9, 4))
            bar_colors = [LABEL_COLORS[l] for l in results]
            bars = ax.bar([l.replace('_', ' ').title() for l in results],
                          list(results.values()), color=bar_colors,
                          edgecolor='black', alpha=0.85)
            ax.axhline(threshold, color='black', linestyle='--', linewidth=1.5,
                       label=f'Threshold = {threshold}')
            ax.set_ylim(0, 1)
            ax.set_ylabel('Toxicity Probability')
            ax.set_title('Toxicity Scores for Submitted Comment',
                         fontsize=13, fontweight='bold')
            ax.legend()
            plt.xticks(rotation=20, ha='right')
            for bar, val in zip(bars, results.values()):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02, f'{val:.2f}', ha='center', fontsize=10)
            plt.tight_layout()
            st.pyplot(fig); plt.close()


# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Bulk Toxicity Prediction via CSV Upload")
    st.markdown("Upload a CSV file with a column named **`comment_text`**.")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            upload_df = pd.read_csv(uploaded_file)
            if 'comment_text' not in upload_df.columns:
                st.error("❌ CSV must have a column named **`comment_text`**.")
            else:
                st.success(f"✅ Loaded {len(upload_df):,} comments.")
                st.dataframe(upload_df[['comment_text']].head(5))

                if st.button("🔍 Run Bulk Analysis", type="primary"):
                    with st.spinner(f"Analyzing {len(upload_df):,} comments..."):
                        results_df = predict_batch(upload_df['comment_text'].tolist(),
                                                   model, vocab)
                    for lbl in LABEL_COLS:
                        results_df[f'{lbl}_flag'] = (results_df[lbl] >= threshold).astype(int)
                    results_df['any_toxic'] = results_df[[f'{l}_flag' for l in LABEL_COLS]].max(axis=1)

                    st.markdown("### Results Preview")
                    st.dataframe(results_df.head(20))

                    total   = len(results_df)
                    n_toxic = results_df['any_toxic'].sum()
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Comments", f"{total:,}")
                    c2.metric("Toxic Comments", f"{n_toxic:,}")
                    c3.metric("Clean Comments", f"{total - n_toxic:,}")
                    c4.metric("Toxic %",        f"{n_toxic/total*100:.1f}%")

                    label_totals = {l: results_df[f'{l}_flag'].sum() for l in LABEL_COLS}
                    fig2, ax2 = plt.subplots(figsize=(9, 4))
                    ax2.bar([l.replace('_', ' ').title() for l in LABEL_COLS],
                            list(label_totals.values()),
                            color=[LABEL_COLORS[l] for l in LABEL_COLS],
                            edgecolor='black', alpha=0.85)
                    ax2.set_title('Detected Toxic Comments per Label',
                                  fontsize=13, fontweight='bold')
                    ax2.set_ylabel('Count')
                    plt.xticks(rotation=20, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig2); plt.close()

                    csv_out = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Download Results as CSV",
                                       data=csv_out,
                                       file_name="toxicity_predictions_pytorch.csv",
                                       mime="text/csv")
        except Exception as e:
            st.error(f"Error reading file: {e}")


# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Dataset Insights")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training Comments", "159,571")
    c2.metric("Test Comments",     "153,164")
    c3.metric("Toxicity Labels",   "6")
    c4.metric("Clean Comments",    "143,346 (89.8%)")

    st.markdown("---")
    st.markdown("### PyTorch vs TensorFlow — Key Differences")
    diff_data = {
        "Feature": ["Model definition", "Tokenizer", "Training loop",
                    "Saving model", "Loading model", "Inference"],
        "TensorFlow / Keras": [
            "keras.Sequential([...])",
            "Keras Tokenizer + pad_sequences()",
            "model.fit() — one line",
            "model.save('model.h5')",
            "load_model('model.h5')",
            "model.predict(numpy_array)"
        ],
        "PyTorch (this version)": [
            "class BiLSTMClassifier(nn.Module)",
            "Manual vocab Counter + encode_and_pad()",
            "Manual for epoch / for batch loop",
            "torch.save(model.state_dict(), path)",
            "Rebuild architecture + load_state_dict()",
            "model(tensor) inside torch.no_grad()"
        ]
    }
    st.dataframe(pd.DataFrame(diff_data), use_container_width=True)

    st.markdown("---")
    st.markdown("### PyTorch Model Architecture")
    st.code("""
BiLSTMClassifier(
  (embedding): Embedding(30002, 64, padding_idx=0)
  (bilstm)   : LSTM(64, 64, batch_first=True, bidirectional=True)
  (fc1)      : Linear(in=128, out=64)
  (dropout)  : Dropout(p=0.3)
  (fc2)      : Linear(in=64, out=6)
  (relu)     : ReLU()
  (sigmoid)  : Sigmoid()
)
    """, language="python")

    st.markdown("---")
    st.markdown("### Label Distribution in Training Data")
    label_data = {'toxic': 15294, 'severe_toxic': 1595, 'obscene': 8449,
                  'threat': 478,  'insult': 7877,  'identity_hate': 1405}
    fig3, ax3 = plt.subplots(figsize=(9, 4))
    bars3 = ax3.bar([l.replace('_', ' ').title() for l in label_data],
                    list(label_data.values()),
                    color=[LABEL_COLORS[l] for l in label_data],
                    edgecolor='black', alpha=0.85)
    for bar, val in zip(bars3, label_data.values()):
        ax3.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 50, f'{val:,}',
                 ha='center', fontsize=10, fontweight='bold')
    ax3.set_title('Number of Comments per Toxicity Label',
                  fontsize=13, fontweight='bold')
    ax3.set_ylabel('Count')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    st.pyplot(fig3); plt.close()
