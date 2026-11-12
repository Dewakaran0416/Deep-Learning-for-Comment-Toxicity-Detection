# 🚨 Deep Learning for Comment Toxicity Detection

> A **Bidirectional LSTM** deep learning model built in **PyTorch** that detects toxic content in online comments across **6 toxicity categories**, deployed as an interactive **Streamlit web application**.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?style=flat-square&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-Educational-green?style=flat-square)

---

## 📌 Project Overview

Online platforms face a massive challenge moderating millions of user comments every day. This project automates toxic comment detection using deep learning — analyzing text and predicting the likelihood of a comment being **toxic, obscene, threatening, insulting, or hateful** in real time.

The model is built in **PyTorch** with a hand-written training loop, which gives full control over every step: forward pass, loss, backpropagation and weight updates.

**Results on the 31,915-comment validation set:** macro ROC-AUC **0.974**, Hamming loss **0.018**. See [Results](#-results) for per-label scores and known limitations.

---

## 🗂️ Repository Structure

```
comment-toxicity-detection/
│
├── 📓 Notebook & Script
│   ├── Comment_Toxicity_Detection_PyTorch.ipynb # Jupyter Notebook (step-by-step)
│   └── comment_toxicity_detection_pytorch.py    # Full training pipeline
│
├── 🌐 Streamlit App
│   ├── app_pytorch.py                           # Web app
│   └── sample_comments.csv                      # 12 example comments for the bulk-upload tab
│
├── 📦 Trained Model (included, so the app runs without retraining)
│   ├── toxicity_model_pytorch.pth               # PyTorch model weights
│   ├── vocab_pytorch.pkl                        # Vocabulary dictionary
│   └── model_config_pytorch.pkl                 # Model architecture config
│
├── 📁 Data (add manually — not included, too large for GitHub)
│   ├── train.csv                                # 159,571 labeled comments
│   └── test.csv                                 # 153,164 comments for prediction
│
├── 🖼️ plots/
│   ├── label_distribution.png
│   ├── class_imbalance.png
│   ├── comment_lengths.png
│   ├── label_cooccurrence.png
│   ├── training_history.png
│   └── roc_auc_scores.png
│
├── README.md                                    # This file
├── requirements.txt                             # All Python dependencies
└── .gitignore
```

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Source** | Jigsaw Toxic Comment Classification Challenge (Kaggle) |
| **Training file** | `train.csv` — 159,571 comments with 6 binary labels |
| **Test file** | `test.csv` — 153,164 comments (no labels) |
| **Clean comments** | 143,346 (89.8%) |
| **Toxic comments** | 16,225 (10.2%) |
| **Missing values** | None |
| **Duplicates** | None |

### The 6 Toxicity Labels

| Label | Count | Description |
|---|---|---|
| `toxic` | 15,294 | Rude, disrespectful, or unreasonable content |
| `severe_toxic` | 1,595 | Extremely offensive and harmful content |
| `obscene` | 8,449 | Vulgar or sexually explicit language |
| `threat` | 478 | Threatens a person with harm or violence |
| `insult` | 7,877 | Directly attacks or belittles a person |
| `identity_hate` | 1,405 | Attacks based on race, religion, gender, etc. |

> ⚠️ A single comment can carry **multiple labels simultaneously** — this is a **multi-label classification** problem, not multi-class.

---

## ⚙️ Installation & Setup

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Dewakaran0416/Deep-Learning-for-Comment-Toxicity-Detection.git
cd Deep-Learning-for-Comment-Toxicity-Detection
```

### Step 2 — Install Dependencies

```bash
python -m venv venv
venv\Scripts\activate          # Windows  (macOS/Linux: source venv/bin/activate)
python -m pip install -r requirements.txt
```

> **Windows tip:** if `import torch` fails with `WinError 126 … torch_python.dll`, install the [Microsoft Visual C++ Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe) and reinstall the CPU build: `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`

### Step 3 — Try the App Right Away (optional)

The trained model is included, so you can skip training:

```bash
streamlit run app_pytorch.py
```

### Step 4 — Add the Dataset (only needed to retrain)

Download the dataset from [Kaggle — Jigsaw Toxic Comment Classification](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data) and place both files in the project root:

```
Deep-Learning-for-Comment-Toxicity-Detection/
├── train.csv
└── test.csv
```

### Step 5 — Train the Model

```bash
python comment_toxicity_detection_pytorch.py
```

This generates the model artifacts, plots, and predictions CSV in the project folder. It takes about 15–25 minutes on a CPU. All paths are relative to the script, so you can run it from any folder.

### Step 6 — Launch the Web App

```bash
streamlit run app_pytorch.py
```

Opens at: **http://localhost:8501**

---

## 🧠 Model Architecture

```
BiLSTMClassifier(
  Embedding          →  (batch, 200, 64)      — word ID to dense vector
  Bidirectional LSTM →  (batch, 200, 128)     — reads text forward + backward
  Global Max Pooling →  (batch, 128)          — picks most important feature
  Dense (ReLU)       →  (batch, 64)           — learns complex patterns
  Dropout (0.3)      →  (batch, 64)           — prevents overfitting
  Dense (Sigmoid)    →  (batch, 6)            — probability per label
)

Total Parameters : 1,995,334   (96% in the embedding layer)
Loss Function    : Binary Cross-Entropy  (multi-label)
Optimizer        : Adam  (lr = 0.001)
```

### Why Bidirectional LSTM?

A regular LSTM reads text left-to-right only. A **Bidirectional LSTM** reads it both **forward and backward** — this helps the model understand the full context of each word. For example, in "not a bad comment", the word "bad" needs the context of "not" which comes before it.

### Why Sigmoid instead of Softmax?

- **Softmax** is used when a comment belongs to **exactly one** class (e.g. cat vs dog).
- **Sigmoid** is used when a comment can belong to **multiple** classes simultaneously.
- Since one comment can be both `toxic` AND `insult` at the same time, we use sigmoid on each output independently.

---

## 🔄 Why PyTorch? How it compares to Keras

| Feature | TensorFlow / Keras | PyTorch (this project) |
|---|---|---|
| Model definition | `keras.Sequential([...])` | `class BiLSTMClassifier(nn.Module)` |
| Tokenizer | `Keras Tokenizer` | Manual `collections.Counter` vocab |
| Padding | `pad_sequences()` | Manual `encode_and_pad()` |
| Training loop | `model.fit()` — one line | Manual `for epoch / for batch` loop |
| Early stopping | `EarlyStopping` callback | Manual patience counter |
| Save model | `model.save('model.h5')` | `torch.save(model.state_dict(), path)` |
| Load model | `load_model('model.h5')` | Rebuild class + `load_state_dict()` |
| Inference | `model.predict(numpy_array)` | `model(tensor)` inside `torch.no_grad()` |
| Data pipeline | Raw NumPy arrays | `Dataset` + `DataLoader` classes |

---

## 🚀 Notebook Walkthrough

The Jupyter Notebook `Comment_Toxicity_Detection_PyTorch.ipynb` walks through all 13 steps:

| Step | Description |
|---|---|
| **Step 1** | Import all libraries |
| **Step 2** | Load train.csv and test.csv |
| **Step 3** | EDA — label distributions, class imbalance, comment lengths, co-occurrence heatmap |
| **Step 4** | Text preprocessing — clean_text() function |
| **Step 5** | Build vocabulary, encode texts, pad sequences, train/val split |
| **Step 6** | Build Bidirectional LSTM model |
| **Step 7** | Train model with a manual loop, early stopping (patience 2) and best-model saving |
| **Step 8** | Plot training and validation loss / accuracy curves |
| **Step 9** | Evaluate — Hamming Loss, ROC-AUC, F1 per label |
| **Step 10** | Test on sample comments with predict_toxicity() |
| **Step 11** | Generate predictions on test.csv |
| **Step 12** | Save model weights (.pth) and vocabulary (.pkl) |
| **Step 13** | Print final summary |

---

## 📱 Streamlit App Features

| Tab | What it Does |
|---|---|
| **💬 Single Comment** | Type any comment → see probabilities for all 6 labels as color-coded progress bars + CLEAN / TOXIC verdict |
| **📂 Bulk CSV Upload** | Upload a CSV with `comment_text` column → batch analyze all comments → download results CSV |
| **📊 Dataset Insights** | View dataset statistics, label distribution chart, PyTorch vs TensorFlow comparison table |
| **Sample data** | Upload `sample_comments.csv` to try the bulk tab. Results stay on screen after you download them |
| **Sidebar Slider** | Adjust detection threshold (default 0.5) — higher = stricter detection |

---

## 📊 Results

Validation set: 31,915 comments (20% hold-out, `random_state=42`). Threshold 0.5.

| Label | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| toxic | 0.830 | 0.762 | 0.795 | 0.973 |
| severe_toxic | 0.575 | 0.131 | 0.213 | 0.987 |
| obscene | 0.822 | 0.774 | 0.797 | 0.988 |
| threat | 0.000 | 0.000 | 0.000 | 0.959 |
| insult | 0.718 | 0.685 | 0.701 | 0.980 |
| identity_hate | 0.000 | 0.000 | 0.000 | 0.956 |
| **Overall** | | | micro F1 0.733 | **macro 0.974** |

Hamming loss: **0.018**. Best validation loss 0.0499 at epoch 4 of 5; those weights are restored.

### Known limitations

- **Rare labels:** `threat` (0.3% of data) and `identity_hate` never cross the 0.5 threshold, even though their ROC-AUC is ~0.96. The model ranks them correctly but isn't confident enough. Next step: `BCEWithLogitsLoss(pos_weight=...)` and a tuned threshold per label.
- **Accuracy is misleading:** exact-match accuracy is 91.8%, but predicting "clean" for everything already scores 89.8%. That's why ROC-AUC and F1 are reported.
- **Keyword reliance:** some negations are missed ("You are not stupid at all" scores as toxic). A pre-trained transformer such as BERT would handle context better.
- **Possible identity bias:** comments that mention identity terms are often labelled toxic in this dataset, so harmless mentions can be over-flagged.

---

## 📈 Evaluation Metrics

| Metric | Description | Goal |
|---|---|---|
| **Hamming Loss** | Fraction of labels predicted incorrectly across all comments and all labels | Lower is better |
| **ROC-AUC Score** | Ability to separate toxic from non-toxic per label. 1.0 = perfect, 0.5 = random | Higher is better |
| **F1 Score** | Balance between precision and recall per label. Critical for imbalanced data | Higher is better |
| **Precision** | Of all predicted toxic comments, how many were actually toxic? | Higher is better |
| **Recall** | Of all actual toxic comments, how many did the model catch? | Higher is better |

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.8+ | Core language |
| PyTorch | 2.0+ | Model training and inference |
| pandas | Latest | Data loading and manipulation |
| NumPy | Latest | Array operations |
| scikit-learn | Latest | Evaluation metrics + train/val split |
| Matplotlib | Latest | Visualizations |
| Seaborn | Latest | Heatmaps and styled charts |
| Streamlit | Latest | Interactive web application |
| pickle | Built-in | Saving vocabulary and config |

---

## 📁 Generated Output Files

| File | Description |
|---|---|
| `toxicity_model_pytorch.pth` | Trained PyTorch model weights |
| `vocab_pytorch.pkl` | Vocabulary dictionary (word → integer ID) |
| `model_config_pytorch.pkl` | Architecture config (vocab size, dims, etc.) |
| `test_predictions_pytorch.csv` | Toxicity probability scores for all 153,164 test comments |
| `plots/` | 6 saved visualisation images |

---

## 💼 Business Use Cases

- **Social Media Platforms** — Auto-flag or hide toxic comments before they reach users
- **Online Forums** — Scale content moderation without needing full-time human moderators
- **E-learning Platforms** — Maintain a safe environment for students and teachers
- **News Websites** — Filter reader comments on articles in real time
- **Customer Support Tools** — Flag aggressive or threatening customer messages
- **Brand Safety** — Prevent brand advertisements from appearing near toxic content

---

## 🙋 FAQ

**Q: Why does the model output probabilities instead of just 0 or 1?**
> Probabilities give more flexibility. You can adjust the threshold (default 0.5) depending on how strict you want detection to be. A news website might use 0.3 (catch more), while a general forum might use 0.7 (catch only the most obvious).

**Q: Can I run this without a GPU?**
> Yes. The script automatically detects and uses CPU if no GPU is available. Training will be slower on CPU but will complete correctly.

**Q: Why do I need to save the vocabulary separately?**
> In PyTorch, the model only stores numerical weights — it does not know what words map to what IDs. The vocabulary dictionary (`vocab_pytorch.pkl`) is needed at inference time to convert new text into the same integer IDs used during training.

**Q: What happens to words the model has never seen?**
> Unknown words are replaced with the `<OOV>` (Out-Of-Vocabulary) token during encoding. The model has learned a generic representation for this token during training.

---



## 👤 Author

**Dewakaran** — Senior Lead, Operations (US Healthcare RCM) · GUVI Master Data Science Program
Domain: NLP · Deep Learning · Content Moderation | Stack: Python, PyTorch, Streamlit

---

## 📄 License

For educational purposes. Dataset: [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge) (Kaggle).

---

## ⭐ If this project helped you, give it a star on GitHub!
