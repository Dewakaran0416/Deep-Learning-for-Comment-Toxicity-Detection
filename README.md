# 🚨 Deep Learning for Comment Toxicity Detection

> A **Bidirectional LSTM** deep learning model built in **PyTorch** that detects toxic content in online comments across **6 toxicity categories**, deployed as an interactive **Streamlit web application**.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?style=flat-square&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-Educational-green?style=flat-square)

---

## 📌 Project Overview

Online platforms face a massive challenge moderating millions of user comments every day. This project automates toxic comment detection using deep learning — analyzing text and predicting the likelihood of a comment being **toxic, obscene, threatening, insulting, or hateful** in real time.

**Two complete implementations are provided:**
- ✅ `comment_toxicity_detection_pytorch.py` — **PyTorch** version (primary)
- ✅ `comment_toxicity_detection.py` — **TensorFlow/Keras** version (alternative)

Both produce identical results. The PyTorch version gives you full control over the training loop, which is valuable for learning and customization.

---

## 🗂️ Repository Structure

```
comment-toxicity-detection/
│
├── 📓 Notebooks & Scripts
│   ├── Comment_Toxicity_Detection.ipynb         # Jupyter Notebook (step-by-step)
│   ├── comment_toxicity_detection_pytorch.py    # Full pipeline — PyTorch version
│   └── comment_toxicity_detection.py            # Full pipeline — TensorFlow version
│
├── 🌐 Streamlit Apps
│   ├── app_pytorch.py                           # Web app — PyTorch version
│   └── app.py                                   # Web app — TensorFlow version
│
├── 📁 Data (add manually — not included in repo)
│   ├── train.csv                                # 159,571 labeled comments
│   └── test.csv                                 # 153,164 comments for prediction
│
├── 📦 Model Artifacts (generated after training)
│   ├── toxicity_model_pytorch.pth               # PyTorch model weights
│   ├── vocab_pytorch.pkl                        # Vocabulary dictionary
│   ├── model_config_pytorch.pkl                 # Model architecture config
│   ├── toxicity_model.h5                        # TensorFlow model (alternative)
│   └── tokenizer.pkl                            # Keras tokenizer (alternative)
│
├── 📊 Output
│   └── test_predictions_pytorch.csv             # Toxicity predictions on test set
│
├── 🖼️ plots/
│   ├── label_distribution.png
│   ├── class_imbalance.png
│   ├── comment_lengths.png
│   ├── label_cooccurrence.png
│   ├── training_history.png
│   └── roc_auc_scores.png
│
├── 📄 Docs
│   ├── CommentToxicity_Explanation_Guide.docx   # Full project walkthrough document
│   └── README.md                                # This file
│
└── requirements.txt                             # All Python dependencies
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
git clone https://github.com/your-username/comment-toxicity-detection.git
cd comment-toxicity-detection
```

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn streamlit
```

### Step 3 — Add the Dataset

Download the dataset from [Kaggle — Jigsaw Toxic Comment Classification](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data) and place both files in the project root:

```
comment-toxicity-detection/
├── train.csv
└── test.csv
```

### Step 4 — Train the Model

```bash
# PyTorch version (recommended)
python comment_toxicity_detection_pytorch.py

# OR TensorFlow version
python comment_toxicity_detection.py
```

This will generate all model artifacts, plots, and the predictions CSV automatically.

### Step 5 — Launch the Web App

```bash
# PyTorch app
streamlit run app_pytorch.py

# OR TensorFlow app
streamlit run app.py
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

Total Parameters : ~1,994,758
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

## 🔄 PyTorch vs TensorFlow — Key Differences

| Feature | TensorFlow / Keras | PyTorch (this version) |
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

The Jupyter Notebook `Comment_Toxicity_Detection.ipynb` walks through all 13 steps:

| Step | Description |
|---|---|
| **Step 1** | Import all libraries |
| **Step 2** | Load train.csv and test.csv |
| **Step 3** | EDA — label distributions, class imbalance, comment lengths, co-occurrence heatmap |
| **Step 4** | Text preprocessing — clean_text() function |
| **Step 5** | Build vocabulary, encode texts, pad sequences, train/val split |
| **Step 6** | Build Bidirectional LSTM model |
| **Step 7** | Train model with manual loop, EarlyStopping, and ModelCheckpoint |
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
| **Sidebar Slider** | Adjust detection threshold (default 0.5) — higher = stricter detection |

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



## ⭐ If this project helped you, give it a star on GitHub!
