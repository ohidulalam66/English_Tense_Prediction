# English Tense Classifier: Vanilla RNN for Past, Present, and Future Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://www.tensorflow.org/)

## Overview

This repository implements a lightweight, high-performance **Vanilla Recurrent Neural Network (RNN)** for **sentence-level English tense classification** into three categories: **Past**, **Present**, and **Future**. The model achieves **96.82% test accuracy** and a **0.968 macro F1-score** on a custom-curated dataset of ~13,824 unique sentences (after preprocessing from 26,633 raw samples).

Key highlights:
- **Efficient & Interpretable**: Uses a simple single-layer SimpleRNN (128 units) with embedding (128 dim), dropout (0.5), and L2 regularization (0.001). Total params: ~1.32M (50x smaller than typical transformers).
- **Reproducible**: Fixed seed (42), stratified splits, early stopping, and full pipeline for deterministic results.
- **Applications**: Text summarization, machine translation, temporal extraction, grammar checkers, and low-resource/edge devices (inference <3ms/sentence on CPU).
- **Dataset**: Custom-labeled English sentences from diverse domains (tech, daily life, history, etc.), available via [Mendeley Data](https://data.mendeley.com/datasets/jnb2xp9m4r/2).

This project demonstrates that classical RNNs remain competitive for structured NLP tasks without needing complex architectures like LSTMs, GRUs, or BERT.

**Full Report**: See `report.pdf` (or the provided text) for detailed methodology, ablation studies, results, and analysis.

## Dataset

- **Source**: [EnglishTense Dataset](https://data.mendeley.com/datasets/jnb2xp9m4r/2) (`prediction.xlsx` or `All_Tense(Prediction).xlsx`).
- **Size**: 26,633 raw sentences → 13,824 unique after cleaning.
- **Classes**: Future (36.5%), Present (34.6%), Past (28.8%).
- **Preprocessing**:
  - Lowercasing, typo fixes (e.g., "aienhanced" → "ai enhanced").
  - Duplicate removal, tokenization (vocab: 10k, max len: 50).
  - Label encoding: Future=0, Past=1, Present=2.
- **Examples**:
  | Sentence | Tense |
  |----------|-------|
  | "Tomorrow we will meet the president." | Future |
  | "Yesterday we met the president." | Past |
  | "We meet the president every year." | Present |

Download the dataset and place it in `/data/` (or update the path in `train.py`).

## Model Architecture

```
Model: Sequential
_________________________________________________________________
Layer (type)              Output Shape      Param #
=================================================================
embedding (Embedding)     (None, 50, 128)   1,280,000
simple_rnn (SimpleRNN)    (None, 128)       32,896
dropout (Dropout)         (None, 128)       0
dense (Dense)             (None, 64)        8,256
dropout_1 (Dropout)       (None, 64)        0
dense_1 (Dense)           (None, 3)         195
=================================================================
Total params: 1,321,347
Trainable params: 1,321,347
```

- **Hyperparameters**: Adam (lr=0.001), sparse categorical cross-entropy, batch=32, epochs=20 (early stop patience=3).
- **Saved Artifacts**: `tense_classifier_rnn.h5` (model), `tokenizer.pickle`, `label_encoder.pickle`.

## Installation

1. Clone the repo:
   ```
   git clone https://github.com/yourusername/english-tense-classifier.git
   cd english-tense-classifier
   ```

2. Create a virtual environment (Python 3.10+):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

   **requirements.txt** contents:
   ```
   tensorflow>=2.15.0
   pandas>=2.2.0
   numpy>=1.26.0
   scikit-learn>=1.5.0
   matplotlib>=3.7.0
   seaborn>=0.12.0
   openpyxl>=3.1.0  # For Excel reading
   ```

## Usage

### 1. Training the Model
Run the full pipeline (preprocessing, training, evaluation):
```bash
python train.py
```
- Outputs: Training history plot (`accuracy_vs_epoch.png`), model weights, tokenizer, and label encoder.
- Expected runtime: ~90s on GPU, ~6min on CPU.

### 2. Inference
Load the trained model for predictions:
```bash
python infer.py
```
- Interactive mode: Enter sentences to classify (e.g., "I will go tomorrow" → **FUTURE**, confidence=0.98).
- Or use the function programmatically:
  ```python
  from utils import predict_tense
  tense, conf = predict_tense("I ate breakfast yesterday.")
  print(f"Tense: {tense} | Confidence: {conf:.4f}")  # Tense: PAST | Confidence: 0.97
  ```

### 3. Evaluation
Run test set evaluation and generate confusion matrix/heatmap:
```bash
python evaluate.py
```
- Outputs: Classification report, confusion matrix (`confusion_matrix.png`), metrics table.

### 4. Reproducibility
- Set `SEED=42` in scripts for deterministic results.
- All experiments use stratified 80/20 train-test split.

## Results

| Metric              | Value   |
|---------------------|---------|
| Test Accuracy       | 96.82% |
| Macro F1-Score      | 0.968  |
| Weighted F1-Score   | 0.968  |
| Error Rate          | 2.71%  |
| Misclassifications  | 75/2764|

**Per-Class Performance** (Test Set: 2,764 samples):
| Class    | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Future  | 0.98     | 0.98  | 0.98    | 1,010  |
| Past    | 0.96     | 0.96  | 0.96    | 797    |
| Present | 0.97     | 0.96  | 0.97    | 957    |
| **Macro Avg** | **0.97** | **0.97** | **0.97** | **2,764** |

**Confusion Matrix**:
| True \ Pred | Future | Past | Present |
|-------------|--------|------|---------|
| **Future**  | 992   | 8    | 10     |
| **Past**    | 11    | 763  | 23     |
| **Present** | 12    | 11   | 934    |

- **Strengths**: Excels on clear markers (e.g., "will", "-ed", "tomorrow").
- **Limitations**: Minor errors in complex forms (e.g., "will have been" vs. Present Perfect Progressive).

See `report.pdf` for plots (accuracy curves, heatmap) and ablation studies.

## Project Structure
```
english-tense-classifier/
├── data/                  # Dataset (prediction.xlsx)
├── src/
│   ├── train.py           # Full training pipeline
│   ├── infer.py           # Interactive inference
│   ├── evaluate.py        # Metrics & plots
│   └── utils.py           # Helpers (clean_text, predict_tense)
├── models/                # Saved: tense_classifier_rnn.h5, tokenizer.pickle, label_encoder.pickle
├── plots/                 # Outputs: accuracy_vs_epoch.png, confusion_matrix.png
├── report.pdf             # Full project report
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## Authors
- Mohammad Ohidul Alam
- Tanjilul Islam
- Sayed Hossain

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Dataset: [Mendeley Data](https://data.mendeley.com/datasets/jnb2xp9m4r/2).
- Frameworks: TensorFlow/Keras.
- References: See `report.pdf` for full bibliography.

## Future Work
- Extend to 12 fine-grained tenses/aspects.
- Add attention/Transformer hybrids.
- Multilingual support and on-device deployment (TensorFlow Lite).

Contributions welcome! Fork, star, or open issues for feedback. 🚀
