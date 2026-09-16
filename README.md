# Euphemism Detection with Fine-Tuned ALBERT

A binary text classification project that fine-tunes **ALBERT** (`albert-base-v2`) to detect euphemisms in English text, with class-weighted handling for imbalanced data and a full evaluation pipeline on held-out test data.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Model](#model)
- [Handling Class Imbalance](#handling-class-imbalance)
- [Training Setup](#training-setup)
- [Evaluation](#evaluation)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Key Insights](#key-insights)
- [Future Improvements](#future-improvements)
- [Related Publication](#related-publication)
- [License](#license)

---

## 🩺 Overview

This project fine-tunes **ALBERT** (A Lite BERT), a parameter-efficient transformer that shares weights across layers to reduce model size without sacrificing much performance, for binary euphemism classification. It's a lighter-weight alternative to the XLNet- and LASER+LSTM-based approaches in this portfolio, useful for comparing how a smaller, more efficient transformer performs on the same task.

The pipeline covers the complete workflow: data cleaning, class-weight computation, tokenization, fine-tuning with Hugging Face's `Trainer`, evaluation on a validation split, and final testing against an independently labeled test set.

---

## 📊 Dataset

The project uses the same English-language euphemism classification dataset as the other approaches in this series:

| File | Purpose |
|---|---|
| `EN_train.csv` | Training data with `text` and `label` columns |
| `EN_test.csv` | Unlabeled test data used to generate predictions |
| `EN_reference_test.csv` | Ground-truth labels for the test set |

Labels are binary: **Euphemism** (1) vs. **Non-Euphemism** (0).

---

## 🔄 Pipeline

### 1. Environment Setup
- Fixed random seeds (`SEED = 42`) are set across Python, NumPy, and PyTorch, with deterministic cuDNN settings enabled for reproducibility

### 2. Data Import & Inspection
- Load the training CSV and inspect its structure, summary statistics, and label distribution

### 3. Data Preprocessing
- Drop the irrelevant index column (`Unnamed: 0`)
- Clean text: lowercase, collapse whitespace, strip special characters

### 4. Class Weight Computation
- Compute balanced class weights from the label distribution to address class imbalance during training

### 5. Tokenization
- Tokenize text with the `albert-base-v2` tokenizer (via `AutoTokenizer`), padding/truncating to a maximum length of 512 tokens

### 6. Dataset Preparation
- Split into training (80%) and validation (20%) sets, stratified by label
- Wrap tokenized data into Hugging Face `Dataset` objects

### 7. Model Fine-Tuning
- Fine-tune `AlbertForSequenceClassification` (2 output labels) using the Hugging Face `Trainer` API

### 8. Evaluation & Testing
- Evaluate on the validation split with a classification report and confusion matrix
- Generate predictions on the separate test set and compare against reference labels for final test metrics

### 9. Model Saving
- Save the fine-tuned model and tokenizer for later reuse

---

## 🤖 Model

**Base model:** `albert-base-v2` (Hugging Face Transformers)
**Task:** Binary sequence classification (Euphemism / Non-Euphemism)
**Framework:** PyTorch + Hugging Face `Trainer`

ALBERT's parameter-sharing design makes it significantly smaller than comparable BERT-family models, offering a useful accuracy-vs-efficiency trade-off point to compare against the larger XLNet-based approach used elsewhere in this project series.

---

## ⚖️ Handling Class Imbalance

Balanced class weights are computed with `sklearn.utils.class_weight.compute_class_weight` and converted to a PyTorch tensor. This gives visibility into the degree of imbalance in the dataset and provides the weighting needed if a custom weighted loss function is incorporated into training (see [Future Improvements](#future-improvements)).

---

## ⚙️ Training Setup

| Hyperparameter | Value |
|---|---|
| Base model | albert-base-v2 |
| Epochs | 10 |
| Train batch size | 16 |
| Eval batch size | 16 |
| Learning rate | 2e-5 |
| Warmup steps | 500 |
| Weight decay | 0.01 |
| Adam epsilon | 1e-8 |
| Max sequence length | 512 |
| Metric for best model | Validation loss (minimized) |
| Eval/save strategy | Per epoch, best model reloaded at end |

---

## 📈 Evaluation

- **Validation evaluation:** classification report (precision, recall, F1 per class) and a confusion matrix heatmap
- **Test evaluation:** predictions on the unlabeled test set are compared against the independent reference labels to compute final test accuracy, a full classification report, and a confusion matrix

---

## 🧰 Tech Stack

- **Language:** Python 3
- **NLP / Deep Learning:** Hugging Face Transformers, PyTorch
- **Data Handling:** Pandas, Hugging Face `datasets`
- **ML Utilities:** Scikit-learn (class weights, splitting, metrics)
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Google Colab

---

## 📁 Project Structure

```
euphemism-albert/
│
├── euphemism_albert.ipynb      # Main notebook: preprocessing, training, evaluation
├── data/
│   ├── EN_train.csv            # Training data
│   ├── EN_test.csv             # Test data (unlabeled)
│   └── EN_reference_test.csv   # Ground-truth labels for test set
├── results/                    # Model checkpoints saved during training
├── mALBERTModel/                # Final fine-tuned model + tokenizer
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install transformers datasets torch scikit-learn pandas numpy tqdm seaborn matplotlib
```

### Running the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/euphemism-albert.git
   cd euphemism-albert
   ```
2. Place `EN_train.csv`, `EN_test.csv`, and `EN_reference_test.csv` in the `data/` folder, updating the file paths in the notebook to match your local setup.
3. Open and run the notebook:
   ```bash
   jupyter notebook euphemism_albert.ipynb
   ```

> **Note:** Fine-tuning benefits from GPU acceleration, though ALBERT's smaller size makes it more feasible to run on modest hardware compared to larger transformer variants.

---

## 💡 Key Insights

- ALBERT's parameter-sharing architecture makes it a strong candidate when training/inference efficiency matters, at some potential cost to peak accuracy compared to larger models like XLNet-Large.
- Computing class weights explicitly (even before deciding how to apply them) is good practice — it surfaces the degree of imbalance early, informing decisions like whether oversampling, weighted loss, or a different evaluation metric focus is needed.
- Comparing this ALBERT-based result against the XLNet and LASER+LSTM approaches in this project series provides a useful **model comparison study** for the broader euphemism detection task, which ties directly into the ensemble ("blending") research referenced below.

---

## 🔮 Future Improvements

- **Actually apply the computed class weights** — currently `class_weights_tensor` is computed but never passed into the loss function or `Trainer`; wiring it into a custom weighted `CrossEntropyLoss` (via a custom `Trainer` subclass) would let the imbalance handling take effect during training.
- Switch `metric_for_best_model` from validation loss to **F1-score**, which better reflects performance on the minority class in an imbalanced binary task.
- Remove the duplicate confusion-matrix plotting cell near the end of the test evaluation section.
- Compare inference latency and model size directly against the XLNet-based model to quantify the efficiency trade-off.
- As with the other notebooks in this series, audit for and remove any accidentally hardcoded credentials or tokens before sharing publicly.

---

## 📄 Related Publication

This project is one of several model comparisons feeding into ensemble-based euphemism detection research:

**Blend-XED: A Transformer-Based Blending Ensemble Model for Euphemism Detection**
Oindrela, D., Faiza, S. R., Islam, T., Chy, A. N., & Ahmed, T. — Accepted and published at the IEEE International Black Sea Conference on Communications and Networking (BlackSeaCom), Bucharest, Romania, June 2026.

---

## 📄 License

This project is open-source and available for educational and research purposes. Please cite appropriately if reused.
