# NLP Assignment 3 — Financial Sentiment Analysis

**Team members:** Anh Tran

---

## Problem Statement

Financial news sentiment is critical for algorithmic trading, risk management, and market analysis, yet general-purpose sentiment models trained on social media or product reviews perform poorly on domain-specific financial language. This project fine-tunes **FinBERT** (`ProsusAI/finbert`) on the `financial_phrasebank` dataset to classify sentences from financial news reports as *negative*, *neutral*, or *positive*. We benchmark the fine-tuned model against a zero-shot FinBERT baseline and a TF-IDF + Logistic Regression baseline, reporting accuracy, macro-F1, and weighted-F1 on a held-out test set.

---

## Dataset

[`financial_phrasebank`](https://huggingface.co/datasets/financial_phrasebank) — Malo et al. (2014).  
We use the `sentences_allagree` configuration (~2,264 sentences with 100% inter-annotator agreement) and split it 80 / 10 / 10 into train / val / test, stratified by label.

| Label | Meaning  |
|-------|----------|
| 0     | Negative |
| 1     | Neutral  |
| 2     | Positive |

The split CSVs are saved to `data/` by the notebook on first run.

---

## Repository Structure

```
nlp-a3-financial-sentiment/
├── data/
│   ├── train.csv          # 80% of sentences_allagree
│   ├── val.csv            # 10%
│   └── test.csv           # 10%
├── notebooks/
│   └── finbert_finetune.ipynb   # end-to-end training & evaluation
├── results/
│   └── results_table.csv  # populated by notebook Section 6
├── README.md
└── requirements.txt
```

---

## How to Run

### 1. Clone and install dependencies

```bash
git clone https://github.com/<your-username>/nlp-a3-financial-sentiment.git
cd nlp-a3-financial-sentiment
pip install -r requirements.txt
```

### 2. Launch the notebook

```bash
jupyter notebook notebooks/finbert_finetune.ipynb
```

Run all cells top-to-bottom. The notebook will:

1. Download `financial_phrasebank` from HuggingFace and create the train/val/test CSVs in `data/`.
2. Tokenise the data using the FinBERT tokenizer.
3. Fine-tune FinBERT for up to 5 epochs with early stopping on macro-F1.
4. Print a classification report and save a confusion-matrix PNG to `results/`.
5. Write final metrics to `results/results_table.csv`.

> **GPU recommended.** Fine-tuning takes ~5 min on a T4 (Google Colab free tier). CPU training works but is slower (~45 min).

### 3. (Optional) Run on Google Colab

Upload the notebook to Colab, set runtime to **GPU**, and run:

```python
!pip install -r requirements.txt
```

before executing the first code cell.

---

## Results

See [`results/results_table.csv`](results/results_table.csv) for per-model metrics after running the notebook.

---

## References

- Malo, P. et al. (2014). *Good debt or bad debt: Detecting semantic orientations in economic texts.* JASIST.
- Yang, Z. et al. (2020). *FinBERT: A Pretrained Language Model for Financial Communications.* arXiv:2006.08097.
- HuggingFace `financial_phrasebank`: https://huggingface.co/datasets/financial_phrasebank
