# NLP A3: Financial Sentiment Analysis with FinBERT

**Team Members:** Anh D. Tran

## Problem Statement

Financial sentiment analysis is the task of automatically classifying the sentiment (positive, negative, or neutral) expressed in financial news and reports. In this project, we fine-tune FinBERT — a BERT-based model pre-trained on financial text — on the `financial_phrasebank` dataset from HuggingFace. The goal is to accurately predict the sentiment of financial phrases and evaluate model performance, demonstrating the effectiveness of domain-specific pre-training for NLP tasks in the finance domain.

## Dataset

We use the [financial_phrasebank](https://huggingface.co/datasets/financial_phrasebank) dataset from HuggingFace (`sentences_allagree` split), which contains sentences from financial news annotated with sentiment labels: **positive**, **negative**, and **neutral**.

## How to Run the Notebook

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Anh-D-Tran030/nlp-a3-financial-sentiment.git
   cd nlp-a3-financial-sentiment
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter and open the notebook:**
   ```bash
   jupyter notebook notebooks/finbert_finetune.ipynb
   ```

4. **Run all cells** in `notebooks/finbert_finetune.ipynb` to reproduce data preparation, fine-tuning, and evaluation.

> **Note:** A GPU is recommended for training. The notebook can also be run on [Google Colab](https://colab.research.google.com/) by uploading `notebooks/finbert_finetune.ipynb`.

## Repository Structure

```
nlp-a3-financial-sentiment/
├── data/
│   ├── train.csv          # Training split
│   ├── val.csv            # Validation split
│   └── test.csv           # Test split
├── notebooks/
│   └── finbert_finetune.ipynb  # Fine-tuning notebook
├── results/
│   └── results_table.csv  # Evaluation results
├── README.md
└── requirements.txt
```