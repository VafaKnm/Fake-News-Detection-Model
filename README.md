# Fake News Detection Model

A deep learning project for detecting fake news using text embeddings and an LSTM-based neural network.

This project was originally developed in a Kaggle Notebook environment and demonstrates a complete binary text classification workflow for distinguishing between **fake** and **real** news articles.

---

## Overview

Fake news detection is a Natural Language Processing task where the goal is to classify a news article as either:

```text
FAKE
REAL
```

This project uses a neural network approach based on:

```text
Text preprocessing
Tokenization
Word embeddings
LSTM model
Binary classification
```

The model learns patterns from news article text and predicts whether a given article is likely to be fake or real.

---

## What This Project Does

This project shows how to:

* Load a fake/real news dataset in Kaggle
* Inspect and clean the dataset
* Check for missing values
* Convert text labels into numeric labels
* Prepare news article text for deep learning
* Convert words into numerical sequences
* Use an Embedding layer for text representation
* Train an LSTM model for binary classification
* Predict whether a news article is fake or real

---

## Dataset

The notebook uses the following Kaggle dataset file:

```text
/kaggle/input/textdb3/fake_or_real_news.csv
```

The dataset contains:

```text
6335 rows × 4 columns
```

The columns are:

| Column       | Description                      |
| ------------ | -------------------------------- |
| `Unnamed: 0` | Original index or ID column      |
| `title`      | News article title               |
| `text`       | Full news article text           |
| `label`      | Original label: `FAKE` or `REAL` |

Example rows:

| title                                       | text                 | label |
| ------------------------------------------- | -------------------- | ----- |
| You Can Smell Hillary’s Fear                | Full article text... | FAKE  |
| Kerry to go to Paris in gesture of sympathy | Full article text... | REAL  |

---

## Label Encoding

The original labels are text labels:

```text
FAKE
REAL
```

The notebook converts them into numeric labels:

```text
FAKE news = 1
REAL news = 0
```

This makes the labels usable for binary classification.

Example:

```python
df["label"] = df["label"].replace(["FAKE", "REAL"], [1, 0])
```

---

## Problem Type

This is a **binary text classification** problem.

The model receives a news article and predicts one of two classes:

| Class  | Numeric Label | Meaning   |
| ------ | ------------: | --------- |
| `REAL` |             0 | Real news |
| `FAKE` |             1 | Fake news |

---

## General Pipeline

The full workflow can be described as:

```text
Raw News Dataset
        ↓
Read CSV File
        ↓
Clean and Inspect Data
        ↓
Encode Labels
        ↓
Text Tokenization
        ↓
Text Padding
        ↓
Embedding Layer
        ↓
LSTM Layer
        ↓
Dense Output Layer
        ↓
Fake / Real Prediction
```

---

## Why Use LSTM?

LSTM stands for **Long Short-Term Memory**.

It is a type of recurrent neural network designed to work with sequential data such as text.

News articles are sequences of words, so an LSTM can learn patterns across word order and context.

For example, instead of only looking at individual words, an LSTM can learn from word sequences like:

```text
"officials confirmed the report"
"anonymous sources claimed"
"breaking shocking truth"
"according to the investigation"
```

This makes LSTM useful for text classification tasks.

---

## Why Use Embeddings?

Machine learning models cannot directly understand raw words.

An embedding layer converts words into dense numerical vectors.

Example:

```text
"government" → [0.12, -0.44, 0.31, ...]
"election"   → [0.09,  0.28, -0.11, ...]
"fake"       → [0.41, -0.03, 0.22, ...]
```

These vectors help the model learn relationships between words.

---

## Model Architecture

The project uses an Embedding + LSTM style architecture.

A typical architecture for this project is:

```text
Input Text
    ↓
Tokenizer
    ↓
Integer Sequences
    ↓
Padding
    ↓
Embedding Layer
    ↓
LSTM Layer
    ↓
Dense Layer
    ↓
Sigmoid Output
    ↓
Fake / Real Prediction
```

Conceptual Keras model:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

model = Sequential()

model.add(
    Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_sequence_length
    )
)

model.add(LSTM(128))
model.add(Dropout(0.3))
model.add(Dense(1, activation="sigmoid"))

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

---

## Input and Output

### Input

The input is a news article text.

Example:

```text
The government announced a new policy after officials confirmed the report...
```

### Output

The output is a binary prediction.

Example:

```text
REAL
```

The model may also return a probability.

Example:

```text
fake_probability = 0.87
```

A possible decision rule:

```text
probability >= 0.5 → FAKE
probability < 0.5  → REAL
```

---

## Example Prediction

Input:

```text
Breaking news: anonymous sources claim a shocking secret plan was revealed...
```

Output:

```text
FAKE
```

Input:

```text
The secretary of state said Monday that officials will meet in Paris next week...
```

Output:

```text
REAL
```

---

## Repository Structure

Current repository structure:

```text
Fake-News-Detection-Model/
│
├── README.md
└── fake-news-detector-using-embedding-and-lstm.ipynb
```

Suggested future structure:

```text
Fake-News-Detection-Model/
│
├── README.md
├── requirements.txt
├── notebooks/
│   └── fake-news-detector-using-embedding-and-lstm.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── tokenizer.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── models/
│   ├── fake_news_lstm_model.h5
│   └── tokenizer.pkl
├── assets/
│   ├── label_distribution.png
│   ├── training_history.png
│   └── confusion_matrix.png
└── examples/
    └── sample_predictions.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/VafaKnm/Fake-News-Detection-Model.git
cd Fake-News-Detection-Model
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install common dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras jupyter
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
fake-news-detector-using-embedding-and-lstm.ipynb
```

---

## Suggested `requirements.txt`

A useful `requirements.txt` file for this project could be:

```txt
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
keras
jupyter
```

If additional NLP preprocessing is added later:

```txt
nltk
spacy
beautifulsoup4
regex
```

---

## Example Text Preprocessing

A future script-based version could include a reusable cleaning function:

```python
import re

def clean_text(text):
    """
    Basic text cleaning for fake news detection.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

---

## Example Tokenization

The text can be converted into padded numerical sequences:

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(
    X_train_seq,
    maxlen=300,
    padding="post",
    truncating="post"
)

X_test_pad = pad_sequences(
    X_test_seq,
    maxlen=300,
    padding="post",
    truncating="post"
)
```

---

## Example Inference Function

A future inference function could look like this:

```python
def predict_news(text, model, tokenizer, max_length=300):
    """
    Predict whether a news article is fake or real.
    """
    cleaned_text = clean_text(text)

    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(
        sequence,
        maxlen=max_length,
        padding="post",
        truncating="post"
    )

    probability = model.predict(padded)[0][0]

    if probability >= 0.5:
        label = "FAKE"
    else:
        label = "REAL"

    return {
        "label": label,
        "fake_probability": float(probability)
    }
```

Example:

```python
article = """
The government confirmed the report after an official statement was released.
"""

predict_news(article, model, tokenizer)
```

Example output:

```python
{
    "label": "REAL",
    "fake_probability": 0.21
}
```

---

## Evaluation

The current README does not include a final numerical evaluation report.

A stronger evaluation section should include:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* ROC-AUC
* False positive examples
* False negative examples

Example evaluation code:

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

y_pred_prob = model.predict(X_test_pad)
y_pred = (y_pred_prob >= 0.5).astype(int)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_prob))
```

---

## Recommended Metrics

Fake news detection should not rely only on accuracy.

Important metrics:

| Metric           | Why It Matters                                     |
| ---------------- | -------------------------------------------------- |
| Accuracy         | Overall correctness                                |
| Precision        | How many predicted fake articles are actually fake |
| Recall           | How many fake articles are successfully detected   |
| F1-score         | Balance between precision and recall               |
| ROC-AUC          | Ranking quality across thresholds                  |
| Confusion Matrix | Shows false positives and false negatives          |

In fake news detection, false positives and false negatives can both be harmful.

---

## Suggested Result Table

After running evaluation, the README can be updated with:

| Metric    | Score |
| --------- | ----: |
| Accuracy  |   TBD |
| Precision |   TBD |
| Recall    |   TBD |
| F1-score  |   TBD |
| ROC-AUC   |   TBD |

Per-class table:

| Class | Precision | Recall | F1-score |
| ----- | --------: | -----: | -------: |
| REAL  |       TBD |    TBD |      TBD |
| FAKE  |       TBD |    TBD |      TBD |

---

## Suggested Improvements

This repository can be improved in several practical ways.

### 1. Add a Full Evaluation Report

The notebook should export a classification report and confusion matrix.

Recommended files:

```text
assets/confusion_matrix.png
assets/training_history.png
```

---

### 2. Compare with Classical ML Baselines

A good fake news detection project should compare LSTM with simpler baselines.

Recommended baselines:

| Model               | Feature Type          | Notes                                   |
| ------------------- | --------------------- | --------------------------------------- |
| Logistic Regression | TF-IDF                | Strong simple baseline                  |
| Linear SVM          | TF-IDF                | Often excellent for text classification |
| Naive Bayes         | Bag-of-Words / TF-IDF | Fast baseline                           |
| Random Forest       | TF-IDF                | Nonlinear baseline                      |
| LSTM                | Embedding sequences   | Current deep learning approach          |
| BERT / DistilBERT   | Contextual embeddings | Stronger modern baseline                |

---

### 3. Add BERT-Based Model

A stronger future version could use:

```text
BERT
DistilBERT
RoBERTa
DeBERTa
Longformer
```

Transformer models can capture richer context than a basic LSTM.

---

### 4. Save the Trained Model and Tokenizer

The model and tokenizer should be saved for inference.

Save model:

```python
model.save("models/fake_news_lstm_model.h5")
```

Save tokenizer:

```python
import pickle

with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
```

Load later:

```python
from tensorflow.keras.models import load_model
import pickle

model = load_model("models/fake_news_lstm_model.h5")

with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
```

---

### 5. Add a Web Demo

A small demo would make the project easier to test.

Recommended tools:

* Streamlit
* Gradio
* FastAPI

Example UI:

```text
Paste news article
      ↓
Click Predict
      ↓
Show FAKE / REAL label
      ↓
Show probability score
```

---

### 6. Add Error Analysis

Fake news detection is sensitive, so the project should analyze model mistakes.

Useful questions:

```text
Which real articles are incorrectly predicted as fake?
Which fake articles are incorrectly predicted as real?
Are predictions based on source, writing style, or political keywords?
Does the model overfit to specific names or entities?
Does the model generalize to newer news?
```

---

### 7. Add Data Leakage Checks

Fake news datasets can contain leakage.

Important checks:

* Duplicate articles
* Similar title/text pairs across train and test
* Source-specific artifacts
* Date leakage
* Author/source leakage
* Repeated templates

Without these checks, test accuracy may be overly optimistic.

---

### 8. Add Explainability

For a fake news classifier, explainability is very important.

Possible methods:

```text
LIME
SHAP
Integrated Gradients
Attention visualization
Important words / phrases
```

Example output:

```text
Prediction: FAKE
Important words:
    "shocking"
    "secret"
    "exposed"
    "anonymous"
```

This helps users understand why the model made a prediction.

---

## Ethical Considerations

Fake news detection is a sensitive application.

Important points:

* The model should not be treated as a final truth authority.
* Predictions can be wrong.
* The model may learn dataset bias.
* The model may overfit to writing style instead of factual correctness.
* Real-world fact-checking requires external evidence and source verification.
* A fake news classifier should support human review, not replace it.
* The system should explain uncertainty and avoid overconfident claims.

---

## Limitations

The current project has some limitations:

* It is notebook-based and not yet structured as reusable Python code.
* The README does not currently include full evaluation metrics.
* The trained model checkpoint is not included in the repository.
* The tokenizer is not saved for direct inference.
* The model may not generalize to newer news articles.
* LSTM models may not capture long documents as well as modern Transformer models.
* Fake news detection requires evidence verification, not only text classification.
* Dataset-specific patterns may not reflect real-world misinformation.

---

## Use Cases

This project can be useful for:

* NLP classification learning
* Text preprocessing practice
* LSTM-based text modeling
* Fake news detection experiments
* Binary classification projects
* News filtering research
* Educational misinformation detection demos

This project should be considered an educational baseline, not a production-ready fact-checking system.

---

## Possible Future Work

Recommended future work:

* Add `requirements.txt`
* Add clean Python modules
* Add saved model and tokenizer
* Add inference script
* Add confusion matrix
* Add classification report
* Add ROC curve
* Add model comparison table
* Add TF-IDF baselines
* Add BERT/DistilBERT model
* Add explainability with LIME or SHAP
* Add Streamlit or Gradio demo
* Add Dockerfile
* Add data leakage analysis
* Add real-world external validation set

---

## Conclusion

This repository demonstrates a practical fake news detection pipeline using embeddings and an LSTM neural network.

The project is useful for learning:

```text
text classification
label encoding
tokenization
embedding layers
LSTM models
binary classification
fake news detection workflow
Kaggle-based NLP experimentation
```

With better evaluation, model saving, explainability, baseline comparison, and a clean inference interface, this project can become a much stronger NLP portfolio project.
