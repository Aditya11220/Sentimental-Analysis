# Sentimental Analysis

## Project Overview
This project implements a machine learning solution for **multi-class emotion classification** from text data. The goal is to accurately predict one of several categorical emotions (e.g., sadness, anger, love) associated with a given sentence or phrase.

## Features
* **Data Cleaning:** Comprehensive preprocessing including lowercasing, removal of punctuation, numbers, and emojis.
* **Tokenization & Filtering:** Uses NLTK for tokenization and efficient removal of English stopwords.
* **Feature Engineering:** Compares traditional NLP feature representation methods like **Count Vectorizer (Bag-of-Words)** and **Tf-idf Vectorizer**.
* **Machine Learning Models:** Implementation and comparison of two robust text classification models: **Multinomial Naive Bayes** and **Logistic Regression**.

## Installation

### Prerequisites
Ensure you have Python (3.6+) and pip installed.

### Setup
1.  Clone the repository or download the project files.
2.  Install the required libraries using `pip`:

    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn nltk
    ```

3.  Download necessary NLTK data:

    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    ```

## Dataset

The project uses an emotion dataset, which should be included in the project directory as `train.txt`.

* **File:** `train.txt`
* **Format:** Semi-colon separated (`sep=;`), with two unlabelled columns: `text` and `emotion`.
* **Size:** The dataset contains 16,000 entries.

## Data Preprocessing Pipeline

The following steps are performed on the raw text data:

1.  **Emotion Mapping:** The categorical emotion labels (e.g., 'sadness', 'anger') are converted to numerical integers for modeling purposes.
2.  **Text Normalization:** Text is converted to lowercase.
3.  **Noise Removal:** Punctuation, numbers, and non-ASCII characters (emojis) are removed from the text.
4.  **Stopword Elimination:** Common English stopwords are removed to focus on meaningful keywords.
5.  **Vectorization:** The cleaned text is converted into numerical features using `CountVectorizer` (Bag-of-Words) and subsequently experimented with `TfidfVectorizer`.

## Modeling and Results

The dataset was split into 80% for training and 20% for testing (with `random_state=42`).

### Models Used

| Model | Feature Vectorizer | Purpose |
| :--- | :--- | :--- |
| **Multinomial Naive Bayes (`MultinomialNB`)** | CountVectorizer (BoW) | Baseline and highly effective for sparse feature text data.|
| **Logistic Regression (`LogisticRegression`)** | TfidfVectorizer | Used to leverage the importance of words via TF-IDF weights. |

### Evaluation Metrics

The final model performance was evaluated using **Accuracy Score**.

| Model | Achieved Accuracy |
| :--- | :--- |
| **Logistic Regression (TF-IDF)** | **86.97%** (approx.) |
| **Multinomial Naive Bayes (BoW)** | **76.81%** (approx.) |

*(Note: The Notebook suggests that Logistic Regression with TF-IDF provided the best performance)*.

## How to Run the Project

The entire workflow, from data loading and cleaning to model training and evaluation, is contained in the **`Sentimental_analysis.ipynb`** Jupyter Notebook.

1.  Ensure you have the `train.txt` data file and the `Sentimental_analysis.ipynb` notebook in the same directory.
2.  Open the notebook using a Jupyter environment:
    ```bash
    jupyter notebook
    ```
3.  Select and run all cells in the `Sentimental_analysis.ipynb` notebook.