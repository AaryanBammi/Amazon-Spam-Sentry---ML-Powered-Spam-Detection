# Amazon Spam Sentry – ML‑Powered Spam Detection

## Overview
This repository contains a machine learning pipeline for detecting spam in Amazon reviews. It leverages natural language processing techniques and classical classifiers (logistic regression, random forest, support vector machine) to identify spammy or fraudulent product reviews.

## Data
The dataset consists of Amazon product reviews labelled as “spam” or “non‑spam.” Each record includes review text, star rating and other metadata. (Due to licensing constraints, the full dataset is not included; you can substitute your own labelled review data.)

## Pipeline
1. **Preprocessing:** Clean the text (lowercasing, punctuation removal, stopword filtering) and convert to tokens.
2. **Feature engineering:** Transform tokens into numerical features using TF–IDF.
3. **Model training:** Train logistic regression, random forest and SVM models with scikit‑learn; evaluate with cross‑validation.
4. **Evaluation:** Compute accuracy, precision, recall and F1 scores; plot confusion matrices to compare models.

## Results
The logistic regression model achieved an F1 score of 0.93 on the validation set, with high precision and recall. Random forest and SVM models performed comparably but with slightly lower precision on rare classes.

## Usage
1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Open the notebook:

   ```bash
   jupyter notebook amazon_spam_detection.ipynb
   ```

3. Replace the placeholder dataset loading cell with your own labelled review data, then run through the pipeline to train and evaluate models.

## Next Steps
- Incorporate transformer‑based text embeddings (e.g., BERT) to capture contextual information.
- Deploy a live prediction API or Streamlit app for interactive spam detection.
- Perform error analysis to understand misclassified examples and refine preprocessing.
