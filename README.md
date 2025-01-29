# Text Classification Project

### Overview

This project focuses on text classification using various machine learning models. It involves data preprocessing, feature extraction, model training, and evaluation.

### Technologies Used

Python

Pandas

NumPy

Matplotlib & Seaborn (for visualization)

Scikit-learn

LightGBM

XGBoost

Gensim (for Word2Vec)

NLTK (for tokenization)

Skopt (for Bayesian optimization)



### Data Processing

The dataset consists of text samples with labels.

Features are extracted using:

Word Embeddings (Word2Vec)

Bag of Words (CountVectorizer)

Numeric features (sentiment polarity, presence of negation words)

### Model Training

The following models were trained and evaluated:

LinearSVC

Logistic Regression

LightGBM Classifier

XGBoost Classifier

Random Forest Classifier

AdaBoost Classifier

Ridge Classifier

Evaluation Metrics

### Models were assessed based on:

Accuracy

F1 Score

Precision

Recall

ROC AUC Score

Hyperparameter Optimization

Bayesian Optimization for Logistic Regression

Randomized Search for LightGBM Classifier

Results

Best performing model: Logistic Regression (AUC: 0.8543)

Confusion matrix visualization included

### Future Improvements

Try deep learning approaches (e.g., LSTMs, Transformers)

Experiment with additional feature engineering techniques

Optimize feature selection further

