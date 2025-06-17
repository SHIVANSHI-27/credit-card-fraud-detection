# 💳 Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.11.11-blue)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18.0-brightgreen)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-red)

This project applies machine learning techniques to detect fraudulent credit card transactions. It compares several models (Random Forest, XGBoost, and a Dense Neural Network) using a real-world, highly imbalanced dataset.

## 📝 Introduction

This repository contains a complete pipeline for Credit Card Fraud Detection, from data exploration and scaling to training and evaluating different Machine Learning models.  
The main goals were:

- Handle a **highly imbalanced dataset**
- Compare classical ML (Random Forest, XGBoost) vs Neural Networks
- Provide clear code, metrics, and instructions for reproducibility

## 🔗 Kaggle Notebook

View the full notebook on Kaggle: [Credit Card Fraud Detection (RF, XGBoost, NN)](https://www.kaggle.com/code/axelzar/credit-card-fraud-detection-rf-xgb-nn)

---

## 📁 Contents

- `credit-card-fraud-detection.ipynb`: Main notebook with data analysis, model training, and evaluation.
- `requirements.txt`: Required libraries to run the notebook.

---

## 📊 Dataset

The dataset is from [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). It contains transactions made by European cardholders in September 2013.

- Total records: 284,807
- Fraudulent transactions: ~0.17%
- Features: 28 anonymized numerical features + `Time`, `Amount` and `Class` (target)

---

## ⚙️ Technologies

- Python
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost
- TensorFlow / Keras

---

## 🧪 Models Evaluated

The following models were trained and evaluated:

| Model                | Accuracy | Precision | Recall | F1-Score | ROC AUC |
| -------------------- | -------- | --------- | ------ | -------- | ------- |
| Random Forest        | 0.9995   | 0.9722    | 0.7368 | 0.8383   | 0.9294  |
| XGBoost              | 0.9995   | 0.9459    | 0.7368 | 0.8284   | 0.9749  |
| Dense Neural Network | 0.9994   | 0.8974    | 0.7368 | 0.8092   | 0.9659  |

> 🧠 **XGBoost** achieved the best balance between _recall_ and _ROC AUC_, although Random Forest also performed very well.

---

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/ax-zar/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Open and run the notebook on Jupyter.

## 📌 Notes

- The dataset is highly imbalanced, with fraud cases representing only about 0.17% of all transactions.
- A **stratified train-test split** was used to maintain class distribution.
- The models were evaluated using metrics suitable for imbalanced datasets, including **ROC AUC**, **F1-Score**, and **Recall**.
- No resampling techniques (like **SMOTE** or undersampling) were applied. The focus was on comparing model performance under real-world imbalance conditions.
- The `Amount` feature was scaled using **StandardScaler**, and the `Time` feature was removed.
- Three models were compared: **Random Forest**, **XGBoost**, and a **Dense Neural Network**, all trained and tested using the same split.
