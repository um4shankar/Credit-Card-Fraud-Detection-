# 💳 Credit Card Fraud Detection

## 🧾 Problem Statement

Fraudulent credit card transactions are rare but costly. This project aims to detect such anomalies using machine learning techniques on a highly imbalanced dataset.

---

## 🧪 Approach

- Performed EDA on anonymized credit card transaction data.
- Standardized numerical features (`Amount`, `Time`).
- Used SMOTE to balance classes.
- Trained multiple ML models: Logistic Regression, Random Forest, XGBoost, Isolation Forest.
- Evaluated using F1-Score, AUC-ROC due to imbalance.

---

## 🔧 Tech Stack

- Python (NumPy, Pandas, Scikit-learn, XGBoost)
- Jupyter Notebook
- Matplotlib / Seaborn
- Imbalanced-learn (SMOTE)

---

## 📊 Results

| Model             | Precision | Recall | F1-Score | AUC-ROC |
|------------------|-----------|--------|----------|---------|
| LogisticRegression | 0.92      | 0.89   | 0.90     | 0.98    |
| RandomForest       | 0.97      | 0.94   | 0.95     | 0.99    |
| XGBoost            | 0.96      | 0.95   | 0.95     | 0.99    |
| IsolationForest    | 0.72      | 0.65   | 0.68     | 0.86    |

---

## 📁 File Structure

```bash
├── fraud_detection.ipynb      # Main Jupyter notebook
├── data/
│   └── creditcard.csv         # Dataset (add or link)
├── images/                    # Saved plots
├── README.md
