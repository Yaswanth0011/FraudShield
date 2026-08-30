# 📊 SentinelShield AI — Data Documentation & Dataset Card

## 1. Dataset Overview

This project utilizes transaction data modeled after the widely recognized **Credit Card Fraud Detection Dataset** released by the Machine Learning Group (MLG) at Université Libre de Bruxelles (ULB) and Worldline.

- **Primary Source**: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Timeframe**: Transactions made by European cardholders in September 2013 across a 2-day period.
- **Total Records (Full Dataset)**: 284,807 transactions.
- **Positive (Fraud) Records**: 492 cases (representing ~0.172% of all transactions, presenting extreme class imbalance).

---

## 2. Feature Definitions & Schema

Because financial transaction records contain sensitive personal identifiable information (PII), numerical input variables $V_1, V_2, \dots, V_{28}$ are the result of a **Principal Component Analysis (PCA)** transformation performed by the original researchers.

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `Time` | Float | Number of seconds elapsed between this transaction and the first transaction in the dataset. |
| `V1` – `V28` | Float | Principal components obtained via PCA to protect user privacy and confidentiality. |
| `Amount` | Float | Transaction transaction amount. |
| `Class` | Integer | Binary target variable: `0` for Legitimate transaction, `1` for Fraudulent transaction. |

### Enriched Synthetic Metadata (UI Layer)
For realistic demonstration and simulation in the web application, synthetic contextual metadata fields are dynamically populated using the `Faker` library:
- `Customer_Name`: Generated synthetic cardholder identity.
- `Card_Number` / `Masked_Card`: Synthetic masked card identifier (`•••• •••• •••• 1234`).
- `Merchant`: Synthetic merchant enterprise & category.
- `Location`: Synthetic geographic city and region.

---

## 3. Class Imbalance & Mitigation

Due to the extreme sparsity of fraudulent transactions in real-world payment networks (99.83% legitimate vs 0.17% fraud), standard accuracy is a misleading evaluation metric. This project utilizes:
- **Class-weighted loss functions** (`class_weight='balanced'` in Random Forest and Logistic Regression)
- **Evaluation via Precision, Recall, F1-Score, ROC-AUC, and Precision-Recall AUC (PR-AUC)**
- **Configurable Decision Thresholding** to adjust operational sensitivity based on business tolerance for false positives versus false negatives.

---

## 4. How to Download the Full Benchmark Dataset

To train models on the full 284K transaction dataset:
1. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Place the file directly in this `data/` directory (`data/creditcard.csv`).
3. Run model training:
   ```bash
   python src/model_training.py --data data/creditcard.csv
   ```
