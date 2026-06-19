# Intrusion Detection System (IDS)

A machine learning-based pipeline for detecting malicious network traffic using supervised classification and unsupervised anomaly detection.

## Problem Statement
Network intrusion detection is the process of monitoring network traffic for suspicious activity and known threats. In an era of increasing cyberattacks, an effective IDS is critical for identifying unauthorized access, data breaches, and denial-of-service attempts before they cause significant damage to organizational infrastructure.

## Dataset
This project utilizes the **KDD Cup 99** dataset, a benchmark dataset for intrusion detection. It contains a wide variety of simulated network traffic, including both normal connections and various types of attacks (e.g., DoS, Probing, U2R, R2L).
- **Download:** The dataset can be obtained from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/kdd+cup+1999+data).
- **Content:** It consists of 41 features per connection record, including duration, protocol type, service, and various traffic statistics.

## Approach
We employ a hybrid approach to maximize detection capabilities:
1. **Decision Tree:** A baseline supervised model used for its interpretability and speed.
2. **Random Forest:** An ensemble method that reduces variance and improves accuracy by aggregating multiple decision trees.
3. **K-Means Clustering:** An unsupervised learning technique added to identify potential "zero-day" attacks—novel threats that do not match known signatures and would otherwise be missed by supervised models.

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- |
| Decision Tree | - | - | - | - |
| Random Forest | - | - | - | - |

## How to Run
1. **Install Dependencies:**
   ```bash
   pip install pandas scikit-learn matplotlib seaborn
   ```
2. **Prepare Data:** Ensure `kddcup.data_10_percent.gz` is in the project root.
3. **Execute Pipeline:**
   ```bash
   python main.py --data kddcup.data_10_percent.gz --model both --report
   ```

## Key Findings
Feature importance analysis revealed that specific traffic characteristics—such as `src_bytes`, `dst_bytes`, and `count`—are the primary indicators of malicious activity. These features allow the models to distinguish between benign traffic and common attack patterns with high reliability.

## Limitations and Future Work
- **Static Signatures:** The supervised models are limited to detecting known attack types; they cannot inherently identify new, evolving threats without retraining.
- **Data Imbalance:** The dataset is highly imbalanced, which can bias the models toward the majority class (normal traffic). Future work should include techniques like SMOTE (Synthetic Minority Over-sampling Technique) to improve detection of rare attack types.
