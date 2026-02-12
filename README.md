🔬 Project Overview

Title: Predicting Chemical Biodegradability Using QSAR Data

This project builds and compares machine learning classification models to predict whether a chemical compound is biodegradable based on molecular descriptors.

Dataset:

QSAR Biodegradation Dataset (UCI)

1055 organic compounds

41 molecular descriptors

Binary target: biodegradable (1) or non-biodegradable (0)

⚙️ Data Processing & Validation

Key preprocessing steps:

Checked for missing and infinite values

Verified zero-variance features (none found)

Assessed duplicate rows

Identified moderate class imbalance (699 vs 356)

Performed 80:20 hold-out split

Applied z-score normalisation using training statistics only

Conducted Pearson correlation analysis to assess collinearity

Outliers were detected but retained, as extreme descriptor values may represent valid chemical properties rather than noise.

🤖 Models Implemented

Three classification models were trained and compared:

1️⃣ Logistic Regression

Linear classifier

Interpretable model

Sensitive to scaling and collinearity

2️⃣ Support Vector Machine (RBF Kernel)

Nonlinear decision boundary

Kernel scale automatically estimated

Box constraint = 1

Internal standardisation disabled to prevent leakage

3️⃣ Gaussian Naive Bayes

Assumes conditional independence

Computationally efficient

Used as baseline model

📊 Model Evaluation

Evaluation Metrics:

Accuracy

Confusion Matrix

ROC Curve

Area Under Curve (AUC)

🔹 Logistic Regression

Accuracy: 0.882

AUC: 0.920

Balanced sensitivity and specificity

🔹 SVM (RBF)

Accuracy: 0.872

AUC: 0.920

Similar ranking performance to logistic regression

No clear improvement despite nonlinear boundary

🔹 Naive Bayes

Accuracy: 0.526

AUC: 0.703

Poor specificity

Strong bias toward positive class

Performance limited by violated independence assumption

🏆 Key Findings

Logistic Regression and SVM performed strongly and similarly.

Logistic Regression is preferred due to:

Slightly higher accuracy

Simpler structure

Better interpretability for regulatory contexts

Naive Bayes underperformed due to correlated QSAR descriptors.
