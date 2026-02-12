# Predicting Chemical Biodegradability Using QSAR Data


This project builds and compares machine learning classification models to predict whether a chemical compound is biodegradable based on molecular descriptors.

The dataset used is the QSAR Biodegradation dataset from the UCI Machine Learning Repository. It contains 1055 compounds described by 41 molecular descriptors, with a binary target indicating biodegradable or non-biodegradable.




# Data Processing:

-Checked for missing and infinite values

-Verified zero-variance features

-Assessed duplicate rows

-Identified moderate class imbalance (699 non-biodegradable, 356 biodegradable)

-Performed 80:20 train-test split

-Applied z-score normalization using training statistics only to prevent data leakage

-Conducted Pearson correlation analysis to assess feature collinearity

-Retained outliers as QSAR descriptors can contain meaningful extreme values




# Models Implemented:

1) Logistic Regression

2) Support Vector Machine with RBF kernel

3) Gaussian Naive Bayes

All models were trained on normalized training data and evaluated on a separate test set.




# Model Performance:

Logistic Regression:

-Accuracy: 0.882

-AUC: 0.920

-Balanced sensitivity and specificity



Support Vector Machine (RBF):

-Accuracy: 0.872

-AUC: 0.920

-Similar ranking performance to logistic regression



Naive Bayes:

-Accuracy: 0.526

-AUC: 0.703

-Lower performance due to violation of the conditional independence assumption



# Key Conclusion

Logistic regression is recommended for this dataset. It achieved high predictive performance while maintaining interpretability, which is important for environmental and regulatory decision-making.

1) Data preprocessing and validation

2) Prevention of data leakage

3) Feature scaling and collinearity analysis

4) Implementation of multiple classifiers in MATLAB

5) Model comparison using ROC curves and AUC

6) Interpretation of model assumptions and limitations
