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

-Correlation analysis heatmap:

<img width="718" height="605" alt="correlation_heatmap" src="https://github.com/user-attachments/assets/2abeccdc-777a-4ef5-b9a5-86401317710f" />



# Models Implemented:

1) Logistic Regression

2) Support Vector Machine with RBF kernel

3) Gaussian Naive Bayes

All models were trained on normalized training data and evaluated on a separate test set.




# Model Performance:

## Logistic Regression:

-Accuracy: 0.882

-AUC: 0.920

-Balanced sensitivity and specificity

-ROC Curve: 


<img width="718" height="605" alt="roc_curve_logistic_reg" src="https://github.com/user-attachments/assets/57221cf4-561a-48e3-9f66-1c0c360deba6" />

-Confusion Matrix:


<img width="718" height="605" alt="logistic_reg_confusion_matrix" src="https://github.com/user-attachments/assets/baadff07-96be-4b83-8b94-94f31cae46fd" />




## Support Vector Machine (RBF):

-Accuracy: 0.872

-AUC: 0.920

-Similar ranking performance to logistic regression

-ROC Curve: 


<img width="718" height="605" alt="roc_svm" src="https://github.com/user-attachments/assets/ba409419-83e4-4e0e-a980-4839d017fbed" />

-Confusion Matrix:


<img width="718" height="605" alt="svm_confusion_matrix" src="https://github.com/user-attachments/assets/02d6a8e9-c77f-4741-9fe9-b2f8dbd99e97" />



## Naive Bayes:

-Accuracy: 0.526

-AUC: 0.703

-Lower performance due to violation of the conditional independence assumption

-ROC Curve: 


<img width="718" height="605" alt="roc_naive_bayes" src="https://github.com/user-attachments/assets/7caa385c-dd64-4450-906e-923626a6e013" />


-Confusion Matrix:


<img width="718" height="605" alt="naive_bayes_confusion_matrix" src="https://github.com/user-attachments/assets/ed11d3d3-2ab2-4bb5-ab40-591fbb2f78ec" />


# Key Conclusion

Logistic regression is recommended for this dataset. It achieved high predictive performance while maintaining interpretability, which is important for environmental and regulatory decision-making.

1) Data preprocessing and validation

2) Prevention of data leakage

3) Feature scaling and collinearity analysis

4) Implementation of multiple classifiers in MATLAB

5) Model comparison using ROC curves and AUC

6) Interpretation of model assumptions and limitations
