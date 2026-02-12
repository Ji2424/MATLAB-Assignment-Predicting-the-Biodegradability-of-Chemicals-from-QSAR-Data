%% loading the data
clear
clc
close all
load('QSAR_data.mat')

x= QSAR_data(:, 1:41);
y= QSAR_data(:, 42);
fprintf("X size: %d x %d | Y size: %d x %d\n", size(x,1), size(x,2), size(y,1), size(y,2));

%% Data Check
%Check for missing values (NaN)
 numNaNx= sum(isnan(x), 'all');
 fprintf("Missing values in x: %d\n", numNaNx);
  numNaNy= sum(isnan(y), 'all');
 fprintf("Missing values in y: %d\n", numNaNy);

%Check for infinite values
 numInfx= sum(isinf(x), 'all');
 fprintf("Infinite values in x: %d\n", numInfx);
  numInfy= sum(isinf(y), 'all');
 fprintf("Infinite values in y: %d\n", numInfy);

%Check for duplicate rows
 unique_Rowsx= unique(x, 'rows');
 num_Dupx= size(x,1) - size(unique_Rowsx,1);
 fprintf("Duplicate rows in x: %d\n", num_Dupx);
 unique_Rowsy= unique(y, 'rows');
 num_Dupy= size(y,1) - size(unique_Rowsy,1);
 fprintf("Duplicate rows in y: %d\n", num_Dupy);

%Check for zero variance features
 feature_Std= std(x);
 zero_VarFeatures= sum(feature_Std == 0);
 fprintf("Zero-variance features: %d\n", zero_VarFeatures);

%Class balance check
 fprintf("Class balance:\n");
 tabulate(y)

%Outlier Check
 outlierCount= sum(isoutlier(x), 'all');
 fprintf("Detected outlier entries (not removed): %d\n", outlierCount);

%% Train test split
rng(1)

cv= cvpartition(y, 'HoldOut', 0.2);

x_train= x(training(cv), :);
y_train= y(training(cv), :);

x_test= x(test(cv), :);
y_test= y(test(cv), :);

fprintf("Train size: %d | Test size: %d\n", length(y_train), length(y_test));

%% z score normalization
m= mean(x_train);
s= std(x_train);

x_train= (x_train-m)./s;
x_test= (x_test-m)./s;

%% Collinearity analysis (feature correlation)
 %Find the Pearson correlation between features using training data only
 R= corr(x_train);

%Visual inspection using a heatmap
 figure;
 imagesc(R);
 colorbar;
 axis square;
 title('Correlation Heatmap');
 xlabel('Feature Index');
 ylabel('Feature Index');
 clim([-1 1]);

%Identify highly correlated feature pairs
 threshold= 0.9;
 upperTri= triu(abs(R),1); % avoid diagonal and double counting
 [numPairs]= sum(upperTri(:) > threshold);
fprintf("Highly correlated feature pairs (|r| >= %.2f): %d\n", threshold, numPairs);

%% logistic regression
log_model= fitclinear(x_train, y_train, 'Learner', 'logistic');
y_pred_log= predict(log_model, x_test);
acc_log= mean(y_pred_log == y_test);
fprintf("Logistic Regression Accuracy: %.3f\n", acc_log);

%Confusion matrix 
figure;
confusionchart(y_test, y_pred_log);
title('Logistic Regression Confusion Matrix');

%% ROC Curve
[~, scores] = predict(log_model, x_test);

%Find the ROC and AUC
[X_log, Y_log, ~, AUC_log] = perfcurve(y_test, scores(:,2), 1);

figure;
plot(X_log, Y_log, 'LineWidth', 2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title(sprintf('ROC Curve - Logistic Regression (AUC = %.3f)', AUC_log))
grid on

%% SVM (RBF Kernel)
svm_model = fitcsvm(x_train, y_train, ...
    "KernelFunction","rbf", ...
    "KernelScale","auto", ...
    "BoxConstraint",1, ...
    "Standardize",false);

%SVM Predictions and Accuracy
y_pred_svm = predict(svm_model, x_test);
acc_svm = mean(y_pred_svm == y_test);
fprintf("SVM Accuracy: %.3f\n", acc_svm);

%Confusion matrix 
figure;
confusionchart(y_test, y_pred_svm);
title('SVM Confusion Matrix');

%% SVM ROC Curve
[~, scores_svm] = predict(svm_model, x_test);  % scores_svm(:,2) = score for class 1
[X_svm, Y_svm, ~, AUC_svm] = perfcurve(y_test, scores_svm(:,2), 1);

figure;
plot(X_svm, Y_svm, 'LineWidth', 2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title(sprintf('ROC Curve - SVM (AUC = %.3f)', AUC_svm))
grid on

%% Naive Bayes
nb_model = fitcnb(x_train, y_train);
y_pred_nb = predict(nb_model, x_test);
acc_nb = mean(y_pred_nb == y_test);
fprintf("Naive Bayes Accuracy: %.3f\n", acc_nb);

%Confusion matrix 
figure;
confusionchart(y_test, y_pred_nb);
title('Naive Bayes Confusion Matrix');

%% Naive Bayes ROC
[~, scores_nb] = predict(nb_model, x_test);
[X_nb, Y_nb, ~, AUC_nb] = perfcurve(y_test, scores_nb(:,2), 1);

figure;
plot(X_nb, Y_nb, 'LineWidth', 2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title(sprintf('ROC Curve - Naive Bayes (AUC = %.3f)', AUC_nb))
grid on



