import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.decomposition import PCA

# Load the CSV file into a DataFrame
data = pd.read_csv('Malware_Detection_data.csv', delimiter='|')
print(data.head())

columns_to_drop = ['Name', 'md5', 'Machine', 'LoadConfigurationSize', 'VersionInformationSize']
X = data.drop(columns_to_drop, axis=1)
y = data['legitimate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define functions for KNN and SVM
def KNN_train_test():
    cls = KNeighborsClassifier(n_neighbors=10) 
    cls.fit(X_train, y_train) 
    return cls

def SVM_train_test():
    clf = svm.SVC(C=2.0, gamma='scale', kernel='rbf', random_state=2)
    clf.fit(X_train, y_train)
    return clf

# Create a pipeline for RandomForestClassifier with PCA
pipeline_rf_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),  # Thay đổi số lượng thành phần chính tùy ý
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Create a pipeline for KNN with PCA
pipeline_knn_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),  # Thay đổi số lượng thành phần chính tùy ý
    ('classifier', KNN_train_test())
])

# Create a pipeline for SVM with PCA
pipeline_svm_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),  # Thay đổi số lượng thành phần chính tùy ý
    ('classifier', SVM_train_test())
])

# Fit and predict using RandomForestClassifier pipeline with PCA
pipeline_rf_pca.fit(X_train, y_train)
y_pred_rf_pca = pipeline_rf_pca.predict(X_test)
print("Random Forest Classifier with PCA:")
print(confusion_matrix(y_test, y_pred_rf_pca))

# Calculate metrics
rf_pca_precision = precision_score(y_test, y_pred_rf_pca, average='micro') * 100
rf_pca_recall = recall_score(y_test, y_pred_rf_pca, average='micro') * 100
rf_pca_fmeasure = f1_score(y_test, y_pred_rf_pca, average='micro') * 100
rf_pca_acc = accuracy_score(y_test, y_pred_rf_pca) * 100
print("RF with PCA Precision:", rf_pca_precision)
print("RF with PCA Recall:", rf_pca_recall)
print("RF with PCA FMeasure:", rf_pca_fmeasure)
print("RF with PCA Accuracy:", rf_pca_acc)

# Fit and predict using KNN pipeline with PCA
pipeline_knn_pca.fit(X_train, y_train)
y_pred_knn_pca = pipeline_knn_pca.predict(X_test)
print("K Nearest Neighbors with PCA:")
print(confusion_matrix(y_test, y_pred_knn_pca))

# Calculate metrics
knn_pca_precision = precision_score(y_test, y_pred_knn_pca, average='micro') * 100
knn_pca_recall = recall_score(y_test, y_pred_knn_pca, average='micro') * 100
knn_pca_fmeasure = f1_score(y_test, y_pred_knn_pca, average='micro') * 100
knn_pca_acc = accuracy_score(y_test, y_pred_knn_pca) * 100
print("KNN with PCA Precision:", knn_pca_precision)
print("KNN with PCA Recall:", knn_pca_recall)
print("KNN with PCA FMeasure:", knn_pca_fmeasure)
print("KNN with PCA Accuracy:", knn_pca_acc)

# Fit and predict using SVM pipeline with PCA
pipeline_svm_pca.fit(X_train, y_train)
y_pred_svm_pca = pipeline_svm_pca.predict(X_test)
print("Support Vector Machine with PCA:")
print(confusion_matrix(y_test, y_pred_svm_pca))

# Calculate metrics
svm_pca_precision = precision_score(y_test, y_pred_svm_pca, average='micro') * 100
svm_pca_recall = recall_score(y_test, y_pred_svm_pca, average='micro') * 100
svm_pca_fmeasure = f1_score(y_test, y_pred_svm_pca, average='micro') * 100
svm_pca_acc = accuracy_score(y_test, y_pred_svm_pca) * 100
print("SVM with PCA Precision:", svm_pca_precision)
print("SVM with PCA Recall:", svm_pca_recall)
print("SVM with PCA FMeasure:", svm_pca_fmeasure)
print("SVM with PCA Accuracy:", svm_pca_acc)
