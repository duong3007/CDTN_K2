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

# Create a pipeline for RandomForestClassifier
pipeline_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Create a pipeline for KNN
pipeline_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', KNN_train_test())
])

# Create a pipeline for SVM
pipeline_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVM_train_test())
])

# Fit and predict using RandomForestClassifier pipeline
pipeline_rf.fit(X_train, y_train)
y_pred_rf = pipeline_rf.predict(X_test)
print("Random Forest Classifier:")
print(confusion_matrix(y_test, y_pred_rf))

# Calculate metrics
rf_precision = precision_score(y_test, y_pred_rf, average='micro') * 100
rf_recall = recall_score(y_test, y_pred_rf, average='micro') * 100
rf_fmeasure = f1_score(y_test, y_pred_rf, average='micro') * 100
rf_acc = accuracy_score(y_test, y_pred_rf) * 100
print("RF Precision:", rf_precision)
print("RF Recall:", rf_recall)
print("RF FMeasure:", rf_fmeasure)
print("RF Accuracy:", rf_acc)

# Fit and predict using KNN pipeline
pipeline_knn.fit(X_train, y_train)
y_pred_knn = pipeline_knn.predict(X_test)
print("K Nearest Neighbors:")
print(confusion_matrix(y_test, y_pred_knn))

# Calculate metrics
knn_precision = precision_score(y_test, y_pred_knn, average='micro') * 100
knn_recall = recall_score(y_test, y_pred_knn, average='micro') * 100
knn_fmeasure = f1_score(y_test, y_pred_knn, average='micro') * 100
knn_acc = accuracy_score(y_test, y_pred_knn) * 100
print("KNN Precision:", knn_precision)
print("KNN Recall:", knn_recall)
print("KNN FMeasure:", knn_fmeasure)
print("KNN Accuracy:", knn_acc)

# Fit and predict using SVM pipeline
pipeline_svm.fit(X_train, y_train)
y_pred_svm = pipeline_svm.predict(X_test)
print("Support Vector Machine:")
print(confusion_matrix(y_test, y_pred_svm))

# Calculate metrics
svm_precision = precision_score(y_test, y_pred_svm, average='micro') * 100
svm_recall = recall_score(y_test, y_pred_svm, average='micro') * 100
svm_fmeasure = f1_score(y_test, y_pred_svm, average='micro') * 100
svm_acc = accuracy_score(y_test, y_pred_svm) * 100
print("SVM Precision:", svm_precision)
print("SVM Recall:", svm_recall)
print("SVM FMeasure:", svm_fmeasure)
print("SVM Accuracy:", svm_acc)
