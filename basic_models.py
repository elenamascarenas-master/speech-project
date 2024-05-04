import numpy as np
import os
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC


# Define the path to the directory containing the numpy files
folder_path = './mfccs_files_clean'

mfcc_data = []
labels = []
for file_name in os.listdir(folder_path):
    # Construct the full path to the numpy file
    file_path = os.path.join(folder_path, file_name)
    # fake and real encoded as 0,1
    if "bona-fide" in file_name: labels.append(1)
    if "spoof" in file_name: labels.append(0)
    mfcc_array = np.load(file_path)
    mfcc_data.append(mfcc_array)

# Convert the list of MFCC arrays to a single 3D numpy array
mfcc_data = np.array(mfcc_data)
print("Shape of loaded data:", mfcc_data.shape)

# Reshape MFCC data into 2D array (flatten)
X = mfcc_data.reshape(mfcc_data.shape[0], -1)  # Shape: (30914, 13*1077)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
print("Percentage of real audio in all:", sum(labels)/ len(labels))
print("Percentage of real audio in train:", sum(y_train)/ len(y_train))
print("Percentage of real audio in test:", sum(y_test)/ len(y_test))

# XGboost- 90%

xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("XGboost Accuracy :", accuracy)

# LDA- 77%
lda_model = LinearDiscriminantAnalysis()
# Train LDA classifier
lda_model.fit(X_train, y_train)

# Predictions from LDA classifier
y_pred_lda = lda_model.predict(X_test)

# Evaluate model performance for LDA
accuracy_lda = accuracy_score(y_test, y_pred_lda)
print("LDA Accuracy:", accuracy_lda)

# KNN- 72%

# Create KNN classifier
knn_model = KNeighborsClassifier()
# Train KNN classifier
knn_model.fit(X_train, y_train)

# Predictions from KNN classifier
y_pred_knn = knn_model.predict(X_test)

# Evaluate model performance for KNN
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("KNN Accuracy:", accuracy_knn)

# SVM- 79%

# Create SVM classifier with a polynomial kernel
svm_model = SVC(kernel='poly', degree=3)

# Train SVM classifier
svm_model.fit(X_train, y_train)

# Predictions from SVM classifier
y_pred_svm = svm_model.predict(X_test)

# Evaluate model performance for SVM
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", accuracy_svm)

# Random Forest- 87%
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest model
rf_model.fit(X_train, y_train)

# Make predictions on the testing data
rf_y_pred = rf_model.predict(X_test)

# Evaluate model performance
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print("Random Forest Accuracy:", rf_accuracy)

# LR- 79% maybe more
# Initialize Logistic Regression classifier
lr_model = LogisticRegression(max_iter=1000, random_state=42)

# Train the Logistic Regression model
lr_model.fit(X_train, y_train)

# Make predictions on the testing data
lr_y_pred = lr_model.predict(X_test)

# Evaluate model performance
lr_accuracy = accuracy_score(y_test, lr_y_pred)
print("Logistic Regression Accuracy:", lr_accuracy)
