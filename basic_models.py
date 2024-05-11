import numpy as np
import os
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix
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
    mel_spec_array = np.load(file_path)
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

models = {
    "XGBoost": xgb.XGBClassifier(),
    "LDA": LinearDiscriminantAnalysis(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(kernel='poly', degree=3),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "LogisticRegression": LogisticRegression(penalty='l1', solver='liblinear', C=0.1, max_iter=1000, random_state=42)
}

# Train and evaluate each model separately for male and female subsets
for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)

    # Evaluate on male subset
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)


    # Print results
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(confusion_mat)
    print()