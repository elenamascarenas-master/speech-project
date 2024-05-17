import numpy as np
import pandas as pd
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

# Load gender information
data = pd.read_csv('./celebrity_data.csv.csv')

mfcc_data = []
labels = []
#genders = []
races = []

# Load data, labels, and genders
for file_name in os.listdir(folder_path):
    # Construct the full path to the numpy file
    file_path = os.path.join(folder_path, file_name)
    # Fake and real encoded as 0,1
    if "bona-fide" in file_name:
        labels.append(1)
    if "spoof" in file_name:
        labels.append(0)
    # Extract gender from the CSV file based on the file name
    #gender = data[data['Name Files'] == "_".join(file_name.split('_')[0:-3])]['Gender'].values[0]
    race = data[data['Name Files'] == "_".join(file_name.split('_')[0:-3])]['Race Simple'].values[0]
    #genders.append(gender)
    races.append(race)
    mfcc_array = np.load(file_path)
    mfcc_data.append(mfcc_array)

# Convert the list of MFCC arrays to a single 3D numpy array
mfcc_data = np.array(mfcc_data)

# Reshape MFCC data into 2D array (flatten)
X = mfcc_data.reshape(mfcc_data.shape[0], -1) # Shape: (30914, 13*1077)

# Split the dataset into train and test
#X_train, X_test, y_train, y_test, gender_train, gender_test = train_test_split(
#    X, labels, genders, test_size=0.3, random_state=42)

X_train, X_test, y_train, y_test, race_train, race_test = train_test_split(
    X, labels, races, test_size=0.3, random_state=42)


# Split the test set into male and female subsets
#X_test_male = X_test[np.array(gender_test) == 'Male']
#X_test_female = X_test[np.array(gender_test) == 'Female']
#y_test_male = np.array(y_test)[np.array(gender_test) == 'Male']
#y_test_female = np.array(y_test)[np.array(gender_test) == 'Female']

X_test_non_white = X_test[np.array(race_test) == 'Non-white']
X_test_white = X_test[np.array(race_test) == 'White']
y_test_non_white = np.array(y_test)[np.array(race_test) == 'Non-white']
y_test_white = np.array(y_test)[np.array(race_test) == 'White']
print("done splitting")

# Models
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
    # y_pred_male = model.predict(X_test_male)
    # accuracy_male = accuracy_score(y_test_male, y_pred_male)
    # cm_male = confusion_matrix(y_test_male, y_pred_male)

    # # Evaluate on female subset
    # y_pred_female = model.predict(X_test_female)
    # accuracy_female = accuracy_score(y_test_female, y_pred_female)
    # cm_female = confusion_matrix(y_test_female, y_pred_female)

    y_pred_non_white = model.predict(X_test_non_white)
    accuracy_non_white = accuracy_score(y_test_non_white, y_pred_non_white)
    cm_non_white = confusion_matrix(y_test_non_white, y_pred_non_white)

    # Evaluate on female subset
    y_pred_white = model.predict(X_test_white)
    accuracy_white = accuracy_score(y_test_white, y_pred_white)
    cm_white = confusion_matrix(y_test_white, y_pred_white)

    # Print results
    print(f"Model: {model_name}")
    #print("Male Subset:")
    print("Non-white Subset:")
    #print(f"Accuracy: {accuracy_male}")
    print(f"Accuracy: {accuracy_non_white}")
    print("Confusion Matrix:")
    #print(cm_male)
    print(cm_non_white)
    #print("Female Subset:")
    print("White Subset:")
    #print(f"Accuracy: {accuracy_female}")
    print(f"Accuracy: {accuracy_white}")
    print("Confusion Matrix:")
    #print(cm_female)
    print(cm_white)
    print()