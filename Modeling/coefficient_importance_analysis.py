import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from shap import summary_plot, TreeExplainer
import shap
import xgboost as xgb
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

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

num_mfcc_coefficients = mfcc_data.shape[1]  # The number of MFCC coefficients per array
num_time_stamps = mfcc_data.shape[2]  # The number of time stamps

# Generate feature names based on the indices of MFCC coefficients
feature_names = []
for j in range(num_time_stamps):
    feature_names.append(f'MFCC_{j+1}')

# Reshape MFCC data into 2D array (flatten)

X = mfcc_data.reshape(mfcc_data.shape[0], -1) # Shape: (30914, 1*1077)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
xgb_model_mfcc = xgb.XGBClassifier()
xgb_model_mfcc.fit(X_train, y_train)
explainer_mfcc = TreeExplainer(xgb_model_mfcc)

shap_values_mfcc = explainer_mfcc.shap_values(X_test)
shap_values_mfcc = np.reshape(shap_values_mfcc, (-1, 13, 1077))
# Aggregate SHAP values across timestamps (taking mean)
shap_values_mean = np.mean(shap_values_mfcc, axis=2)

# Visualize
summary_plot(shap_values_mean, feature_names= feature_names, plot_type="bar")

# Generate SHAP values for the test data

y_pred = xgb_model_mfcc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

    # Train the Logistic Regression model without penalty
    # lr_model = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)
    # lr_model.fit(X_train, y_train)

    # Calculate AIC
    # y_pred_train = lr_model.predict(X_train)
    # residuals = y_train - y_pred_train
    # n = len(y_train)
    # k = X_train.shape[1]  # Number of features
    # aic = n * np.log(np.mean(residuals ** 2)) + 2 * k
    # # Calculate R^2
    # y_pred_test = lr_model.predict(X_test)
    # r2 = r2_score(y_test, y_pred_test)
    # print(f'R^2 for the {i+1} coefficient:', r2)
    # print(f'AIC for the {i+1} coefficient:', aic)
