# Unsupervised Learning
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


folder_path = './mfccs_files_clean'

mfcc_data = []
labels = []
for file_name in os.listdir(folder_path):
    # Construct the full path to the numpy file
    file_path = os.path.join(folder_path, file_name)
    if "bona-fide" in file_name: labels.append(1)
    if "spoof" in file_name: labels.append(0)
    mfcc_array = np.load(file_path)
    mfcc_data.append(mfcc_array)

# Convert the list of MFCC arrays to a single 3D numpy array
mfcc_data = np.array(mfcc_data)

# Reshape MFCC data into 2D array (flatten)
X = mfcc_data.reshape(mfcc_data.shape[0], -1)  # Shape: (30914, 13*1077)

# PCA
#pca = PCA(n_components=3)
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(reduced_features, labels, test_size=0.2, random_state=42)

model = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
print(accuracy)
print(confusion_mat)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# ax.scatter(reduced_features[:, 0], reduced_features[:, 1], reduced_features[:, 2])
#
# # Adding labels
# ax.set_xlabel('Principal Component 1')
# ax.set_ylabel('Principal Component 2')
# ax.set_zlabel('Principal Component 3')
# # Show legend
# ax.legend()
# # Show plot
# plt.show()
#

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# Plot in 3D the actual labels
# for label in range(2):
#     label_indices = np.array(labels) == label
#     label_data = reduced_features[label_indices]
#     ax.scatter(label_data[:, 0], label_data[:, 1], label_data[:, 2], label=f'Label {label == 1} Audio')
#
# # Adding labels
# ax.set_xlabel('Principal Component 1')
# ax.set_ylabel('Principal Component 2')
# ax.set_zlabel('Principal Component 3')
#
# # Show legend
# ax.legend()
#
# # Show plot
# plt.show()


# Initialize KMeans with 4 clusters
# kmeans = KMeans(n_clusters=3, random_state=12)
#
# # Fit KMeans on your training data
# kmeans.fit(reduced_features)
#
# # Predict clusters for training data
# clusters = kmeans.predict(reduced_features)
#
# # Create a 3D plot
#
# # Plotting the data points for each cluster
# for cluster_label in range(3):
#     cluster_indices = clusters == cluster_label
#     cluster_data = reduced_features[cluster_indices]
#     ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2], label=f'Cluster {cluster_label}')
#
# # Adding labels
# ax.set_xlabel('Principal Component 1')
# ax.set_ylabel('Principal Component 2')
# ax.set_zlabel('Principal Component 3')
# # Show legend
# ax.legend()
# # Show plot
# plt.show()


# print("\nHierarchical Clustering - Cluster Method:")
# Z = linkage(reduced_features, method='ward')
#
# # Plot the dendrogram
# plt.figure(figsize=(10, 6))
# dendrogram(Z)
# plt.title('Dendrogram for Hierarchical Clustering')
# plt.xlabel('Sample Index')
# plt.ylabel('Distance')
# plt.show()
