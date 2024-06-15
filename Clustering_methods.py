# Unsupervised Learning
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd

folder_path = './mfccs_files_clean'

mfcc_data = []
labels = []
genders = []
races = []
professions = []
socioecos = []
data = pd.read_csv('./celebrity_data.csv')

for file_name in os.listdir(folder_path):
    # Construct the full path to the numpy file
    file_path = os.path.join(folder_path, file_name)
    if "bona-fide" in file_name: labels.append(1)
    if "spoof" in file_name: labels.append(0)

    gender = data[data['Name Files'] == "_".join(file_name.split('_')[0:-3])]['Gender'].values[0]
    genders.append(gender)
    race = data[data['Name Files'] == "_".join(file_name.split('_')[0:-3])]['Race Simple'].values[0]
    races.append(race)
    profession = data[data['Name Files'] == "_".join(file_name.split('_')[0:-3])]['Profession Simple'].values[0]
    professions.append(profession)
    socioeco = data[data['Name Files'] == "_".join(file_name.split('_')[0:-3])]['Family Socioeconomic Status'].values[0]
    socioecos.append(socioeco)


    mfcc_array = np.load(file_path)
    mfcc_data.append(mfcc_array)

# Convert the list of MFCC arrays to a single 3D numpy array
mfcc_data = np.array(mfcc_data)

# Reshape MFCC data into 2D array (flatten)
X = mfcc_data.reshape(mfcc_data.shape[0], -1)  # Shape: (30914, 13*1077)

# PCA
pca = PCA(n_components=3)
reduced_features = pca.fit_transform(X)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Plot in 3D the actual labels
for label in range(2):
     label_indices = np.array(labels) == label
     label_data = reduced_features[label_indices]
     ax.scatter(label_data[:, 0], label_data[:, 1], label_data[:, 2], label=f'Label {label == 1} Audio')
#
# # Adding labels
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
#
# # Show legend
ax.legend()
#
# # Show plot
plt.show()

# print("\nHierarchical Clustering - Cluster Method:")
Z = linkage(reduced_features, method='ward')

# Plot the dendrogram
plt.figure(figsize=(10, 6))
dendrogram(Z)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# Initialize KMeans with 4 clusters
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
kmeans = KMeans(n_clusters=2, random_state=12)

# Fit KMeans on your training data
kmeans.fit(reduced_features)

# Predict clusters for training data
clusters = kmeans.predict(reduced_features)

# Create a 3D plot

# Plotting the data points for each cluster
for cluster_label in range(2):
    cluster_indices = clusters == cluster_label
    cluster_data = reduced_features[cluster_indices]
    ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2], label=f'Cluster {cluster_label}')

# Adding labels
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
# Show legend
ax.legend()
# Show plot
plt.show()


#K-means 2 clusters discovery
# Calculate the mean across the timeframe axis
mfcc_data_mean = np.mean(mfcc_data, axis=2)

# Create a dictionary from the lists
data = {
    'cluster': list(clusters),
    'gender': genders,
    'race': races,
    'profession': professions,
    'socioeconomic_status': socioecos,
    'mfcc_1': mfcc_data_mean[:, 0],
    'mfcc_2': mfcc_data_mean[:, 1],
    'mfcc_3': mfcc_data_mean[:, 2],
    'mfcc_4': mfcc_data_mean[:, 3],
    'mfcc_5': mfcc_data_mean[:, 4],
    'mfcc_6': mfcc_data_mean[:, 5],
    'mfcc_7': mfcc_data_mean[:, 6],
    'mfcc_8': mfcc_data_mean[:, 7],
    'mfcc_9': mfcc_data_mean[:, 8],
    'mfcc_10': mfcc_data_mean[:, 9],
    'mfcc_11': mfcc_data_mean[:, 10],
    'mfcc_12': mfcc_data_mean[:, 11],
    'mfcc_13': mfcc_data_mean[:, 12],
}

# Create the DataFrame
df = pd.DataFrame(data)

# Define categories to loop through
categories = ['gender', 'race', 'profession', 'socioeconomic_status']
for category in categories:
    # Group by cluster and category, then count occurrences
    grouped = df.groupby(['cluster', category]).size().unstack().fillna(0)

    # Normalize each row to get percentages
    grouped = grouped.div(grouped.sum(axis=1), axis=0) * 100

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    grouped.plot(kind='bar', stacked=True, ax=ax)

    # Add percentage labels
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        ax.annotate(f'{height:.1f}%', (x + width / 2, y + height / 2), ha='center', va='center', fontsize=8,
                    color='white')

    # Add title and labels
    ax.set_title(f'Bar Plot of {category.capitalize()} by Cluster')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Percentage')
    ax.legend(title=category.capitalize(), bbox_to_anchor=(1.05, 1), loc='upper left')

    # Show the plot (no need to save or store)
    plt.tight_layout()
    plt.show()

#Boxplots
# Plotting the boxplot
num_coefficients = mfcc_data_mean.shape[1]
plots = []
for i in range(num_coefficients):
    plt.figure(figsize=(10, 6))
    df.boxplot(column=f'mfcc_{i+1}', by='cluster', grid=False)

    # Add title and labels
    plt.title(f'Distribution of MFCC {i+1} Coefficient by Cluster')
    plt.suptitle('')  # Suppress the automatic 'Boxplot grouped by Cluster' title
    plt.xlabel('Cluster')
    plt.ylabel(f'MFCC {i+1}')

# Show the plot
    plt.show()









