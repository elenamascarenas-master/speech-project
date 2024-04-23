    import numpy as np
    import os
    from sklearn.model_selection import train_test_split
    import xgboost as xgb
    from sklearn.metrics import accuracy_score
    import time

    start_time = time.time()
    # Define the path to the directory containing the numpy files
    folder_path = './melspec_files_clean'

    melspec_data = []
    labels = []
    for file_name in os.listdir(folder_path):
        # Construct the full path to the numpy file
        file_path = os.path.join(folder_path, file_name)
        # fake and real encoded as 0,1
        if "bona-fide" in file_name: labels.append(1)
        if "spoof" in file_name: labels.append(0)
        melspec_array = np.load(file_path)
        melspec_data.append(melspec_array)

    # Convert the list of MFCC arrays to a single 3D numpy array
    melspec_data = np.array(melspec_data)
    print("Shape of loaded data:", melspec_data.shape)

    # Reshape MFCC data into 2D array (flatten)
    X = melspec_data.reshape(melspec_data.shape[0], -1)  # Shape: (30914, 13*`1077)

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    print("Percentage of real audio in all:", sum(labels)/ len(labels))
    print("Percentage of real audio in train:", sum(y_train)/ len(y_train))
    print("Percentage of real audio in test:", sum(y_test)/ len(y_test))

    ## XGboost
    model = xgb.XGBClassifier()

    # Train the model
    model.fit(X_train[0:2000], y_train[0:2000])

    # Make predictions on the testing data
    y_pred = model.predict(X_test[0:2000])

    # Evaluate model performance
    accuracy = accuracy_score(y_test[0:2000], y_pred)
    print("Accuracy:", accuracy)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time} seconds")
