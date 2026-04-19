#Task 1: k-NN classifier 
# - Design a k -NN classifier (k =5) for all ten genres using only the following four features; 
#       spectral rolloff mean, mfcc 1 mean, spectral centroid mean and tempo.
# - Evaluate the performance of the classification mode

import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from file_to_array import train_30s, test_30s, mahalanobis_distance, cov_matrix_30s, data_to_array_30s, features

def k_NN_classifier(train_set,test_set,cov_m,k):
    
    predictions = []

    for i in range(len(test_set[0])):
        test_features = test_set[0][i] 

        distances = []
        for j in range(len(train_set[0])):
            train_features = train_set[0][j]
            train_label = train_set[1][j]

            #Calculating the mahalanobis distance   
            distance = mahalanobis_distance(test_features, train_features, cov_m)
            distances.append((distance, train_label))
        
        #sorting based on distance
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:k]

        labels = []
        for _,label in k_nearest:
             labels.append(label)

        #choosing the most common label prediction
        prediction = Counter(labels).most_common(1)[0][0]
        predictions.append(prediction)

    return np.array(predictions)

#Mark that following functions are inspired by code in notebook Problem Set 2 Solutions in TTT4275
def evaluating_k_NN_classifier(train_set, test_set, cov_m, k):
     predicted_genre = k_NN_classifier(train_set, test_set, cov_m, k)
     true_genre = np.array(test_set[1])
     error_rate = np.mean(true_genre != predicted_genre)

     labels = sorted(np.unique(true_genre))

     cm = confusion_matrix(true_genre, predicted_genre, labels=labels)

     return error_rate, cm, labels, predicted_genre

#
def plot_confusion_matrix(cm, labels, title):
    """
    Plots the confusion matrix as a heatmap
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()

# k = 5

# error_rate, cm, labels, predictions = evaluating_k_NN_classifier(train_30s, test_30s, cov_matrix_30s, k)

# accuracy = 1 - error_rate

# print("Error rate:", error_rate)
# print("Accuracy:", accuracy)
#print("Labels:", labels)
#print("Confusion matrix:\n", cm)
#plot_confusion_matrix(cm, labels, f"Confusion Matrix for k-NN (k={k})")


def rank_single_features(feature_list, file_path, k=5):
    results = []

    for feature in feature_list:
        print(f"Testing feature: {feature}")

        # use ONLY one feature
        train_set, test_set = data_to_array_30s(
            file_path,
            [feature],
            label_col="GenreID",
            include_track_id=False
        )

        # compute covariance (handle 1D case)
        X_train = train_set[0]
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)

        cov_matrix = np.cov(X_train, rowvar=False)
        if np.ndim(cov_matrix) == 0:
            cov_matrix = np.array([[cov_matrix]])

        error_rate, _, _, _ = evaluating_k_NN_classifier(
            train_set, test_set, cov_matrix, k
        )

        accuracy = 1 - error_rate

        results.append({
            "Feature": feature,
            "Accuracy": accuracy,
            "Error rate": error_rate
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

    return results_df

results = rank_single_features(features, "../data/GenreClassData_30s.txt")
print(results.to_string())