#Task 1: k-NN classifier 
# - Design a k -NN classifier (k =5) for all ten genres using only the following four features; 
#       spectral rolloff mean, mfcc 1 mean, spectral centroid mean and tempo.
# - Evaluate the performance of the classification mode


import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

k = 5


features = [ 
        "spectral_rolloff_mean",
        "mfcc_1_mean",
        "spectral_centroid_mean",
        "tempo"
        ]


def k_NN_classifier(train_set,test_set,k):
    
    predictions = []

    for i in range(len(test_set[0])):
        test_features = test_set[0][i] 

        distances = []
        for j in range(len(train_set[0])):
            train_features = train_set[0][j]
            train_label = train_set[1][j]

            #Calculating the eucledian distance    
            distance = np.linalg.norm(np.array(test_features)-np.array(train_features))
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
def evaluating_k_NN_classifier(train_set, test_set, k):
     predicted_genre = k_NN_classifier(train_set, test_set, k)
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


train_set = 
test_set = 
error_rate, cm, labels, predictions = evaluating_k_NN_classifier(train_set, test_set, k)

accuracy = 1 - error_rate

print("Error rate:", error_rate)
print("Accuracy:", accuracy)
print("Labels:", labels)
print("Confusion matrix:\n", cm)

plot_confusion_matrix(cm, labels, "Confusion Matrix for k-NN (k=5)")

