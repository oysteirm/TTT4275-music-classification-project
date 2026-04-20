import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from file_to_array import (
    data_to_array_30s,
    split_train_into_train_validation,
    mahalanobis_distance,
    train_30s,
    cov_matrix_30s,
    test_30s,
)


FILE_PATH = "../data/GenreClassData_30s.txt"
K = 5


ALL_FEATURES = [
    "zero_cross_rate_mean",
    "zero_cross_rate_std",
    "rmse_mean",
    "rmse_var",
    "spectral_centroid_mean",
    "spectral_centroid_var",
    "spectral_bandwidth_mean",
    "spectral_bandwidth_var",
    "spectral_rolloff_mean",
    "spectral_rolloff_var",
    "spectral_contrast_mean",
    "spectral_contrast_var",
    "spectral_flatness_mean",
    "spectral_flatness_var",
    "chroma_stft_1_mean",
    "chroma_stft_2_mean",
    "chroma_stft_3_mean",
    "chroma_stft_4_mean",
    "chroma_stft_5_mean",
    "chroma_stft_6_mean",
    "chroma_stft_7_mean",
    "chroma_stft_8_mean",
    "chroma_stft_9_mean",
    "chroma_stft_10_mean",
    "chroma_stft_11_mean",
    "chroma_stft_12_mean",
    "chroma_stft_1_std",
    "chroma_stft_2_std",
    "chroma_stft_3_std",
    "chroma_stft_4_std",
    "chroma_stft_5_std",
    "chroma_stft_6_std",
    "chroma_stft_7_std",
    "chroma_stft_8_std",
    "chroma_stft_9_std",
    "chroma_stft_10_std",
    "chroma_stft_11_std",
    "chroma_stft_12_std",
    "tempo",
    "mfcc_1_mean",
    "mfcc_2_mean",
    "mfcc_3_mean",
    "mfcc_4_mean",
    "mfcc_5_mean",
    "mfcc_6_mean",
    "mfcc_7_mean",
    "mfcc_8_mean",
    "mfcc_9_mean",
    "mfcc_10_mean",
    "mfcc_11_mean",
    "mfcc_12_mean",
    "mfcc_1_std",
    "mfcc_2_std",
    "mfcc_3_std",
    "mfcc_4_std",
    "mfcc_5_std",
    "mfcc_6_std",
    "mfcc_7_std",
    "mfcc_8_std",
    "mfcc_9_std",
    "mfcc_10_std",
    "mfcc_11_std",
    "mfcc_12_std",
]

TRIPLE_SETS = [
    ["spectral_rolloff_mean", "mfcc_1_mean", "spectral_centroid_mean"],
    ["spectral_rolloff_mean", "spectral_centroid_mean", "tempo"],
    ["spectral_rolloff_mean", "mfcc_1_mean", "tempo"],
    ["spectral_centroid_mean", "mfcc_1_mean", "tempo"],
]

def get_X_y(dataset):
    return dataset[0], dataset[1]


def load_data(features):
    train_set, test_set = data_to_array_30s(FILE_PATH, features, "GenreID", True)

    train_set, val_set = split_train_into_train_validation(train_set)

    X_train, _ = get_X_y(train_set)

    cov = np.cov(X_train, rowvar=False)

    if np.ndim(cov) == 0:
        cov = np.array([[cov]])
    # to avoid singularity issues
    cov = cov + 1e-8 * np.eye(cov.shape[0])

    return train_set, val_set, cov


def k_NN(train_set, eval_set, cov, k):
    X_train, y_train = get_X_y(train_set)
    X_eval, _ = get_X_y(eval_set)

    predictions = []

    for x in X_eval:
        distances = []

        for xt, yt in zip(X_train, y_train):
            d = mahalanobis_distance(x, xt, cov)
            distances.append((d, yt))

        distances.sort(key=lambda t: t[0])
        k_labels = [label for _, label in distances[:k]]
        pred = Counter(k_labels).most_common(1)[0][0]
        predictions.append(pred)

    return np.array(predictions)


def compute_accuracy(train_set, val_set, cov, k):
    _, y_val = get_X_y(val_set)
    y_pred = k_NN(train_set, val_set, cov, k)
    return np.mean(y_pred == np.array(y_val))


results = []

for triple in TRIPLE_SETS:
    for feature_4 in ALL_FEATURES:
        if feature_4 in triple:
            continue
        features = triple + [feature_4]
        train_set, val_set, cov = load_data(features)
        acc = compute_accuracy(train_set, val_set, cov, K)
        results.append((features, acc))

results.sort(key=lambda x: x[1], reverse=True)

print("\n----TOP 10 FEATURE COMBINATIONS---\n")
for i, (features, acc) in enumerate(results[:10], 1):
    print(f"{i}. {features} -> {acc:.4f}")

best_features = results[0][0]
print("\nBest features:", best_features)

#Mark that the two following functions are inspired by code in notebook Problem Set 2 Solutions in TTT4275, provided via course webpage
def evaluating_k_NN_classifier(train_set, test_set, cov_m, k):
     predicted_genre = k_NN(train_set, test_set, cov_m, k)
     true_genre = np.array(test_set[1])
     error_rate = np.mean(true_genre != predicted_genre)

     labels = sorted(np.unique(true_genre))

     cm = confusion_matrix(true_genre, predicted_genre, labels=labels)

     return error_rate, cm, labels, predicted_genre

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

error_rate, cm, labels, predictions = evaluating_k_NN_classifier(train_30s, test_30s, cov_matrix_30s, K)

accuracy = 1 - error_rate

print("Error rate:", error_rate)
print("Accuracy:", accuracy)
print("Confusion matrix:\n", cm)
plot_confusion_matrix(cm, labels, f"Confusion Matrix for k-NN (k={K})")
