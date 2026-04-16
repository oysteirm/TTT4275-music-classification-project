import numpy as np
import pandas as pd
from collections import Counter

from file_to_array import (
    data_to_array_30s,
    data_to_array_10s,
    data_to_array_5s,
    mahalanobis_distance,
)


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


def load_dataset(file_path, feature_cols, dataset_type="30s"):
    if dataset_type == "30s":
        train_set, test_set = data_to_array_30s(
            file_path,
            feature_cols,
            label_col="GenreID",
            include_track_id=True,
        )
    elif dataset_type == "10s":
        train_set, test_set = data_to_array_10s(
            file_path,
            feature_cols,
            label_col="GenreID",
            include_track_id=True,
        )
    elif dataset_type == "5s":
        train_set, test_set = data_to_array_5s(
            file_path,
            feature_cols,
            label_col="GenreID",
            include_track_id=True,
        )
    else:
        raise ValueError("dataset_type must be '30s', '10s', or '5s'")

    cov_matrix = np.cov(train_set[0], rowvar=False)
    return train_set, test_set, cov_matrix


def k_NN_classifier(train_set, test_set, cov_m, k):
    predictions = []

    for i in range(len(test_set[0])):
        test_features = test_set[0][i]
        distances = []

        for j in range(len(train_set[0])):
            train_features = train_set[0][j]
            train_label = train_set[1][j]

            distance = mahalanobis_distance(test_features, train_features, cov_m)
            distances.append((distance, train_label))

        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:k]

        labels = [label for _, label in k_nearest]
        prediction = Counter(labels).most_common(1)[0][0]
        predictions.append(prediction)

    return np.array(predictions)


def evaluate_kNN(file_path, features, dataset_type="30s", k=5):
    train_set, test_set, cov_m = load_dataset(
        file_path=file_path,
        feature_cols=features,
        dataset_type=dataset_type,
    )

    predicted_genre = k_NN_classifier(train_set, test_set, cov_m, k)
    true_genre = np.array(test_set[1])
    error_rate = np.mean(true_genre != predicted_genre)
    accuracy = 1 - error_rate

    return accuracy


def test_feature4_candidates(file_path, fixed_features, dataset_type="30s", candidate_features=None, k=5):
    if candidate_features is None:
        candidate_features = ALL_FEATURES

    results = []
    fixed_set = set(fixed_features)

    for candidate in candidate_features:
        if candidate in fixed_set:
            continue

        features = fixed_features + [candidate]

        try:
            accuracy = evaluate_kNN(
                file_path=file_path,
                features=features,
                dataset_type=dataset_type,
                k=k,
            )
            results.append({
                "Feature_4": candidate,
                "Accuracy": accuracy,
            })
            print(f"Done: {candidate} -> {accuracy:.12f}")
        except Exception as e:
            print(f"Failed: {candidate} -> {e}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    return results_df


if __name__ == "__main__":
    FILE_PATH = "../data/GenreClassData_30s.txt"
    DATASET_TYPE = "30s"
    K = 5

    FIXED_FEATURES = [
        "spectral_rolloff_mean",
        "tempo",
        "spectral_centroid_mean",
    ]

    results_df = test_feature4_candidates(
        file_path=FILE_PATH,
        fixed_features=FIXED_FEATURES,
        dataset_type=DATASET_TYPE,
        k=K,
    )

    print("________________________________________________")
    print("\nTop 10:")
    print(results_df.head(10).to_string(index=False))
    print("________________________________________________")

    print("\nExcel format:")
    print("Feature_4\tAccuracy")
    for _, row in results_df.iterrows():
        print(f"{row['Feature_4']}\t{row['Accuracy']}")

    output_name = f"feature4_results_{DATASET_TYPE}_k{K}.csv"
    results_df.to_csv(output_name, index=False)
    print(f"\nSaved to {output_name}")