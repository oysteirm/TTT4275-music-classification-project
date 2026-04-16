import numpy as np
import pandas as pd
from collections import Counter

from file_to_array import data_to_array_30s, mahalanobis_distance


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

BASE_FEATURES = [
    "spectral_rolloff_mean",
    "mfcc_1_mean",
    "spectral_centroid_mean",
    "tempo",
]


def load_data_30s(file_path, feature_cols):
    train_set, test_set = data_to_array_30s(
        file_path,
        feature_cols,
        label_col="GenreID",
        include_track_id=True,
    )
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


def evaluate_kNN_30s(file_path, features, k=5):
    train_set, test_set, cov_m = load_data_30s(file_path, features)
    predicted_genre = k_NN_classifier(train_set, test_set, cov_m, k)
    true_genre = np.array(test_set[1])
    error_rate = np.mean(true_genre != predicted_genre)
    return 1 - error_rate


def test_single_features(file_path, all_features, k=5):
    results = []

    for feature in all_features:
        try:
            accuracy = evaluate_kNN_30s(file_path, [feature], k=k)
            results.append({"Single feature": feature, "Accuracy": accuracy})
            print(f"[single] Done: {feature} -> {accuracy:.12f}")
        except Exception as e:
            print(f"[single] Failed: {feature} -> {e}")

    df = pd.DataFrame(results)
    df = df.sort_values(by="Single feature").reset_index(drop=True)
    return df


def test_feature4_candidates(file_path, fixed_features, all_features, k=5):
    results = []
    fixed_set = set(fixed_features)

    for candidate in all_features:
        # hopp over faste features, unntatt dersom du vil ha med duplikat-linja som i arket ditt
        if candidate in fixed_set:
            continue

        features = fixed_features + [candidate]

        try:
            accuracy = evaluate_kNN_30s(file_path, features, k=k)
            results.append({"Feature 4": candidate, "Accuracy": accuracy})
            print(f"[triple] Done: {fixed_features} + {candidate} -> {accuracy:.12f}")
        except Exception as e:
            print(f"[triple] Failed: {fixed_features} + {candidate} -> {e}")

    df = pd.DataFrame(results)
    df = df.sort_values(by="Feature 4").reset_index(drop=True)
    return df


def make_excel_wide_table(file_path, k=5):
    triple_sets = [
        ["spectral_rolloff_mean", "mfcc_1_mean", "spectral_centroid_mean"],
        ["spectral_rolloff_mean", "spectral_centroid_mean", "tempo"],
        ["spectral_rolloff_mean", "mfcc_1_mean", "tempo"],
        ["spectral_centroid_mean", "mfcc_1_mean", "tempo"],
    ]

    all_tables = []

    for fixed_features in triple_sets:
        df = test_feature4_candidates(file_path, fixed_features, ALL_FEATURES, k=k)

        header_feature = (
            f"Feature 4 (other used fretures: "
            f"{fixed_features[0]}, {fixed_features[1]}, {fixed_features[2]})"
        )
        df.columns = [header_feature, "Accuracy"]
        all_tables.append(df)

    single_df = test_single_features(file_path, ALL_FEATURES, k=k)
    all_tables.append(single_df)

    max_len = max(len(df) for df in all_tables)

    padded_tables = []
    for df in all_tables:
        df = df.reindex(range(max_len))
        padded_tables.append(df)

    combined = pd.concat(padded_tables, axis=1)
    return combined


def get_top5_combinations(combined_df):
    combo_cols = [(i, i + 1) for i in range(0, 8, 2)]  # de 4 triple-tabellene

    rows = []

    for feature_col_idx, acc_col_idx in combo_cols:
        feature_col_name = combined_df.columns[feature_col_idx]
        acc_col_name = combined_df.columns[acc_col_idx]

        for _, row in combined_df[[feature_col_name, acc_col_name]].dropna().iterrows():
            feature_4 = row[feature_col_name]
            acc = row[acc_col_name]

            prefix = feature_col_name.replace("Feature 4 (other used fretures: ", "").replace(")", "")
            full_combo = f"{prefix}, {feature_4}"

            rows.append({
                "Combination": full_combo,
                "Accuracy": acc
            })

    top5 = pd.DataFrame(rows).sort_values(by="Accuracy", ascending=False).head(5).reset_index(drop=True)
    return top5


if __name__ == "__main__":
    FILE_PATH = "../data/GenreClassData_30s.txt"
    K = 5

    combined_df = make_excel_wide_table(FILE_PATH, k=K)

    print("\nFull Excel-style table:")
    print(combined_df.to_string(index=False))

    combined_df.to_csv("feature_tables_k5_30s.csv", index=False)
    print("\nSaved: feature_tables_k5_30s.csv")

    top5_df = get_top5_combinations(combined_df)

    print("\nTop 5 from the table:")
    print(top5_df.to_string(index=False))

    top5_df.to_csv("top5_feature_combinations_k5_30s.csv", index=False)
    print("\nSaved: top5_feature_combinations_k5_30s.csv")