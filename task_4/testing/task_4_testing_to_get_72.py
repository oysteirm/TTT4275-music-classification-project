from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# CONFIG - ENDRE HER
# =========================================================

DATA_DIR = Path("../data")

DATA_FILES = {
    "5s": DATA_DIR / "GenreClassData_5s.txt",
    "10s": DATA_DIR / "GenreClassData_10s.txt",
    "30s": DATA_DIR / "GenreClassData_30s.txt",
}

# Velg k selv
K = 15

# Velg features selv
FEATURES_TO_USE = [
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

# Filnavn for output
SEGMENT_OUTPUT_CSV = "simple_knn_segment_predictions.csv"
TRACK_OUTPUT_CSV = "simple_knn_track_predictions.csv"


# =========================================================
# DATA
# =========================================================

def split_index_for_dataset(dataset_name: str) -> int:
    if dataset_name == "30s":
        return 792
    if dataset_name == "10s":
        return 792 * 3
    if dataset_name == "5s":
        return 792 * 6
    raise ValueError(f"Unknown dataset: {dataset_name}")


def get_train_test_track_ids_from_30s() -> tuple[set, set]:
    """
    Bruker 30s-filen til å definere hvilke Track ID-er som er train og test.
    Samme split brukes så på 5s og 10s for å unngå leakage.
    """
    df_30 = pd.read_csv(DATA_FILES["30s"], sep="\t")
    split_idx = split_index_for_dataset("30s")

    train_ids = set(df_30.iloc[:split_idx]["Track ID"].tolist())
    test_ids = set(df_30.iloc[split_idx:]["Track ID"].tolist())

    overlap = train_ids.intersection(test_ids)
    if overlap:
        raise ValueError("Noen Track ID-er finnes både i train og test.")

    return train_ids, test_ids


def read_dataset(dataset_name: str, feature_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(DATA_FILES[dataset_name], sep="\t")
    keep_cols = ["Track ID", "GenreID"] + feature_cols
    df = df[keep_cols].copy()
    df["source"] = dataset_name
    return df


def load_all_datasets(feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_ids, test_ids = get_train_test_track_ids_from_30s()

    dfs = []
    for source in ["5s", "10s", "30s"]:
        dfs.append(read_dataset(source, feature_cols))

    all_df = pd.concat(dfs, ignore_index=True)

    train_df = all_df[all_df["Track ID"].isin(train_ids)].copy()
    test_df = all_df[all_df["Track ID"].isin(test_ids)].copy()

    if set(train_df["Track ID"]).intersection(set(test_df["Track ID"])):
        raise ValueError("Leakage: noen Track ID-er er både i train og test.")

    return train_df, test_df


# =========================================================
# STANDARDIZATION
# =========================================================

def standardize(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]):
    X_train = train_df[feature_cols].to_numpy(dtype=float)
    X_test = test_df[feature_cols].to_numpy(dtype=float)

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test


# =========================================================
# MAHALANOBIS KNN
# =========================================================

def mahalanobis_prepare(X_train: np.ndarray) -> np.ndarray:
    cov = np.cov(X_train, rowvar=False)
    if np.ndim(cov) == 0:
        cov = np.array([[cov]])
    return np.linalg.pinv(cov)


def mahalanobis_distances(test_vec: np.ndarray, X_train: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    diffs = X_train - test_vec
    if diffs.ndim == 1:
        diffs = diffs.reshape(-1, 1)

    # Kvadrert Mahalanobis-avstand. Helt fint for rangering.
    return np.einsum("ij,jk,ik->i", diffs, cov_inv, diffs)


def knn_predict_mahalanobis(X_train, y_train, X_test, k=5):
    predictions = []
    cov_inv = mahalanobis_prepare(X_train)

    for test_vec in X_test:
        dists = mahalanobis_distances(test_vec, X_train, cov_inv)
        nn_idx = np.argsort(dists)[:k]
        nn_labels = y_train[nn_idx]
        pred = Counter(nn_labels).most_common(1)[0][0]
        predictions.append(pred)

    return np.array(predictions)


# =========================================================
# EVALUATION
# =========================================================

def segment_level_accuracy(pred_df: pd.DataFrame) -> float:
    return float((pred_df["Predicted GenreID"] == pred_df["True GenreID"]).mean())


def majority_vote_all_segments(pred_df: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """
    Alle segmenter for samme Track ID stemmer likt.
    """
    rows = []

    for track_id, group in pred_df.groupby("Track ID"):
        voted_label = Counter(group["Predicted GenreID"]).most_common(1)[0][0]
        true_label = group["True GenreID"].iloc[0]

        rows.append({
            "Track ID": track_id,
            "True GenreID": true_label,
            "Predicted GenreID": voted_label,
            "n_segments": len(group),
        })

    track_df = pd.DataFrame(rows)
    acc = float((track_df["Predicted GenreID"] == track_df["True GenreID"]).mean())
    return acc, track_df


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    print("Running simple Mahalanobis k-NN...")
    print(f"k = {K}")
    print(f"Number of features = {len(FEATURES_TO_USE)}")
    print("Datasets = 5s + 10s + 30s")
    print("Distance metric = mahalanobis")

    train_df, test_df = load_all_datasets(FEATURES_TO_USE)

    X_train, X_test = standardize(train_df, test_df, FEATURES_TO_USE)
    y_train = train_df["GenreID"].to_numpy()

    preds = knn_predict_mahalanobis(X_train, y_train, X_test, k=K)

    segment_pred_df = test_df[["Track ID", "GenreID", "source"]].copy()
    segment_pred_df.rename(columns={"GenreID": "True GenreID"}, inplace=True)
    segment_pred_df["Predicted GenreID"] = preds

    seg_acc = segment_level_accuracy(segment_pred_df)
    track_acc, track_pred_df = majority_vote_all_segments(segment_pred_df)

    print(f"\nSegment-level accuracy: {seg_acc:.6f}")
    print(f"Track-level accuracy (majority_all_segments): {track_acc:.6f}")

    segment_pred_df.to_csv(SEGMENT_OUTPUT_CSV, index=False)
    track_pred_df.to_csv(TRACK_OUTPUT_CSV, index=False)

    print("\nSaved files:")
    print(f"- {SEGMENT_OUTPUT_CSV}")
    print(f"- {TRACK_OUTPUT_CSV}")