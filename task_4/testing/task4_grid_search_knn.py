import itertools
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


DATA_DIR = Path("../data")

DATA_FILES = {
    "5s": DATA_DIR / "GenreClassData_5s.txt",
    "10s": DATA_DIR / "GenreClassData_10s.txt",
    "30s": DATA_DIR / "GenreClassData_30s.txt",
}

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

K_VALUES = [5, 10, 20, 30, 40, 50, 60, 70, 80]

# Bruk nøyaktig 6 features
EXACT_FEATURES_IN_COMBO = 6

DISTANCE_METRIC = "mahalanobis"
SAVE_BEST_CONFUSION_MATRIX = True


# =========================
# DATA LOADING
# =========================

def split_index_for_dataset(dataset_name: str) -> int:
    if dataset_name == "30s":
        return 792
    if dataset_name == "10s":
        return 792 * 3
    if dataset_name == "5s":
        return 792 * 6
    raise ValueError(f"Unknown dataset: {dataset_name}")


def read_single_dataset(dataset_name: str, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(DATA_FILES[dataset_name], sep="\t")
    split_idx = split_index_for_dataset(dataset_name)

    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    keep_cols = feature_cols + ["GenreID", "Track ID"]
    return df_train[keep_cols], df_test[keep_cols]


def load_dataset_combo(dataset_combo: tuple[str, ...], feature_cols: list[str]):
    train_parts = []
    test_parts = []

    for dataset_name in dataset_combo:
        tr, te = read_single_dataset(dataset_name, feature_cols)
        train_parts.append(tr)
        test_parts.append(te)

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df["GenreID"].to_numpy()
    ids_train = train_df["Track ID"].to_numpy()

    X_test = test_df[feature_cols].to_numpy(dtype=float)
    y_test = test_df["GenreID"].to_numpy()
    ids_test = test_df["Track ID"].to_numpy()

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return (X_train, y_train, ids_train), (X_test, y_test, ids_test)


# =========================
# k-NN
# =========================

def mahalanobis_prepare(X_train: np.ndarray) -> np.ndarray:
    cov = np.cov(X_train, rowvar=False)

    # Hvis bare én feature eller numpy returnerer skalar
    if np.ndim(cov) == 0:
        cov = np.array([[cov]])

    return np.linalg.pinv(cov)


def pairwise_distances(
    test_vec: np.ndarray,
    X_train: np.ndarray,
    metric="euclidean",
    cov_inv=None
) -> np.ndarray:
    if metric == "euclidean":
        return np.linalg.norm(X_train - test_vec, axis=1)

    if metric == "mahalanobis":
        diffs = X_train - test_vec

        # Hvis én feature, sørg for 2D-form
        if diffs.ndim == 1:
            diffs = diffs.reshape(-1, 1)

        return np.einsum("ij,jk,ik->i", diffs, cov_inv, diffs)

    raise ValueError(f"Unknown metric: {metric}")


def knn_predict(X_train, y_train, X_test, k=5, metric="euclidean"):
    predictions = []

    cov_inv = None
    if metric == "mahalanobis":
        cov_inv = mahalanobis_prepare(X_train)

    for test_vec in X_test:
        dists = pairwise_distances(test_vec, X_train, metric=metric, cov_inv=cov_inv)
        nn_idx = np.argsort(dists)[:k]
        nn_labels = y_train[nn_idx]
        pred = Counter(nn_labels).most_common(1)[0][0]
        predictions.append(pred)

    return np.array(predictions)


def evaluate_model(train_set, test_set, k, metric):
    X_train, y_train, _ = train_set
    X_test, y_test, ids_test = test_set

    preds = knn_predict(X_train, y_train, X_test, k=k, metric=metric)
    accuracy = float(np.mean(preds == y_test))
    labels = sorted(np.unique(y_test))
    cm = confusion_matrix(y_test, preds, labels=labels)

    return {
        "accuracy": accuracy,
        "predictions": preds,
        "true_labels": y_test,
        "track_ids": ids_test,
        "labels": labels,
        "confusion_matrix": cm,
    }


# =========================
# SEARCH SPACE
# =========================

def all_dataset_combinations():
    names = list(DATA_FILES.keys())
    combos = []
    for r in range(1, len(names) + 1):
        combos.extend(itertools.combinations(names, r))
    return combos


def all_feature_combinations_exact(features: list[str], exact_size: int):
    for combo in itertools.combinations(features, exact_size):
        yield combo


def estimate_total_runs():
    n_dataset_combos = sum(math.comb(3, r) for r in range(1, 4))  # 7
    n_feature_combos = math.comb(len(ALL_FEATURES), EXACT_FEATURES_IN_COMBO)
    return n_dataset_combos * len(K_VALUES) * n_feature_combos


# =========================
# MAIN SEARCH
# =========================

def run_grid_search():
    dataset_combos = all_dataset_combinations()
    feature_combos = list(all_feature_combinations_exact(ALL_FEATURES, EXACT_FEATURES_IN_COMBO))

    total_runs = len(dataset_combos) * len(K_VALUES) * len(feature_combos)
    print(f"Total runs: {total_runs:,}")

    results = []
    best_result = None
    run_no = 0

    for dataset_combo in dataset_combos:
        dataset_name = "+".join(dataset_combo)

        data_cache = {}

        for feature_combo in feature_combos:
            feature_key = tuple(feature_combo)

            if feature_key not in data_cache:
                train_set, test_set = load_dataset_combo(dataset_combo, list(feature_combo))
                data_cache[feature_key] = (train_set, test_set)

            train_set, test_set = data_cache[feature_key]

            for k in K_VALUES:
                run_no += 1
                eval_out = evaluate_model(train_set, test_set, k=k, metric=DISTANCE_METRIC)

                row = {
                    "dataset_combo": dataset_name,
                    "k": k,
                    "n_features": len(feature_combo),
                    "features": ", ".join(feature_combo),
                    "accuracy": eval_out["accuracy"],
                }
                results.append(row)

                if best_result is None or row["accuracy"] > best_result["accuracy"]:
                    best_result = {
                        **row,
                        "labels": eval_out["labels"],
                        "confusion_matrix": eval_out["confusion_matrix"],
                        "track_ids": eval_out["track_ids"],
                        "predictions": eval_out["predictions"],
                        "true_labels": eval_out["true_labels"],
                    }

                if run_no % 100 == 0 or run_no == total_runs:
                    print(f"Run {run_no:,}/{total_runs:,} | best accuracy so far: {best_result['accuracy']:.6f}")

    results_df = pd.DataFrame(results).sort_values(
        by=["accuracy", "n_features"],
        ascending=[False, True]
    ).reset_index(drop=True)

    return results_df, best_result


def save_outputs(results_df: pd.DataFrame, best_result: dict):
    results_df.to_csv("task4_knn_gridsearch_results.csv", index=False)

    top100_df = results_df.head(100).copy()
    top100_df.to_csv("task4_knn_gridsearch_top100.csv", index=False)

    summary_rows = [{
        "best_dataset_combo": best_result["dataset_combo"],
        "best_k": best_result["k"],
        "best_n_features": best_result["n_features"],
        "best_features": best_result["features"],
        "best_accuracy": best_result["accuracy"],
        "distance_metric": DISTANCE_METRIC,
        "exact_features_in_combo": EXACT_FEATURES_IN_COMBO,
    }]
    pd.DataFrame(summary_rows).to_csv("task4_knn_best_summary.csv", index=False)

    if SAVE_BEST_CONFUSION_MATRIX:
        cm_df = pd.DataFrame(
            best_result["confusion_matrix"],
            index=[f"true_{x}" for x in best_result["labels"]],
            columns=[f"pred_{x}" for x in best_result["labels"]],
        )
        cm_df.to_csv("task4_knn_best_confusion_matrix.csv")

    pred_df = pd.DataFrame({
        "Track ID": best_result["track_ids"],
        "True GenreID": best_result["true_labels"],
        "Predicted GenreID": best_result["predictions"],
    })
    pred_df.to_csv("task4_knn_best_predictions.csv", index=False)


if __name__ == "__main__":
    print(f"Distance metric: {DISTANCE_METRIC}")
    print(f"K values: {K_VALUES}")
    print(f"Testing feature combinations of exact size: {EXACT_FEATURES_IN_COMBO}")

    results_df, best_result = run_grid_search()
    save_outputs(results_df, best_result)

    print("\nBest model:")
    print(f"Dataset combo: {best_result['dataset_combo']}")
    print(f"k: {best_result['k']}")
    print(f"n_features: {best_result['n_features']}")
    print(f"features: {best_result['features']}")
    print(f"accuracy: {best_result['accuracy']:.6f}")
    print("\nSaved files:")
    print("- task4_knn_gridsearch_results.csv")
    print("- task4_knn_gridsearch_top100.csv")
    print("- task4_knn_best_summary.csv")
    print("- task4_knn_best_confusion_matrix.csv")
    print("- task4_knn_best_predictions.csv")