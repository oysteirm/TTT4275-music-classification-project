from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


# =========================================================
# CONFIG
# =========================================================

DATA_DIR = Path("../data")

DATA_FILES = {
    "5s": DATA_DIR / "GenreClassData_5s.txt",
    "10s": DATA_DIR / "GenreClassData_10s.txt",
    "30s": DATA_DIR / "GenreClassData_30s.txt",
}

# Velg hvilke k-verdier du vil teste
K_VALUES = [3, 5, 7, 9, 15, 21]

# Start gjerne med euclidean
DISTANCE_METRIC = "mahalanobis"   # "euclidean" eller "mahalanobis"

# Velg hvilke segmentkilder som skal brukes
SOURCES_TO_USE = ["5s", "10s", "30s"]

# Velg features du vil teste
# Du kan starte med alle, eller bytte til et håndplukket sett.
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

# Bytt denne hvis du vil teste et håndplukket feature-sett
FEATURES_TO_USE = ALL_FEATURES

# Forslag til vekter for group-vote.
# Disse tolkes som én stemme fra hver varighet etter intern voting.
WEIGHT_CONFIGS = [
    {"5s": 1, "10s": 1, "30s": 1},
    {"5s": 1, "10s": 2, "30s": 3},
    {"5s": 1, "10s": 1, "30s": 2},
    {"5s": 1, "10s": 3, "30s": 5},
]

# Filnavn for output
RESULTS_CSV = "task4_tracklevel_results.csv"
BEST_TRACK_PRED_CSV = "task4_best_track_predictions.csv"
BEST_SEGMENT_PRED_CSV = "task4_best_segment_predictions.csv"
BEST_TRACK_CM_CSV = "task4_best_track_confusion_matrix.csv"


# =========================================================
# HELPERS
# =========================================================

def split_index_for_dataset(dataset_name: str) -> int:
    if dataset_name == "30s":
        return 792
    if dataset_name == "10s":
        return 792 * 3
    if dataset_name == "5s":
        return 792 * 6
    raise ValueError(f"Unknown dataset: {dataset_name}")


def read_dataset(dataset_name: str, feature_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(DATA_FILES[dataset_name], sep="\t")
    keep_cols = ["Track ID", "GenreID"] + feature_cols
    df = df[keep_cols].copy()
    df["source"] = dataset_name
    return df


def get_train_test_track_ids_from_30s() -> tuple[set, set]:
    """
    Viktig:
    Vi bruker 30s-filen til å definere hvilke Track ID-er som er train og test.
    Deretter brukes samme Track ID-splitt på 5s og 10s.
    Dette gjør at samme sang aldri havner både i train og test.
    """
    df_30 = pd.read_csv(DATA_FILES["30s"], sep="\t")
    split_idx = split_index_for_dataset("30s")

    train_ids = set(df_30.iloc[:split_idx]["Track ID"].tolist())
    test_ids = set(df_30.iloc[split_idx:]["Track ID"].tolist())

    overlap = train_ids.intersection(test_ids)
    if overlap:
        raise ValueError("Track IDs overlap between train and test, which should not happen.")

    return train_ids, test_ids


def load_segment_data(
    sources: list[str],
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_ids, test_ids = get_train_test_track_ids_from_30s()

    dfs = []
    for source in sources:
        df = read_dataset(source, feature_cols)
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    train_df = all_df[all_df["Track ID"].isin(train_ids)].copy()
    test_df = all_df[all_df["Track ID"].isin(test_ids)].copy()

    # Sikkerhetssjekk
    if set(train_df["Track ID"]).intersection(set(test_df["Track ID"])):
        raise ValueError("Leakage detected: some Track IDs are in both train and test.")

    return train_df, test_df


def standardize(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]):
    X_train = train_df[feature_cols].to_numpy(dtype=float)
    X_test = test_df[feature_cols].to_numpy(dtype=float)

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test


def mahalanobis_prepare(X_train: np.ndarray) -> np.ndarray:
    cov = np.cov(X_train, rowvar=False)
    if np.ndim(cov) == 0:
        cov = np.array([[cov]])
    return np.linalg.pinv(cov)


def pairwise_distances(test_vec: np.ndarray, X_train: np.ndarray, metric="euclidean", cov_inv=None) -> np.ndarray:
    if metric == "euclidean":
        return np.linalg.norm(X_train - test_vec, axis=1)

    if metric == "mahalanobis":
        diffs = X_train - test_vec
        if diffs.ndim == 1:
            diffs = diffs.reshape(-1, 1)
        # Merk: dette gir kvadrert Mahalanobis-avstand, men det er helt fint for rangering
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


# =========================================================
# AGGREGATION METHODS
# =========================================================

def segment_level_accuracy(pred_df: pd.DataFrame) -> float:
    return float((pred_df["Predicted GenreID"] == pred_df["True GenreID"]).mean())


def majority_vote_all_segments(pred_df: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """
    Alle segmenter for samme Track ID får én stemme hver.
    Ulempen er at 5s vil dominere fordi de er flest.
    """
    rows = []

    for track_id, group in pred_df.groupby("Track ID"):
        pred = Counter(group["Predicted GenreID"]).most_common(1)[0][0]
        true_label = group["True GenreID"].iloc[0]

        rows.append({
            "Track ID": track_id,
            "True GenreID": true_label,
            "Predicted GenreID": pred,
            "method": "majority_all_segments",
            "n_segments": len(group),
        })

    out_df = pd.DataFrame(rows)
    acc = float((out_df["Predicted GenreID"] == out_df["True GenreID"]).mean())
    return acc, out_df


def majority_vote_per_source_then_equal(pred_df: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """
    Først finner vi én vinner innen hver varighet.
    Så stemmer 5s-vinner, 10s-vinner og 30s-vinner likt.
    Dermed får ikke 5s dominere bare fordi de er flere.
    """
    rows = []

    for track_id, track_group in pred_df.groupby("Track ID"):
        source_votes = []

        for source, source_group in track_group.groupby("source"):
            source_winner = Counter(source_group["Predicted GenreID"]).most_common(1)[0][0]
            source_votes.append(source_winner)

        final_pred = Counter(source_votes).most_common(1)[0][0]
        true_label = track_group["True GenreID"].iloc[0]

        rows.append({
            "Track ID": track_id,
            "True GenreID": true_label,
            "Predicted GenreID": final_pred,
            "method": "group_vote_equal",
            "source_votes": str(source_votes),
        })

    out_df = pd.DataFrame(rows)
    acc = float((out_df["Predicted GenreID"] == out_df["True GenreID"]).mean())
    return acc, out_df


def majority_vote_per_source_weighted(pred_df: pd.DataFrame, weights: dict[str, int]) -> tuple[float, pd.DataFrame]:
    """
    Først finner vi én vinner innen hver varighet.
    Deretter summerer vi vekter:
        5s-vinner får weight["5s"]
        10s-vinner får weight["10s"]
        30s-vinner får weight["30s"]
    """
    rows = []

    for track_id, track_group in pred_df.groupby("Track ID"):
        weighted_counter = Counter()
        source_winners = {}

        for source, source_group in track_group.groupby("source"):
            source_winner = Counter(source_group["Predicted GenreID"]).most_common(1)[0][0]
            source_winners[source] = source_winner
            weighted_counter[source_winner] += weights.get(source, 1)

        final_pred = weighted_counter.most_common(1)[0][0]
        true_label = track_group["True GenreID"].iloc[0]

        rows.append({
            "Track ID": track_id,
            "True GenreID": true_label,
            "Predicted GenreID": final_pred,
            "method": f"group_vote_weighted_{weights}",
            "source_winners": str(source_winners),
        })

    out_df = pd.DataFrame(rows)
    acc = float((out_df["Predicted GenreID"] == out_df["True GenreID"]).mean())
    return acc, out_df


# =========================================================
# EVALUATION
# =========================================================

def evaluate_one_setup(
    feature_cols: list[str],
    sources: list[str],
    k: int,
    metric: str,
    weight_configs: list[dict[str, int]],
):
    train_df, test_df = load_segment_data(sources=sources, feature_cols=feature_cols)

    X_train, X_test = standardize(train_df, test_df, feature_cols)
    y_train = train_df["GenreID"].to_numpy()
    y_test = test_df["GenreID"].to_numpy()

    preds = knn_predict(X_train, y_train, X_test, k=k, metric=metric)

    segment_pred_df = test_df[["Track ID", "GenreID", "source"]].copy()
    segment_pred_df.rename(columns={"GenreID": "True GenreID"}, inplace=True)
    segment_pred_df["Predicted GenreID"] = preds

    results = []

    # 1) Segmentnivå
    seg_acc = segment_level_accuracy(segment_pred_df)
    results.append({
        "method": "segment_level",
        "k": k,
        "metric": metric,
        "sources": "+".join(sources),
        "n_features": len(feature_cols),
        "features": ", ".join(feature_cols),
        "weights": "",
        "accuracy": seg_acc,
    })

    # 2) Alle segmenter stemmer likt
    acc_all, track_df_all = majority_vote_all_segments(segment_pred_df)
    results.append({
        "method": "majority_all_segments",
        "k": k,
        "metric": metric,
        "sources": "+".join(sources),
        "n_features": len(feature_cols),
        "features": ", ".join(feature_cols),
        "weights": "",
        "accuracy": acc_all,
    })

    # 3) Én stemme per varighet
    acc_equal, track_df_equal = majority_vote_per_source_then_equal(segment_pred_df)
    results.append({
        "method": "group_vote_equal",
        "k": k,
        "metric": metric,
        "sources": "+".join(sources),
        "n_features": len(feature_cols),
        "features": ", ".join(feature_cols),
        "weights": "",
        "accuracy": acc_equal,
    })

    # 4) Vektet per varighet
    weighted_outputs = []
    for weights in weight_configs:
        acc_w, track_df_w = majority_vote_per_source_weighted(segment_pred_df, weights)
        results.append({
            "method": "group_vote_weighted",
            "k": k,
            "metric": metric,
            "sources": "+".join(sources),
            "n_features": len(feature_cols),
            "features": ", ".join(feature_cols),
            "weights": str(weights),
            "accuracy": acc_w,
        })
        weighted_outputs.append((weights, acc_w, track_df_w))

    return pd.DataFrame(results), segment_pred_df, track_df_all, track_df_equal, weighted_outputs


def make_confusion_matrix_df(track_pred_df: pd.DataFrame) -> pd.DataFrame:
    labels = sorted(track_pred_df["True GenreID"].unique())
    cm = confusion_matrix(
        track_pred_df["True GenreID"],
        track_pred_df["Predicted GenreID"],
        labels=labels,
    )
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{x}" for x in labels],
        columns=[f"pred_{x}" for x in labels],
    )
    return cm_df


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    print("Running track-level voting experiment...")
    print(f"Sources: {SOURCES_TO_USE}")
    print(f"Metric: {DISTANCE_METRIC}")
    print(f"Number of features: {len(FEATURES_TO_USE)}")

    all_results = []
    saved_runs = []

    for k in K_VALUES:
        print(f"\nTesting k={k} ...")

        results_df, segment_pred_df, track_df_all, track_df_equal, weighted_outputs = evaluate_one_setup(
            feature_cols=FEATURES_TO_USE,
            sources=SOURCES_TO_USE,
            k=k,
            metric=DISTANCE_METRIC,
            weight_configs=WEIGHT_CONFIGS,
        )

        all_results.append(results_df)

        saved_runs.append({
            "k": k,
            "segment_pred_df": segment_pred_df,
            "track_df_all": track_df_all,
            "track_df_equal": track_df_equal,
            "weighted_outputs": weighted_outputs,
        })

        print(results_df[["method", "weights", "accuracy"]].to_string(index=False))

    final_results = pd.concat(all_results, ignore_index=True)
    final_results = final_results.sort_values(by="accuracy", ascending=False).reset_index(drop=True)

    print("\nBest results:")
    print(final_results.head(20).to_string(index=False))

    final_results.to_csv(RESULTS_CSV, index=False)

    # Finn beste oppsett og lagre tilhørende prediksjoner
    best_row = final_results.iloc[0]
    best_k = int(best_row["k"])
    best_method = best_row["method"]
    best_weights = best_row["weights"]

    chosen_run = None
    for run in saved_runs:
        if run["k"] == best_k:
            chosen_run = run
            break

    if chosen_run is None:
        raise RuntimeError("Could not find saved run for best k.")

    chosen_run["segment_pred_df"].to_csv(BEST_SEGMENT_PRED_CSV, index=False)

    if best_method == "majority_all_segments":
        best_track_df = chosen_run["track_df_all"]
    elif best_method == "group_vote_equal":
        best_track_df = chosen_run["track_df_equal"]
    elif best_method == "group_vote_weighted":
        best_track_df = None
        for weights, acc, track_df_w in chosen_run["weighted_outputs"]:
            if str(weights) == best_weights:
                best_track_df = track_df_w
                break
        if best_track_df is None:
            raise RuntimeError("Could not find weighted output for best weights.")
    else:
        # Hvis segment_level er best, lagrer vi bare segmentprediksjoner.
        # Men vi kan fortsatt også lagre en enkel majority_all_segments som track-output.
        best_track_df = chosen_run["track_df_all"]

    best_track_df.to_csv(BEST_TRACK_PRED_CSV, index=False)
    cm_df = make_confusion_matrix_df(best_track_df)
    cm_df.to_csv(BEST_TRACK_CM_CSV)

    print("\nSaved files:")
    print(f"- {RESULTS_CSV}")
    print(f"- {BEST_SEGMENT_PRED_CSV}")
    print(f"- {BEST_TRACK_PRED_CSV}")
    print(f"- {BEST_TRACK_CM_CSV}")

    print("\nBest setup:")
    print(best_row.to_string())