import itertools
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


# =========================
# CONFIG
# =========================

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

K_VALUES = [3, 5, 7, 9, 15, 21]
DISTANCE_METRIC = "mahalanobis"   # "euclidean" eller "mahalanobis"

# Hvor aggressivt søket skal snevres inn
TOP_SINGLE_FEATURES = 15
TOP_DATASET_COMBOS = 3
TOP_K_VALUES = 3
MAX_FINAL_FEATURES = 6

# Hvor mange topprader som skal lagres/vises
TOP_RESULTS_TO_SAVE = 100


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

    # Standardiser med train-statistikk
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return (X_train, y_train, ids_train), (X_test, y_test, ids_test)


# =========================
# DISTANCES / KNN
# =========================

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
# SEARCH HELPERS
# =========================

def all_dataset_combinations():
    names = list(DATA_FILES.keys())
    combos = []
    for r in range(1, len(names) + 1):
        combos.extend(itertools.combinations(names, r))
    return combos


def dataset_combo_name(dataset_combo: tuple[str, ...]) -> str:
    return "+".join(dataset_combo)


def eval_cached(cache, dataset_combo, features, k, metric):
    key = (dataset_combo, tuple(features), k, metric)
    if key in cache:
        return cache[key]

    train_set, test_set = load_dataset_combo(dataset_combo, list(features))
    out = evaluate_model(train_set, test_set, k=k, metric=metric)
    cache[key] = out
    return out


# =========================
# STAGE 1: SINGLE FEATURES
# =========================

def run_single_feature_search(cache):
    rows = []
    dataset_combos = all_dataset_combinations()

    total = len(ALL_FEATURES) * len(dataset_combos) * len(K_VALUES)
    run_no = 0

    print(f"Stage 1: single features | total runs = {total:,}")

    for feature in ALL_FEATURES:
        for dataset_combo in dataset_combos:
            for k in K_VALUES:
                run_no += 1
                out = eval_cached(cache, dataset_combo, [feature], k, DISTANCE_METRIC)

                rows.append({
                    "feature": feature,
                    "dataset_combo": dataset_combo_name(dataset_combo),
                    "k": k,
                    "accuracy": out["accuracy"],
                })

                if run_no % 100 == 0 or run_no == total:
                    print(f"  single: {run_no:,}/{total:,}")

    df = pd.DataFrame(rows).sort_values(by="accuracy", ascending=False).reset_index(drop=True)
    return df


def pick_top_candidates_from_single(df_single):
    top_features = (
        df_single.groupby("feature")["accuracy"]
        .max()
        .sort_values(ascending=False)
        .head(TOP_SINGLE_FEATURES)
        .index
        .tolist()
    )

    top_dataset_combos = (
        df_single.groupby("dataset_combo")["accuracy"]
        .max()
        .sort_values(ascending=False)
        .head(TOP_DATASET_COMBOS)
        .index
        .tolist()
    )

    top_k_values = (
        df_single.groupby("k")["accuracy"]
        .max()
        .sort_values(ascending=False)
        .head(TOP_K_VALUES)
        .index
        .tolist()
    )

    return top_features, top_dataset_combos, top_k_values


# =========================
# STAGE 2: FORWARD SELECTION
# =========================

def run_forward_selection(cache, candidate_features, dataset_combo_names, k_values, max_features):
    dataset_combo_lookup = {dataset_combo_name(c): c for c in all_dataset_combinations()}
    dataset_combos = [dataset_combo_lookup[name] for name in dataset_combo_names]

    all_runs = []
    final_models = []

    print("\nStage 2: forward selection")

    for dataset_combo in dataset_combos:
        for k in k_values:
            selected = []
            remaining = candidate_features.copy()
            best_acc_so_far = -1.0

            print(f"\n  dataset={dataset_combo_name(dataset_combo)}, k={k}")

            for step in range(1, max_features + 1):
                best_feature_this_step = None
                best_out_this_step = None
                best_acc_this_step = -1.0

                for candidate in remaining:
                    trial_features = selected + [candidate]
                    out = eval_cached(cache, dataset_combo, trial_features, k, DISTANCE_METRIC)
                    acc = out["accuracy"]

                    all_runs.append({
                        "dataset_combo": dataset_combo_name(dataset_combo),
                        "k": k,
                        "step": step,
                        "features": ", ".join(trial_features),
                        "accuracy": acc,
                    })

                    if acc > best_acc_this_step:
                        best_acc_this_step = acc
                        best_feature_this_step = candidate
                        best_out_this_step = out

                if best_feature_this_step is None:
                    break

                selected.append(best_feature_this_step)
                remaining.remove(best_feature_this_step)
                best_acc_so_far = best_acc_this_step

                print(f"    step {step}: added {best_feature_this_step} -> accuracy {best_acc_so_far:.6f}")

                final_models.append({
                    "dataset_combo": dataset_combo_name(dataset_combo),
                    "k": k,
                    "n_features": len(selected),
                    "features": ", ".join(selected),
                    "accuracy": best_acc_so_far,
                    "labels": best_out_this_step["labels"],
                    "confusion_matrix": best_out_this_step["confusion_matrix"],
                    "track_ids": best_out_this_step["track_ids"],
                    "predictions": best_out_this_step["predictions"],
                    "true_labels": best_out_this_step["true_labels"],
                })

    all_runs_df = pd.DataFrame(all_runs).sort_values(by="accuracy", ascending=False).reset_index(drop=True)
    final_models_df = pd.DataFrame([
        {
            "dataset_combo": x["dataset_combo"],
            "k": x["k"],
            "n_features": x["n_features"],
            "features": x["features"],
            "accuracy": x["accuracy"],
        }
        for x in final_models
    ]).sort_values(by=["accuracy", "n_features"], ascending=[False, True]).reset_index(drop=True)

    best_final = max(final_models, key=lambda x: x["accuracy"])
    return all_runs_df, final_models_df, best_final


# =========================
# SAVE / PRINT
# =========================

def save_outputs(df_single, df_forward_all, df_forward_models, best_model):
    df_single.to_csv("task4_single_feature_results.csv", index=False)
    df_forward_all.to_csv("task4_forward_selection_all_runs.csv", index=False)
    df_forward_models.to_csv("task4_forward_selection_models.csv", index=False)
    df_forward_models.head(TOP_RESULTS_TO_SAVE).to_csv("task4_forward_selection_top_models.csv", index=False)

    summary_df = pd.DataFrame([{
        "best_dataset_combo": best_model["dataset_combo"],
        "best_k": best_model["k"],
        "best_n_features": best_model["n_features"],
        "best_features": best_model["features"],
        "best_accuracy": best_model["accuracy"],
        "distance_metric": DISTANCE_METRIC,
    }])
    summary_df.to_csv("task4_best_model_summary.csv", index=False)

    cm_df = pd.DataFrame(
        best_model["confusion_matrix"],
        index=[f"true_{x}" for x in best_model["labels"]],
        columns=[f"pred_{x}" for x in best_model["labels"]],
    )
    cm_df.to_csv("task4_best_model_confusion_matrix.csv")

    pred_df = pd.DataFrame({
        "Track ID": best_model["track_ids"],
        "True GenreID": best_model["true_labels"],
        "Predicted GenreID": best_model["predictions"],
    })
    pred_df.to_csv("task4_best_model_predictions.csv", index=False)


def print_best_overview(df_forward_models, top_n=15):
    print("\nBest combinations:")
    print(df_forward_models.head(top_n).to_string(index=False))


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    print(f"Distance metric: {DISTANCE_METRIC}")
    print(f"K values: {K_VALUES}")
    print(f"Max final features: {MAX_FINAL_FEATURES}")
    print(f"Top single features kept: {TOP_SINGLE_FEATURES}")
    print(f"Top dataset combos kept: {TOP_DATASET_COMBOS}")
    print(f"Top k-values kept: {TOP_K_VALUES}")

    cache = {}

    df_single = run_single_feature_search(cache)

    top_features, top_dataset_combos, top_k_values = pick_top_candidates_from_single(df_single)

    print("\nSelected candidates from single-feature stage:")
    print("Top features:")
    print(top_features)
    print("Top dataset combos:")
    print(top_dataset_combos)
    print("Top k-values:")
    print(top_k_values)

    df_forward_all, df_forward_models, best_model = run_forward_selection(
        cache=cache,
        candidate_features=top_features,
        dataset_combo_names=top_dataset_combos,
        k_values=top_k_values,
        max_features=MAX_FINAL_FEATURES,
    )

    save_outputs(df_single, df_forward_all, df_forward_models, best_model)

    print_best_overview(df_forward_models, top_n=15)

    print("\nBest model:")
    print(f"Dataset combo: {best_model['dataset_combo']}")
    print(f"k: {best_model['k']}")
    print(f"n_features: {best_model['n_features']}")
    print(f"features: {best_model['features']}")
    print(f"accuracy: {best_model['accuracy']:.6f}")

    print("\nSaved files:")
    print("- task4_single_feature_results.csv")
    print("- task4_forward_selection_all_runs.csv")
    print("- task4_forward_selection_models.csv")
    print("- task4_forward_selection_top_models.csv")
    print("- task4_best_model_summary.csv")
    print("- task4_best_model_confusion_matrix.csv")
    print("- task4_best_model_predictions.csv")