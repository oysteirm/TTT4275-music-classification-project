from __future__ import annotations

from collections import Counter
from itertools import combinations, product
from pathlib import Path
import math
import random

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


# =========================================================
# 1. FILER OG HOVEDINNSTILLINGER
# =========================================================

DATA_DIR = Path("../data")

DATA_FILES = {
    "5s": DATA_DIR / "GenreClassData_5s.txt",
    "10s": DATA_DIR / "GenreClassData_10s.txt",
    "30s": DATA_DIR / "GenreClassData_30s.txt",
}

DISTANCE_METRIC = "mahalanobis"

# Her kan du sette hvilke k-verdier som er lov å teste per kilde
K_VALUES_PER_SOURCE = {
    "5s": [3, 5, 7, 9, 11],
    "10s": [3, 5, 7, 9, 11],
    "30s": [3, 5, 7, 9,11],
}

# Hvilke kilder som skal være med i eksperimentene
SOURCE_COMBINATIONS = [
    ["5s"],
    ["10s"],
    ["30s"],
    ["5s", "10s"],
    ["5s", "30s"],
    ["10s", "30s"],
    ["5s", "10s", "30s"],
]

# Voting-konfigurasjoner på track-nivå
WEIGHT_CONFIGS = [
    {"5s": 1, "10s": 1, "30s": 1},
    {"5s": 1, "10s": 1, "30s": 2},
    {"5s": 1, "10s": 2, "30s": 3},
    {"5s": 1, "10s": 3, "30s": 5},
]

# Hvor stor del av den opprinnelige train-delen som skal brukes som validation
VALIDATION_RATIO = 0.20
RANDOM_SEED = 42

# Hvor filer lagres
VALIDATION_RESULTS_CSV = "task4_validation_results.csv"
TEST_RESULTS_CSV = "task4_test_results.csv"
BEST_TEST_TRACK_PRED_CSV = "task4_best_test_track_predictions.csv"
BEST_TEST_SEGMENT_PRED_CSV = "task4_best_test_segment_predictions.csv"
BEST_TEST_CM_CSV = "task4_best_test_confusion_matrix.csv"


# =========================================================
# 2. FEATURE-KANDIDATER
# =========================================================

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

# Her velger du hvilket feature-område du vil søke i
FEATURE_POOL = [
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
    "chroma_stft_6_mean",
    "chroma_stft_4_std",
    "tempo",
    "mfcc_4_mean",
    "mfcc_6_std",

]

# Min og maks størrelse på feature-settene som skal genereres
MIN_FEATURES = 10
MAX_FEATURES = 14

# Maks antall feature-sett totalt
MAX_NUMBER_OF_FEATURE_SETS = 300


# =========================================================
# 3. HJELPEFUNKSJONER FOR FEATURE-SETT
# =========================================================

def count_total_combinations(n: int, min_size: int, max_size: int) -> int:
    total = 0
    for r in range(min_size, max_size + 1):
        total += math.comb(n, r)
    return total


def generate_feature_sets(
    feature_pool: list[str],
    min_size: int,
    max_size: int,
    max_feature_sets: int,
    random_seed: int = 42,
) -> list[tuple[str, list[str]]]:
    """
    Lager feature-sett fra et feature-område.
    Vi bruker kombinasjoner, ikke permutasjoner.
    Hvis det finnes for mange kombinasjoner, trekkes et tilfeldig utvalg.
    """
    rng = random.Random(random_seed)

    all_feature_sets = []
    n = len(feature_pool)
    total_possible = count_total_combinations(n, min_size, max_size)

    print("=" * 70)
    print("Genererer feature-sett")
    print(f"Antall features i pool: {n}")
    print(f"Min størrelse: {min_size}")
    print(f"Maks størrelse: {max_size}")
    print(f"Totalt antall mulige kombinasjoner: {total_possible}")

    if total_possible <= max_feature_sets:
        print("Alle kombinasjoner brukes.\n")
        counter = 0
        for r in range(min_size, max_size + 1):
            for comb in combinations(feature_pool, r):
                counter += 1
                name = f"auto_fs_{counter}"
                all_feature_sets.append((name, list(comb)))
    else:
        print(f"For mange kombinasjoner. Trekker tilfeldig ut {max_feature_sets} feature-sett.\n")

        all_candidates = []
        for r in range(min_size, max_size + 1):
            for comb in combinations(feature_pool, r):
                all_candidates.append(comb)

        sampled = rng.sample(all_candidates, max_feature_sets)

        for i, comb in enumerate(sampled, start=1):
            name = f"auto_fs_{i}"
            all_feature_sets.append((name, list(comb)))

    return all_feature_sets


# =========================================================
# 4. DATALESING OG SPLITT
# =========================================================

def split_index_for_dataset(dataset_name: str) -> int:
    if dataset_name == "30s":
        return 792
    elif dataset_name == "10s":
        return 792 * 3
    elif dataset_name == "5s":
        return 792 * 6
    else:
        raise ValueError(f"Ukjent datasett: {dataset_name}")


def read_dataset(dataset_name: str, feature_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(DATA_FILES[dataset_name], sep="\t")
    keep_cols = ["Track ID", "GenreID"] + feature_cols
    df = df[keep_cols].copy()
    df["source"] = dataset_name
    return df


def get_original_train_test_track_ids() -> tuple[list[int], list[int]]:
    """
    Bruker den opprinnelige 30s-filen for å definere train/test track IDs.
    """
    df_30 = pd.read_csv(DATA_FILES["30s"], sep="\t")
    split_idx = split_index_for_dataset("30s")

    original_train_ids = df_30.iloc[:split_idx]["Track ID"].tolist()
    original_test_ids = df_30.iloc[split_idx:]["Track ID"].tolist()

    overlap = set(original_train_ids).intersection(set(original_test_ids))
    if overlap:
        raise ValueError("Noen Track ID-er finnes i både opprinnelig train og test.")

    return original_train_ids, original_test_ids


def split_train_into_train_and_validation(
    original_train_ids: list[int],
    validation_ratio: float,
    random_seed: int,
) -> tuple[set[int], set[int]]:
    """
    Deler den opprinnelige train-delen i:
    - ny train
    - validation
    """
    rng = random.Random(random_seed)

    unique_ids = list(original_train_ids)
    rng.shuffle(unique_ids)

    n_val = int(len(unique_ids) * validation_ratio)

    validation_ids = set(unique_ids[:n_val])
    train_ids = set(unique_ids[n_val:])

    if len(train_ids.intersection(validation_ids)) > 0:
        raise ValueError("Overlap mellom train og validation.")

    return train_ids, validation_ids


def get_all_splits() -> tuple[set[int], set[int], set[int]]:
    original_train_ids, original_test_ids = get_original_train_test_track_ids()
    train_ids, validation_ids = split_train_into_train_and_validation(
        original_train_ids=original_train_ids,
        validation_ratio=VALIDATION_RATIO,
        random_seed=RANDOM_SEED,
    )
    test_ids = set(original_test_ids)

    if len(train_ids.intersection(test_ids)) > 0:
        raise ValueError("Overlap mellom train og test.")
    if len(validation_ids.intersection(test_ids)) > 0:
        raise ValueError("Overlap mellom validation og test.")

    return train_ids, validation_ids, test_ids


def load_data_for_split(
    sources: list[str],
    feature_cols: list[str],
    split_name: str,
) -> pd.DataFrame:
    """
    split_name kan være 'train', 'validation', 'test', eller 'train_plus_validation'
    """
    train_ids, validation_ids, test_ids = get_all_splits()

    if split_name == "train":
        selected_ids = train_ids
    elif split_name == "validation":
        selected_ids = validation_ids
    elif split_name == "test":
        selected_ids = test_ids
    elif split_name == "train_plus_validation":
        selected_ids = train_ids.union(validation_ids)
    else:
        raise ValueError(f"Ukjent split_name: {split_name}")

    dfs = []
    for source in sources:
        df = read_dataset(source, feature_cols)
        df = df[df["Track ID"].isin(selected_ids)].copy()
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


# =========================================================
# 5. STANDARDISERING OG AVSTAND
# =========================================================

def standardize_using_reference(
    reference_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Standardiserer target basert på statistikk fra reference_df.
    """
    X_ref = reference_df[feature_cols].to_numpy(dtype=float)
    X_target = target_df[feature_cols].to_numpy(dtype=float)

    mean = X_ref.mean(axis=0)
    std = X_ref.std(axis=0)
    std[std == 0] = 1.0

    X_ref_std = (X_ref - mean) / std
    X_target_std = (X_target - mean) / std

    return X_ref_std, X_target_std


def compute_inverse_covariance(X_train: np.ndarray) -> np.ndarray:
    cov = np.cov(X_train, rowvar=False)
    if np.ndim(cov) == 0:
        cov = np.array([[cov]])
    return np.linalg.pinv(cov)


def compute_distances_to_train(
    x_test: np.ndarray,
    X_train: np.ndarray,
    metric: str,
    cov_inv: np.ndarray | None,
) -> np.ndarray:
    if metric == "euclidean":
        return np.linalg.norm(X_train - x_test, axis=1)

    if metric == "mahalanobis":
        diffs = X_train - x_test
        return np.einsum("ij,jk,ik->i", diffs, cov_inv, diffs)

    raise ValueError(f"Ukjent metric: {metric}")


def predict_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    k: int,
    metric: str,
) -> np.ndarray:
    predictions = []

    cov_inv = None
    if metric == "mahalanobis":
        cov_inv = compute_inverse_covariance(X_train)

    for x_test in X_eval:
        distances = compute_distances_to_train(
            x_test=x_test,
            X_train=X_train,
            metric=metric,
            cov_inv=cov_inv,
        )

        nearest_idx = np.argsort(distances)[:k]
        nearest_labels = y_train[nearest_idx]
        predicted_label = Counter(nearest_labels).most_common(1)[0][0]
        predictions.append(predicted_label)

    return np.array(predictions)


# =========================================================
# 6. PREDIKSJON PER KILDE
# =========================================================

def predict_for_one_source(
    source: str,
    feature_cols: list[str],
    k: int,
    metric: str,
    train_split_name: str,
    eval_split_name: str,
) -> pd.DataFrame:
    """
    Trener og evaluerer for én kilde, f.eks. bare 5s eller bare 30s.
    """
    train_df = load_data_for_split(
        sources=[source],
        feature_cols=feature_cols,
        split_name=train_split_name,
    )

    eval_df = load_data_for_split(
        sources=[source],
        feature_cols=feature_cols,
        split_name=eval_split_name,
    )

    X_train, X_eval = standardize_using_reference(
        reference_df=train_df,
        target_df=eval_df,
        feature_cols=feature_cols,
    )

    y_train = train_df["GenreID"].to_numpy()
    preds = predict_knn(
        X_train=X_train,
        y_train=y_train,
        X_eval=X_eval,
        k=k,
        metric=metric,
    )

    out_df = eval_df[["Track ID", "GenreID", "source"]].copy()
    out_df.rename(columns={"GenreID": "True GenreID"}, inplace=True)
    out_df["Predicted GenreID"] = preds

    return out_df


def predict_for_source_combination(
    sources: list[str],
    feature_cols: list[str],
    k_config: dict[str, int],
    metric: str,
    train_split_name: str,
    eval_split_name: str,
) -> pd.DataFrame:
    """
    Lager segmentprediksjoner for alle valgte kilder.
    Hver kilde får sin egen k-verdi.
    """
    all_predictions = []

    for source in sources:
        source_pred_df = predict_for_one_source(
            source=source,
            feature_cols=feature_cols,
            k=k_config[source],
            metric=metric,
            train_split_name=train_split_name,
            eval_split_name=eval_split_name,
        )
        all_predictions.append(source_pred_df)

    return pd.concat(all_predictions, ignore_index=True)


# =========================================================
# 7. AGGREGERING FRA SEGMENT TIL TRACK
# =========================================================

def segment_level_accuracy(pred_df: pd.DataFrame) -> float:
    return float((pred_df["Predicted GenreID"] == pred_df["True GenreID"]).mean())


def majority_vote_all_segments(pred_df: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    rows = []

    for track_id, group in pred_df.groupby("Track ID"):
        predicted = Counter(group["Predicted GenreID"]).most_common(1)[0][0]
        true_label = group["True GenreID"].iloc[0]

        rows.append({
            "Track ID": track_id,
            "True GenreID": true_label,
            "Predicted GenreID": predicted,
            "Method": "majority_all_segments",
        })

    track_df = pd.DataFrame(rows)
    accuracy = float((track_df["Predicted GenreID"] == track_df["True GenreID"]).mean())

    return accuracy, track_df


def majority_vote_per_source_equal(pred_df: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    rows = []

    for track_id, track_group in pred_df.groupby("Track ID"):
        source_votes = []

        for source, source_group in track_group.groupby("source"):
            source_winner = Counter(source_group["Predicted GenreID"]).most_common(1)[0][0]
            source_votes.append(source_winner)

        final_prediction = Counter(source_votes).most_common(1)[0][0]
        true_label = track_group["True GenreID"].iloc[0]

        rows.append({
            "Track ID": track_id,
            "True GenreID": true_label,
            "Predicted GenreID": final_prediction,
            "Method": "group_vote_equal",
            "Source votes": str(source_votes),
        })

    track_df = pd.DataFrame(rows)
    accuracy = float((track_df["Predicted GenreID"] == track_df["True GenreID"]).mean())

    return accuracy, track_df


def majority_vote_per_source_weighted(
    pred_df: pd.DataFrame,
    weights: dict[str, int],
) -> tuple[float, pd.DataFrame]:
    rows = []

    for track_id, track_group in pred_df.groupby("Track ID"):
        weighted_votes = Counter()
        winners_per_source = {}

        for source, source_group in track_group.groupby("source"):
            source_winner = Counter(source_group["Predicted GenreID"]).most_common(1)[0][0]
            winners_per_source[source] = source_winner
            weighted_votes[source_winner] += weights.get(source, 1)

        final_prediction = weighted_votes.most_common(1)[0][0]
        true_label = track_group["True GenreID"].iloc[0]

        rows.append({
            "Track ID": track_id,
            "True GenreID": true_label,
            "Predicted GenreID": final_prediction,
            "Method": "group_vote_weighted",
            "Weights": str(weights),
            "Source winners": str(winners_per_source),
        })

    track_df = pd.DataFrame(rows)
    accuracy = float((track_df["Predicted GenreID"] == track_df["True GenreID"]).mean())

    return accuracy, track_df


# =========================================================
# 8. EVALUERING AV ETT OPPSETT PÅ VALIDATION
# =========================================================

def evaluate_one_validation_setup(
    feature_set_name: str,
    feature_cols: list[str],
    sources: list[str],
    k_config: dict[str, int],
    metric: str,
    weight_configs: list[dict[str, int]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[tuple[dict[str, int], float, pd.DataFrame]]]:
    """
    Trener på train og evaluerer på validation.
    """
    segment_pred_df = predict_for_source_combination(
        sources=sources,
        feature_cols=feature_cols,
        k_config=k_config,
        metric=metric,
        train_split_name="train",
        eval_split_name="validation",
    )

    results = []

    seg_acc = segment_level_accuracy(segment_pred_df)
    results.append({
        "Feature set": feature_set_name,
        "Sources": "+".join(sources),
        "k_config": str(k_config),
        "Metric": metric,
        "Method": "segment_level",
        "Weights": "",
        "Number of features": len(feature_cols),
        "Validation accuracy": seg_acc,
    })

    acc_all, track_df_all = majority_vote_all_segments(segment_pred_df)
    results.append({
        "Feature set": feature_set_name,
        "Sources": "+".join(sources),
        "k_config": str(k_config),
        "Metric": metric,
        "Method": "majority_all_segments",
        "Weights": "",
        "Number of features": len(feature_cols),
        "Validation accuracy": acc_all,
    })

    acc_equal, track_df_equal = majority_vote_per_source_equal(segment_pred_df)
    results.append({
        "Feature set": feature_set_name,
        "Sources": "+".join(sources),
        "k_config": str(k_config),
        "Metric": metric,
        "Method": "group_vote_equal",
        "Weights": "",
        "Number of features": len(feature_cols),
        "Validation accuracy": acc_equal,
    })

    weighted_outputs = []
    for weights in weight_configs:
        acc_weighted, track_df_weighted = majority_vote_per_source_weighted(segment_pred_df, weights)

        results.append({
            "Feature set": feature_set_name,
            "Sources": "+".join(sources),
            "k_config": str(k_config),
            "Metric": metric,
            "Method": "group_vote_weighted",
            "Weights": str(weights),
            "Number of features": len(feature_cols),
            "Validation accuracy": acc_weighted,
        })

        weighted_outputs.append((weights, acc_weighted, track_df_weighted))

    results_df = pd.DataFrame(results)

    return results_df, segment_pred_df, track_df_all, track_df_equal, weighted_outputs


# =========================================================
# 9. EVALUERING AV BESTE MODELL PÅ TEST
# =========================================================

def evaluate_best_setup_on_test(
    feature_cols: list[str],
    sources: list[str],
    k_config: dict[str, int],
    metric: str,
    method: str,
    weights_str: str,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Trener på train+validation og evaluerer på test.
    """
    segment_pred_df = predict_for_source_combination(
        sources=sources,
        feature_cols=feature_cols,
        k_config=k_config,
        metric=metric,
        train_split_name="train_plus_validation",
        eval_split_name="test",
    )

    if method == "segment_level":
        acc = segment_level_accuracy(segment_pred_df)
        track_df = None

    elif method == "majority_all_segments":
        acc, track_df = majority_vote_all_segments(segment_pred_df)

    elif method == "group_vote_equal":
        acc, track_df = majority_vote_per_source_equal(segment_pred_df)

    elif method == "group_vote_weighted":
        weights = eval(weights_str)
        acc, track_df = majority_vote_per_source_weighted(segment_pred_df, weights)

    else:
        raise ValueError(f"Ukjent method: {method}")

    return segment_pred_df, track_df, acc


def make_confusion_matrix_df(track_df: pd.DataFrame) -> pd.DataFrame:
    labels = sorted(track_df["True GenreID"].unique())

    cm = confusion_matrix(
        track_df["True GenreID"],
        track_df["Predicted GenreID"],
        labels=labels,
    )

    return pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )


# =========================================================
# 10. GENERERING AV k-KONFIGURASJONER
# =========================================================

def generate_k_configs_for_sources(
    sources: list[str],
    k_values_per_source: dict[str, list[int]],
) -> list[dict[str, int]]:
    """
    Lager alle kombinasjoner av k-verdier for de valgte kildene.
    Eksempel:
    sources = ["5s", "30s"]
    kan gi:
    {"5s":3, "30s":1}, {"5s":3, "30s":3}, ...
    """
    value_lists = [k_values_per_source[source] for source in sources]

    k_configs = []
    for values in product(*value_lists):
        config = {}
        for source, k in zip(sources, values):
            config[source] = k
        k_configs.append(config)

    return k_configs


# =========================================================
# 11. HOVEDPROGRAM
# =========================================================

if __name__ == "__main__":
    print("\nStarter oppgave 4-eksperimenter")
    print(f"Avstandsmetode: {DISTANCE_METRIC}")
    print(f"Validation-ratio: {VALIDATION_RATIO}")
    print(f"Random seed: {RANDOM_SEED}")
    print()

    train_ids, validation_ids, test_ids = get_all_splits()
    print("Datasplitt:")
    print(f"Antall train tracks      : {len(train_ids)}")
    print(f"Antall validation tracks : {len(validation_ids)}")
    print(f"Antall test tracks       : {len(test_ids)}")
    print()

    feature_sets = generate_feature_sets(
        feature_pool=FEATURE_POOL,
        min_size=MIN_FEATURES,
        max_size=MAX_FEATURES,
        max_feature_sets=MAX_NUMBER_OF_FEATURE_SETS,
        random_seed=RANDOM_SEED,
    )

    print(f"Antall feature-sett som faktisk testes: {len(feature_sets)}")
    print()

    all_validation_results = []
    saved_validation_runs = []

    experiment_counter = 0

    for feature_set_name, feature_cols in feature_sets:
        for sources in SOURCE_COMBINATIONS:
            k_configs = generate_k_configs_for_sources(
                sources=sources,
                k_values_per_source=K_VALUES_PER_SOURCE,
            )

            for k_config in k_configs:
                experiment_counter += 1

                print("=" * 80)
                print(f"Eksperiment {experiment_counter}")
                print(f"Feature set : {feature_set_name}")
                print(f"Antall features: {len(feature_cols)}")
                print(f"Sources      : {sources}")
                print(f"k per source : {k_config}")
                print(f"Metric       : {DISTANCE_METRIC}")
                print("Trener på TRAIN, evaluerer på VALIDATION")

                results_df, segment_pred_df, track_df_all, track_df_equal, weighted_outputs = evaluate_one_validation_setup(
                    feature_set_name=feature_set_name,
                    feature_cols=feature_cols,
                    sources=sources,
                    k_config=k_config,
                    metric=DISTANCE_METRIC,
                    weight_configs=WEIGHT_CONFIGS,
                )

                print("Validation-resultater:")
                print(results_df[["Method", "Weights", "Validation accuracy"]].to_string(index=False))
                print()

                all_validation_results.append(results_df)

                saved_validation_runs.append({
                    "feature_set_name": feature_set_name,
                    "feature_cols": feature_cols,
                    "sources": sources,
                    "k_config": k_config,
                    "segment_pred_df": segment_pred_df,
                    "track_df_all": track_df_all,
                    "track_df_equal": track_df_equal,
                    "weighted_outputs": weighted_outputs,
                })

    validation_results_df = pd.concat(all_validation_results, ignore_index=True)
    validation_results_df = validation_results_df.sort_values(
        by="Validation accuracy",
        ascending=False
    ).reset_index(drop=True)

    validation_results_df.to_csv(VALIDATION_RESULTS_CSV, index=False)

    print("\nBeste oppsett på validation:")
    print(validation_results_df.head(20).to_string(index=False))

    best_row = validation_results_df.iloc[0]

    best_feature_set_name = best_row["Feature set"]
    best_sources = best_row["Sources"].split("+")
    best_k_config = eval(best_row["k_config"])
    best_method = best_row["Method"]
    best_weights_str = best_row["Weights"]

    best_feature_cols = None
    for fs_name, fs_cols in feature_sets:
        if fs_name == best_feature_set_name:
            best_feature_cols = fs_cols
            break

    if best_feature_cols is None:
        raise RuntimeError("Fant ikke feature-settet for beste oppsett.")

    print("\nValgt beste modell basert på validation:")
    print(best_row.to_string())
    print("\nNå trenes modellen på TRAIN + VALIDATION, og evalueres på TEST.\n")

    best_test_segment_df, best_test_track_df, best_test_acc = evaluate_best_setup_on_test(
        feature_cols=best_feature_cols,
        sources=best_sources,
        k_config=best_k_config,
        metric=DISTANCE_METRIC,
        method=best_method,
        weights_str=best_weights_str,
    )

    test_summary = pd.DataFrame([{
        "Feature set": best_feature_set_name,
        "Sources": "+".join(best_sources),
        "k_config": str(best_k_config),
        "Metric": DISTANCE_METRIC,
        "Method": best_method,
        "Weights": best_weights_str,
        "Test accuracy": best_test_acc,
        "Number of features": len(best_feature_cols),
    }])

    test_summary.to_csv(TEST_RESULTS_CSV, index=False)
    best_test_segment_df.to_csv(BEST_TEST_SEGMENT_PRED_CSV, index=False)

    if best_test_track_df is not None:
        best_test_track_df.to_csv(BEST_TEST_TRACK_PRED_CSV, index=False)

        cm_df = make_confusion_matrix_df(best_test_track_df)
        cm_df.to_csv(BEST_TEST_CM_CSV)

    print("Sluttresultat på TEST:")
    print(test_summary.to_string(index=False))
    print()

    print("Lagrede filer:")
    print(f"- {VALIDATION_RESULTS_CSV}")
    print(f"- {TEST_RESULTS_CSV}")
    print(f"- {BEST_TEST_SEGMENT_PRED_CSV}")

    if best_test_track_df is not None:
        print(f"- {BEST_TEST_TRACK_PRED_CSV}")
        print(f"- {BEST_TEST_CM_CSV}")



"""
To avoid biasing the final performance estimate, the original training data was further split into a training subset and a validation subset based on Track ID. The validation subset was used for model selection, while the original test subset was kept untouched until the final evaluation.

We evaluated multiple k-NN classifiers using Mahalanobis distance. The model design variables were:
(1) the feature subset,
(2) the segment source combination (5s, 10s, 30s, or combinations),
(3) the number of neighbors k, where different values of k were allowed for different segment durations, and 
(4) the track-level voting scheme.

After selecting the best configuration based on validation accuracy, the classifier was retrained on the combined training and validation data, and finally evaluated on the test set.


Dette oppsettet gjør at du kan svare ryddig på:

Hvordan valgte dere k?
Ved å teste flere k-konfigurasjoner på validation-settet.
Hvordan valgte dere features?
Ved å generere og teste mange feature-kombinasjoner fra et definert feature-pool.
Hvordan unngikk dere å tilpasse dere test-settet?
Ved å spare test-settet helt til slutt.
Hvorfor ulik k for ulike datasett?
Fordi 5s-, 10s- og 30s-segmenter har ulik informasjonsmengde og kan egne seg best med forskjellige nabotall.
"""