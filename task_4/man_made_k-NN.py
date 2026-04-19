from collections import Counter
from itertools import combinations, product
from pathlib import Path
import math

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

DATA_DIR = Path("../data")

DATA_FILES = {
    "5s": DATA_DIR / "GenreClassData_5s.txt",
    "10s": DATA_DIR / "GenreClassData_10s.txt",
    "30s": DATA_DIR / "GenreClassData_30s.txt",
}

DATASET_CACHE = {}

K_VALUES_PER_SOURCE = {
    "5s": [5, 10, 15],
    "10s": [5, 10, 15],
    "30s": [5, 10, 15],
}

SOURCE_COMBINATIONS = [
    ["5s", "10s", "30s"],
]

WEIGHT_CONFIGS = [
    {"5s": 1, "10s": 1, "30s": 1},
    {"5s": 1, "10s": 2, "30s": 3}
]

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
RANKED_FEATURES = [
    "spectral_centroid_mean", #1  : 0.292929  0.707071
    "mfcc_1_mean", #2             : 0.272727  0.727273
    "spectral_flatness_var", #3   : 0.257576  0.742424
    "rmse_mean", #4               : 0.252525  0.747475
    "spectral_bandwidth_mean", #5 : 0.247475  0.752525
    "spectral_rolloff_mean", #6   : 0.247475  0.752525
    "rmse_var", #7                : 0.232323  0.767677
    "zero_cross_rate_std", #8     : 0.222222  0.777778
    "spectral_flatness_mean", #9  : 0.207071  0.792929
    "mfcc_5_std", #10             : 0.207071  0.792929
    "chroma_stft_2_std", #11      : 0.207071  0.792929
    "spectral_centroid_var", #12  : 0.202020  0.797980
    "chroma_stft_5_std", #13      : 0.202020  0.797980
    "mfcc_5_mean", #14            : 0.202020  0.797980
    "chroma_stft_12_mean", #15    : 0.202020  0.797980
    "mfcc_2_mean", #16            : 0.202020  0.797980
    "spectral_contrast_mean", #17 : 0.196970  0.803030
    "mfcc_9_mean", #18            : 0.196970  0.803030
    "mfcc_10_mean", #19           : 0.196970  0.803030
    "spectral_contrast_var", #20  : 0.196970  0.803030
    "mfcc_4_mean", #21            : 0.196970  0.803030
    "chroma_stft_2_mean", #22     : 0.196970  0.803030
    "mfcc_8_std", #23             : 0.196970  0.803030
    "chroma_stft_9_std", #24      : 0.196970  0.803030
    "mfcc_1_std", #25             : 0.191919  0.808081
    "mfcc_11_std", #26            : 0.191919  0.808081
    "mfcc_12_mean", #27           : 0.186869  0.813131
    "mfcc_8_mean", #28            : 0.186869  0.813131
    "mfcc_12_std", #29            : 0.181818  0.818182
    "chroma_stft_1_std", #30      : 0.181818  0.818182
    "mfcc_2_std", #31             : 0.181818  0.818182
    "chroma_stft_4_mean", #32     : 0.176768  0.823232
    # "mfcc_7_std", #33             : 0.176768  0.823232
    # "mfcc_3_std", #34             : 0.176768  0.823232
    # "mfcc_6_mean", #35            : 0.166667  0.833333
    # "chroma_stft_12_std", #36     : 0.166667  0.833333
    # "zero_cross_rate_mean", #37   : 0.166667  0.833333
    # "chroma_stft_10_std", #38     : 0.166667  0.833333
    # "chroma_stft_9_mean", #39     : 0.166667  0.833333
    # "chroma_stft_11_std", #40     : 0.161616  0.838384
    # "chroma_stft_1_mean", #41     : 0.161616  0.838384
    # "chroma_stft_5_mean", #42     : 0.161616  0.838384
    # "chroma_stft_11_mean", #43    : 0.161616  0.838384
    # "spectral_bandwidth_var", #44 : 0.156566  0.843434
    # "spectral_rolloff_var", #45   : 0.156566  0.843434
    # "mfcc_4_std", #46             : 0.156566  0.843434
    # "chroma_stft_6_mean", #47     : 0.156566  0.843434
    # "chroma_stft_7_mean", #48     : 0.151515  0.848485
    # "mfcc_6_std", #49             : 0.151515  0.848485
    # "chroma_stft_8_mean", #50     : 0.151515  0.848485
    # "mfcc_3_mean", #51            : 0.151515  0.848485
    # "chroma_stft_4_std", #52      : 0.146465  0.853535
    # "mfcc_11_mean", #53           : 0.141414  0.858586
    # "mfcc_9_std", #54             : 0.141414  0.858586
    # "chroma_stft_6_std", #55      : 0.141414  0.858586
    # "chroma_stft_10_mean", #56    : 0.136364  0.863636
    # "mfcc_7_mean", #57            : 0.136364  0.863636
    # "mfcc_10_std", #58            : 0.136364  0.863636
    # "chroma_stft_8_std", #59      : 0.136364  0.863636
    # "chroma_stft_3_std", #60      : 0.126263  0.873737
    # "chroma_stft_3_mean", #61     : 0.121212  0.878788
    # "chroma_stft_7_std", #62      : 0.106061  0.893939
    # "tempo", #63                  : 0.085859  0.914141
]

#features to search in
FEATURE_POOL = RANKED_FEATURES

# feature set sizing 
MIN_FEATURES = 30
MAX_FEATURES = 32

# Max number of feature sets to test on
MAX_NUMBER_OF_FEATURE_SETS = 200

#Functions start

def count_total_combinations(n, min_size, max_size):
    total = 0
    for r in range(min_size, max_size + 1):
        total += math.comb(n, r)
    return total

#genrate feature sets
def generate_feature_sets(feature_pool, min_size, max_size, max_feature_sets):
    all_feature_sets = []
    n = len(feature_pool)
    total_possible = count_total_combinations(n, min_size, max_size)

    print("=" * 70)
    print("Genererer feature-sett")
    print(f"Antall features i pool: {n}")
    print(f"Min størrelse: {min_size}")
    print(f"Maks størrelse: {max_size}")
    print(f"Totalt antall mulige kombinasjoner: {total_possible}")

    counter = 0

    for r in range(min_size, max_size + 1):
        for comb in combinations(feature_pool, r):
            counter += 1
            name = f"auto_fs_{counter}"
            all_feature_sets.append((name, list(comb)))

            # Stop early if we hit the max limit
            if counter >= max_feature_sets:
                print(f"Reached max_feature_sets={max_feature_sets}, stopping early.\n")
                return all_feature_sets

    print("Alle kombinasjoner brukt.\n")
    return all_feature_sets

#Data reading
def split_index_for_dataset(dataset_name):
    if dataset_name == "30s":
        return 792
    elif dataset_name == "10s":
        return 792 * 3
    elif dataset_name == "5s":
        return 792 * 6
    else:
        return
    
def get_dataset(source):
    if source not in DATASET_CACHE:
        DATASET_CACHE[source] = pd.read_csv(DATA_FILES[source], sep="\t")
    return DATASET_CACHE[source]

def read_dataset(dataset_name, feature_cols):
    df = get_dataset(dataset_name)
    keep_cols = ["Track ID", "GenreID"] + feature_cols
    df = df[keep_cols].copy()
    df["source"] = dataset_name
    return df


def get_original_train_test_track_ids():
    df_30 = pd.read_csv(DATA_FILES["30s"], sep="\t")
    split_idx = split_index_for_dataset("30s")

    #.iloc used to split rows into train and test
    original_train_ids = df_30.iloc[:split_idx]["Track ID"].tolist()
    original_test_ids = df_30.iloc[split_idx:]["Track ID"].tolist()

    return original_train_ids, original_test_ids

def get_all_splits():
    original_train_ids, original_test_ids = get_original_train_test_track_ids()

    #every 4 song is validation, making the the total split
    #train:600, validation:200, test: 200
    validation_ids = set(original_train_ids[::4])
    train_ids = set(id for id in original_train_ids if id not in validation_ids)
    test_ids = set(original_test_ids)

    return train_ids, validation_ids, test_ids


def load_data_for_split(sources ,feature_cols, split_name):
    train_ids, validation_ids, test_ids = get_all_splits()

    if split_name == "train":
        selected_ids = train_ids
    elif split_name == "validation":
        selected_ids = validation_ids
    elif split_name == "test":
        selected_ids = test_ids
    elif split_name == "train_plus_validation":
        selected_ids = train_ids.union(validation_ids)

    dfs = []
    for source in sources:
        df = read_dataset(source, feature_cols)
        df = df[df["Track ID"].isin(selected_ids)].copy()
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

#standization 
def standardize(reference_df, target_df, feature_cols):
    X_ref = reference_df[feature_cols].to_numpy(dtype=float)
    X_target = target_df[feature_cols].to_numpy(dtype=float)

    mean = X_ref.mean(axis=0)
    std = X_ref.std(axis=0)
    std[std == 0] = 1.0

    X_ref_std = (X_ref - mean) / std
    X_target_std = (X_target - mean) / std

    return X_ref_std, X_target_std

def compute_inverse_covariance(X_train):
    cov = np.cov(X_train, rowvar=False)
    if np.ndim(cov) == 0:
        cov = np.array([[cov]])
    return np.linalg.pinv(cov)


def compute_Mahalanobis_distances_to_train(x_test, X_train, cov_inv):
    diffs = X_train - x_test
    return np.sum(diffs @ cov_inv * diffs, axis=1)


def predict_knn(X_train, y_train, X_eval, k):
    predictions = []
    cov_inv = compute_inverse_covariance(X_train)

    for x_test in X_eval:
        distances = compute_Mahalanobis_distances_to_train(x_test, X_train, cov_inv)

        nearest_idx = np.argsort(distances)[:k]
        nearest_labels = y_train[nearest_idx]
        predicted_label = Counter(nearest_labels).most_common(1)[0][0]
        predictions.append(predicted_label)

    return np.array(predictions)

#Predictions
def predict_for_one_source(source, feature_cols, k, train_split_name, eval_split_name):

    train_df = load_data_for_split([source], feature_cols, train_split_name)
    eval_df = load_data_for_split([source], feature_cols, eval_split_name)

    X_train, X_eval = standardize(train_df, eval_df, feature_cols)

    y_train = train_df["GenreID"].to_numpy()
    preds = predict_knn(X_train, y_train, X_eval, k)

    out_df = eval_df[["Track ID", "GenreID", "source"]].copy()
    out_df.rename(columns={"GenreID": "True GenreID"}, inplace=True)
    out_df["Predicted GenreID"] = preds

    return out_df


def predict_for_source_combination(sources, feature_cols, k_config, train_split_name, eval_split_name):
    all_predictions = []

    for source in sources:
        source_pred_df = predict_for_one_source(source, feature_cols, k_config[source], train_split_name, eval_split_name)
        all_predictions.append(source_pred_df)

    return pd.concat(all_predictions, ignore_index=True)

#voting
# every segment is weighted acording to source
def majority_vote_all_segments_weighted(pred_df, weights):
    rows = []

    for track_id, group in pred_df.groupby("Track ID"):
        weighted_votes = Counter()

        for _, row in group.iterrows():
            genre = row["Predicted GenreID"]
            source = row["source"]

            weight = weights[source]

            weighted_votes[genre] += weight

        final_prediction = weighted_votes.most_common(1)[0][0]
        true_label = group["True GenreID"].iloc[0]

        rows.append({
            "Track ID": track_id,
            "True GenreID": true_label,
            "Predicted GenreID": final_prediction,
            "Method": "weighted_segments",
            "Weights": str(weights),
        })

    track_df = pd.DataFrame(rows)
    accuracy = float((track_df["Predicted GenreID"] == track_df["True GenreID"]).mean())

    return accuracy, track_df


#every source is weighted eqaully
def majority_vote_per_source_equal(pred_df):
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


#evaluate one
def evaluate_one_validation_setup(feature_set_name, feature_cols, sources, k_config, weight_configs):

    segment_pred_df = predict_for_source_combination(sources, feature_cols, k_config, "train", "validation")

    results = []

    seg_acc = float((segment_pred_df["Predicted GenreID"] == segment_pred_df["True GenreID"]).mean())
    results.append({
        "Feature set": feature_set_name,
        "Sources": "+".join(sources),
        "k_config": str(k_config),
        "Method": "segment_level",
        "Weights": "",
        "Number of features": len(feature_cols),
        "Validation accuracy": seg_acc,
    })

    for weights in weight_configs:
        acc_weighted, _ = majority_vote_all_segments_weighted(segment_pred_df, weights)
        results.append({ 
        "Feature set": feature_set_name,
        "Sources": "+".join(sources),
        "k_config": str(k_config),
        "Method": "weighted_segments",
        "Weights": str(weights),
        "Number of features": len(feature_cols),
        "Validation accuracy": acc_weighted,
        })

    acc_equal, _ = majority_vote_per_source_equal(segment_pred_df)
    results.append({
        "Feature set": feature_set_name,
        "Sources": "+".join(sources),
        "k_config": str(k_config),
        "Method": "group_vote_equal",
        "Weights": "",
        "Number of features": len(feature_cols),
        "Validation accuracy": acc_equal,
    })

    results_df = pd.DataFrame(results)

    return results_df

#final test based on model chosen
def evaluate_best_setup_on_test(feature_cols, sources, k_config, method, weights_str):

    segment_pred_df = predict_for_source_combination(sources, feature_cols, k_config, "train_plus_validation", "test")

    if method == "segment_level":
        acc = float((segment_pred_df["Predicted GenreID"] == segment_pred_df["True GenreID"]).mean())
        track_df = None

    elif method == "weighted_segments":
        weights = eval(weights_str)
        acc, track_df = majority_vote_all_segments_weighted(segment_pred_df, weights)

    elif method == "group_vote_equal":
        acc, track_df = majority_vote_per_source_equal(segment_pred_df)

    return segment_pred_df, track_df, acc


def make_confusion_matrix_df(track_df):
    labels = sorted(track_df["True GenreID"].unique())

    cm = confusion_matrix(track_df["True GenreID"], track_df["Predicted GenreID"], labels=labels)

    return pd.DataFrame(cm, [f"true_{label}" for label in labels], [f"pred_{label}" for label in labels])

#make k konfigs
def generate_k_configs_for_sources(sources, k_values_per_source):

    value_lists = [k_values_per_source[source] for source in sources]

    k_configs = []
    for values in product(*value_lists):
        config = {}
        for source, k in zip(sources, values):
            config[source] = k
        k_configs.append(config)

    return k_configs

#main
train_ids, validation_ids, test_ids = get_all_splits()
print("Data information:")
print(f"Number of train tracks      : {len(train_ids)}")
print(f"Number of validation tracks : {len(validation_ids)}")
print(f"Number of test tracks       : {len(test_ids)}")
print()

feature_sets = generate_feature_sets(FEATURE_POOL, MIN_FEATURES, MAX_FEATURES, MAX_NUMBER_OF_FEATURE_SETS)

print(f"Number of features that is tested: {len(feature_sets)}")
print()

all_validation_results = []

experiment_counter = 0

for feature_set_name, feature_cols in feature_sets:
    for sources in SOURCE_COMBINATIONS:
        k_configs = generate_k_configs_for_sources(sources, K_VALUES_PER_SOURCE)

        for k_config in k_configs:
            experiment_counter += 1

            print("=" * 80)
            print(f"Experiment {experiment_counter}")
            print(f"Feature set : {feature_set_name}")
            print(f"Number of features: {len(feature_cols)}")
            print(f"Sources      : {sources}")
            print(f"k per source : {k_config}")
            print("TRAIN and VALIDATION")

            results_df = evaluate_one_validation_setup(feature_set_name, feature_cols, sources, k_config, WEIGHT_CONFIGS)

            print("Validation results:")
            print(results_df[["Method", "Weights", "Validation accuracy"]].to_string(index=False))
            print()

            all_validation_results.append(results_df)


validation_results_df = pd.concat(all_validation_results, ignore_index=True)
validation_results_df = validation_results_df.sort_values(by="Validation accuracy", ascending=False).reset_index(drop=True)


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


print("\nChoose best model based on validation:")
print(best_row.to_string())
print("\nTraining on TRAIN + VALIDATION, and evaluating on TEST.\n")

best_test_segment_df, best_test_track_df, best_test_acc = evaluate_best_setup_on_test(best_feature_cols, best_sources, best_k_config, best_method, best_weights_str)

cm_df = make_confusion_matrix_df(best_test_track_df)

print("End result TEST:")
print(f"Feature set: {best_feature_cols}\n"
f"Sources: {'+'.join(best_sources)}\n"
f"k_config: {best_k_config}\n"
f"Method: {best_method}\n"
f"Weights: {best_weights_str}\n"
f"Test accuracy: {best_test_acc}\n"
f"Number of features: {len(best_feature_cols)}")