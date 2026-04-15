import numpy as np
import pandas as pd

#Example of usage
#outputs_train[0] = normalize_standard(outputs_train[0])
#outputs_test[0] = normalize_standard(outputs_test[0])

def normalize_standard(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)

    # avoid division by zero
    std[std == 0] = 1

    return (X - mean) / std

def mahalanobis_distance(x, y, cov_matrix):
    diff = x - y
    cov_inv = np.linalg.inv(cov_matrix)
    d = diff.T @ cov_inv @ diff
    return d

def data_to_array_30s(file_path, feature_cols, label_col=None, include_track_id=False):

    df = pd.read_csv(file_path, sep="\t")

    # Features
    X = df[feature_cols].values
    # Optional outputs
    y = df[label_col].values if label_col else None
    track_id = df["Track ID"].values if include_track_id else None

    train_test_split = 792
    max_idx = len(X)

    # --- SPLIT ---
    X_train = X[:train_test_split]
    X_test = X[train_test_split:max_idx]

    outputs_train = [X_train]
    outputs_test = [X_test]

    if y is not None:
        y_train = y[:train_test_split]
        y_test = y[train_test_split:max_idx]
        outputs_train.append(y_train)
        outputs_test.append(y_test)

    if track_id is not None:
        id_train = track_id[:train_test_split]
        id_test = track_id[train_test_split:max_idx]
        outputs_train.append(id_train)
        outputs_test.append(id_test)

    return tuple(outputs_train), tuple(outputs_test)


def data_to_array_10s(file_path, feature_cols, label_col=None, include_track_id=False):
    df = pd.read_csv(file_path, sep="\t")

    # Features
    X = df[feature_cols].values
    # Optional outputs
    y = df[label_col].values if label_col else None
    track_id = df["Track ID"].values if include_track_id else None

    train_test_split = 792*3
    max_idx = len(X)

    # --- SPLIT ---
    X_train = X[:train_test_split]
    X_test = X[train_test_split:max_idx]

    outputs_train = [X_train]
    outputs_test = [X_test]

    if y is not None:
        y_train = y[:train_test_split]
        y_test = y[train_test_split:max_idx]
        outputs_train.append(y_train)
        outputs_test.append(y_test)

    if track_id is not None:
        id_train = track_id[:train_test_split]
        id_test = track_id[train_test_split:max_idx]
        outputs_train.append(id_train)
        outputs_test.append(id_test)

    return tuple(outputs_train), tuple(outputs_test)


def data_to_array_5s(file_path, feature_cols, label_col=None, include_track_id=False):

    df = pd.read_csv(file_path, sep="\t")

    # Features
    X = df[feature_cols].values
    # Optional outputs
    y = df[label_col].values if label_col else None
    track_id = df["Track ID"].values if include_track_id else None

    train_test_split = 792*6
    max_idx = len(X)

    # --- SPLIT ---
    X_train = X[:train_test_split]
    X_test = X[train_test_split:max_idx]

    outputs_train = [X_train]
    outputs_test = [X_test]

    if y is not None:
        y_train = y[:train_test_split]
        y_test = y[train_test_split:max_idx]
        outputs_train.append(y_train)
        outputs_test.append(y_test)

    if track_id is not None:
        id_train = track_id[:train_test_split]
        id_test = track_id[train_test_split:max_idx]
        outputs_train.append(id_train)
        outputs_test.append(id_test)

    return tuple(outputs_train), tuple(outputs_test)


features = [
    # zero crossing
#    "zero_cross_rate_mean",
#    "zero_cross_rate_std",

    # energy
#    "rmse_mean",
#    "rmse_var",

#    # spectral
    "spectral_centroid_mean",
#    "spectral_centroid_var",
#    "spectral_bandwidth_mean",
#    "spectral_bandwidth_var",
    "spectral_rolloff_mean",
#    "spectral_rolloff_var",
#    "spectral_contrast_mean",
#    "spectral_contrast_var",
#    "spectral_flatness_mean",
#    "spectral_flatness_var",

    # chroma mean
#    "chroma_stft_1_mean",
#    "chroma_stft_2_mean",
#    "chroma_stft_3_mean",
#    "chroma_stft_4_mean",
#    "chroma_stft_5_mean",
#    "chroma_stft_6_mean",
#    "chroma_stft_7_mean",
#    "chroma_stft_8_mean",
#    "chroma_stft_9_mean",
#    "chroma_stft_10_mean",
#    "chroma_stft_11_mean",
#    "chroma_stft_12_mean",

    # chroma std
#    "chroma_stft_1_std",
#    "chroma_stft_2_std",
#    "chroma_stft_3_std",
#    "chroma_stft_4_std",
#    "chroma_stft_5_std",
#    "chroma_stft_6_std",
#    "chroma_stft_7_std",
#    "chroma_stft_8_std",
#    "chroma_stft_9_std",
#    "chroma_stft_10_std",
#    "chroma_stft_11_std",
#    "chroma_stft_12_std",

    # rhythm
#    "tempo",

    # mfcc mean
   "mfcc_1_mean",
#   "mfcc_2_mean",
#    "mfcc_3_mean",
#    "mfcc_4_mean",
#    "mfcc_5_mean",
#    "mfcc_6_mean",
#    "mfcc_7_mean",
#    "mfcc_8_mean",
#    "mfcc_9_mean",
#    "mfcc_10_mean",
#    "mfcc_11_mean",
#    "mfcc_12_mean",

    # mfcc std
#    "mfcc_1_std",
#    "mfcc_2_std",
#    "mfcc_3_std",
    "mfcc_4_std",
#    "mfcc_5_std",
#    "mfcc_6_std",
#    "mfcc_7_std",
#    "mfcc_8_std",
#    "mfcc_9_std",
#    "mfcc_10_std",
#    "mfcc_11_std",
#    "mfcc_12_std",
]

train_30s, test_30s = data_to_array_30s(
    "../data/GenreClassData_30s.txt",
    features,
    label_col="GenreID",
    include_track_id=True)

# train and test on this form
# [[data_from_features], [GenreID], [Track ID]]

train_10s, test_10s = data_to_array_10s(
    "../data/GenreClassData_10s.txt",
    features, 
    label_col="GenreID",
    include_track_id=True
)

train_5s, test_5s = data_to_array_5s(
    "../data/GenreClassData_5s.txt",
    features, 
    label_col="GenreID",
    include_track_id=True
)

cov_matrix_30s = np.cov(train_30s[0], rowvar=False)
cov_matrix_10s = np.cov(train_10s[0], rowvar=False)
cov_matrix_5s = np.cov(train_5s[0], rowvar=False)
