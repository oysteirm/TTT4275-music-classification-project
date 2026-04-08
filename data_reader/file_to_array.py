import numpy as np
import pandas as pd

def data_to_array(file_path, feature_cols, label_col=None, include_track_id=False):
    """
    Reads dataset and returns numpy arrays.

    Parameters:
        file_path (str): path to .txt/.csv file
        feature_cols (list): list of column names to use as features
        label_col (str, optional): column name for labels

    Returns:
        X (np.ndarray): shape [n_samples, n_features]
        y (np.ndarray, optional): shape [n_samples]
    """
    # Read file (works for both .txt and .csv)
    df = pd.read_csv(file_path, sep="\t")


    # Extract features
    X = df[feature_cols].values

    outputs = [X]

    if label_col:
        y = df[label_col].values
        outputs.append(y)

    if include_track_id:
        track_id = df["Track ID"].values
        outputs.append(track_id)

    return outputs


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
    "tempo",

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
#    "mfcc_4_std",
#    "mfcc_5_std",
#    "mfcc_6_std",
#    "mfcc_7_std",
#    "mfcc_8_std",
#    "mfcc_9_std",
#    "mfcc_10_std",
#    "mfcc_11_std",
#    "mfcc_12_std",
]


ar = data_to_array("../data/GenreClassData_30s.txt", features)

print(ar)