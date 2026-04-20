
#function used with the file_to_array.py and k-NN.py in task 1 to get the ranked features
def rank_single_features(feature_list, file_path, k=5):
    results = []

    for feature in feature_list:
        train_set, test_set = data_to_array_30s(
            file_path,
            [feature],
            label_col="GenreID",
            include_track_id=False
        )

        X_train = train_set[0]
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)

        cov_matrix = np.cov(X_train, rowvar=False)
        if np.ndim(cov_matrix) == 0:
            cov_matrix = np.array([[cov_matrix]])

        error_rate, _, _, _ = evaluating_k_NN_classifier(
            train_set, test_set, cov_matrix, k
        )

        accuracy = 1 - error_rate

        results.append({
            "Feature": feature,
            "Accuracy": accuracy,
            "Error rate": error_rate})

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

    return results_df