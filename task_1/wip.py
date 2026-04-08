from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def evaluate_classifier(model, X_test, y_test):
    """
    Returns:
        error_rate : float
        cm         : confusion matrix
        labels     : class labels in sorted order
    """
    predictions = model.predict(X_test)

    labels = sorted(np.unique(y_test))

    error_rate = np.mean(np.array(y_test) != np.array(predictions))
    cm = confusion_matrix(y_test, predictions, labels=labels)

    return error_rate, cm, labels

def print_evaluation(model, X_test, y_test):
    error_rate, cm, labels = evaluate_classifier(model, X_test, y_test)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("Evaluation of k-NN classifier")
    print(f"Accuracy:   {accuracy:.4f}")
    print(f"Error rate: {error_rate:.4f}")
    print("\nLabels:")
    print(labels)

    print("\nConfusion matrix:")
    print(cm)

    print("\nClassification report:")
    print(classification_report(y_test, predictions))