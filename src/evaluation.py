import numpy as np


def train_test_split_manual(X, y, test_size=0.2, random_seed=42):
    np.random.seed(random_seed)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    test_count = int(len(indices) * test_size)

    train_indices = indices[:-test_count]
    test_indices = indices[-test_count:]

    return (
        X[train_indices],
        X[test_indices],
        y[train_indices],
        y[test_indices],
    )


def calculate_f1_score_per_class(y_true, y_pred):
    unique_classes = np.unique(y_true)
    f1_scores = {}

    for class_label in unique_classes:
        tp = np.sum(
            (y_true == class_label) & (y_pred == class_label)
        )
        fp = np.sum(
            (y_true != class_label) & (y_pred == class_label)
        )
        fn = np.sum(
            (y_true == class_label) & (y_pred != class_label)
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0
        )

        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        f1_scores[class_label] = f1

    return f1_scores


def calculate_f1_macro(y_true, y_pred):
    f1_scores = calculate_f1_score_per_class(y_true, y_pred)
    return np.mean(list(f1_scores.values()))
