import numpy as np


def gini_impurity_node(y):
    classes, counts = np.unique(y, return_counts=True)
    prob = counts / counts.sum()
    return 1 - np.sum(prob ** 2)


def gini_gain(y, y_left, y_right):
    parent_impurity = gini_impurity_node(y)
    n = len(y)

    left_impurity = gini_impurity_node(y_left)
    right_impurity = gini_impurity_node(y_right)

    n_left, n_right = len(y_left), len(y_right)

    weighted_impurity = (
        (n_left / n) * left_impurity
        + (n_right / n) * right_impurity
    )

    return parent_impurity - weighted_impurity


def split(X, y, feature_index, threshold):
    left = X[:, feature_index] <= threshold
    right = X[:, feature_index] > threshold
    return X[left], y[left], X[right], y[right]


class Node:
    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None
        self.leaf = True


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.tree = self.build_tree(X, y)

    def build_tree(self, X, y, depth=0):
        num_samples_per_class = [
            np.sum(y == i) for i in range(self.n_classes)
        ]

        predicted_class = np.argmax(num_samples_per_class)

        node = Node(
            gini=gini_impurity_node(y),
            num_samples=len(y),
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )

        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or len(set(y)) == 1
        ):
            return node

        best_gini_gain = 0
        best_index, best_threshold = None, None

        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])

            for threshold in thresholds:
                X_left, y_left, X_right, y_right = split(
                    X, y, feature_index, threshold
                )

                if len(y_left) > 0 and len(y_right) > 0:
                    gain = gini_gain(y, y_left, y_right)

                    if gain > best_gini_gain:
                        best_gini_gain = gain
                        best_index = feature_index
                        best_threshold = threshold

        if best_gini_gain > 0:
            node.feature_index = best_index
            node.threshold = best_threshold

            X_left, y_left, X_right, y_right = split(
                X, y, best_index, best_threshold
            )

            node.leaf = False
            node.left = self.build_tree(X_left, y_left, depth + 1)
            node.right = self.build_tree(X_right, y_right, depth + 1)

        return node

    def predict(self, X):
        return [
            self.predict_single_input(inputs)
            for inputs in X
        ]

    def predict_single_input(self, inputs):
        node = self.tree

        while not node.leaf:
            if inputs[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right

        return node.predicted_class
