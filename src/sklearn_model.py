from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def train_sklearn_tree(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    classifier = DecisionTreeClassifier(
        max_depth=5,
        random_state=42,
    )

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    return classifier, X_train, X_test, y_train, y_test, y_pred
