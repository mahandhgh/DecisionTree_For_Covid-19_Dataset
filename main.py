from sklearn.metrics import classification_report
from src.data_preprocessing import load_and_prepare_data
from src.evaluation import train_test_split_manual, calculate_f1_macro
from src.custom_decision_tree import DecisionTree
from src.sklearn_model import train_sklearn_tree
from src.visualization import plot_custom_tree
import matplotlib.pyplot as plt


def main():
    # Step 1: Changes on the dataset
    covid_data_cleaned, X, y = load_and_prepare_data(
        "Covid Data.csv"
    )

    # Step 2: Custom Decision Tree + F1 without sklearn
    X_train, X_test, y_train, y_test = train_test_split_manual(
        X,
        y,
        test_size=0.2,
    )

    tree = DecisionTree(max_depth=5)
    tree.fit(X_train, y_train)

    y_pred_custom = tree.predict(X_test)

    f1_custom = calculate_f1_macro(
        y_test,
        y_pred_custom,
    )

    print(
        "F1 Score (Custom Decision Tree):",
        round(f1_custom, 3),
    )

    # Step 3: sklearn Decision Tree + classification report
    (
        classfy,
        X_train,
        X_test,
        y_train,
        y_test,
        y_pred,
    ) = train_sklearn_tree(X, y)

    print(classification_report(
        y_test,
        y_pred,
        digits=3,
    ))

    # Plot custom Decision Tree
    plot_custom_tree(tree.tree)
    plt.show()


if __name__ == "__main__":
    main()
