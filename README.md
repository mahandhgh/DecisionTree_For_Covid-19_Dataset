# Decision Tree Classification — Modular Implementation

A modular implementation of the **Decision Tree classification homework** using the COVID-19 dataset.

The project is organized so that data preparation, the custom Decision Tree, evaluation, the scikit-learn implementation, and visualization are separated into independent Python modules. The original notebook workflow is preserved in `main_modular.ipynb`.

---

## 1. Project Overview

This project performs binary classification on the **COVID-19 dataset** (`Covid Data.csv`) using two Decision Tree implementations:

1. **Custom Decision Tree**
   - Implemented from scratch with NumPy.
   - Uses Gini impurity and Gini gain to select splits.
   - Includes a manually implemented train/test split.
   - Calculates the F1 score without using scikit-learn for the evaluation itself.

2. **Scikit-learn Decision Tree**
   - Uses `DecisionTreeClassifier` from scikit-learn.
   - Uses scikit-learn's `train_test_split`.
   - Produces a standard classification report.

The project also contains a visualization module for displaying the custom Decision Tree.

---

## 2. Project Structure

```text
HW01_DecisionTree/
│
├── Covid Data.csv
├── main.py
├── main_modular.ipynb
├── requirements.txt
├── README.md
│
└── src/
    ├── __init__.py
    ├── data_preprocessing.py
    ├── custom_decision_tree.py
    ├── evaluation.py
    ├── sklearn_model.py
    └── visualization.py
```

### File responsibilities

| File | Responsibility |
|---|---|
| `main.py` | Runs the complete project from start to finish |
| `main_modular.ipynb` | Notebook version of the same modular workflow |
| `data_preprocessing.py` | Loads and preprocesses the COVID-19 dataset |
| `custom_decision_tree.py` | Contains the custom Gini-based Decision Tree |
| `evaluation.py` | Contains the manual train/test split and F1 calculations |
| `sklearn_model.py` | Trains and evaluates the scikit-learn Decision Tree |
| `visualization.py` | Draws the custom Decision Tree |
| `requirements.txt` | Lists the required Python packages |
| `Covid Data.csv` | Input dataset |

---

## 3. Dataset

The project expects a file named:

```text
Covid Data.csv
```

The file must be located in the project root, next to `main.py`.

### Dataset preprocessing

The preprocessing module performs the following operations:

- Loads the CSV file with pandas.
- Removes:
  - `DATE_DIED`
  - `INTUBED`
  - `ICU`
- Treats `97` and `99` as missing values for the selected categorical health-related columns.
- Sets `PREGNANT = 2` for records where `SEX = 2`.
- Converts `CLASIFFICATION_FINAL` into a binary target:
  - values `1–3` → `1`
  - values greater than `3` → `0`
- Converts the selected categorical columns to numeric values.
- Removes rows with a missing target.
- Separates the processed data into:
  - `X`: input features
  - `y`: binary target

The preprocessing logic is isolated in:

```text
src/data_preprocessing.py
```

---

## 4. Custom Decision Tree

The custom implementation is located in:

```text
src/custom_decision_tree.py
```

It contains the following main components:

### Gini impurity

`gini_impurity_node()` calculates the impurity of a node based on the class distribution.

### Gini gain

`gini_gain()` compares the impurity before and after a split and measures how useful that split is.

### Split

`split()` divides the samples according to:

```text
feature value <= threshold
```

and

```text
feature value > threshold
```

### Node

The `Node` class stores the information required for each tree node, including:

- Gini impurity
- Number of samples
- Number of samples per class
- Predicted class
- Feature index
- Threshold
- Left child
- Right child
- Leaf status

### DecisionTree

The `DecisionTree` class:

- Builds the tree recursively.
- Searches through available features and thresholds.
- Selects the split with the highest Gini gain.
- Stops when the maximum depth is reached or the node contains a single class.
- Predicts classes by traversing the tree from the root to a leaf.

The current configuration uses:

```python
DecisionTree(max_depth=5)
```

---

## 5. Evaluation

Evaluation functions are located in:

```text
src/evaluation.py
```

### Manual train/test split

`train_test_split_manual()`:

- Creates an index array.
- Shuffles it using a fixed random seed.
- Uses 20% of the samples as the test set.
- Returns training and testing subsets.

The default parameters are:

```text
test_size = 0.2
random_seed = 42
```

### F1 score

`calculate_f1_score_per_class()` calculates precision, recall, and F1 score separately for each class.

`calculate_f1_macro()` calculates the macro-average F1 score across the classes.

This evaluation does not use `sklearn.metrics.f1_score`.

---

## 6. Scikit-learn Implementation

The scikit-learn workflow is located in:

```text
src/sklearn_model.py
```

It uses:

```python
DecisionTreeClassifier(max_depth=5, random_state=42)
```

The data is split with:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

The resulting predictions are evaluated using:

```python
classification_report(...)
```

This provides precision, recall, F1 score, and support for each class, along with the macro and weighted averages.

---

## 7. Visualization

The custom tree visualization is implemented in:

```text
src/visualization.py
```

`plot_custom_tree()` recursively displays the nodes of the custom Decision Tree.

For internal nodes, the visualization shows:

- Feature index
- Threshold
- Number of samples

For leaf nodes, it shows:

- Predicted class
- Number of samples

---

## 8. Running the Project

### Step 1 — Clone or copy the project

Make sure the project has the structure shown above.

### Step 2 — Add the dataset

Place:

```text
Covid Data.csv
```

in the project root:

```text
HW01_DecisionTree/
├── Covid Data.csv
├── main.py
└── src/
```

### Step 3 — Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — Run the Python program

```bash
python main.py
```

The program will:

1. Load and preprocess the dataset.
2. Train the custom Decision Tree.
3. Calculate the custom F1 score.
4. Train the scikit-learn Decision Tree.
5. Print the classification report.
6. Display the custom tree visualization.

---

## 9. Running the Notebook

Open:

```text
main_modular.ipynb
```

in Jupyter Notebook, JupyterLab, or VS Code.

Run the cells in order.

The notebook intentionally keeps the assignment's presentation flow:

```text
Imports
   ↓
Step 1 — Dataset preprocessing
   ↓
Step 2 — Custom Decision Tree and F1 score
   ↓
Step 3 — Scikit-learn Decision Tree
   ↓
Decision Tree visualization
```

The notebook imports the implementation from `src/` instead of defining the functions and classes directly inside notebook cells.

---

## 10. Reproducibility

The project uses fixed random seeds where randomness is involved.

The main values are:

```text
test_size = 0.2
random_seed = 42
max_depth = 5
random_state = 42
```

Using the same dataset and environment should therefore produce reproducible results for the corresponding workflows.

---

## 11. Custom vs. Scikit-learn Implementation

The two implementations are intentionally kept separate.

| Aspect | Custom implementation | Scikit-learn |
|---|---|---|
| Decision Tree | Implemented from scratch | `DecisionTreeClassifier` |
| Split criterion | Gini gain | Scikit-learn implementation |
| Train/test split | Manual | `train_test_split` |
| Random seed | `42` | `42` |
| Test size | `20%` | `20%` |
| Stratification | No | Yes |
| F1 calculation | Manual | `classification_report` |
| Maximum depth | `5` | `5` |

Because the train/test split procedures are not identical, the numerical results of the two models are not expected to be exactly the same.

---

## 12. Dependencies

The project requires:

- Python 3
- NumPy
- pandas
- Matplotlib
- scikit-learn

The exact package requirements are listed in:

```text
requirements.txt
```

---

## 13. Notes

- Keep `Covid Data.csv` in the project root unless the dataset path is changed in `main.py` or the notebook.
- Run the project from the project root so that imports such as `from src...` work correctly.
- `src/__init__.py` marks `src` as the project's Python package.
- The modular structure separates responsibilities without changing the intended assignment workflow.
