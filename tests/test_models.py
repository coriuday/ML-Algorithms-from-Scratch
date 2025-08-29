import numpy as np
import pytest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import (
    train_test_split,
    LogisticRegressionScratch,
    DecisionTreeScratch,
    RandomForestScratch,
    DecisionTreeNode,
    cross_val_score,
    GridSearchCV,
    roc_curve,
    roc_auc_score,
    learning_curve
)

@pytest.fixture
def simple_classification_data():
    """A simple, linearly separable dataset."""
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3], [4, 5], [5, 5], [5, 6], [6, 6]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return X, y

# --- Test for train_test_split ---
def test_train_test_split_sizes(simple_classification_data):
    X, y = simple_classification_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    assert len(X_train) == 6
    assert len(X_test) == 2
    assert len(y_train) == 6
    assert len(y_test) == 2

def test_train_test_split_reproducibility(simple_classification_data):
    X, y = simple_classification_data
    X_train1, _, y_train1, _ = train_test_split(X, y, test_size=0.25, random_state=42)
    X_train2, _, y_train2, _ = train_test_split(X, y, test_size=0.25, random_state=42)
    assert np.array_equal(X_train1, X_train2)
    assert np.array_equal(y_train1, y_train2)

# --- Test for LogisticRegressionScratch ---
def test_logistic_regression_simple_fit(simple_classification_data):
    X, y = simple_classification_data
    model = LogisticRegressionScratch(lr=0.1, n_iter=1000)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.mean(preds == y) > 0.9, "Model should learn the simple dataset"

# --- Test for DecisionTreeScratch ---
def test_decision_tree_perfect_fit():
    """Test if the tree can perfectly fit a noise-free dataset."""
    X = np.array([[0], [1], [0], [1]])
    y = np.array([0, 1, 0, 1])
    model = DecisionTreeScratch()
    model.fit(X, y)
    preds = model.predict(X)
    assert np.array_equal(preds, y)

def test_decision_tree_max_depth():
    """Test if the max_depth parameter is respected."""
    X = np.array([[0], [1], [2], [3], [4], [5]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = DecisionTreeScratch(max_depth=1)
    model.fit(X, y)
    # A tree with depth 1 can only have one split (2 leaf nodes)
    assert model.root.left is not None
    assert model.root.right is not None
    assert model.root.left.left is None
    assert model.root.left.right is None

# --- Test for RandomForestScratch ---
def test_random_forest_simple_fit(simple_classification_data):
    X, y = simple_classification_data
    # Set a seed for numpy's random functions used in the model
    np.random.seed(42)
    model = RandomForestScratch(n_trees=5, max_depth=3)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.mean(preds == y) > 0.9, "Random Forest should learn the simple dataset"

# --- Tests for New Functionality ---
def test_cross_val_score(simple_classification_data):
    X, y = simple_classification_data
    model = LogisticRegressionScratch()
    scores = cross_val_score(model, X, y, cv=3)
    assert len(scores) == 3
    assert np.all(scores >= 0)
    assert np.all(scores <= 1)

def test_grid_search_cv(simple_classification_data):
    X, y = simple_classification_data
    model = DecisionTreeScratch()
    param_grid = {'max_depth': [1, 2], 'min_samples_split': [2, 3]}
    grid_search = GridSearchCV(model, param_grid, cv=2)
    grid_search.fit(X, y)
    assert grid_search.best_params_ is not None
    assert grid_search.best_score_ >= 0

def test_roc_curve_and_auc():
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    fpr, tpr = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    assert len(fpr) == len(tpr)
    assert auc == pytest.approx(0.75)

def test_learning_curve(simple_classification_data):
    X, y = simple_classification_data
    model = LogisticRegressionScratch()
    train_sizes, train_scores, test_scores = learning_curve(model, X, y)
    assert len(train_sizes) == 4
    assert train_scores.shape == (4,)
    assert test_scores.shape == (4,)
