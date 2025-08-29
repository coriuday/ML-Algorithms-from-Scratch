import numpy as np
from collections import Counter
import copy

# --- Preprocessing ---
class StandardScaler:
    """Standardizes features by removing the mean and scaling to unit variance."""
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        # Handle zero variance columns
        self.scale_[self.scale_ == 0] = 1

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class MinMaxScaler:
    """Transforms features by scaling each feature to a given range, typically [0, 1]."""
    def fit(self, X):
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        self.range_ = self.max_ - self.min_
        # Handle zero range columns
        self.range_[self.range_ == 0] = 1

    def transform(self, X):
        return (X - self.min_) / self.range_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# --- Utility Functions ---
def train_test_split(X, y, test_size=0.2, random_state=None):
    """Splits data into training and testing sets."""
    if random_state:
        np.random.seed(random_state)
    n_samples = X.shape[0]
    shuffled_indices = np.random.permutation(n_samples)
    test_samples = int(n_samples * test_size)
    test_indices = shuffled_indices[:test_samples]
    train_indices = shuffled_indices[test_samples:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

# --- Logistic Regression ---
class LogisticRegressionScratch:
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        # Clip z to prevent overflow in np.exp
        z_clipped = np.clip(z, -709, 709) # Approximate limits for float64
        return 1 / (1 + np.exp(-z_clipped))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iter):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X):
        y_predicted = self.predict_proba(X)
        return [1 if i > 0.5 else 0 for i in y_predicted]
    
    def clone(self):
        return copy.deepcopy(self)

# --- Decision Tree ---
class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeScratch:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def entropy(self, y):
        counts = Counter(y)
        total = len(y)
        return -sum((count/total) * np.log2(count/total) for count in counts.values() if count)

    def best_split(self, X, y):
        best_gain = -1
        split_idx, split_thresh = None, None
        n_features = X.shape[1]
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = X[:, feature] > threshold
                if sum(left_idx) == 0 or sum(right_idx) == 0:
                    continue
                y_left, y_right = y[left_idx], y[right_idx]
                gain = self.information_gain(y, y_left, y_right)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature
                    split_thresh = threshold
        return split_idx, split_thresh

    def information_gain(self, y, y_left, y_right):
        p = len(y_left) / len(y)
        return self.entropy(y) - (p * self.entropy(y_left) + (1 - p) * self.entropy(y_right))

    def build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(set(y))
        if (self.max_depth is not None and depth >= self.max_depth) or num_labels == 1 or num_samples < self.min_samples_split:
            leaf_value = Counter(y).most_common(1)[0][0]
            return DecisionTreeNode(value=leaf_value)
        feature, threshold = self.best_split(X, y)
        if feature is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return DecisionTreeNode(value=leaf_value)
        left_idx = X[:, feature] <= threshold
        right_idx = X[:, feature] > threshold
        left = self.build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self.build_tree(X[right_idx], y[right_idx], depth + 1)
        return DecisionTreeNode(feature, threshold, left, right)

    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self.predict_one(x, node.left)
        else:
            return self.predict_one(x, node.right)

    def predict(self, X):
        return np.array([self.predict_one(x, self.root) for x in X])
    
    def clone(self):
        return copy.deepcopy(self)

# --- Random Forest ---
class RandomForestScratch:
    def __init__(self, n_trees=5, max_depth=None, min_samples_split=2, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def fit(self, X, y):
        self.trees = []
        n_features = X.shape[1]
        max_feats = self.max_features or int(np.sqrt(n_features))
        for _ in range(self.n_trees):
            X_sample, y_sample = self.bootstrap_sample(X, y)
            features = np.random.choice(n_features, max_feats, replace=False)
            tree = DecisionTreeScratch(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample[:, features], y_sample)
            self.trees.append((tree, features))

    def predict(self, X):
        tree_preds = np.array([tree.predict(X[:, features]) for tree, features in self.trees])
        return [Counter(tree_preds[:, i]).most_common(1)[0][0] for i in range(X.shape[0])]

    def clone(self):
        return copy.deepcopy(self)

# --- Model Evaluation ---
def cross_val_score(model, X, y, cv=5):
    """Performs k-fold cross-validation."""
    scores = []
    fold_size = len(X) // cv
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    for i in range(cv):
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
        
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        model_clone = model.clone()
        model_clone.fit(X_train, y_train)
        preds = model_clone.predict(X_test)
        scores.append(np.mean(preds == y_test))
        
    return np.array(scores)

class GridSearchCV:
    """Performs grid search to find the best hyperparameters."""
    def __init__(self, model, param_grid, cv=5):
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.best_params_ = None
        self.best_score_ = -1

    def fit(self, X, y):
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        
        from itertools import product
        for params in product(*values):
            param_dict = dict(zip(keys, params))
            
            # Set params for the model
            for key, value in param_dict.items():
                setattr(self.model, key, value)
            
            scores = cross_val_score(self.model, X, y, cv=self.cv)
            avg_score = np.mean(scores)
            
            if avg_score > self.best_score_:
                self.best_score_ = avg_score
                self.best_params_ = param_dict
        
        # Set the best params to the model
        for key, value in self.best_params_.items():
            setattr(self.model, key, value)
        
        self.model.fit(X, y)

# --- Metrics ---
def compute_confusion_matrix(y_true, y_pred):
    classes = np.unique(np.concatenate([y_true, y_pred]))
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for i, true in enumerate(classes):
        for j, pred in enumerate(classes):
            matrix[i, j] = np.sum((y_true == true) & (y_pred == pred))
    return matrix

def compute_precision(y_true, y_pred):
    y_pred = np.array(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def compute_recall(y_true, y_pred):
    y_pred = np.array(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def compute_f1(y_true, y_pred):
    precision = compute_precision(y_true, y_pred)
    recall = compute_recall(y_true, y_pred)
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

def roc_curve(y_true, y_scores):
    """Computes the Receiver Operating Characteristic curve."""
    thresholds = np.sort(np.unique(y_scores))[::-1]
    tpr = [0]
    fpr = [0]
    
    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
        
    return np.array(fpr), np.array(tpr)

def roc_auc_score(y_true, y_scores):
    """Computes the Area Under the ROC Curve."""
    fpr, tpr = roc_curve(y_true, y_scores)
    return np.trapezoid(tpr, fpr)

def learning_curve(model, X, y, train_sizes=np.linspace(0.1, 1.0, 5)):
    """Generates the data for a learning curve."""
    train_scores = []
    test_scores = []
    
    actual_train_sizes = []
    for size in train_sizes:
        n_samples = int(len(X) * size)
        if n_samples == 0:
            continue
        
        actual_train_sizes.append(n_samples)
        X_train, y_train = X[:n_samples], y[:n_samples]
        
        model_clone = model.clone()
        model_clone.fit(X_train, y_train)
        
        train_preds = model_clone.predict(X_train)
        train_scores.append(np.mean(train_preds == y_train))
        
        # This is a simplification; in a real scenario, you'd have a fixed test set.
        # For this educational tool, we'll use the whole dataset as a stand-in for a test set.
        test_preds = model_clone.predict(X)
        test_scores.append(np.mean(test_preds == y))
        
    return np.array(actual_train_sizes), np.array(train_scores), np.array(test_scores)