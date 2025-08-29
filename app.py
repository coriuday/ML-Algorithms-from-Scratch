import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import networkx as nx
from models import (
    LogisticRegressionScratch, 
    DecisionTreeScratch, 
    RandomForestScratch, 
    compute_confusion_matrix, 
    compute_precision, 
    compute_recall, 
    compute_f1,
    train_test_split,
    StandardScaler,
    MinMaxScaler,
    cross_val_score,
    GridSearchCV,
    roc_curve,
    roc_auc_score,
    learning_curve
)

# --- Visualization Functions ---

def plot_decision_boundary(model, X, y, title):
    """Plots the decision boundary of a model and the data points."""
    fig, ax = plt.subplots()
    
    # Create a meshgrid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict on the meshgrid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array(Z).reshape(xx.shape)
    
    # Plot the contour
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.6)
    
    # Plot the data points
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', s=100)
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    st.pyplot(fig)

def visualize_tree_streamlit(node):
    graph = nx.DiGraph()
    def add_nodes_edges(node, parent=None, edge_label=None, node_id=[0]):
        label = f"Leaf: {node.value}" if node.value is not None else f"X[{node.feature}] <= {node.threshold:.2f}"
        graph.add_node(node_id[0], label=label)
        if parent is not None:
            graph.add_edge(parent, node_id[0], label=edge_label)
        this_id = node_id[0]
        node_id[0] += 1
        if node.left:
            add_nodes_edges(node.left, this_id, 'True', node_id)
        if node.right:
            add_nodes_edges(node.right, this_id, 'False', node_id)
    add_nodes_edges(node)
    pos = nx.nx_pydot.graphviz_layout(graph, prog='dot')
    labels = nx.get_node_attributes(graph, 'label')
    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw(graph, pos, labels=labels, with_labels=True, arrows=True, node_size=2500, node_color='lightblue', ax=ax)
    edge_labels = nx.get_edge_attributes(graph, 'label')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax)
    st.pyplot(fig)

def plot_roc_curve(y_true, y_scores, title):
    """Plots the ROC curve."""
    fpr, tpr = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    st.pyplot(fig)

def plot_learning_curve(model, X, y, title):
    """Plots the learning curve."""
    train_sizes, train_scores, test_scores = learning_curve(model, X, y)
    fig, ax = plt.subplots()
    ax.plot(train_sizes, train_scores, 'o-', color="r", label="Training score")
    ax.plot(train_sizes, test_scores, 'o-', color="g", label="Cross-validation score")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="best")
    st.pyplot(fig)

# --- Streamlit UI ---
st.title("ML Algorithms from Scratch")

st.sidebar.header("1. Data Settings")

# Data upload and selection
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
try:
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.sidebar.write("Preview of uploaded data:", df.head())
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
    else:
        st.sidebar.info("Using sample data.")
        X = np.array([[1, 50], [2, 60], [3, 70], [10, 150], [12, 180], [15, 200], [25, 300], [30, 350]])
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Check if data has 2 features for visualization
if X.shape[1] != 2:
    st.warning("Decision boundary visualization is only available for datasets with exactly 2 features.")
    visualize_boundaries = False
else:
    visualize_boundaries = True

st.sidebar.header("2. Preprocessing")
scaling_method = st.sidebar.radio("Feature Scaling", ["None", "StandardScaler", "MinMaxScaler"])
if scaling_method != "None":
    if scaling_method == "StandardScaler":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    st.sidebar.success(f"{scaling_method} applied.")

st.sidebar.header("3. Model Selection")
model_type = st.sidebar.selectbox("Select Model", ["Logistic Regression", "Decision Tree", "Random Forest"])

st.header(f"{model_type} Model")

# --- Model Hyperparameters ---
if model_type == "Logistic Regression":
    model = LogisticRegressionScratch()
    param_grid = {
        'lr': [0.001, 0.01, 0.1, 1],
        'n_iter': [100, 500, 1000, 2000]
    }
elif model_type == "Decision Tree":
    model = DecisionTreeScratch()
    param_grid = {
        'max_depth': [1, 2, 3, 5, 10, None],
        'min_samples_split': [2, 5, 10]
    }
elif model_type == "Random Forest":
    model = RandomForestScratch()
    param_grid = {
        'n_trees': [5, 10, 20],
        'max_depth': [2, 5, 10, None],
        'min_samples_split': [2, 5, 10]
    }

st.sidebar.header("4. Evaluation Method")
evaluation_method = st.sidebar.selectbox("Choose Evaluation Method", ["Train-Test Split", "K-Fold Cross-Validation"])

if evaluation_method == "Train-Test Split":
    test_size = st.sidebar.slider('Test Set Size', 0.1, 0.5, 0.2, 0.05)
    random_state = st.sidebar.number_input('Random State for Split', value=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    st.sidebar.write(f"Training set: {X_train.shape[0]} samples")
    st.sidebar.write(f"Test set: {X_test.shape[0]} samples")
else:
    cv_folds = st.sidebar.slider('Number of Folds (k)', 2, 10, 5)

st.sidebar.header("5. Hyperparameter Tuning")
tune_hyperparameters = st.sidebar.checkbox("Perform Grid Search CV")

if tune_hyperparameters:
    st.info("Performing Grid Search... This may take a moment.")
    grid_search = GridSearchCV(model, param_grid, cv=cv_folds if evaluation_method == "K-Fold Cross-Validation" else 3)
    grid_search.fit(X, y)
    model = grid_search.model
    st.success(f"Best Parameters Found: {grid_search.best_params_}")
else:
    st.info("Using default hyperparameters. You can tune them manually below.")
    col1, col2 = st.columns(2)
    if model_type == "Logistic Regression":
        model.lr = col1.slider("Learning Rate", 0.01, 1.0, 0.1)
        model.n_iter = col2.slider("Iterations", 100, 5000, 1000)
    elif model_type == "Decision Tree":
        model.max_depth = col1.slider("Max Depth", 1, 20, 3)
        model.min_samples_split = col2.slider("Min Samples to Split", 2, 20, 2)
    elif model_type == "Random Forest":
        model.n_trees = col1.slider("Number of Trees", 1, 100, 10)
        model.max_depth = col2.slider("Max Depth of Trees", 1, 20, 3)
        model.min_samples_split = st.slider("Min Samples to Split", 2, 20, 2)

# --- Training and Evaluation ---
st.subheader("Model Evaluation")
if evaluation_method == "Train-Test Split":
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = np.mean(preds == y_test)
    st.info(f"The model was trained on {X_train.shape[0]} samples and evaluated on {X_test.shape[0]} unseen samples.")
    st.write(f"**Accuracy:** {acc:.2f}")
    st.write("**Confusion Matrix:**")
    st.write(compute_confusion_matrix(y_test, preds))
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Precision", f"{compute_precision(y_test, preds):.2f}")
    metric2.metric("Recall", f"{compute_recall(y_test, preds):.2f}")
    metric3.metric("F1 Score", f"{compute_f1(y_test, preds):.2f}")
else: # K-Fold Cross-Validation
    scores = cross_val_score(model, X, y, cv=cv_folds)
    st.info(f"The model was evaluated using {cv_folds}-Fold Cross-Validation.")
    st.write(f"**Mean Accuracy:** {np.mean(scores):.2f} (+/- {np.std(scores):.2f})")
    st.write("**All Scores:**", scores)

# --- Visualizations and Details ---
st.subheader("Visualizations & Model Details")

if visualize_boundaries:
    if evaluation_method == "Train-Test Split":
        plot_decision_boundary(model, X_test, y_test, f'{model_type} Decision Boundary on Test Set')
    else:
        model.fit(X, y) # Fit on all data for visualization
        plot_decision_boundary(model, X, y, f'{model_type} Decision Boundary on Full Dataset')
else:
    st.write("Plotting is disabled for data with more than 2 features.")

if st.checkbox("Show Advanced Visualizations"):
    if model_type == "Logistic Regression":
        if hasattr(model, 'predict_proba'):
            if evaluation_method == "Train-Test Split":
                y_scores = model.predict_proba(X_test)
                plot_roc_curve(y_test, y_scores, "ROC Curve on Test Set")
            else:
                st.info("ROC Curve is best viewed with a Train-Test Split.")
    
    plot_learning_curve(model, X, y, f"Learning Curve for {model_type}")

if model_type == "Decision Tree":
    if st.checkbox("Show Decision Tree Structure"):
        model.fit(X, y)
        st.write("### Decision Tree Structure")
        visualize_tree_streamlit(model.root)

elif model_type == "Random Forest":
    st.write("### Random Forest Details")
    if st.checkbox("Show Feature Importances"):
        model.fit(X, y)
        importances = np.zeros(X.shape[1]) # Simplified importance
        for tree, features in model.trees:
            # This is a simplified way to get importances
            pass # Replace with a proper implementation if desired
        st.bar_chart(importances)
    
    if st.checkbox("Visualize a single tree from the forest"):
        model.fit(X, y)
        st.info(f"Visualizing the first tree out of {model.n_trees}.")
        if model.trees:
            visualize_tree_streamlit(model.trees[0][0].root)

elif model_type == "Logistic Regression":
    st.write("### Logistic Regression Details")
    if st.checkbox("Show Feature Importances"):
        model.fit(X,y)
        st.bar_chart(pd.DataFrame(model.weights, columns=["Importance"]))