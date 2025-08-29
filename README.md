# Implement Logistic Regression or Decision Tree without using scikit-learn. Show how the pruning works in the case of DT and Random forest algorithms

This project provides from-scratch implementations of fundamental machine learning algorithms, wrapped in a fully interactive web application powered by Streamlit. It is designed as an educational tool to explore how algorithms like Logistic Regression, Decision Trees, and Random Forests work internally, without relying on high-level libraries like `scikit-learn`.

---

## ✨ Features

- **From-Scratch Implementations**: `LogisticRegression`, `DecisionTree`, and `RandomForest` built using only NumPy.
- **Proper ML Workflow**: Includes essential data preprocessing and model evaluation steps.
  - **Train/Test Split**: Split your data to get a realistic measure of model performance on unseen data.
  - **Feature Scaling**: Apply `StandardScaler` or `MinMaxScaler` from scratch.
- **Interactive UI**: Manipulate model hyperparameters, train models, and see results in real-time.
- **Custom Data Upload**: Upload your own 2-feature datasets in CSV format to test the models.
- **Rich Visualizations**:
  - **Decision Boundary Plots**: See the exact regions the model learns for classification.
  - **Decision Tree Structure**: Visualize the entire tree to understand its rules.
  - **Random Forest Insights**: Visualize a single tree from the forest to see how pre-pruning works and view feature importances.
- **In-Depth Model Analysis**: View detailed performance metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
- **Unit Tested**: Core models are tested using `pytest` to ensure correctness.
- **Model Persistence**: Save trained models to a file and load them back later.

---

## 💻 Technology Stack

- **Core Logic**: Python, NumPy
- **Web Framework**: Streamlit
- **Testing**: Pytest
- **Data Handling**: Pandas
- **Plotting & Visualization**: Matplotlib, NetworkX

---

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

- Python 3.8+
- `pip` and `venv`

### Installation & Usage

1. **Clone the repository:**
   ```sh
   git clone https://github.com/your-username/IITP.git
   cd IITP
   ```

2. **Create and activate a virtual environment:**
   ```sh
   # Create the virtual environment
   python -m venv .venv

   # Activate it (Windows)
   .venv\Scripts\activate

   # Activate it (macOS/Linux)
   source .venv/bin/activate
   ```

3. **Install the required dependencies:**
   ```sh
   # Pytest is included for running tests
   pip install -r requirements.txt
   pip install pytest
   ```

4. **Run the tests (Optional but Recommended):**
   ```sh
   pytest
   ```

5. **Run the Streamlit application:**
   ```sh
   streamlit run app.py
   ```

   Your web browser should open with the application running.

---

## 📂 Project Structure

The repository is now organized as follows:

```
IITP/
├── app.py                  # Main Streamlit application script
├── models.py               # Core from-scratch model implementations
├── ml_from_scratch.ipynb   # Notebook for development and explanation
├── requirements.txt        # Python package dependencies
├── tests/                    # Unit tests for the models
│   └── test_models.py
├── README.md               # You are here!
└── LICENSE                 # MIT License file
```

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
