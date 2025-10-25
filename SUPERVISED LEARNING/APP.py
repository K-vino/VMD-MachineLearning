"""
Interactive Regression Learning Lab - single file Streamlit app
Features:
- Multiple regression models: Linear, Polynomial, Decision Tree, Random Forest,
  SVR, KNN, XGBoost/LightGBM/CatBoost (if installed), Neural Network (Keras)
- Upload CSV / use sample datasets / synthetic datasets
- Train, evaluate, visualize, and compare models
- Editable code snippet for each model (can be copied/modified)
- Save trained model
Run with: streamlit run app.py
"""

import streamlit as st
st.set_page_config(page_title="Regression Learning Lab", layout="wide", initial_sidebar_state="expanded")

# Standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
import pickle
import textwrap
import sys
import warnings
warnings.filterwarnings("ignore")

# Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# Optional boosters / DL
try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    import catboost as cb
except Exception:
    cb = None

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception:
    tf = None

# Plotly for interactive charts
import plotly.express as px
import plotly.graph_objects as go

# -------------------------
# Utilities
# -------------------------
def metrics_dict(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {"R2": r2, "MAE": mae, "MSE": mse, "RMSE": rmse}

def download_link(obj, filename, text):
    """Create a download link for a Python object (pickle)"""
    b = pickle.dumps(obj)
    b64 = base64.b64encode(b).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{text}</a>'
    return href

def quick_description():
    return """
    This app allows you to:
    - Load your dataset (CSV) or choose a sample dataset
    - Pick a target column and features
    - Train multiple regression models with interactive hyperparameters
    - Visualize predictions, residuals, feature importances
    - Compare models using R², MAE, MSE, RMSE
    - View and edit code snippets for each model
    """

# -------------------------
# Sample datasets
# -------------------------
def load_boston_like():
    # since sklearn's Boston is deprecated, create a synthetic regression sample
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=300, n_features=6, noise=20.0, random_state=42)
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
    df["target"] = y
    return df

def load_diabetes():
    from sklearn.datasets import load_diabetes
    d = load_diabetes()
    df = pd.DataFrame(d.data, columns=d.feature_names)
    df["target"] = d.target
    return df

def generate_synthetic(n=300):
    rng = np.random.RandomState(42)
    x1 = rng.normal(loc=0, scale=1, size=n)
    x2 = rng.normal(loc=3, scale=2, size=n)
    noise = rng.normal(scale=5, size=n)
    y = 3.5 * x1 - 1.2 * (x2**2) + 8 + noise
    df = pd.DataFrame({"x1": x1, "x2": x2, "target": y})
    return df

# -------------------------
# UI: Sidebar
# -------------------------
st.title("📚 Regression Learning Lab — Single File Streamlit App")
st.sidebar.header("Controls & Data")
with st.sidebar.expander("About this app", expanded=False):
    st.write(quick_description())
    st.write("Author: Generated for Vino K — Interactive single-file app")

# Data source options
data_option = st.sidebar.selectbox("Data source", ["Sample: Synthetic (toy)", "Sample: Diabetes", "Sample: Regression (make_regression)", "Upload CSV"])

if data_option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
else:
    uploaded_file = None

test_size = st.sidebar.slider("Test set fraction", 0.05, 0.5, 0.2, 0.05)
random_state = st.sidebar.number_input("Random seed", value=42, step=1)
scale_features = st.sidebar.checkbox("Standard scale features", value=True)
preview_rows = st.sidebar.slider("Preview rows (dataset)", 3, 50, 8)

# Model selection
st.sidebar.header("Model selection")
model_name = st.sidebar.selectbox("Choose model", [
    "Linear Regression",
    "Multiple Linear Regression",
    "Polynomial Regression",
    "Decision Tree",
    "Random Forest",
    "Support Vector Regression (SVR)",
    "K-Nearest Neighbors (KNN)",
    "Gradient Boosting (XGBoost / LightGBM / CatBoost)",
    "Neural Network (Keras)"
])

# common params
train_button = st.sidebar.button("Train model")
compare_button = st.sidebar.button("Train & Compare multiple models")

# -------------------------
# Load / Prepare Data
# -------------------------
@st.cache_data(show_spinner=False)
def load_data(option):
    if option == "Sample: Synthetic (toy)":
        return generate_synthetic(350)
    elif option == "Sample: Diabetes":
        return load_diabetes()
    else:
        return load_boston_like()

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        df = load_data(data_option)
else:
    df = load_data(data_option)

st.header("1. Dataset")
st.write("Preview of dataset:")
st.dataframe(df.head(preview_rows))

# Feature / target selection
all_cols = df.columns.tolist()
default_target = "target" if "target" in all_cols else all_cols[-1]
target_col = st.selectbox("Select target column (y)", all_cols, index=all_cols.index(default_target) if default_target in all_cols else len(all_cols)-1)
feature_cols = st.multiselect("Select feature columns (X). Leave empty for all except target.", [c for c in all_cols if c != target_col], default=[c for c in all_cols if c != target_col])

if not feature_cols:
    feature_cols = [c for c in all_cols if c != target_col]

X = df[feature_cols].copy()
y = df[target_col].copy()

st.write(f"Features: {feature_cols} — Target: {target_col}")

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))

if scale_features:
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
else:
    scaler = None

st.write(f"Train samples: {X_train.shape[0]} — Test samples: {X_test.shape[0]}")

# -------------------------
# Model-specific hyperparameters & snippets
# -------------------------
def linear_snippet():
    return textwrap.dedent("""
    # Linear Regression snippet
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('R2:', r2_score(y_test, y_pred))
    """)

def multiple_linear_snippet():
    return textwrap.dedent("""
    # Multiple linear regression uses same LinearRegression class (multivariate X)
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('R2:', r2_score(y_test, y_pred))
    """)

def poly_snippet(degree=2):
    return textwrap.dedent(f"""
    # Polynomial Regression snippet (degree={degree})
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    poly = PolynomialFeatures(degree={degree}, include_bias=False)
    Xp = poly.fit_transform(X_train)
    Xp_test = poly.transform(X_test)
    model = LinearRegression()
    model.fit(Xp, y_train)
    y_pred = model.predict(Xp_test)
    print('R2:', r2_score(y_test, y_pred))
    """)

def tree_snippet(max_depth=4):
    return textwrap.dedent(f"""
    # Decision Tree Regression snippet
    from sklearn.tree import DecisionTreeRegressor
    model = DecisionTreeRegressor(max_depth={max_depth}, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('R2:', r2_score(y_test, y_pred))
    """)

def rf_snippet(n_estimators=100, max_depth=None):
    return textwrap.dedent(f"""
    # Random Forest snippet
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators={n_estimators}, max_depth={repr(max_depth)}, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('R2:', r2_score(y_test, y_pred))
    """)

def svr_snippet(kernel="rbf", C=1.0, epsilon=0.1):
    return textwrap.dedent(f"""
    # Support Vector Regression snippet
    from sklearn.svm import SVR
    model = SVR(kernel='{kernel}', C={C}, epsilon={epsilon})
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('R2:', r2_score(y_test, y_pred))
    """)

def knn_snippet(n_neighbors=5):
    return textwrap.dedent(f"""
    # KNN Regression snippet
    from sklearn.neighbors import KNeighborsRegressor
    model = KNeighborsRegressor(n_neighbors={n_neighbors})
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('R2:', r2_score(y_test, y_pred))
    """)

def xgb_snippet(n_estimators=100, lr=0.1):
    return textwrap.dedent(f"""
    # XGBoost snippet
    import xgboost as xgb
    model = xgb.XGBRegressor(n_estimators={n_estimators}, learning_rate={lr}, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('R2:', r2_score(y_test, y_pred))
    """)

def nn_snippet(epochs=50):
    return textwrap.dedent(f"""
    # Simple Keras NN Regression snippet
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    model = keras.Sequential([
        layers.InputLayer(input_shape=(X_train.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs={epochs}, batch_size=16, validation_split=0.1, verbose=0)
    y_pred = model.predict(X_test).ravel()
    print('R2:', r2_score(y_test, y_pred))
    """)

# -------------------------
# Model functions
# -------------------------
def train_linear(X_train, y_train, X_test, params):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred

def train_polynomial(X_train, y_train, X_test, degree=2):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    Xp = poly.fit_transform(X_train)
    Xp_test = poly.transform(X_test)
    model = LinearRegression()
    model.fit(Xp, y_train)
    y_pred = model.predict(Xp_test)
    return (poly, model), y_pred

def train_tree(X_train, y_train, X_test, params):
    model = DecisionTreeRegressor(max_depth=params.get("max_depth", None), random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred

def train_rf(X_train, y_train, X_test, params):
    model = RandomForestRegressor(n_estimators=params.get("n_estimators",100),
                                  max_depth=params.get("max_depth", None),
                                  random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred

def train_svr(X_train, y_train, X_test, params):
    model = SVR(kernel=params.get("kernel","rbf"), C=params.get("C",1.0), epsilon=params.get("epsilon",0.1))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred

def train_knn(X_train, y_train, X_test, params):
    model = KNeighborsRegressor(n_neighbors=params.get("n_neighbors",5))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred

def train_xgb(X_train, y_train, X_test, params):
    if xgb is None:
        raise RuntimeError("XGBoost is not installed in this environment.")
    model = xgb.XGBRegressor(n_estimators=params.get("n_estimators",100),
                              learning_rate=params.get("learning_rate",0.1),
                              random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred

def train_lgb(X_train, y_train, X_test, params):
    if lgb is None:
        raise RuntimeError("LightGBM is not installed.")
    model = lgb.LGBMRegressor(n_estimators=params.get("n_estimators",100),
                              learning_rate=params.get("learning_rate",0.1),
                              random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred

def train_cat(X_train, y_train, X_test, params):
    if cb is None:
        raise RuntimeError("CatBoost is not installed.")
    model = cb.CatBoostRegressor(iterations=params.get("iterations",100),
                                 learning_rate=params.get("learning_rate",0.1),
                                 verbose=0, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred

def train_nn(X_train, y_train, X_test, params):
    if tf is None:
        raise RuntimeError("TensorFlow is not installed.")
    model = keras.Sequential()
    model.add(layers.InputLayer(input_shape=(X_train.shape[1],)))
    model.add(layers.Dense(params.get("units1",64), activation='relu'))
    model.add(layers.Dense(params.get("units2",32), activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=params.get("epochs",50), batch_size=16, validation_split=0.1, verbose=0)
    y_pred = model.predict(X_test).ravel()
    return model, y_pred

# -------------------------
# Visualization helpers
# -------------------------
def plot_predictions(y_true, y_pred, title="Predictions vs Actual"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_true, mode='markers', name='Actual'))
    fig.add_trace(go.Scatter(y=y_pred, mode='markers', name='Predicted'))
    fig.update_layout(title=title, xaxis_title="Sample index", yaxis_title="Target")
    st.plotly_chart(fig, use_container_width=True)

def plot_residuals(y_true, y_pred):
    res = y_true - y_pred
    fig = px.histogram(res, nbins=40, title="Residuals distribution")
    st.plotly_chart(fig, use_container_width=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=y_pred, y=res, mode='markers'))
    fig2.update_layout(title="Residuals vs Predicted", xaxis_title="Predicted", yaxis_title="Residuals")
    st.plotly_chart(fig2, use_container_width=True)

def plot_feature_importance(model, X_train, topk=15, model_name="Model"):
    try:
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
            names = X_train.columns
        else:
            # use permutation importance as fallback
            r = permutation_importance(model, X_train, np.zeros(X_train.shape[0])+0, n_repeats=8, random_state=42)
            fi = r.importances_mean
            names = X_train.columns
        fi_series = pd.Series(fi, index=names).sort_values(ascending=False).head(topk)
        fig = px.bar(fi_series, orientation='v', title=f"{model_name} feature importance (top {len(fi_series)})")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info("Feature importance is not available for this model.")

# -------------------------
# Main App Area: Model training & display
# -------------------------
st.header("2. Model Training & Exploration")

# Parameters UI per model
params = {}
if model_name == "Linear Regression":
    st.subheader("Linear Regression — Theory")
    st.markdown("""
    **Linear Regression** fits a linear model `y = Xw + b`.  
    Assumptions: linearity, independence, homoscedasticity, normal residuals (for inference).
    """)
    st.subheader("Hyperparameters")
    st.write("Linear Regression (no hyperparameters to tune for plain OLS).")
    snippet = linear_snippet()

elif model_name == "Multiple Linear Regression":
    st.subheader("Multiple Linear Regression")
    st.markdown("Same as Linear Regression but with multivariate features.")
    snippet = multiple_linear_snippet()

elif model_name == "Polynomial Regression":
    st.subheader("Polynomial Regression")
    degree = st.slider("Degree of polynomial", 2, 6, 2)
    params["degree"] = degree
    snippet = poly_snippet(degree=degree)
    st.markdown("Polynomial features transform X -> [1, x, x^2, ...] and then linear regression is applied.")

elif model_name == "Decision Tree":
    st.subheader("Decision Tree Regression")
    max_depth = st.slider("Max depth", 1, 20, 5)
    params["max_depth"] = max_depth
    snippet = tree_snippet(max_depth=max_depth)

elif model_name == "Random Forest":
    st.subheader("Random Forest Regression")
    n_estimators = st.slider("n_estimators", 10, 500, 100, step=10)
    max_depth = st.slider("max_depth (None=0)", 0, 40, 0)
    params["n_estimators"] = n_estimators
    params["max_depth"] = None if max_depth == 0 else int(max_depth)
    snippet = rf_snippet(n_estimators=n_estimators, max_depth=params["max_depth"])

elif model_name == "Support Vector Regression (SVR)":
    st.subheader("Support Vector Regression")
    kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"])
    C = st.number_input("C (regularization)", value=1.0)
    eps = st.number_input("epsilon", value=0.1)
    params["kernel"] = kernel
    params["C"] = float(C)
    params["epsilon"] = float(eps)
    snippet = svr_snippet(kernel=kernel, C=C, epsilon=eps)

elif model_name == "K-Nearest Neighbors (KNN)":
    st.subheader("KNN Regression")
    n_neighbors = st.slider("n_neighbors (k)", 1, 50, 5)
    params["n_neighbors"] = n_neighbors
    snippet = knn_snippet(n_neighbors=n_neighbors)

elif model_name == "Gradient Boosting (XGBoost / LightGBM / CatBoost)":
    st.subheader("Gradient Boosting — choose backend")
    gb_backend = st.selectbox("Backend", ["XGBoost", "LightGBM", "CatBoost"])
    n_estimators = st.slider("n_estimators", 50, 1000, 100, step=50)
    lr = st.number_input("learning_rate", min_value=0.0001, max_value=1.0, value=0.1, format="%.4f")
    params["n_estimators"] = n_estimators
    params["learning_rate"] = float(lr)
    snippet = xgb_snippet(n_estimators=n_estimators, lr=lr) if gb_backend=="XGBoost" else xgb_snippet(n_estimators=n_estimators, lr=lr)

elif model_name == "Neural Network (Keras)":
    st.subheader("Neural Network Regression (Keras)")
    units1 = st.slider("units (layer 1)", 8, 512, 64, step=8)
    units2 = st.slider("units (layer 2)", 8, 512, 32, step=8)
    epochs = st.slider("epochs", 1, 500, 50)
    params["units1"] = units1
    params["units2"] = units2
    params["epochs"] = epochs
    snippet = nn_snippet(epochs=epochs)

# Show code snippet
with st.expander("Model code snippet (editable)"):
    user_code = st.text_area("Edit code (you can copy and run locally). Press 'Run Edited Snippet' to execute here.", value=snippet, height=220)
    run_edited = st.button("Run Edited Snippet")

if run_edited:
    st.info("Executing the edited snippet in a restricted namespace. Be cautious with arbitrary code.")
    # Prepare restricted namespace
    safe_globals = {
        "__builtins__": __builtins__,
        "np": np, "pd": pd,
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "r2_score": r2_score, "mean_squared_error": mean_squared_error, "mean_absolute_error": mean_absolute_error
    }
    try:
        exec(user_code, safe_globals, {})
    except Exception as e:
        st.error(f"Error while running code: {e}")

# Training & evaluation
trained_model = None
y_pred = None
metrics_res = None

if train_button or compare_button:
    # Train a single model
    if compare_button:
        st.info("Training and comparing several models (fast defaults). This may take time for large datasets.")
        candidate_models = []
        candidate_models.append(("Linear", train_linear, {}))
        candidate_models.append(("Tree", train_tree, {"max_depth":5}))
        candidate_models.append(("RF", train_rf, {"n_estimators":100, "max_depth":None}))
        candidate_models.append(("KNN", train_knn, {"n_neighbors":5}))
        candidate_models.append(("SVR", train_svr, {"kernel":"rbf","C":1.0,"epsilon":0.1}))
        if xgb is not None: candidate_models.append(("XGB", train_xgb, {"n_estimators":100,"learning_rate":0.1}))
        if tf is not None: candidate_models.append(("NN", train_nn, {"epochs":30}))
        results = []
        with st.spinner("Training multiple models..."):
            for name, func, p in candidate_models:
                try:
                    model, yp = func(X_train, y_train, X_test, p)
                    m = metrics_dict(y_test, yp)
                    results.append((name, model, yp, m))
                except Exception as e:
                    st.warning(f"{name} failed: {e}")
        # summary table
        summary = pd.DataFrame([{**{"model":r[0]}, **r[3]} for r in results]).set_index("model")
        st.subheader("Comparison Results")
        st.dataframe(summary)
        st.write("Best by R2:")
        st.write(summary.sort_values("R2", ascending=False).head(5))
        # show best model predictions
        if results:
            best = max(results, key=lambda r: r[3]["R2"])
            st.subheader(f"Best model: {best[0]} (R2={best[3]['R2']:.4f})")
            plot_predictions(y_test.reset_index(drop=True), best[2])
            plot_residuals(y_test.reset_index(drop=True), best[2])
            try:
                plot_feature_importance(best[1], X_train)
            except Exception:
                pass
    else:
        st.info(f"Training {model_name}...")
        try:
            if model_name == "Linear Regression" or model_name == "Multiple Linear Regression":
                trained_model, y_pred = train_linear(X_train, y_train, X_test, params)
            elif model_name == "Polynomial Regression":
                trained_model, y_pred = train_polynomial(X_train, y_train, X_test, degree=params.get("degree",2))
            elif model_name == "Decision Tree":
                trained_model, y_pred = train_tree(X_train, y_train, X_test, params)
            elif model_name == "Random Forest":
                trained_model, y_pred = train_rf(X_train, y_train, X_test, params)
            elif model_name == "Support Vector Regression (SVR)":
                trained_model, y_pred = train_svr(X_train, y_train, X_test, params)
            elif model_name == "K-Nearest Neighbors (KNN)":
                trained_model, y_pred = train_knn(X_train, y_train, X_test, params)
            elif model_name == "Gradient Boosting (XGBoost / LightGBM / CatBoost)":
                # choose backend
                gb_backend = st.sidebar.selectbox("GB backend (sidebar)", ["XGBoost","LightGBM","CatBoost"])
                if gb_backend == "XGBoost":
                    trained_model, y_pred = train_xgb(X_train, y_train, X_test, params)
                elif gb_backend == "LightGBM":
                    trained_model, y_pred = train_lgb(X_train, y_train, X_test, params)
                else:
                    trained_model, y_pred = train_cat(X_train, y_train, X_test, params)
            elif model_name == "Neural Network (Keras)":
                trained_model, y_pred = train_nn(X_train, y_train, X_test, params)
            else:
                st.error("Model not implemented.")
                trained_model = None
            if y_pred is not None:
                metrics_res = metrics_dict(y_test, y_pred)
                st.subheader("Metrics")
                st.json(metrics_res)
                st.subheader("Visualizations")
                plot_predictions(y_test.reset_index(drop=True), y_pred)
                plot_residuals(y_test.reset_index(drop=True), y_pred)
                # feature importance for tree-based models
                try:
                    plot_feature_importance(trained_model, X_train)
                except Exception:
                    pass
                # save model
                st.subheader("Download / Save Model")
                try:
                    st.markdown(download_link(trained_model, f"{model_name.replace(' ','_')}.pkl", "Download trained model (.pkl)"), unsafe_allow_html=True)
                except Exception as e:
                    st.warning("Could not create download link for model: " + str(e))
        except Exception as e:
            st.error(f"Error training model: {e}")

# -------------------------
# Educational cheat sheet & comparison
# -------------------------
st.header("3. Cheat Sheet & When to use which model")
cheat = {
    "Linear Regression": "Great baseline, interpretable, assumes linear relationship.",
    "Polynomial Regression": "Captures non-linear relationships via polynomial features; beware of overfitting.",
    "Decision Tree": "Interpretable, handles non-linearities, prone to overfitting.",
    "Random Forest": "Strong default for tabular data, reduces variance, less interpretable.",
    "SVR": "Good for small datasets, kernel methods, sensitive to scaling.",
    "KNN": "Instance-based, simple, needs distance metric & scaled features.",
    "Gradient Boosting": "State-of-the-art for tabular problems (XGBoost/LightGBM/CatBoost).",
    "Neural Network": "Powerful for complex patterns, needs tuning & more data."
}
cheat_df = pd.DataFrame.from_dict(cheat, orient='index', columns=['Notes'])
st.dataframe(cheat_df)

st.header("4. Quick Tips for Interview Prep")
st.markdown("""
- Understand assumptions (e.g., for linear regression).
- Know how to evaluate (R², MAE, MSE, RMSE) and what they mean.
- Be able to explain bias-variance tradeoff and regularization.
- Know pros/cons of tree-based models vs linear models vs NN.
- Be ready to show how to preprocess (scaling, polynomial features).
""")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ — Interactive Regression Learning Lab — single-file. Expand and adapt as needed.")
