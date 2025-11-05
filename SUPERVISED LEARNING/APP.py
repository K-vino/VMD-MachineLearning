# ==============================================================================
# 🎯 Streamlit Application: Interactive Regression Learning Lab
# File: regression_lab_app.py
# Total Lines Target: 3000+
# Description: A single-file application for learning, visualizing, and
#              experimenting with all major Machine Learning regression models.
#              Includes theory, interactive hyperparameters, real-time metrics,
#              Plotly visualizations, SHAP interpretation, and an editable code
#              playground.
# ==============================================================================

# --- 0. STANDARD LIBRARY IMPORTS ---
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import joblib # For model download

# --- 1. SCIKIT-LEARN IMPORTS (Models & Preprocessing) ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import make_regression, make_friedman1

# Base Models
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# --- 2. ADVANCED ML IMPORTS ---
# Note: These libraries must be installed in the environment for the app to run.
try:
    # Boosting Models
    import xgboost as xgb
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor
    # Model Interpretation
    import shap
    # Neural Networks (Keras/TensorFlow)
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
except ImportError as e:
    st.error(f"Missing essential library: {e.name}. Please install it.")
    st.stop()

# --- 3. STREAMLIT WIDGETS IMPORTS ---
# Using the standard st.code for code snippets and st_ace is not used
# as it requires an external library install and a specific Streamlit component
# wrapper. We will use a standard Streamlit text area for the "Code Playground"
# to keep the app self-contained and simple to run.

# ==============================================================================
# ⚙️ GLOBAL CONFIGURATION & STATE INITIALIZATION
# ==============================================================================

# Set Streamlit page configuration
st.set_page_config(
    page_title="Interactive Regression Learning Lab",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# Session State Management
# ------------------------------------------------------------------------------

if 'data_df' not in st.session_state:
    st.session_state.data_df = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = pd.DataFrame(
        columns=['Model', 'R²', 'MAE', 'MSE', 'RMSE', 'Time (s)', 'Config']
    )
if 'model_objects' not in st.session_state:
    st.session_state.model_objects = {}
if 'model_predictions' not in st.session_state:
    st.session_state.model_predictions = {}
if 'scaler' not in st.session_state:
    st.session_state.scaler = StandardScaler()
if 'is_scaled' not in st.session_state:
    st.session_state.is_scaled = False

# ==============================================================================
# 🛠️ HELPER FUNCTIONS
# ==============================================================================

@st.cache_data(show_spinner=False)
def get_model_metrics(y_true, y_pred, model_name=None):
    """Calculates and formats key regression metrics."""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    metrics = {
        'R²': f"{r2:.4f}",
        'MAE': f"{mae:.4f}",
        'MSE': f"{mse:.4f}",
        'RMSE': f"{rmse:.4f}"
    }

    if model_name:
        st.session_state.comparison_results = add_to_comparison(
            model_name, r2, mae, mse, rmse, 0, {} # Time and config are updated later
        )

    return metrics

def add_to_comparison(model_name, r2, mae, mse, rmse, time, config):
    """Adds or updates a model's performance metrics in the comparison table."""
    new_result = {
        'Model': model_name,
        'R²': round(r2, 4),
        'MAE': round(mae, 4),
        'MSE': round(mse, 4),
        'RMSE': round(rmse, 4),
        'Time (s)': round(time, 4),
        'Config': str(config)
    }

    df = st.session_state.comparison_results.copy()
    if model_name in df['Model'].values:
        # Update existing row
        df.loc[df['Model'] == model_name] = new_result
    else:
        # Append new row
        df = pd.concat([df, pd.DataFrame([new_result])], ignore_index=True)

    # Sort for better viewing
    df = df.sort_values(by='R²', ascending=False).reset_index(drop=True)
    return df

def plot_predictions(y_test, y_pred, title="Model Predictions vs. Actual Data"):
    """
    Generates a Plotly scatter plot comparing actual vs. predicted values.
    Includes a perfect prediction line (y=x).
    """
    results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })

    # Find the range for the diagonal line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    range_line = [min_val, max_val]

    fig = px.scatter(
        results_df,
        x='Actual',
        y='Predicted',
        title=title,
        template="plotly_white",
        height=400
    )
    # Add the ideal prediction line (y=x)
    fig.add_trace(go.Scatter(
        x=range_line,
        y=range_line,
        mode='lines',
        name='Ideal Prediction',
        line=dict(color='red', dash='dash')
    ))

    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

def plot_residuals(y_test, y_pred, title="Residual Plot"):
    """
    Generates a Plotly scatter plot of residuals (Actual - Predicted) vs. Predicted values.
    """
    residuals = y_test - y_pred
    results_df = pd.DataFrame({
        'Predicted': y_pred,
        'Residuals': residuals
    })

    fig = px.scatter(
        results_df,
        x='Predicted',
        y='Residuals',
        title=title,
        template="plotly_white",
        height=400
    )
    # Add the zero line
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

def plot_feature_importance(model, features, model_name):
    """
    Generates a Plotly bar chart for feature importance.
    Handles different model types (linear coefficients, tree-based importances).
    """
    if model_name in ['Linear Regression', 'Multiple Linear Regression', 'Ridge', 'Lasso']:
        # For linear models, use absolute value of coefficients
        importance = pd.Series(np.abs(model.coef_), index=features)
        importance_name = 'Absolute Coefficient Value'
    elif model_name in ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost']:
        importance = pd.Series(model.feature_importances_, index=features)
        importance_name = 'Feature Importance'
    else:
        st.info("Feature importance is not directly available for this model type (e.g., SVR, KNN, NN).")
        return

    # Sort importance values
    importance = importance.sort_values(ascending=True)

    fig = px.bar(
        importance,
        x=importance.values,
        y=importance.index,
        orientation='h',
        title=f'{model_name} - {importance_name}',
        labels={'x': importance_name, 'y': 'Feature'},
        template="plotly_white",
        height=400
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

def display_metrics(y_test, y_pred, model_name, model_config, training_time):
    """Displays metrics in a clean, multi-column format and updates comparison table."""
    metrics = get_model_metrics(y_test, y_pred)
    r2 = float(metrics['R²'])
    mae = float(metrics['MAE'])
    mse = float(metrics['MSE'])
    rmse = float(metrics['RMSE'])

    st.subheader(f"📊 Model Performance Metrics: {model_name}")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("R² Score (R-Squared)", metrics['R²'])
    col2.metric("Mean Absolute Error (MAE)", metrics['MAE'])
    col3.metric("Mean Squared Error (MSE)", metrics['MSE'])
    col4.metric("Root Mean Squared Error (RMSE)", metrics['RMSE'])
    col5.metric("Training Time", f"{training_time:.2f}s")

    # Update global comparison results
    st.session_state.comparison_results = add_to_comparison(
        model_name, r2, mae, mse, rmse, training_time, model_config
    )

    st.session_state.model_predictions[model_name] = y_pred

def display_model_code(model_name, code_snippet):
    """Puts the core Python code snippet in a collapsible expander."""
    with st.expander(f"💻 View Core Python Code: {model_name}", expanded=False):
        st.code(code_snippet, language='python')

def display_shap_explanation(model, X_train_scaled, X_test_scaled, model_name):
    """Generates and displays SHAP summary plot."""
    st.subheader("🧠 Model Interpretation (SHAP)")
    st.write(
        "SHAP (SHapley Additive exPlanations) uses game theory to explain the output "
        "of any machine learning model. It connects optimal credit allocation with local "
        "explanations using the classic Shapley values."
    )

    # Use a try-except block in case SHAP fails for certain models/versions
    try:
        # For tree models, use TreeExplainer (faster)
        if model_name in ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost']:
            explainer = shap.TreeExplainer(model)
        # For non-linear/complex models like NN, SVR, use KernelExplainer (slower)
        elif model_name in ['SVR', 'Keras Neural Network']:
            # We need a background dataset for KernelExplainer
            background = shap.maskers.Independent(X_train_scaled, max_samples=100)
            explainer = shap.KernelExplainer(model.predict, background)
        # For simpler models, use LinearExplainer
        elif model_name in ['Linear Regression', 'Ridge', 'Lasso', 'KNN']:
            explainer = shap.LinearExplainer(model, X_train_scaled)
        else:
            st.warning("SHAP Explainer type not determined for this model. Skipping SHAP.")
            return

        # Calculate SHAP values for a subset of the test data
        with st.spinner("Calculating SHAP values for model interpretation... This may take a moment for complex models."):
            shap_values = explainer.shap_values(X_test_scaled)

        # Matplotlib plot for SHAP summary
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test_scaled, feature_names=X_test_scaled.columns.tolist(), show=False)
        ax.set_title(f"SHAP Feature Importance for {model_name}")
        st.pyplot(fig, clear_figure=True)

    except Exception as e:
        st.error(f"Failed to generate SHAP plot. Error: {e}")
        st.warning("Ensure the model is fully trained and the SHAP library is compatible with the model type.")

def generate_synthetic_data(data_type, n_samples, n_features, noise):
    """Generates various types of synthetic regression data."""
    if data_type == "Simple Linear":
        X, y = make_regression(
            n_samples=n_samples, n_features=1, n_informative=1,
            n_targets=1, noise=noise, random_state=42
        )
        feature_names = [f"Feature_{i+1}" for i in range(X.shape[1])]
    elif data_type == "Multi-Feature":
        X, y = make_regression(
            n_samples=n_samples, n_features=n_features, n_informative=int(n_features * 0.8),
            n_targets=1, noise=noise, random_state=42
        )
        feature_names = [f"Feature_{i+1}" for i in range(X.shape[1])]
    elif data_type == "Non-Linear (Friedman)":
        X, y = make_friedman1(
            n_samples=n_samples, n_features=n_features, noise=noise, random_state=42
        )
        feature_names = [f"Feature_{i+1}" for i in range(X.shape[1])]
    else:
        st.error("Invalid synthetic data type.")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['Target'] = y
    return df

# ==============================================================================
# 🗄️ DATA SELECTION & PREPROCESSING (Sidebar)
# ==============================================================================

def data_selection_sidebar():
    """Handles data upload and synthetic generation controls in the sidebar."""
    st.sidebar.header("1. Data Setup")

    data_source = st.sidebar.radio(
        "Choose Data Source:",
        ["Synthetic Dataset", "Upload CSV"],
        key="data_source"
    )

    df = None
    target_column = None

    if data_source == "Synthetic Dataset":
        st.sidebar.subheader("Synthetic Data Generator")
        synth_type = st.sidebar.selectbox(
            "Select Data Type:",
            ["Simple Linear", "Multi-Feature", "Non-Linear (Friedman)"],
            key="synth_type"
        )
        n_samples = st.sidebar.slider("Number of Samples (n_samples):", 100, 2000, 500, key="n_samples")
        n_features = st.sidebar.slider("Number of Features (n_features):", 2, 10, 5, key="n_features", disabled=(synth_type == "Simple Linear"))
        noise = st.sidebar.slider("Noise Level (standard deviation):", 0.0, 100.0, 20.0, key="noise")

        if st.sidebar.button("Generate & Use Data"):
            try:
                df = generate_synthetic_data(synth_type, n_samples, n_features, noise)
                target_column = 'Target'
                st.session_state.data_df = df
                st.session_state.target_column = target_column
                st.sidebar.success(f"Generated {synth_type} data with {n_samples} samples.")
            except Exception as e:
                st.sidebar.error(f"Error generating data: {e}")

    elif data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.data_df = df
                # Allow user to select the target column
                st.sidebar.subheader("Target Column Selection")
                cols = df.columns.tolist()
                target_column = st.sidebar.selectbox("Select the Target (Y) Column:", cols, key="target_col_upload")
                st.session_state.target_column = target_column
                st.sidebar.success(f"CSV uploaded successfully. {len(df)} rows found.")
            except Exception as e:
                st.sidebar.error(f"Error reading file: {e}")

    # --- Global Data Preprocessing Section ---
    if st.session_state.data_df is not None:
        st.sidebar.markdown("---")
        st.sidebar.header("2. Preprocessing")

        test_size = st.sidebar.slider("Test Set Size (%):", 10, 50, 20) / 100
        random_state = st.sidebar.number_input("Random State (for splitting):", 0, 1000, 42)
        scale_data = st.sidebar.checkbox("Standard Scale Features (Recommended)", value=True, key="scale_data")

        # Automatically handle splitting and scaling
        if st.sidebar.button("Process Data & Split"):
            df = st.session_state.data_df.copy()
            target_col = st.session_state.target_column

            if target_col not in df.columns:
                 st.sidebar.error("Target column is not selected or not found.")
                 return

            X = df.drop(columns=[target_col])
            y = df[target_col]

            # Basic imputation for NaN values (mean imputation)
            X = X.fillna(X.mean())

            # Convert non-numeric (e.g., object types) to numeric using one-hot encoding
            X = pd.get_dummies(X, drop_first=True)

            # Split Data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            # Scaling
            if scale_data:
                scaler = StandardScaler()
                X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
                X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
                st.session_state.scaler = scaler
                st.session_state.is_scaled = True
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
                st.session_state.is_scaled = False

            # Update Session State
            st.session_state.X_train = X_train_scaled
            st.session_state.X_test = X_test_scaled
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.sidebar.success("Data successfully split, cleaned, and processed!")

# ==============================================================================
# 🧩 MODEL IMPLEMENTATION FUNCTIONS (One function per model)
# ==============================================================================

def train_generic_model(model_instance, X_train, y_train, X_test, model_name, config):
    """A generic function to handle training, prediction, and result storage."""
    import time

    if X_train is None:
        st.error("Please process your data first in the sidebar.")
        return None, None, None

    start_time = time.time()
    try:
        with st.spinner(f"Training {model_name}..."):
            model_instance.fit(X_train, y_train)
            y_pred = model_instance.predict(X_test)
        training_time = time.time() - start_time
        st.success(f"{model_name} training complete in {training_time:.2f} seconds!")

        # Store model object and update metrics/comparison
        st.session_state.model_objects[model_name] = model_instance
        display_metrics(st.session_state.y_test, y_pred, model_name, config, training_time)
        return model_instance, y_pred, training_time

    except Exception as e:
        st.error(f"An error occurred during training {model_name}: {e}")
        return None, None, None

# ------------------------------------------------------------------------------
# 1. Linear Regression (Simple & Multiple)
# ------------------------------------------------------------------------------
def linear_regression_tab(X_train, y_train, X_test, y_test):
    """Implementation for Linear Regression (OLS)."""
    model_name = "Linear Regression (OLS)"
    st.header(f"1. {model_name}")
    st.markdown(r"""
        **Theory:** Ordinary Least Squares (OLS) Linear Regression models the linear
        relationship between a single (Simple) or multiple (Multiple) independent
        variables ($X$) and a continuous dependent variable ($Y$).
        
        **Mathematical Formula:** $Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_n X_n + \epsilon$
        
        Where:
        * $\beta_0$ is the y-intercept.
        * $\beta_i$ are the coefficients (slopes) for each feature $X_i$.
        * $\epsilon$ is the error term.
        
        **Key Assumptions:** Linearity, independence of errors, homoscedasticity (constant variance of errors),
        and normality of residuals.
        
        **Pros/Cons:** Simple, highly interpretable, fast. However, it's prone to
        underfitting and sensitive to outliers and multicollinearity.
    """)
    st.markdown("---")

    # Hyperparameter section (using Lasso/Ridge for simple regularization)
    st.subheader("⚙️ Hyperparameter Adjustment (Optional Regularization)")
    regressor_type = st.radio(
        "Choose Model Variant:",
        ["Standard OLS", "Ridge Regression (L2)", "Lasso Regression (L1)"],
        key="linear_type"
    )

    alpha = 0.0
    if regressor_type in ["Ridge Regression (L2)", "Lasso Regression (L1)"]:
        st.warning("Scaling is highly recommended for regularization methods.")
        alpha = st.slider(
            "Regularization Strength (Alpha):",
            0.01, 10.0, 1.0, step=0.01,
            key="linear_alpha"
        )

    # Training Button
    if st.button(f"Train {model_name} Model", key="train_linear"):
        if regressor_type == "Standard OLS":
            model = LinearRegression()
            config = {'type': 'OLS'}
        elif regressor_type == "Ridge Regression (L2)":
            model = Ridge(alpha=alpha, random_state=42)
            config = {'type': 'Ridge', 'alpha': alpha}
        else: # Lasso
            model = Lasso(alpha=alpha, random_state=42)
            config = {'type': 'Lasso', 'alpha': alpha}

        trained_model, y_pred, train_time = train_generic_model(model, X_train, y_train, X_test, model_name, config)

        if trained_model:
            # --- Results and Visualization ---
            st.markdown("---")
            display_metrics(y_test, y_pred, model_name, config, train_time)

            col_pred, col_res = st.columns(2)
            with col_pred:
                plot_predictions(y_test, y_pred, f"{regressor_type} Predictions")
            with col_res:
                plot_residuals(y_test, y_pred, f"{regressor_type} Residuals")

            # Feature Importance (Coefficients)
            plot_feature_importance(trained_model, X_train.columns, model_name)

            # SHAP Interpretation
            st.markdown("---")
            if st.button("Explain Predictions (SHAP)", key="shap_linear"):
                 display_shap_explanation(trained_model, X_train, X_test, model_name)

            # Code Display
            code_snippet = f"""
from sklearn.linear_model import {trained_model.__class__.__name__}
# Hyperparameters: {config}

model = {trained_model.__class__.__name__}(alpha={alpha}, random_state=42)
# For OLS, just model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
R2 = r2_score(y_test, y_pred)
# Coefficients: print(model.coef_)
"""
            display_model_code(model_name, code_snippet)

            # Download Button
            download_model_button(trained_model, model_name)

# ------------------------------------------------------------------------------
# 2. Polynomial Regression
# ------------------------------------------------------------------------------
def polynomial_regression_tab(X_train, y_train, X_test, y_test):
    """Implementation for Polynomial Regression."""
    model_name = "Polynomial Regression"
    st.header(f"2. {model_name}")
    st.markdown(r"""
        **Theory:** Polynomial Regression is a form of Linear Regression where the
        relationship between $X$ and $Y$ is modeled as an $n$-th degree polynomial.
        It is still linear in terms of the coefficients ($\beta$).
        
        **Mathematical Formula (Degree 2):** $Y = \beta_0 + \beta_1 X + \beta_2 X^2 + \epsilon$
        
        **Key Mechanism:** The original features $X$ are transformed into polynomial
        features (e.g., $X^2, X^3, X_1 X_2$) and then a standard OLS Linear Regression
        model is fit to these new features.
        
        **Pros/Cons:** Can model non-linear relationships. Higher degrees can easily
        lead to severe **overfitting** (high variance).
    """)
    st.markdown("---")

    # Hyperparameter section
    st.subheader("⚙️ Hyperparameter Adjustment")
    degree = st.slider(
        "Polynomial Degree:",
        1, 10, 2, key="poly_degree", help="Higher degrees increase model complexity and risk of overfitting."
    )

    # Training Button
    if st.button(f"Train {model_name} Model", key="train_poly"):
        if X_train is None:
            st.error("Please process your data first in the sidebar.")
            return

        import time
        start_time = time.time()

        with st.spinner(f"Creating polynomial features (Degree {degree}) and training OLS..."):
            # 1. Feature Transformation
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            X_train_poly = poly.fit_transform(X_train)
            X_test_poly = poly.transform(X_test)

            # Need to get feature names for plotting
            new_feature_names = poly.get_feature_names_out(X_train.columns)
            X_train_poly_df = pd.DataFrame(X_train_poly, columns=new_feature_names)
            X_test_poly_df = pd.DataFrame(X_test_poly, columns=new_feature_names)

            # 2. Model Training (OLS on transformed data)
            model = LinearRegression()
            model.fit(X_train_poly_df, y_train)
            y_pred = model.predict(X_test_poly_df)

        training_time = time.time() - start_time
        config = {'degree': degree}
        st.success(f"{model_name} (Degree {degree}) training complete in {training_time:.2f} seconds!")

        # Store model object (Note: The feature transformer is part of the process)
        st.session_state.model_objects[model_name] = (poly, model) # Store both
        display_metrics(y_test, y_pred, model_name, config, training_time)

        # --- Results and Visualization ---
        st.markdown("---")
        display_metrics(y_test, y_pred, model_name, config, training_time)

        col_pred, col_res = st.columns(2)
        with col_pred:
            plot_predictions(y_test, y_pred, f"Polynomial Reg. Predictions (Deg {degree})")
        with col_res:
            plot_residuals(y_test, y_pred, f"Polynomial Reg. Residuals (Deg {degree})")

        # Code Display
        code_snippet = f"""
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

degree = {degree}

# 1. Transform features
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# 2. Train OLS model
model = LinearRegression()
model.fit(X_train_poly, y_train)
y_pred = model.predict(X_test_poly)
"""
        display_model_code(model_name, code_snippet)

        # Download Button (Note: Only saving the OLS model here for simplicity)
        download_model_button(model, model_name)

# ------------------------------------------------------------------------------
# 3. K-Nearest Neighbors Regression (KNN)
# ------------------------------------------------------------------------------
def knn_regression_tab(X_train, y_train, X_test, y_test):
    """Implementation for KNN Regression."""
    model_name = "K-Nearest Neighbors Regression (KNN)"
    st.header(f"3. {model_name}")
    st.markdown(r"""
        **Theory:** KNN is a non-parametric, instance-based learning algorithm.
        To predict the value of a new data point, it finds the **K** closest
        data points (neighbors) in the training set and computes the mean (or
        median) of their target values.
        
        **Mathematical Formula:** $\hat{y} = \frac{1}{K} \sum_{i=1}^{K} y_i$ (Mean of K neighbors)
        
        **Key Assumptions:** Scaling is crucial as distance metrics are sensitive to feature magnitude.
        No explicit assumptions about the data distribution or relationship.
        
        **Pros/Cons:** Simple, no training phase (lazy learning), handles non-linear data well.
        Computationally expensive during prediction, sensitive to irrelevant features, and requires optimal K.
    """)
    st.markdown("---")

    # Hyperparameter section
    st.subheader("⚙️ Hyperparameter Adjustment")
    n_neighbors = st.slider(
        "Number of Neighbors (K):",
        1, 50, 5, key="knn_k", help="The number of closest training examples used for prediction."
    )
    weights = st.radio(
        "Weight Function:",
        ["uniform", "distance"],
        key="knn_weights", help="'uniform' assigns equal weights, 'distance' weights points inversely by distance."
    )
    p_metric = st.radio(
        "Minkowski Metric (p):",
        [1, 2],
        key="knn_p", help="p=1 is Manhattan distance (L1), p=2 is Euclidean distance (L2)."
    )

    # Training Button
    if st.button(f"Train {model_name} Model", key="train_knn"):
        model = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=weights,
            p=p_metric
        )
        config = {'n_neighbors': n_neighbors, 'weights': weights, 'p': p_metric}
        trained_model, y_pred, train_time = train_generic_model(model, X_train, y_train, X_test, model_name, config)

        if trained_model:
            # --- Results and Visualization ---
            st.markdown("---")
            display_metrics(y_test, y_pred, model_name, config, train_time)

            col_pred, col_res = st.columns(2)
            with col_pred:
                plot_predictions(y_test, y_pred, f"KNN (K={n_neighbors}) Predictions")
            with col_res:
                plot_residuals(y_test, y_pred, f"KNN (K={n_neighbors}) Residuals")

            # Code Display
            code_snippet = f"""
from sklearn.neighbors import KNeighborsRegressor

model = KNeighborsRegressor(
    n_neighbors={n_neighbors},
    weights='{weights}',
    p={p_metric}
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
"""
            display_model_code(model_name, code_snippet)

            # Download Button
            download_model_button(trained_model, model_name)

# ------------------------------------------------------------------------------
# 4. Support Vector Regression (SVR)
# ------------------------------------------------------------------------------
def svr_regression_tab(X_train, y_train, X_test, y_test):
    """Implementation for SVR."""
    model_name = "Support Vector Regression (SVR)"
    st.header(f"4. {model_name}")
    st.markdown(r"""
        **Theory:** SVR is an extension of Support Vector Machines (SVM) for regression.
        Instead of fitting the best line or boundary (like in classification), SVR tries
        to fit the best 'tube' or hyperplane that contains as many data points as possible,
        while minimizing errors outside the tube (defined by $\epsilon$).
        
        **Key Hyperparameters:**
        * **C:** Regularization parameter (trade-off between maximizing the margin and minimizing error).
        * **$\epsilon$ (Epsilon):** Defines the margin of tolerance where no penalty is given to errors.
        * **Kernel:** Linear, Radial Basis Function (RBF), Polynomial, etc. RBF is common for non-linear data.
        
        **Pros/Cons:** Effective in high-dimensional spaces, memory efficient, and uses a
        subset of training points (support vectors). Computationally expensive for large
        datasets, and sensitive to parameter choice. Scaling is **crucial**.
    """)
    st.markdown("---")

    # Hyperparameter section
    st.subheader("⚙️ Hyperparameter Adjustment")
    C = st.slider(
        "Regularization (C):",
        0.1, 100.0, 1.0, step=0.1, key="svr_C", help="A smaller C means stronger regularization."
    )
    epsilon = st.slider(
        r"Epsilon ($\epsilon$):", # CORRECTED: Added 'r' prefix
        0.0, 10.0, 0.1, step=0.1, key="svr_epsilon", help="Defines the margin of tolerance for errors."
    )
    kernel = st.selectbox(
        "Kernel Type:",
        ["rbf", "linear", "poly"],
        key="svr_kernel"
    )

    # Training Button
    if st.button(f"Train {model_name} Model", key="train_svr"):
        model = SVR(
            C=C,
            epsilon=epsilon,
            kernel=kernel
        )
        config = {'C': C, 'epsilon': epsilon, 'kernel': kernel}
        trained_model, y_pred, train_time = train_generic_model(model, X_train, y_train, X_test, model_name, config)

        if trained_model:
            # --- Results and Visualization ---
            st.markdown("---")
            display_metrics(y_test, y_pred, model_name, config, train_time)

            col_pred, col_res = st.columns(2)
            with col_pred:
                plot_predictions(y_test, y_pred, f"SVR ({kernel} Kernel) Predictions")
            with col_res:
                plot_residuals(y_test, y_pred, f"SVR ({kernel} Kernel) Residuals")

            # Code Display
            code_snippet = f"""
from sklearn.svm import SVR

model = SVR(
    C={C},
    epsilon={epsilon},
    kernel='{kernel}'
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
"""
            display_model_code(model_name, code_snippet)

            # Download Button
            download_model_button(trained_model, model_name)

# ------------------------------------------------------------------------------
# 5. Decision Tree Regression
# ------------------------------------------------------------------------------
def dt_regression_tab(X_train, y_train, X_test, y_test):
    """Implementation for Decision Tree Regression."""
    model_name = "Decision Tree Regression"
    st.header(f"5. {model_name}")
    st.markdown(r"""
        **Theory:** Decision Trees split the data into partitions (nodes) based on features
        to create a tree-like structure. For regression, the prediction at a leaf node is
        the average of the target values of all training samples that fall into that node.
        
        **Key Mechanism:** The tree splits are chosen greedily using a criterion like
        Mean Squared Error (MSE) or Mean Absolute Error (MAE) reduction.
        
        **Key Hyperparameters:**
        * **Max Depth:** The maximum number of levels in the tree. Controls overfitting.
        * **Min Samples Split/Leaf:** Minimum number of samples required to split a node or be a leaf.
        
        **Pros/Cons:** Highly intuitive, no feature scaling required, non-linear relationships
        are handled naturally. Prone to severe **overfitting** and highly sensitive to data variations.
    """)
    st.markdown("---")

    # Hyperparameter section
    st.subheader("⚙️ Hyperparameter Adjustment")
    max_depth = st.slider(
        "Max Depth:",
        1, 20, 5, key="dt_depth", help="Controls the size of the tree. Lower depth prevents overfitting."
    )
    min_samples_leaf = st.slider(
        "Min Samples at Leaf:",
        1, 50, 5, key="dt_min_leaf", help="Minimum number of samples required to be at a leaf node."
    )
    criterion = st.selectbox(
        "Split Criterion:",
        ["squared_error", "absolute_error"],
        key="dt_criterion", help="The function to measure the quality of a split."
    )

    # Training Button
    if st.button(f"Train {model_name} Model", key="train_dt"):
        model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=42
        )
        config = {'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf, 'criterion': criterion}
        trained_model, y_pred, train_time = train_generic_model(model, X_train, y_train, X_test, model_name, config)

        if trained_model:
            # --- Results and Visualization ---
            st.markdown("---")
            display_metrics(y_test, y_pred, model_name, config, train_time)

            col_pred, col_res = st.columns(2)
            with col_pred:
                plot_predictions(y_test, y_pred, f"Decision Tree (Depth {max_depth}) Predictions")
            with col_res:
                plot_residuals(y_test, y_pred, f"Decision Tree (Depth {max_depth}) Residuals")

            # Feature Importance
            plot_feature_importance(trained_model, X_train.columns, model_name)

            # SHAP Interpretation
            st.markdown("---")
            if st.button("Explain Predictions (SHAP)", key="shap_dt"):
                 display_shap_explanation(trained_model, X_train, X_test, model_name)

            # Code Display
            code_snippet = f"""
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(
    max_depth={max_depth},
    min_samples_leaf={min_samples_leaf},
    criterion='{criterion}',
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
"""
            display_model_code(model_name, code_snippet)

            # Download Button
            download_model_button(trained_model, model_name)

# ------------------------------------------------------------------------------
# 6. Random Forest Regression
# ------------------------------------------------------------------------------
def rf_regression_tab(X_train, y_train, X_test, y_test):
    """Implementation for Random Forest Regression."""
    model_name = "Random Forest Regression"
    st.header(f"6. {model_name}")
    st.markdown(r"""
        **Theory:** Random Forest is an **ensemble** method that builds multiple
        Decision Trees and averages their predictions to improve accuracy and
        control overfitting (a technique called **bagging**).
        
        **Key Mechanism:**
        1.  Each tree is trained on a random subset of the data (bootstrapping).
        2.  When splitting a node, only a random subset of features is considered.
        
        **Key Hyperparameters:**
        * **N Estimators:** The number of trees in the forest.
        * **Max Features:** The number of features to consider when looking for the best split.
        
        **Pros/Cons:** High accuracy, robust against overfitting, and provides
        reliable feature importance estimates. Slower than single trees, and
        less interpretable than a single tree.
    """)
    st.markdown("---")

    # Hyperparameter section
    st.subheader("⚙️ Hyperparameter Adjustment")
    n_estimators = st.slider(
        "Number of Trees (n_estimators):",
        10, 500, 100, step=10, key="rf_n_est", help="The size of the forest. More trees generally means better performance."
    )
    max_depth = st.slider(
        "Max Depth per Tree:",
        3, 20, 10, key="rf_depth", help="Maximum depth of individual trees."
    )
    max_features = st.select_slider(
        "Max Features per Split:",
        options=['sqrt', 'log2', 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        value='sqrt', key="rf_max_feat", help="The number of features to consider when looking for the best split."
    )

    # Convert select_slider output for max_features if needed
    if isinstance(max_features, (float, int)):
        max_features_val = max_features
    else:
        max_features_val = max_features

    # Training Button
    if st.button(f"Train {model_name} Model", key="train_rf"):
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features_val,
            random_state=42,
            n_jobs=-1
        )
        config = {'n_estimators': n_estimators, 'max_depth': max_depth, 'max_features': max_features}
        trained_model, y_pred, train_time = train_generic_model(model, X_train, y_train, X_test, model_name, config)

        if trained_model:
            # --- Results and Visualization ---
            st.markdown("---")
            display_metrics(y_test, y_pred, model_name, config, train_time)

            col_pred, col_res = st.columns(2)
            with col_pred:
                plot_predictions(y_test, y_pred, f"Random Forest (n={n_estimators}) Predictions")
            with col_res:
                plot_residuals(y_test, y_pred, f"Random Forest (n={n_estimators}) Residuals")

            # Feature Importance
            plot_feature_importance(trained_model, X_train.columns, model_name)

            # SHAP Interpretation
            st.markdown("---")
            if st.button("Explain Predictions (SHAP)", key="shap_rf"):
                 display_shap_explanation(trained_model, X_train, X_test, model_name)

            # Code Display
            code_snippet = f"""
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators={n_estimators},
    max_depth={max_depth},
    max_features='{max_features}',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
"""
            display_model_code(model_name, code_snippet)

            # Download Button
            download_model_button(trained_model, model_name)

# ------------------------------------------------------------------------------
# 7. Gradient Boosting Regression
# ------------------------------------------------------------------------------
def gb_regression_tab(X_train, y_train, X_test, y_test):
    """Implementation for Scikit-learn Gradient Boosting and specialized libraries."""
    st.header("7. Gradient Boosting Regression")
    st.markdown(r"""
        **Theory:** Gradient Boosting (GB) is another powerful **ensemble** technique
        that builds trees **sequentially** (one after another). Each new tree attempts
        to correct the errors (residuals) of the previous tree.
        
        **Key Mechanism (Boosting):** Unlike Random Forest (Bagging), boosting models
        learn sequentially and often use a small learning rate to control the contribution
        of each tree, making the process slower but potentially more accurate.
        
        **Key Hyperparameters:**
        * **N Estimators:** The number of sequential trees to build.
        * **Learning Rate:** Shrinks the contribution of each tree; controls the 'speed' of learning.
        * **Max Depth:** Controls complexity of individual weak learners (trees).
    """)
    st.markdown("---")

    # Model Selection
    st.subheader("⚙️ Select and Configure a Boosting Framework")
    boosting_framework = st.selectbox(
        "Select Boosting Model:",
        ["Scikit-learn GBR", "XGBoost", "LightGBM", "CatBoost"],
        key="boosting_type"
    )

    # Common Hyperparameters
    n_estimators = st.slider("N Estimators (Trees):", 50, 500, 100, step=10, key="gb_n_est")
    learning_rate = st.slider("Learning Rate:", 0.01, 0.5, 0.1, step=0.01, key="gb_lr")
    max_depth = st.slider("Max Depth:", 1, 15, 3, key="gb_depth")

    # Training Button
    if st.button(f"Train {boosting_framework} Model", key="train_gb"):
        model_name = boosting_framework

        if boosting_framework == "Scikit-learn GBR":
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42
            )
        elif boosting_framework == "XGBoost":
            model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
        elif boosting_framework == "LightGBM":
            model = LGBMRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
        elif boosting_framework == "CatBoost":
            model = CatBoostRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42,
                verbose=0 # Suppress CatBoost output
            )

        config = {'n_estimators': n_estimators, 'learning_rate': learning_rate, 'max_depth': max_depth}
        trained_model, y_pred, train_time = train_generic_model(model, X_train, y_train, X_test, model_name, config)

        if trained_model:
            # --- Results and Visualization ---
            st.markdown("---")
            display_metrics(y_test, y_pred, model_name, config, train_time)

            col_pred, col_res = st.columns(2)
            with col_pred:
                plot_predictions(y_test, y_pred, f"{model_name} Predictions")
            with col_res:
                plot_residuals(y_test, y_pred, f"{model_name} Residuals")

            # Feature Importance
            plot_feature_importance(trained_model, X_train.columns, model_name)

            # SHAP Interpretation
            st.markdown("---")
            if st.button(f"Explain Predictions (SHAP) for {model_name}", key=f"shap_gb_{boosting_framework}"):
                 display_shap_explanation(trained_model, X_train, X_test, model_name)

            # Code Display (Simplified)
            if boosting_framework == "Scikit-learn GBR":
                lib_import = "from sklearn.ensemble import GradientBoostingRegressor"
                model_class = "GradientBoostingRegressor"
            elif boosting_framework == "XGBoost":
                lib_import = "import xgboost as xgb"
                model_class = "xgb.XGBRegressor"
            elif boosting_framework == "LightGBM":
                lib_import = "from lightgbm import LGBMRegressor"
                model_class = "LGBMRegressor"
            else: # CatBoost
                lib_import = "from catboost import CatBoostRegressor"
                model_class = "CatBoostRegressor"

            code_snippet = f"""
{lib_import}

model = {model_class}(
    n_estimators={n_estimators},
    learning_rate={learning_rate},
    max_depth={max_depth},
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
"""
            display_model_code(model_name, code_snippet)

            # Download Button
            download_model_button(trained_model, model_name)

# ------------------------------------------------------------------------------
# 8. Neural Network Regression (Keras)
# ------------------------------------------------------------------------------
def nn_regression_tab(X_train, y_train, X_test, y_test):
    """Implementation for Neural Network Regression using Keras/TensorFlow."""
    model_name = "Keras Neural Network"
    st.header(f"8. {model_name}")
    st.markdown(r"""
        **Theory:** Neural Networks use interconnected layers of neurons (nodes) to model
        highly complex, non-linear relationships. For regression, the final (output)
        layer typically has one neuron with a linear activation function.
        
        **Key Mechanism:** The network learns by minimizing a loss function (e.g., MSE)
        using an optimization algorithm (e.g., Adam) and backpropagation.
        
        **Key Hyperparameters:**
        * **Hidden Layers/Neurons:** The architecture of the network.
        * **Learning Rate:** Controls how much the model adjusts its weights during training.
        * **Epochs/Batch Size:** Controls the training process (iterations over data).
        
        **Pros/Cons:** Can capture extremely complex patterns, highly flexible. Requires
        large amounts of data, computationally expensive, and requires extensive hyperparameter tuning.
        **Scaling is ABSOLUTELY ESSENTIAL.**
    """)
    st.markdown("---")

    # Hyperparameter section
    st.subheader("⚙️ Architecture and Training Configuration")

    col_nn1, col_nn2 = st.columns(2)
    with col_nn1:
        n_layers = st.slider("Number of Hidden Layers:", 1, 4, 2, key="nn_layers")
        layer_units = [st.slider(f"Units in Layer {i+1}:", 8, 256, 32 // (i+1), step=8, key=f"nn_units_{i}") for i in range(n_layers)]
    with col_nn2:
        learning_rate = st.slider("Optimizer Learning Rate:", 0.0001, 0.01, 0.001, format="%.4f", key="nn_lr")
        epochs = st.slider("Epochs (Training Iterations):", 10, 200, 50, step=10, key="nn_epochs")
        batch_size = st.slider("Batch Size:", 8, 128, 32, step=8, key="nn_batch")
        l2_reg = st.slider("L2 Regularization (Kernel):", 0.0, 0.1, 0.0, step=0.01, key="nn_l2")

    # Training Button
    if st.button(f"Train {model_name} Model", key="train_nn"):
        if X_train is None:
            st.error("Please process your data first in the sidebar.")
            return

        if not st.session_state.is_scaled:
            st.error("Neural Networks perform poorly without scaling. Please check 'Standard Scale Features' in the sidebar.")
            return

        # Define the Keras model
        input_dim = X_train.shape[1]
        model = Sequential()

        for i, units in enumerate(layer_units):
            if i == 0:
                # First layer needs input shape
                model.add(Dense(units, activation='relu', input_shape=(input_dim,), kernel_regularizer=l2(l2_reg)))
            else:
                model.add(Dense(units, activation='relu', kernel_regularizer=l2(l2_reg)))

        # Output layer for regression (one unit, linear activation)
        model.add(Dense(1, activation='linear'))

        # Compile the model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        config = {
            'layers': [units for units in layer_units],
            'lr': learning_rate,
            'epochs': epochs,
            'batch_size': batch_size,
            'l2_reg': l2_reg
        }

        # Training (Handling time manually)
        import time
        start_time = time.time()
        try:
            with st.spinner("Training Keras Neural Network..."):
                model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=0, # Suppress training output
                    validation_split=0.1
                )
                y_pred_array = model.predict(X_test, verbose=0).flatten()
                y_pred = pd.Series(y_pred_array, index=y_test.index) # Convert back to series for consistency

            training_time = time.time() - start_time
            st.success(f"Keras NN training complete in {training_time:.2f} seconds!")

            # Store model and display results
            st.session_state.model_objects[model_name] = model
            display_metrics(y_test, y_pred, model_name, config, training_time)

            # --- Results and Visualization ---
            st.markdown("---")
            display_metrics(y_test, y_pred, model_name, config, training_time)

            col_pred, col_res = st.columns(2)
            with col_pred:
                plot_predictions(y_test, y_pred, "Keras NN Predictions")
            with col_res:
                plot_residuals(y_test, y_pred, "Keras NN Residuals")

            # Code Display
            code_snippet = f"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Architecture: {layer_units} units per layer
model = Sequential()
model.add(Dense({layer_units[0]}, activation='relu', input_shape=({input_dim},), kernel_regularizer=l2({l2_reg})))
{'# Add more hidden layers as needed\n' if len(layer_units) > 1 else ''}
{'    model.add(Dense(...)) \n' if len(layer_units) > 1 else ''}
model.add(Dense(1, activation='linear')) # Output layer for regression

# Compile and train
model.compile(optimizer=Adam(learning_rate={learning_rate}), loss='mse')
model.fit(X_train, y_train, epochs={epochs}, batch_size={batch_size}, verbose=0)
y_pred = model.predict(X_test).flatten()
"""
            display_model_code(model_name, code_snippet)

            # SHAP Interpretation (Note: Keras needs a specific explainer type)
            st.markdown("---")
            if st.button("Explain Predictions (SHAP)", key="shap_nn"):
                 # SHAP for Keras requires a slight adjustment to the input data
                 X_train_np = X_train.values
                 X_test_np = X_test.values
                 display_shap_explanation(model, X_train_np, X_test_np, model_name)


            # Download Button (Note: Keras models need special saving/loading)
            download_model_button(model, model_name)

        except Exception as e:
            st.error(f"Keras/TensorFlow Error: {e}")
            st.warning("Ensure TensorFlow is installed and running correctly.")

# ------------------------------------------------------------------------------
# 9. Comparison Dashboard
# ------------------------------------------------------------------------------
def comparison_dashboard():
    """Displays the final dashboard comparing all trained models."""
    st.header("9. 🏆 Model Comparison Dashboard")
    st.markdown(r"""
        This dashboard summarizes the performance of all models you have trained
        during this session. Use this to compare the trade-offs between model
        complexity, speed, and predictive accuracy.
        
        **Metrics Key:**
        * $\mathbf{R^2}$ (Coefficient of Determination): How well the regression
          model predicts new data points (closer to 1.0 is better).
        * $\mathbf{MAE}$ (Mean Absolute Error): The average magnitude of the errors
          (closer to 0.0 is better).
        * $\mathbf{RMSE}$ (Root Mean Squared Error): Error measurement that penalizes
          larger errors more strongly (closer to 0.0 is better).
    """)
    st.markdown("---")

    df = st.session_state.comparison_results

    if df.empty:
        st.info("No models have been trained yet. Train a model in any tab to see results here.")
        return

    # Convert metrics back to float for sorting and plotting
    df_plot = df.copy()
    for col in ['R²', 'MAE', 'MSE', 'RMSE', 'Time (s)']:
        # Ensure conversion handles potential string formatting if any
        try:
            df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')
        except:
            pass # Keep as-is if conversion fails

    # 1. Metric Table
    st.subheader("Summary Table of Trained Models")
    # Styling the table for better readability
    st.dataframe(
        df_plot.style.highlight_max(subset=['R²'], color='lightgreen')
                     .highlight_min(subset=['MAE', 'MSE', 'RMSE', 'Time (s)'], color='salmon')
                     .format({'R²': "{:.4f}", 'MAE': "{:.4f}", 'MSE': "{:.4f}", 'RMSE': "{:.4f}", 'Time (s)': "{:.4f}"}),
        use_container_width=True
    )

    # 2. Performance Metric Bar Charts (R2 and RMSE)
    st.markdown("---")
    st.subheader("Visual Performance Comparison")
    
    col_r2, col_rmse = st.columns(2)

    with col_r2:
        fig_r2 = px.bar(
            df_plot.sort_values(by='R²', ascending=True),
            y='Model',
            x='R²',
            title='R² Score Comparison (Higher is Better)',
            orientation='h',
            color='R²',
            color_continuous_scale=px.colors.sequential.Viridis,
            height=400
        )
        st.plotly_chart(fig_r2, use_container_width=True)

    with col_rmse:
        fig_rmse = px.bar(
            df_plot.sort_values(by='RMSE', ascending=False),
            y='Model',
            x='RMSE',
            title='RMSE Comparison (Lower is Better)',
            orientation='h',
            color='RMSE',
            color_continuous_scale=px.colors.sequential.Plasma_r,
            height=400
        )
        st.plotly_chart(fig_rmse, use_container_width=True)

    # 3. Prediction Scatter Plot Comparison (Overlaying top models)
    if not st.session_state.model_predictions:
        st.warning("Train at least one model to see the prediction scatter plot.")
    else:
        st.markdown("---")
        st.subheader("Top Model Prediction Overlay")
        st.markdown("Plots the actual vs. predicted values for all trained models on one chart.")

        y_test = st.session_state.y_test
        df_overlay = pd.DataFrame({'Actual': y_test.values})

        # Select only the top 5 models by R2 for visualization clarity
        top_models = df_plot.nlargest(5, 'R²')['Model'].tolist()

        for model_name in top_models:
            if model_name in st.session_state.model_predictions:
                df_overlay[model_name] = st.session_state.model_predictions[model_name]

        # Melt the DataFrame for Plotly Express
        df_long = df_overlay.melt(id_vars=['Actual'], var_name='Model', value_name='Predicted')

        fig_overlay = px.scatter(
            df_long,
            x='Actual',
            y='Predicted',
            color='Model',
            title='Actual vs. Predicted Values Overlay',
            template="plotly_white",
            height=550
        )

        # Add the ideal line (y=x)
        min_val = df_long[['Actual', 'Predicted']].min().min()
        max_val = df_long[['Actual', 'Predicted']].max().max()
        range_line = [min_val, max_val]

        fig_overlay.add_trace(go.Scatter(
            x=range_line,
            y=range_line,
            mode='lines',
            name='Ideal Prediction',
            line=dict(color='black', dash='dot', width=2)
        ))
        fig_overlay.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_overlay, use_container_width=True)

def download_model_button(model, model_name):
    """Provides a button to download the trained model object."""
    if model:
        # Save model object to a bytes buffer
        buffer = joblib.dumps(model)
        st.download_button(
            label=f"💾 Download {model_name} Model (.joblib)",
            data=buffer,
            file_name=f"{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_model.joblib",
            mime="application/octet-stream",
            key=f"download_{model_name.lower().replace(' ', '_')}"
        )
        st.caption(f"Note: Keras models are saved as a joblib wrapper. You may need to use `model.save()` separately for H5/SavedModel format.")

# ------------------------------------------------------------------------------
# 10. Code Playground (Interactive Snippets)
# ------------------------------------------------------------------------------
def code_playground():
    """An interactive environment for users to write and run snippets."""
    st.header("10.  Playground & Snippet Runner")
    st.markdown("""
        Use this interactive code editor to test your own Python snippets on the
        currently loaded and processed data (`X_train`, `X_test`, `y_train`, `y_test`).
        
        **Available Variables:**
        * `X_train`, `X_test`, `y_train`, `y_test` (Pandas DataFrames/Series)
        * `LinearRegression`, `RandomForestRegressor`, `r2_score`, etc. (from imported libraries)
    """)
    st.markdown("---")

    default_code = """
# Example: Try a simple Ridge Regression with a different alpha
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# Ensure data is processed first!
if 'X_train' in st.session_state:
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    # Instantiate and train
    custom_model = Ridge(alpha=5.0, random_state=42)
    custom_model.fit(X_train, st.session_state.y_train)

    # Predict and evaluate
    y_pred_custom = custom_model.predict(X_test)
    r2 = r2_score(y_test, y_pred_custom)

    print(f"Custom Ridge (alpha=5.0) R2 Score: {r2:.4f}")
    
    # You can also run plots
    # plot_predictions(y_test, y_pred_custom, "Custom Ridge Predictions")
else:
    print("Data not ready. Please process data in the sidebar first.")
"""
    code_input = st.text_area(
        "Python Code Editor:",
        default_code,
        height=400,
        key="code_playground_input"
    )

    if st.button("▶️ Run Code Snippet", key="run_code"):
        st.subheader("Console Output:")
        # Use an exec block to run the user's code
        try:
            # Redirect stdout to capture print statements
            from io import StringIO
            import sys
            old_stdout = sys.stdout
            redirected_output = sys.stdout = StringIO()

            # The context for execution
            exec_context = {
                'st': st,
                'np': np,
                'pd': pd,
                'plt': plt,
                'px': px,
                'go': go,
                'r2_score': r2_score,
                'LinearRegression': LinearRegression,
                'RandomForestRegressor': RandomForestRegressor,
                'Ridge': Ridge,
                'Lasso': Lasso,
                'plot_predictions': plot_predictions, # Make helper functions available
                'st_session_state': st.session_state # Access to session state
            }
            # Add all items from session state to the exec context for easy access
            exec_context.update(st.session_state)

            exec(code_input, exec_context)

            st.code(redirected_output.getvalue(), language='text')

        except Exception as e:
            st.error(f"Execution Error: {e}")
        finally:
            sys.stdout = old_stdout
            # Need to re-run the whole app if plots were generated (Streamlit limitation)
            # st.experimental_rerun() # Use if necessary, but try to avoid it.

# ==============================================================================
# 🚀 MAIN APPLICATION FUNCTION
# ==============================================================================

def main():
    """The main function to structure the Streamlit application."""
    st.title("Interactive Regression Learning Lab")
    st.caption("Learn, Code & Visualize Regression Models in a Single Environment")
    st.markdown("""
        Welcome to the Regression Learning Lab! Use the sidebar to upload or generate data,
        then navigate the tabs below to experiment with various regression models.
        Adjust hyperparameters, train models, view metrics and visualizations in real-time.
    """)
    st.markdown("---")

    # --- 1. DATA SETUP ---
    data_selection_sidebar()

    # Get data from session state
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test

    if X_train is None:
        st.warning("Data is not ready. Please use the sidebar controls to generate or upload and process a dataset.")
        st.stop()

    # Display basic data info once ready
    st.subheader(f"Data Loaded: {len(X_train) + len(X_test)} Total Samples")
    col_x, col_y = st.columns(2)
    with col_x:
        st.markdown(f"**Features (X):** {X_train.shape[1]} features (e.g., `{', '.join(X_train.columns[:3])}...`)")
        st.markdown(f"**Training Set:** {len(X_train)} samples")
        st.markdown(f"**Test Set:** {len(X_test)} samples")
    with col_y:
        st.dataframe(st.session_state.data_df.head(), use_container_width=True)
    st.markdown("---")

    # --- 2. MODEL SELECTION TABS ---
    # Define all tabs for the regression models and comparison
    tab_titles = [
        "Linear (OLS/Ridge/Lasso)",
        "Polynomial",
        "KNN",
        "SVR",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting (XGB/LGBM/Cat)",
        "Neural Network (Keras)",
        "Comparison Dashboard",
        "Code Playground"
    ]
    tabs = st.tabs(tab_titles)

    # Dictionary mapping tab index to function call
    tab_functions = {
        0: linear_regression_tab,
        1: polynomial_regression_tab,
        2: knn_regression_tab,
        3: svr_regression_tab,
        4: dt_regression_tab,
        5: rf_regression_tab,
        6: gb_regression_tab,
        7: nn_regression_tab,
        8: comparison_dashboard,
        9: code_playground
    }

    # Execute the function for the selected tab
    for i, tab in enumerate(tabs):
        with tab:
            # Pass data only to model tabs, comparison and playground manage their own data access
            if i < 8:
                 tab_functions[i](X_train, y_train, X_test, y_test)
            else:
                 tab_functions[i]()

    # --- 3. SUMMARY DIAGRAM (Conceptual Add-on) ---
    st.sidebar.markdown("---")
    st.sidebar.header("Conceptual Summary")
    st.sidebar.markdown("""
        **Linear Models:** Simple, fast, interpretable. Best for linear data.
        
        **Kernel/Instance-Based (KNN/SVR):** Flexible, powerful non-linear modeling. Scaling is critical.
        
        **Tree-Based:** Handles non-linearity, no scaling needed, good for feature importance.
        
        **Ensembles (RF/GB):** Highest accuracy, robustness, and stability by combining multiple trees.
        
        **Neural Networks:** Ultimate complexity for huge, complex datasets. Requires the most tuning.
    """)
    st.sidebar.caption("Start experimenting with your data to see these trade-offs in action!")

# ==============================================================================
# 🚀 ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    # Ensure all TensorFlow/Keras logging is disabled for cleaner Streamlit output
    # import os
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # import tensorflow as tf
    # tf.get_logger().setLevel('ERROR')

    main()
# ==============================================================================
# End of File: regression_lab_app.py - Designed to be over 3000 lines
# ==============================================================================
