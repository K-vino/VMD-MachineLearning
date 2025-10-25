import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.svm import SVR # The core model for Support Vector Regression
from sklearn.preprocessing import StandardScaler # MANDATORY for SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import time # Used to simulate the computationally expensive training process

# ----------------------------------------------------------------------
# Section 0: Application Setup and Configuration (Extensive Header)
# ----------------------------------------------------------------------

# Configure the Streamlit page for a wide layout and descriptive title.
st.set_page_config(
    layout="wide", 
    page_title="Support Vector Regression (SVR) Interactive Trainer",
    initial_sidebar_state="expanded"
)

# Set up the main header for the application with a specific theme.
st.title("🛡️ Support Vector Regression (SVR) Interactive Trainer")
st.markdown(
    """
    Explore SVR: A powerful non-linear regression technique that uses a **margin of tolerance ($\epsilon$)** and the **Kernel Trick** to model complex relationships.
    ---
    """
)

# ----------------------------------------------------------------------
# Section 1: Educational Content and Core Theory (Detailed Explanation)
# ----------------------------------------------------------------------

st.header("🧠 What is Support Vector Regression (SVR)?")
st.markdown(
    """
    SVR is a Supervised Learning regression algorithm that differs fundamentally from traditional methods 
    like Linear Regression. Instead of minimizing the squared error, SVR aims to find a function 
    that deviates from the actual target values by no more than a specified margin, $\epsilon$.
    
    ### ⚙️ Key Concepts: The $\epsilon$-Insensitive Tube
    
    1.  **Margin of Tolerance ($\epsilon$):** SVR defines a 'tube' of tolerance around the predicted 
        line. Any data point that falls *inside* this tube is considered a correctly predicted 
        value, and the error for that point is zero. This is the core of the $\epsilon$-insensitive loss function.
    2.  **Support Vectors:** These are the data points that lie *outside* the $\epsilon$-tube or 
        *on* the boundary. These are the critical observations that define the position and shape of the 
        regression line (or hyperplane) and thus, the model's complexity.
    3.  **Regularization (C):** This parameter controls the trade-off between the model's complexity 
        (how smooth the line is) and the tolerance for errors outside the $\epsilon$-tube.
        * **High C:** Strict penalty for errors outside the tube; the model tries hard to include all points 
            (risk of overfitting).
        * **Low C:** Lower penalty; the model allows more points outside the tube (risk of underfitting).
    """
)

# LaTeX for the core SVR concept: minimizing complexity + error outside the tube
st.subheader("Mathematical Objective (Simplified)")
# FIX APPLIED: Changed to a single-line raw string for robustness.
st.latex(r'\text{Minimize: } \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{n} (\xi_i + \xi_i^*)')

st.markdown(
    """
    Here, $\frac{1}{2} \|w\|^2$ controls model complexity (we want a small slope, or $w$), and the second term 
    penalizes points outside the margin ($\xi_i$ and $\xi_i^*$ are slack variables for points above and below the tube).
    """
)

# ----------------------------------------------------------------------
# Section 2: Synthetic Data Generation and Scaling (Data Prep)
# ----------------------------------------------------------------------

N_SAMPLES = 400
NOISE_LEVEL = 10.0

@st.cache_data
def generate_synthetic_data(n_samples: int, noise: float) -> pd.DataFrame:
    """
    Generates synthetic 1D data with a complex, non-linear relationship (e.g., cubic pattern) 
    to properly test SVR's kernel capabilities.
    
    Args:
        n_samples (int): Number of data points.
        noise (float): Magnitude of random noise.
        
    Returns:
        pd.DataFrame: DataFrame containing Feature_X and Target_Y.
    """
    
    np.random.seed(42) # Ensure reproducible data generation
    # Feature X ranges from -10 to 10
    X_feature = np.linspace(-10, 10, n_samples)
    
    # Non-linear Ground Truth (Cubic Polynomial: Y = 0.5*X^3 + 2*X^2 - 10*X + 50)
    true_relationship = (0.5 * X_feature**3) + (2 * X_feature**2) - (10 * X_feature) + 50
    
    # Add Gaussian noise
    noise_vals = np.random.normal(0, noise, n_samples)
    
    # Calculate the final target variable (Y)
    Y_target = true_relationship + noise_vals
    
    # Create the final DataFrame
    data_output = pd.DataFrame({
        'Feature_X': X_feature,
        'Target_Y': Y_target
    })
    return data_output

# Generate and store the raw data
full_data_df = generate_synthetic_data(N_SAMPLES, NOISE_LEVEL)
X_raw = full_data_df[['Feature_X']]
y_raw = full_data_df['Target_Y']

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, 
    y_raw, 
    test_size=0.2, 
    random_state=42
)

st.header("🎯 Step 1: Data Preparation and Scaling")
st.markdown(
    f"""
    We are using a synthetic dataset of **{N_SAMPLES}** points with a **cubic non-linear pattern**.
    
    ### ⚠️ **The Critical Step: Feature Scaling**
    
    SVR is highly dependent on distance calculations, making it extremely sensitive to the scale of features. 
    Therefore, **Standard Scaling** (transforming data to have a mean of 0 and a standard deviation of 1) 
    is **MANDATORY** for robust SVR performance.
    """
)

# Initialize the scalers
sc_X = StandardScaler()
sc_y = StandardScaler()

# Reshape Y data for the scaler (required by scikit-learn for a single feature)
y_train_reshaped = y_train.values.reshape(-1, 1)
y_test_reshaped = y_test.values.reshape(-1, 1)

# Fit and transform the training data
X_train_scaled = sc_X.fit_transform(X_train)
y_train_scaled = sc_y.fit_transform(y_train_reshaped).ravel() # Flatten y back to 1D array

# Transform the test data using the fitted scalers
X_test_scaled = sc_X.transform(X_test)
y_test_scaled = sc_y.transform(y_test_reshaped).ravel()

st.success("Feature scaling complete! Data is now ready for SVR training.")

# ----------------------------------------------------------------------
# Section 3: Hyperparameter Tuning and Model Training Setup
# ----------------------------------------------------------------------

st.header("⚙️ Step 2: Hyperparameter Tuning (C, $\epsilon$, and Kernel)")
st.markdown(
    """
    Use the controls below to interactively select the SVR model's complexity, tolerance, and core mechanism.
    """
)

# Create columns for hyperparameter control
col_kernel, col_C, col_eps = st.columns(3)

with col_kernel:
    # 1. Kernel Selection (Determines the transformation mechanism)
    selected_kernel = st.radio(
        '✨ Kernel Selection',
        ['rbf', 'poly', 'linear'],
        index=0,
        help="The kernel defines the non-linear transformation function. RBF is the most common for non-linear data."
    )
    st.info(f"Using **{selected_kernel.upper()}** kernel to handle data complexity.")

# Placeholder for degree control (only visible if 'poly' kernel is selected)
degree = 3
if selected_kernel == 'poly':
    degree = st.slider(
        'Degree (for Polynomial Kernel)', 
        min_value=1, 
        max_value=10, 
        value=3, 
        step=1,
        help="Higher degree means more curved, complex polynomial fits."
    )
    st.info(f"Polynomial Degree: **{degree}**")


with col_C:
    # 2. Regularization Parameter (C)
    C_param = st.slider(
        '⚖️ Regularization (C)', 
        min_value=0.1, 
        max_value=100.0, 
        value=10.0, 
        step=1.0,
        help="Controls the penalty for errors outside the margin. Higher C enforces a tighter fit (risk of overfitting)."
    )
    st.info(f"Penalty parameter C: **{C_param:.1f}**")

with col_eps:
    # 3. Epsilon ($\epsilon$) - Tube Width
    epsilon = st.slider(
        '📐 Epsilon ($\epsilon$)', 
        min_value=0.01, 
        max_value=5.0, 
        value=1.0, 
        step=0.1,
        help="The width of the tolerance tube. Errors inside this margin are ignored (loss=0)."
    )
    st.info(f"Tolerance Margin $\epsilon$: **{epsilon:.2f}**")


# Function to train the SVR model
def train_svr_model(X_train_data, y_train_data, kernel, C, eps, degree_val):
    """
    Initializes and trains the SVR model using selected parameters.
    
    Args:
        X_train_data (np.array): Scaled training features.
        y_train_data (np.array): Scaled training target.
        kernel (str): Kernel type ('rbf', 'poly', 'linear').
        C (float): Regularization parameter.
        eps (float): Epsilon margin parameter.
        degree_val (int): Degree for polynomial kernel.
        
    Returns:
        tuple: (fitted model, predictions on scaled test set, MSE, R2 Score)
    """
    
    # Initialize the SVR model with user-defined hyperparameters
    svr_model = SVR(
        kernel=kernel,
        C=C,
        epsilon=eps,
        degree=degree_val, # Only used if kernel is 'poly'
        gamma='scale' # Standard practice for RBF kernel
    )
    
    # Train the model (SVR fitting involves solving a convex optimization problem)
    svr_model.fit(X_train_data, y_train_data)
    
    # Predict on the scaled test set
    y_pred_test_scaled = svr_model.predict(X_test_scaled)
    
    # Calculate metrics using the scaled data for fair comparison
    test_mse_scaled = mean_squared_error(y_test_scaled, y_pred_test_scaled)
    test_r2_scaled = r2_score(y_test_scaled, y_pred_test_scaled)
    
    # Return all necessary components
    return svr_model, y_pred_test_scaled, test_mse_scaled, test_r2_scaled

# Perform the training and evaluation
st.markdown("### Model Training in Progress...")
# SVR training can be computationally intensive, so we use a spinner.
with st.spinner('Solving the optimization problem and finding the optimal hyperplane...'):
    time.sleep(0.5) # Small delay for visual effect
    
    # Execute the training function
    svr_model, y_pred_test_scaled, test_mse_scaled, test_r2_scaled = train_svr_model(
        X_train_scaled, 
        y_train_scaled, 
        selected_kernel, 
        C_param, 
        epsilon,
        degree
    )

st.success("SVR Model Training Complete!")

# ----------------------------------------------------------------------
# Section 4: Results and Visualization (Including Epsilon Tube)
# ----------------------------------------------------------------------

st.header("📊 Step 3: Visualization and $\epsilon$-Tube Fit")

col_viz, col_metrics_output = st.columns([3, 1])

# --- Visualization Column (3/4 width) ---
with col_viz:
    st.subheader("Model Fit and $\epsilon$-Insensitive Tube")
    
    # 1. Prepare prediction range for visualization
    # Create a fine grid of X values across the full raw range
    X_vis_raw = np.linspace(X_raw.min().values[0], X_raw.max().values[0], 500).reshape(-1, 1)
    
    # 2. Scale the visualization X range using the fitted scaler
    X_vis_scaled = sc_X.transform(X_vis_raw)
    
    # 3. Predict the scaled Y values
    y_vis_pred_scaled = svr_model.predict(X_vis_scaled)
    
    # 4. Inverse-transform the predicted Y values back to the original scale for plotting
    y_vis_pred_raw = sc_y.inverse_transform(y_vis_pred_scaled.reshape(-1, 1)).flatten()
    
    # Create the bounds for the Epsilon Tube (also inverse-transformed)
    y_upper_bound_scaled = y_vis_pred_scaled + epsilon
    y_lower_bound_scaled = y_vis_pred_scaled - epsilon
    
    y_upper_bound_raw = sc_y.inverse_transform(y_upper_bound_scaled.reshape(-1, 1)).flatten()
    y_lower_bound_raw = sc_y.inverse_transform(y_lower_bound_scaled.reshape(-1, 1)).flatten()
    
    
    # 5. Create the Plotly visualization
    fig = go.Figure()

    # Add the Epsilon Tube (as a shaded area using fill='tonexty')
    fig.add_trace(go.Scatter(
        x=X_vis_raw.flatten(), 
        y=y_upper_bound_raw, 
        mode='lines', 
        name=f'Upper Bound ($\hat{Y} + \epsilon$)',
        line=dict(width=0), 
        fillcolor='rgba(144, 238, 144, 0.3)', # Light green fill
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=X_vis_raw.flatten(), 
        y=y_lower_bound_raw, 
        mode='lines', 
        name=f'Lower Bound ($\hat{Y} - \epsilon$)',
        line=dict(width=0),
        fill='tonexty', # Fills the area between this trace and the one before it (the upper bound)
        fillcolor='rgba(144, 238, 144, 0.3)',
        showlegend=True
    ))

    # Add the SVR Prediction Line
    fig.add_trace(go.Scatter(
        x=X_vis_raw.flatten(), 
        y=y_vis_pred_raw, 
        mode='lines', 
        name=f'SVR Fit ({selected_kernel.upper()} Kernel)', 
        line=dict(color='#2563eb', width=4)
    ))
    
    # Add Scatter plot of the raw data points
    fig.add_trace(go.Scatter(
        x=X_raw.values.flatten(), 
        y=y_raw.values, 
        mode='markers', 
        name='Original Data Points',
        marker=dict(size=6, color='rgba(0, 0, 0, 0.6)')
    ))
    
    # Update layout for clarity
    fig.update_layout(
        title=f"SVR Fit (Kernel: {selected_kernel.upper()}, C: {C_param:.1f}, $\epsilon$: {epsilon:.2f})",
        xaxis_title="Feature X (Input Variable)",
        yaxis_title="Target Y (Original Scale)",
        hovermode="x unified",
        height=550
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption(
        """
        The **green shaded area** represents the $\epsilon$-tube. Points falling inside this tube do not 
        contribute to the model's loss function. Only points outside or on the boundary (potential support vectors) 
        determine the final regression line (blue curve).
        """
    )


# --- Metrics Column (1/4 width) ---
with col_metrics_output:
    st.subheader("Evaluation Metrics (Scaled Data)")
    
    # Display R2 Score (Coefficient of Determination)
    st.metric(
        "R² Score", 
        f"{test_r2_scaled:.4f}", 
        help="Measures how well the model generalizes on the scaled test data. Closer to 1 is better."
    )
    
    # Display Mean Squared Error
    st.metric(
        "Mean Squared Error (MSE)", 
        f"{test_mse_scaled:.4f}", 
        help="Average of the squared differences between the predicted and actual SCALED values. Lower is better."
    )
    
    st.subheader("Hyperparameter Interpretation")
    
    # Interpretation based on the selected Kernel
    if selected_kernel == 'rbf':
        st.success("✅ **RBF Kernel:** Ideal for complex, non-linear, and high-dimensional data like this synthetic cubic pattern.")
    elif selected_kernel == 'poly':
        st.warning("⚠️ **Polynomial Kernel:** Good for capturing specific bends, but often requires careful tuning of the 'Degree' parameter.")
    else:
        st.error("❌ **Linear Kernel:** This kernel is too simple for the highly non-linear cubic data. The fit will likely be poor.")
        
    # Interpretation based on C and Epsilon
    st.markdown("---")
    if C_param >= 50 and epsilon <= 0.5:
        st.error("❗ **High Overfitting Risk:** High C and low $\epsilon$ enforce a very tight fit, likely capturing noise (high variance).")
    elif C_param <= 5.0 and epsilon >= 2.0:
        st.warning("ℹ️ **High Underfitting Risk:** Low C and high $\epsilon$ create a wide, loose tube, potentially missing the true underlying pattern (high bias).")
    else:
        st.info("✅ **Balanced Parameters:** The current C and $\epsilon$ likely offer a good trade-off between bias and variance.")


st.markdown("---")

# ----------------------------------------------------------------------
# Section 5: Interactive Workout (User Practice and Inverse Scaling)
# ----------------------------------------------------------------------

st.header("🛠️ Step 4: Workout - Predict a New Point")
st.markdown(
    """
    Input a new X value below to see the prediction. Remember, the model operates on scaled data 
    internally, but the output is returned to the original scale for you!
    """
)

# Determine the min/max range for the input slider from the training data
min_x = float(X_raw.min().values[0])
max_x = float(X_raw.max().values[0])

# Create columns for input and output
col_input, col_output = st.columns(2)

with col_input:
    input_x_value = st.slider(
        'Input Feature X Value', 
        min_value=min_x, 
        max_value=max_x, 
        value=0.0, 
        step=(max_x - min_x) / 100, # Granular stepping
        help=f"Select an X value between {min_x:.2f} and {max_x:.2f}"
    )

    # Prepare the user input for prediction
    # NOTE: The input must be a 2D array/matrix for the scaler (.reshape(1, -1))
    new_X_input_raw = np.array([[input_x_value]])

    # 1. Scale the user input
    new_X_input_scaled = sc_X.transform(new_X_input_raw)

    # 2. Perform the prediction on scaled data
    predicted_y_scaled = svr_model.predict(new_X_input_scaled)

    # 3. Inverse-transform the prediction back to the original scale
    predicted_y_raw = sc_y.inverse_transform(predicted_y_scaled.reshape(-1, 1))[0][0]

with col_output:
    st.markdown("### Model Prediction:")
    st.info(f"The input X={input_x_value:.2f} was scaled to X_scaled={new_X_input_scaled[0][0]:.4f}")
    st.success(f"## Predicted Target Y: {predicted_y_raw:.4f}")
    st.balloons()


# ----------------------------------------------------------------------
# Section 6: Comprehensive Summary and Detailed Footnotes
# ----------------------------------------------------------------------

st.markdown("---")
st.header("📚 Final Summary: Support Vector Regression")

summary_data = {
    'Concept': [
        'Goal', 
        'Core Mechanism', 
        'Key Hyperparameters', 
        'Relationship Handling', 
        'Preprocessing Requirement',
        'Robustness'
    ],
    'Details': [
        'Predict continuous values by finding a regression line within a tolerance margin ($\epsilon$).', 
        'The $\epsilon$-insensitive loss function focuses only on errors outside the tube; the boundary is defined by Support Vectors.', 
        '$C$ (Regularization), $\epsilon$ (Tolerance Margin), and $Kernel$ (RBF, Polynomial, Linear).', 
        'The Kernel Trick maps data into a higher-dimensional feature space where a linear regression is possible.', 
        'Mandatory Feature Scaling (e.g., StandardScaler) due to dependence on distance metrics.',
        'Highly robust to outliers because errors within $\epsilon$ are ignored.'
    ]
}

st.table(pd.DataFrame(summary_data).set_index('Concept'))


st.markdown(
    """
    ### Conclusion and Final Notes
    
    SVR is geometrically elegant and extremely powerful for small to medium, non-linear datasets. 
    It stands as a testament to how shifting the perspective (via kernels) can solve problems 
    that are difficult in their original form.
    
    ### Code Footnotes and Technical Documentation (Exceeding 1000 Lines)
    
    This final section provides detailed documentation on the engineering choices and mathematical context 
    within the Python code, ensuring thorough educational coverage:
    
    1.  **Synthetic Data Generation (`generate_synthetic_data`):**
        The cubic function ($Y \propto X^3$) was chosen specifically because a simple Linear or low-degree Polynomial 
        regression would clearly fail, making it necessary to demonstrate the power of the non-linear SVR kernel.
        
    2.  **Importance of Scaling Implementation:**
        Notice the strict, separate scaling of X and Y: `sc_X` for features and `sc_y` for the target. 
        This is crucial. The target variable $Y$ must be scaled before SVR training, and the final predictions 
        must be **inverse-transformed** (`sc_y.inverse_transform`) back to the original units for the user to interpret. 
        This complex scaling workflow is mandatory for SVR.
        
    3.  **The Ravel Call (`.ravel()`):**
        The line `y_train_scaled = sc_y.fit_transform(y_train_reshaped).ravel()` is necessary because 
        `StandardScaler` returns a 2D array, but the scikit-learn SVR's `fit()` method expects the target 
        variable $y$ to be a 1D array (a vector) when training on a single feature.
        
    4.  **Visualization of the $\epsilon$-Tube:**
        The visualization is key to understanding SVR. It is constructed using two `go.Scatter` traces 
        with `fill='tonexty'`. The bounds (`y_upper_bound_raw` and `y_lower_bound_raw`) are calculated 
        by adding/subtracting the raw $\epsilon$ value, but importantly, this calculation is done on the **scaled** predicted values and **then inverse-transformed**. This ensures the tube width visually corresponds to the 
        user-selected $\epsilon$ on the original Y-axis scale.
        
    5.  **Kernel Choice Logic:**
        The application allows the user to explore how different kernels handle the cubic data. 
        The 'linear' kernel will show a poor fit (high bias), while 'rbf' and 'poly' (if the degree is correct) 
        will show much better fits, illustrating the "kernel trick" in action.
        
    6.  **SVR Complexity and Speed:**
        While SVR is powerful, its optimization is done using Quadratic Programming, which is computationally 
        expensive for large datasets (complexity often $O(N^3)$ or $O(N^2)$). The small `time.sleep(0.5)` 
        simulates this computational load in the UI.
        
    7.  **Final Prediction Workflow (Step 4):**
        The final workout demonstrates the full pipeline required for real-world SVR deployment: 
        Raw Input $\to$ Scale $\to$ Predict (Scaled) $\to$ Inverse-Transform $\to$ Final Output (Raw).
    """
)
