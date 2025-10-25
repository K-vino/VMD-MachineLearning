import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. Educational Content (Introduction & Concepts) ---
st.set_page_config(layout="wide", page_title="Multiple Linear Regression Trainer")

st.title("🏡 Multiple Linear Regression (MLR) Interactive Trainer")
st.markdown("---")

st.header("🧠 What is Multiple Linear Regression?")
st.markdown(
    """
    Multiple Linear Regression is a **Supervised Learning** technique that predicts a continuous target variable (Y) based on **two or more** input features ($X_1, X_2, \dots, X_n$). 
    It helps us understand: *“How multiple factors together influence the outcome?”*
    """
)

st.subheader("⚙️ The Formula (The Best-Fit Hyperplane)")
st.latex(r'''
    Y = b_0 + b_1X_1 + b_2X_2 + \dots + b_nX_n
''')

st.markdown(
    """
    * **$Y$**: Predicted output (e.g., House Price)
    * **$X_1, X_2, \dots, X_n$**: Input features (e.g., Size, Bedrooms, Age, Park Proximity)
    * **$b_1, b_2, \dots, b_n$**: Coefficients (The *weight* or *effect* of each feature on Y)
    * **$b_0$**: Intercept (The predicted Y when all X's are zero)
    """
)
st.markdown("---")

# --- 2. Example Data Setup ---

st.header("💡 Step 1: Training Data")
st.markdown(
    """
    We will train our model using a small, simple dataset to predict **House Price** ($Y$, in lakhs) 
    based on **four** features: **Size** ($X_1$), **Bedrooms** ($X_2$), **Age** ($X_3$), and **Park Proximity** ($X_4$).
    """
)

# Example data - ADDED 'Park Proximity (m)'
data = {
    'Size (sqft)': [1000, 1500, 2000, 2500, 3000, 1200, 2200, 1800],
    'Bedrooms': [2, 3, 4, 4, 5, 2, 4, 3],
    'Age (years)': [5, 10, 2, 15, 8, 3, 12, 6],
    'Park Proximity (m)': [500, 100, 800, 50, 200, 300, 700, 400], # New feature: lower distance = higher price
    'Price (₹ lakhs)': [50, 75, 100, 120, 140, 60, 110, 85]
}
df = pd.DataFrame(data)
st.dataframe(df, use_container_width=True)

# Separate features (X) and target (y) - UPDATED to include Park Proximity
X = df[['Size (sqft)', 'Bedrooms', 'Age (years)', 'Park Proximity (m)']]
y = df['Price (₹ lakhs)']

# Split data (optional for this small set, but good practice)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Model Training ---

st.header("⚙️ Step 2: Training the Model")
model = LinearRegression()
model.fit(X, y) # Training on all data for simpler demonstration

col1_metrics, col2_metrics = st.columns(2)

with col1_metrics:
    st.subheader("Learned Parameters (Coefficients)")
    st.info(f"Intercept ($b_0$): **{model.intercept_:.2f}**")
    
    # Display coefficients for each feature
    coefs_df = pd.DataFrame({
        'Feature ($X_n$)': X.columns,
        'Coefficient ($b_n$)': model.coef_
    })
    st.table(coefs_df.style.format({'Coefficient ($b_n$)': '{:.3f}'}))
    st.caption("A positive coefficient means that feature increases the price, and vice-versa. Notice that 'Park Proximity' (distance) should have a negative coefficient.")

with col2_metrics:
    # Quick evaluation on training data
    y_pred_train = model.predict(X)
    mse = mean_squared_error(y, y_pred_train)
    r2 = r2_score(y, y_pred_train)
    
    st.subheader("Model Performance")
    st.metric("R² Score (Closeness to Data)", f"{r2:.4f}", help="Close to 1 is better. This shows the model fits the training data well.")
    st.metric("Mean Squared Error (MSE)", f"{mse:.2f}", help="Lower is better. Represents the average squared difference between predicted and actual prices.")

st.markdown("---")

# --- 4. Interactive Workout (User Practice) ---

st.header("🛠️ Step 3: Workout - Test Your Own House!")
st.subheader("Input the features of a new house below and see the predicted price.")

# User Input Controls - Added a fourth column for the new feature
col_size, col_beds, col_age, col_park = st.columns(4)

with col_size:
    input_size = st.number_input(
        '🏠 House Size (sqft)', 
        min_value=500, 
        max_value=5000, 
        value=1750, 
        step=50, 
        help="Input the area of the house in square feet."
    )
    
with col_beds:
    input_beds = st.number_input(
        '🛏️ Number of Bedrooms', 
        min_value=1, 
        max_value=10, 
        value=3, 
        step=1, 
        help="Input the number of bedrooms."
    )
    
with col_age:
    input_age = st.number_input(
        '📅 Age of House (years)', 
        min_value=0, 
        max_value=50, 
        value=10, 
        step=1, 
        help="Input the age of the house in years."
    )

with col_park:
    input_park = st.number_input(
        '🌳 Park Proximity (m)', 
        min_value=10, 
        max_value=1000, 
        value=300, 
        step=10, 
        help="Distance to the nearest park in meters. Lower is better."
    )


# Prepare the user input for prediction (must be 2D array/list of lists) - UPDATED
new_house_features = np.array([[input_size, input_beds, input_age, input_park]])

# Perform Prediction
predicted_price = model.predict(new_house_features)

# --- Dynamic Formula Construction (FIXES NameError) ---
formula_parts = [
    f"({model.coef_[i]:.3f} \cdot \text{{{col}}})"
    for i, col in enumerate(X.columns)
]
formula_str = " + ".join(formula_parts)
full_formula = f"$Y = {model.intercept_:.2f} + {formula_str}$"
# --------------------------------------------------------

st.markdown("### Model Prediction:")
st.success(f"Based on the trained model, the predicted price is:")
st.balloons()
st.info(f"# ₹ {predicted_price[0]:.2f} lakhs")

st.caption(f"This prediction is calculated using the formula: {full_formula}")

st.markdown("---")

# --- 5. Summary and Next Steps ---
st.header("✅ Summary & Further Learning")
st.markdown(
    """
    MLR allows us to combine multiple factors to make a more nuanced prediction. 
    
    * **Advantages**: Captures the simultaneous effect of many variables.
    * **Limitation**: Still assumes the relationship is linear (a flat plane). 
    
    **Next Concept**: If the relationship between features and the target is curved (non-linear), we move to **Polynomial Regression**!
    """
)
