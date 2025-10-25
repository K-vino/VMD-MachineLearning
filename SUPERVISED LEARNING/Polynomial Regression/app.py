import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. Educational Content (Introduction & Concepts) ---
st.set_page_config(layout="wide", page_title="Polynomial Regression Trainer")

st.title("📈 Polynomial Regression Interactive Trainer")
st.markdown("---")

st.header("🧠 What is Polynomial Regression?")
st.markdown(
    """
    Polynomial Regression is a powerful extension of Linear Regression used when the relationship between the 
    input ($X$) and the output ($Y$) is **non-linear** (curved).
    
    **💡 Key takeaway:** If your data looks curved, Linear Regression will fail, and Polynomial Regression is your solution.
    """
)

st.subheader("⚙️ The Formula (Degree $n$)")
st.latex(r'''
    Y = b_0 + b_1X + b_2X^2 + b_3X^3 + \dots + b_nX^n
''')

st.markdown(
    """
    The **Degree ($n$)** of the polynomial is the key parameter, controlling the curve's flexibility:
    * Degree 1: Simple Linear Regression ($Y = b_0 + b_1X$).
    * Degree 2: Quadratic Curve ($Y = b_0 + b_1X + b_2X^2$).
    * Higher degrees allow for more bends, increasing the risk of **overfitting**!
    """
)
st.markdown("---")

# --- 2. Example Data Setup ---

st.header("💡 Step 1: Data and Degree Selection")
st.markdown(
    """
    We will use a simple, non-linear dataset where **Sales** grow faster as **Ad Spend** increases.
    """
)

# Example data
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1) # Ad Spend (₹k)
y = np.array([2, 5, 9, 16, 25, 38, 55, 76, 102, 130]) # Sales (₹k)

data_df = pd.DataFrame({
    'Ad Spend (₹k)': X.flatten(),
    'Sales (₹k)': y
})
st.dataframe(data_df, use_container_width=True)


# User chooses the polynomial degree
st.subheader("Select the Degree (n)")
degree = st.slider(
    'Adjust the degree to see how the curve fits the data:', 
    min_value=1, 
    max_value=9, 
    value=2, 
    step=1
)
st.markdown(f"**Current Degree Selected:** **{degree}**")
st.markdown("---")


# --- 3. Model Training and Visualization ---

st.header(f"⚙️ Step 2: Training and Visualization (Degree $n={degree}$)")

# 1. Transform features
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)

# 2. Train model
model = LinearRegression()
model.fit(X_poly, y)

# 3. Create smooth line for plotting the fitted curve
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_range_poly = poly.transform(X_range)
y_range_pred = model.predict(X_range_poly)

# Prepare plot data
plot_df = pd.DataFrame({
    'Ad Spend (₹k)': X_range.flatten(),
    'Predicted Sales (₹k)': y_range_pred
})

# Create Plotly figure
fig = px.scatter(
    data_df, 
    x='Ad Spend (₹k)', 
    y='Sales (₹k)', 
    title=f"Polynomial Regression Fit (Degree {degree})"
)
# Add the fitted curve
fig.add_scatter(
    x=plot_df['Ad Spend (₹k)'], 
    y=plot_df['Predicted Sales (₹k)'], 
    mode='lines', 
    name=f'Fitted Curve (Degree {degree})', 
    line=dict(color='red', width=3)
)
fig.update_layout(xaxis_title="Ad Spend (₹k)", yaxis_title="Sales (₹k)")

# Display Plot
st.plotly_chart(fig, use_container_width=True)
st.caption("Try setting the degree to 1 (underfitting), 2 or 3 (good fit), and 9 (overfitting).")

# 4. Display Metrics
y_pred_train = model.predict(X_poly)
mse = mean_squared_error(y, y_pred_train)
r2 = r2_score(y, y_pred_train)

col1_metrics, col2_metrics = st.columns(2)

with col1_metrics:
    st.subheader("Model Performance")
    st.metric("R² Score (Closeness to Data)", f"{r2:.4f}", help="Close to 1 is better. Note how it approaches 1.0 with high degrees (overfitting).")
    st.metric("Mean Squared Error (MSE)", f"{mse:.2f}", help="Lower is better.")

with col2_metrics:
    st.subheader("Interpretation")
    if degree == 1:
        st.warning("Degree 1 (Linear) is too simple; it **underfits** the curved data.")
    elif degree == 2 or degree == 3:
        st.success("Degrees 2 or 3 typically provide a good balance for this type of curve.")
    elif degree >= 8:
        st.error("High degrees lead to **overfitting**. The curve fits the training points perfectly but fails to generalize to new data.")
    else:
        st.info("The model is currently using a degree that balances fit and complexity.")

st.markdown("---")

# --- 4. Interactive Workout (User Practice) ---

st.header("🛠️ Step 3: Workout - Predict New Sales!")
st.subheader("Input a new Ad Spend value and see the predicted Sales.")

input_ad_spend = st.number_input(
    '💰 New Ad Spend (₹k)', 
    min_value=0.0, 
    max_value=20.0, 
    value=12.0, 
    step=0.5, 
    help="Enter an Ad Spend value between 0 and 20."
)

# Prepare the user input for prediction
new_X = np.array([[input_ad_spend]])

# Transform the new input using the *same* PolynomialFeatures object
new_X_poly = poly.transform(new_X)

# Perform Prediction
predicted_sales = model.predict(new_X_poly)

st.markdown("### Model Prediction:")
st.success(f"Based on the Degree {degree} polynomial model:")
st.balloons()
st.info(f"# Sales: ₹ {predicted_sales[0]:.2f}k")

st.markdown("---")

# --- 5. Summary and Conclusion ---
st.header("🧩 Summary")
st.table(pd.DataFrame({
    'Feature': ['Polynomial Regression', 'Relationship', 'Key Parameter', 'Advantage'],
    'Value': ['Predict continuous value with a curve', 'Non-linear (curved)', 'Degree of the polynomial', 'Can model complex non-linear relationships']
}).set_index('Feature'))
