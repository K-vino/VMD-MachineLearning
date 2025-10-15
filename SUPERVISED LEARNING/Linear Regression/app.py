# -----------------------------------------------------------
# STREAMLIT APP: LINEAR REGRESSION - HOUSE PRICE PREDICTION
# -----------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------------------------------------
# 🌟 Page Config
# -----------------------------------------------------------
st.set_page_config(page_title="House Price Predictor", layout="wide", page_icon="🏠")

# -----------------------------------------------------------
# 🏠 App Title
# -----------------------------------------------------------
st.title("🏠 House Price Prediction using Linear Regression")
st.markdown("""
This web app predicts **house prices** based on the **size of the house**.
It uses **Linear Regression** to find the best fit line between House Size and Price.
""")

# -----------------------------------------------------------
# 📊 Dataset
# -----------------------------------------------------------
st.header("📊 Sample Dataset")
data = {
    'House_Size': [1000, 1500, 2000, 2500, 3000],
    'Price': [200000, 250000, 300000, 350000, 400000]
}
df = pd.DataFrame(data)
st.dataframe(df)

# -----------------------------------------------------------
# ⚙️ Linear Regression Model
# -----------------------------------------------------------
X = df[['House_Size']]
y = df['Price']

model = LinearRegression()
model.fit(X, y)

intercept = model.intercept_
slope = model.coef_[0]

st.markdown(f"**Regression Equation:** Price = {intercept:.2f} + {slope:.2f} × House_Size")

# -----------------------------------------------------------
# 🔮 Prediction Input
# -----------------------------------------------------------
st.header("🔮 Predict Price for New House Size")
house_size = st.number_input("Enter House Size (sq.ft):", min_value=500, max_value=10000, value=2000, step=100)

predicted_price = model.predict([[house_size]])[0]
st.success(f"Predicted House Price: ₹{predicted_price:,.2f}")

# -----------------------------------------------------------
# 📈 Visualization
# -----------------------------------------------------------
st.header("📈 Visualization")
fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(X, y, color='blue', label='Actual Data')
ax.plot(X, model.predict(X), color='red', label='Regression Line')
ax.scatter(house_size, predicted_price, color='green', s=100, label='Predicted Price')
ax.set_xlabel("House Size (sq.ft)")
ax.set_ylabel("Price (₹)")
ax.set_title("Linear Regression - House Price Prediction")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# -----------------------------------------------------------
# 📏 Model Evaluation
# -----------------------------------------------------------
st.header("📏 Model Evaluation Metrics")

y_pred = model.predict(X)
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

st.write(f"- **Mean Absolute Error (MAE):** {mae:.2f}")
st.write(f"- **Mean Squared Error (MSE):** {mse:.2f}")
st.write(f"- **Root Mean Squared Error (RMSE):** {rmse:.2f}")
st.write(f"- **R² Score:** {r2:.2f}")

# -----------------------------------------------------------
# 🎨 Styling (Optional)
# -----------------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #f0f8ff;
}
h1, h2, h3 {
    color: #003366;
}
</style>
""", unsafe_allow_html=True)

st.markdown("✅ **App is ready! Enter house size to predict price dynamically.**")
