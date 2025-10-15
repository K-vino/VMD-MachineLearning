# -----------------------------------------------------------
# STREAMLIT APP: SALES FORECASTING WITH MULTIVARIATE ML MODEL
# -----------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------
# 🌟 Page Config
# -----------------------------------------------------------
st.set_page_config(page_title="Sales Forecasting App", layout="wide", page_icon="📈")

# -----------------------------------------------------------
# 🏷️ App Title
# -----------------------------------------------------------
st.title("📈 Sales Forecasting ML Model")
st.markdown("""
Predict store sales based on multiple features including marketing spend, store size, seasonality, and more.
This interactive app allows you to enter your own feature values and see predicted sales.
""")

# -----------------------------------------------------------
# 1️⃣ Simulate Dataset
# -----------------------------------------------------------
st.header("📊 Sample Dataset")

np.random.seed(42)
n_samples = 500

data = {
    'Store_Size': np.random.randint(1000, 5000, n_samples),
    'Marketing_Spend': np.random.randint(1000, 50000, n_samples),
    'Season': np.random.choice(['Winter','Spring','Summer','Fall'], n_samples),
    'Holiday': np.random.randint(0,2,n_samples),
    'Competitor_Price': np.random.randint(50, 500, n_samples),
    'Discount': np.random.randint(0,30,n_samples),
    'Number_of_Employees': np.random.randint(5,50,n_samples),
    'Economic_Index': np.random.uniform(50,150,n_samples),
    'Day_of_Week': np.random.randint(1,8,n_samples),
    'Online_Ads_Clicks': np.random.randint(100,10000,n_samples),
    'Past_Sales': np.random.randint(500,5000,n_samples)
}

df = pd.DataFrame(data)

# Encode categorical variable 'Season'
df = pd.get_dummies(df, columns=['Season'], drop_first=True)
st.dataframe(df.head())

# -----------------------------------------------------------
# 2️⃣ Define Features and Target
# -----------------------------------------------------------
X = df.drop(columns=['Past_Sales'])
y = df['Past_Sales']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------------------------------
# 3️⃣ Build Linear Regression Model
# -----------------------------------------------------------
model = LinearRegression()
model.fit(X_scaled, y)

st.markdown("**Model trained successfully with 10+ features.**")

# -----------------------------------------------------------
# 4️⃣ User Input Section
# -----------------------------------------------------------
st.header("🔮 Enter Features to Predict Sales")

# Function to collect user input
def user_input_features():
    Store_Size = st.number_input("Store Size (sq.ft)", 500, 10000, 2000)
    Marketing_Spend = st.number_input("Marketing Spend ($)", 0, 100000, 10000)
    Holiday = st.selectbox("Holiday (0=No, 1=Yes)", [0,1])
    Competitor_Price = st.number_input("Competitor Price ($)", 0, 1000, 100)
    Discount = st.slider("Discount (%)", 0, 50, 10)
    Number_of_Employees = st.number_input("Number of Employees", 1, 100, 10)
    Economic_Index = st.number_input("Economic Index", 0.0, 200.0, 100.0)
    Day_of_Week = st.selectbox("Day of the Week", list(range(1,8)))
    Online_Ads_Clicks = st.number_input("Online Ads Clicks", 0, 20000, 500)
    Season = st.selectbox("Season", ["Winter","Spring","Summer","Fall"])

    # Encode Season
    Season_Winter = 1 if Season=="Winter" else 0
    Season_Spring = 1 if Season=="Spring" else 0
    Season_Summer = 1 if Season=="Summer" else 0
    Season_Fall = 0  # reference

    data = {
        'Store_Size': Store_Size,
        'Marketing_Spend': Marketing_Spend,
        'Holiday': Holiday,
        'Competitor_Price': Competitor_Price,
        'Discount': Discount,
        'Number_of_Employees': Number_of_Employees,
        'Economic_Index': Economic_Index,
        'Day_of_Week': Day_of_Week,
        'Online_Ads_Clicks': Online_Ads_Clicks,
        'Season_Spring': Season_Spring,
        'Season_Summer': Season_Summer,
        'Season_Winter': Season_Winter
    }

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Scale input features
input_scaled = scaler.transform(input_df)

# Predict Sales
predicted_sales = model.predict(input_scaled)[0]
st.success(f"Predicted Sales: {predicted_sales:,.2f}")

# -----------------------------------------------------------
# 5️⃣ Model Evaluation Metrics
# -----------------------------------------------------------
y_pred = model.predict(X_scaled)
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

st.header("📏 Model Evaluation Metrics")
st.write(f"- Mean Absolute Error (MAE): {mae:.2f}")
st.write(f"- Mean Squared Error (MSE): {mse:.2f}")
st.write(f"- Root Mean Squared Error (RMSE): {rmse:.2f}")
st.write(f"- R² Score: {r2:.2f}")

# -----------------------------------------------------------
# 6️⃣ Feature Importance
# -----------------------------------------------------------
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)

st.header("📈 Feature Importance")
st.dataframe(feature_importance)

# -----------------------------------------------------------
# 7️⃣ Visualization
# -----------------------------------------------------------
st.header("📊 Actual vs Predicted Sales")
fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(y, y_pred, color='blue')
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
ax.set_xlabel("Actual Sales")
ax.set_ylabel("Predicted Sales")
ax.set_title("Actual vs Predicted Sales")
ax.grid(True)
st.pyplot(fig)

# Correlation heatmap
st.header("🔍 Feature Correlation Heatmap")
fig2, ax2 = plt.subplots(figsize=(12,6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
st.pyplot(fig2)

st.markdown("✅ **Interactive Sales Forecasting Model is ready! Enter values to predict sales dynamically.**")
