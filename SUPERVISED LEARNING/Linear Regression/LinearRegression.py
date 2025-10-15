# -----------------------------------------------------------
# 📘 LINEAR REGRESSION - HOUSE PRICE PREDICTION
# -----------------------------------------------------------

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------------------------------------
# 🧠 STEP 1: Create the Dataset
# -----------------------------------------------------------
data = {
    'House_Size': [1000, 1500, 2000, 2500, 3000],
    'Price': [200000, 250000, 300000, 350000, 400000]
}

# Convert to DataFrame
df = pd.DataFrame(data)
print("📊 Dataset:")
print(df)

# -----------------------------------------------------------
# 🧩 STEP 2: Define Input (X) and Output (y)
# -----------------------------------------------------------
X = df[['House_Size']]   # Independent variable (2D)
y = df['Price']           # Dependent variable (1D)

# -----------------------------------------------------------
# ⚙️ STEP 3: Create Linear Regression Model
# -----------------------------------------------------------
model = LinearRegression()

# Train the model
model.fit(X, y)

# -----------------------------------------------------------
# 📈 STEP 4: Get Coefficients (β₀ and β₁)
# -----------------------------------------------------------
intercept = model.intercept_
slope = model.coef_[0]

print("\n📏 Regression Equation:")
print(f"Price = {intercept:.2f} + {slope:.2f} × House_Size")

# -----------------------------------------------------------
# 🔮 STEP 5: Make Predictions
# -----------------------------------------------------------
y_pred = model.predict(X)

# Predict for a new house
new_house_size = 2200
predicted_price = model.predict([[new_house_size]])
print(f"\n🏠 Predicted price for {new_house_size} sq.ft house: ₹{predicted_price[0]:,.2f}")

# -----------------------------------------------------------
# 📊 STEP 6: Visualization
# -----------------------------------------------------------
plt.figure(figsize=(8,6))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.scatter(new_house_size, predicted_price, color='green', label='Predicted Point', s=100)
plt.xlabel("House Size (sq.ft)")
plt.ylabel("Price (₹)")
plt.title("Linear Regression - House Price Prediction")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------------------------------------
# 📏 STEP 7: Evaluate the Model
# -----------------------------------------------------------
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print("\n📊 Model Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# -----------------------------------------------------------
# ✅ STEP 8: Manual Check with Formula
# -----------------------------------------------------------
# For verification, calculate prediction manually:
manual_pred = intercept + slope * new_house_size
print(f"\n🧮 Manual formula result: ₹{manual_pred:,.2f}")

print("\n✅ Linear Regression Model Executed Successfully!")



'''
output:
📊 Dataset:
   House_Size   Price
0        1000  200000
1        1500  250000
2        2000  300000
3        2500  350000
4        3000  400000

📏 Regression Equation:
Price = 100000.00 + 100.00 × House_Size

🏠 Predicted price for 2200 sq.ft house: ₹320,000.00

📊 Model Evaluation Metrics:
Mean Absolute Error (MAE): 0.00
Mean Squared Error (MSE): 0.00
Root Mean Squared Error (RMSE): 0.00
R² Score: 1.00

🧮 Manual formula result: ₹320,000.00
✅ Linear Regression Model Executed Successfully!

'''