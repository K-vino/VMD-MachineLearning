# Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Example Dataset (Sales/Growth Data)
# You can replace this with your CSV or database
data = {
    'Advertising': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    'Price': [10, 12, 15, 20, 25, 30, 35, 40, 45, 50],
    'Season_Index': [1, 2, 3, 4, 1, 2, 3, 4, 1, 2],
    'Sales': [110, 220, 305, 410, 500, 620, 710, 820, 900, 1020]
}

df = pd.DataFrame(data)

# Features and Target
X = df[['Advertising', 'Price', 'Season_Index']]  # multiple features
y = df['Sales']

# Polynomial Features Transformation (degree=2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Split Data into Training and Test
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Train Polynomial Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on Test Set
y_pred = model.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

# User Input for Prediction
print("\n--- Predict Sales / Growth ---")
adv = float(input("Enter Advertising Spend: "))
price = float(input("Enter Product Price: "))
season = int(input("Enter Season Index (1-4): "))

user_data = np.array([[adv, price, season]])
user_data_poly = poly.transform(user_data)

prediction = model.predict(user_data_poly)
print(f"Predicted Sales / Growth: {prediction[0]:.2f}")
