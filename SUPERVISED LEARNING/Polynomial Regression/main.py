from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Example data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 5, 9, 16, 25])

# Transform input to polynomial features (degree 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_poly, y)

# Predict
y_pred = model.predict(X_poly)

print("Predictions:", y_pred)
print("Mean Squared Error:", mean_squared_error(y, y_pred))
