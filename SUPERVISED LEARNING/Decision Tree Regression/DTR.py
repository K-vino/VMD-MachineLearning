from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Example data
X = [[1000], [1500], [2000], [2500], [3000]]  # Size
y = [50, 75, 100, 120, 150]                   # Price

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Predictions:", y_pred)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
