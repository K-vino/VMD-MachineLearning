from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Example data
X = [[1000,2,5],[1500,3,10],[2000,4,2],[2500,4,15],[3000,5,8]]  # Features: Size, Bedrooms, Age
y = [50, 75, 100, 120, 150]  # Price

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Predictions:", y_pred)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
