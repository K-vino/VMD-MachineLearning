from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import numpy as np

# Example data
X = np.array([[1],[2],[3],[4],[5]]).astype(float)
y = np.array([1.5, 3.7, 7.4, 8.0, 12.1]).astype(float)

# Feature Scaling (important for SVR)
sc_X = StandardScaler()
sc_y = StandardScaler()
X_scaled = sc_X.fit_transform(X)
y_scaled = sc_y.fit_transform(y.reshape(-1,1))

# Train SVR model with RBF kernel
svr = SVR(kernel='rbf')
svr.fit(X_scaled, y_scaled.ravel())

# Predict
y_pred_scaled = svr.predict(sc_X.transform(np.array([[6]])))
y_pred = sc_y.inverse_transform(y_pred_scaled.reshape(-1,1))
print("Predicted value for X=6:", y_pred[0][0])
