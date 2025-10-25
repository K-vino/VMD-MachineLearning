


from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Sample data: let's say we have a list of fruits
data = np.array(["Apple", "Banana", "Cherry", "Apple", "Cherry"]).reshape(-1, 1)

# Initialize the OneHotEncoder
encoder = OneHotEncoder(sparse=False)

# Fit and transform the data
one_hot_encoded = encoder.fit_transform(data)

# Print the original categories
print("Original Categories:\n", encoder.categories_)

# Print the one-hot encoded data
print("\nOne-Hot Encoded Data:\n", one_hot_encoded)
