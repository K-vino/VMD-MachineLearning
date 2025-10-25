

from sklearn.preprocessing import LabelEncoder

# Define the custom order
order = ["Low", "Medium", "High"] #"Low"-1, "Medium"-2, "High"-0

# Create a LabelEncoder instance
encoder = LabelEncoder()

# Fit the encoder with the custom order
encoder.fit(order)

# Sample data
data = ["Low", "Medium", "High", "Low","Low", "Medium", "High", "Low"]

# Encode the data
encoded_data = encoder.transform(data)
print(data)
print(encoded_data)




