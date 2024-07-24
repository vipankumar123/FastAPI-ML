# train_model.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Generate a synthetic dataset
np.random.seed(0)
n_samples = 100
X = pd.DataFrame({
    'num_rooms': np.random.randint(1, 10, size=n_samples),
    'square_footage': np.random.randint(500, 4000, size=n_samples),
    'age_of_house': np.random.randint(1, 100, size=n_samples)
})
y = 5000 + (X['num_rooms'] * 1000) + (X['square_footage'] * 0.5) - (X['age_of_house'] * 100) + np.random.randn(n_samples) * 1000

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model to a file
with open('house_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as house_price_model.pkl")
