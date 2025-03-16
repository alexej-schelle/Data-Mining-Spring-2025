import pandas as pd
import numpy as np

import tensorflow

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import warnings
warnings.simplefilter("ignore", UserWarning)

# Load the dataset
file_path = "Energy_consumption_dataset.csv"  # Replace with your dataset file path
try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print(f"File not found: {file_path}")
   # exit()

# Step 1: Data Preprocessing
df_reduced = df.replace({'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 
                         'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7, 
                         'Yes': 1, 'No': 0, 'On': 0, 'Off': 1}).infer_objects(copy=False)

# Step 2: Verify Data
print(df_reduced)

# Step 3: Model Processing

# Define input features (X) and target variable (y)
X = df_reduced.drop(columns=["EnergyConsumption"])  # Features
y = df_reduced["EnergyConsumption"]  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the numerical features for better performance
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 2: Build the TensorFlow Model
model = Sequential([
    Dense(4, input_dim=X_train.shape[1], activation="relu"),  # Input layer with 4 neurons
    Dense(128, activation="relu"),                              # Hidden layer with 128 neurons
    Dense(4, activation="linear")                              # Output layer (4 neuron for regression)
])

# Compile the model
model.compile(optimizer="adam", loss="mse", metrics=["mae"])  # Loss: Mean Squared Error, Metric: Mean Absolute Error

# Step 3: Train the Model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# Step 4: Evaluate the Model
loss, mae = model.evaluate(X_test, y_test, verbose=0)

print(f"Test Loss (MSE): {loss:.4f}")
print(f"Test Mean Absolute Error (MAE): {mae:.4f}")

# Step 5: Make Predictions
predictions = model.predict(X_test)

print("Predictions for EnergyConsumption:")
print(predictions[:10])  # Print the first 10 predictions
