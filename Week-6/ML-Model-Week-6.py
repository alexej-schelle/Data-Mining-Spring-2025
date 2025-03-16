import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np

# Example: Generate synthetic data for demonstration
def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'Month': np.random.randint(1, 13, num_samples),
        'Hour': np.random.randint(0, 24, num_samples),
        'DayOfWeek': np.random.randint(0, 7, num_samples),
        'Holiday': np.random.choice([0, 1], num_samples),
        'Temperature': np.random.uniform(15, 35, num_samples),
        'Humidity': np.random.uniform(30, 70, num_samples),
        'SquareFootage': np.random.randint(500, 5000, num_samples),
        'Occupancy': np.random.randint(1, 200, num_samples),
        'HVACUsage': np.random.uniform(0, 1, num_samples),
        'LightingUsage': np.random.uniform(0, 1, num_samples),
        'RenewableEnergy': np.random.uniform(0, 1, num_samples),
        'EnergyConsumption': np.random.uniform(100, 1000, num_samples)
    }
    return pd.DataFrame(data)

# Load or generate data
data = generate_synthetic_data()

# Features and target variable
X = data[['Month', 'Hour', 'DayOfWeek', 'Holiday', 'Temperature', 'Humidity', 'SquareFootage',
          'Occupancy', 'HVACUsage', 'LightingUsage', 'RenewableEnergy']]
y = data['EnergyConsumption']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the SVM model

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np

svm_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svm_model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = svm_model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation results
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

# Define a single new data point
new_data = [[7, 14, 3, 0, 25.0, 50.0, 2000, 50, 0.7, 0.5, 0.3]]

# Scale the new data using the same scaler as the training set
new_data_scaled = scaler.transform(new_data)

# Predict the EnergyConsumption for the new data point
predicted_value = svm_model.predict(new_data_scaled)

print(f"Predicted EnergyConsumption: {predicted_value[0]:.2f}")

# Create Random Forest Model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

print(' ')
print(' ')

# Create and train the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf_model.predict(X_test)

# Calculate metrics
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Print the evaluation results
print("Random Forest Model Performance:")
print(f"Mean Squared Error (MSE): {mse_rf:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_rf:.2f}")
print(f"R-squared (R²): {r2_rf:.2f}")

# Define a single new data point
new_data_rf = [[7, 14, 3, 0, 25.0, 50.0, 2000, 50, 0.7, 0.5, 0.3]]  # Example values

# Predict the EnergyConsumption for the new data point using the Random Forest model
predicted_value_rf = rf_model.predict(new_data_rf)

print(f"Predicted EnergyConsumption (Random Forest): {predicted_value_rf[0]:.2f}")

print(' ')
print(' ')

# Create and train the K-Nearest Neighbors Regressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

knn_model = KNeighborsRegressor(n_neighbors=5)  # Default: k=5
knn_model.fit(X_train_scaled, y_train)  # Use scaled data for KNN

# Predict on the test set
y_pred_knn = knn_model.predict(X_test_scaled)

# Calculate metrics
mse_knn = mean_squared_error(y_test, y_pred_knn)
rmse_knn = np.sqrt(mse_knn)
r2_knn = r2_score(y_test, y_pred_knn)

# Print the evaluation results
print("K-Nearest Neighbors Model Performance:")
print(f"Mean Squared Error (MSE): {mse_knn:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_knn:.2f}")
print(f"R-squared (R²): {r2_knn:.2f}")

# Define a single new data point
new_data_knn = [[7, 14, 3, 0, 25.0, 50.0, 2000, 50, 0.7, 0.5, 0.3]]  # Example input values

# Scale the new data using the same scaler
new_data_knn_scaled = scaler.transform(new_data_knn)

# Predict the EnergyConsumption for the new data point
predicted_value_knn = knn_model.predict(new_data_knn_scaled)

print(f"Predicted EnergyConsumption (KNN): {predicted_value_knn[0]:.2f}")

print(' ')
print(' ')

# Create and train the Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)

# Predict on the test set
y_pred_gb = gb_model.predict(X_test)

# Calculate metrics
mse_gb = mean_squared_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mse_gb)
r2_gb = r2_score(y_test, y_pred_gb)

# Print the evaluation results
print("Gradient Boosting Model Performance:")
print(f"Mean Squared Error (MSE): {mse_gb:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_gb:.2f}")
print(f"R-squared (R²): {r2_gb:.2f}")

# Define a single new data point (example values)
new_data_gb = [[7, 14, 3, 0, 25.0, 50.0, 2000, 50, 0.7, 0.5, 0.3]]  # Example input values

# Predict the EnergyConsumption for the new data point using the Gradient Boosting model
predicted_value_gb = gb_model.predict(new_data_gb)

# Print the predicted value
print(f"Predicted EnergyConsumption (Gradient Boosting): {predicted_value_gb[0]:.2f}")

print(' ')
print(' ')

# Create and train the Kernel Ridge Regression model
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

kr_model = KernelRidge(alpha=1.0, kernel='rbf')  # Use Radial Basis Function (RBF) kernel
kr_model.fit(X_train_scaled, y_train)  # Make sure to scale data for kernel methods

# Predict on the test set
y_pred_kr = kr_model.predict(X_test_scaled)

# Calculate metrics
mse_kr = mean_squared_error(y_test, y_pred_kr)
rmse_kr = np.sqrt(mse_kr)
r2_kr = r2_score(y_test, y_pred_kr)

# Print the evaluation results
print("Kernel Ridge Regression Model Performance:")
print(f"Mean Squared Error (MSE): {mse_kr:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_kr:.2f}")
print(f"R-squared (R²): {r2_kr:.2f}")

print(' ')
print(' ')

# Define a single new data point (example values)
new_data_kr = [[7, 14, 3, 0, 25.0, 50.0, 2000, 50, 0.7, 0.5, 0.3]]  # Example input values

# Scale the new data using the same scaler (important for kernel methods)
new_data_kr_scaled = scaler.transform(new_data_kr)

# Predict the EnergyConsumption for the new data point using the Kernel Ridge Regression model
predicted_value_kr = kr_model.predict(new_data_kr_scaled)

# Print the predicted value
print(f"Predicted EnergyConsumption (Kernel Ridge Regression): {predicted_value_kr[0]:.2f}")

print(' ')
print(' ')

# Create and train the Stochastic Gradient Descent Regressor Model
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

sgd_model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
sgd_model.fit(X_train_scaled, y_train)  # Make sure to scale the data as SGD is sensitive to feature scaling

# Predict on the test set
y_pred_sgd = sgd_model.predict(X_test_scaled)

# Calculate metrics
mse_sgd = mean_squared_error(y_test, y_pred_sgd)
rmse_sgd = np.sqrt(mse_sgd)
r2_sgd = r2_score(y_test, y_pred_sgd)

# Print the evaluation results
print("Stochastic Gradient Descent Model Performance:")
print(f"Mean Squared Error (MSE): {mse_sgd:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_sgd:.2f}")
print(f"R-squared (R²): {r2_sgd:.2f}")

# Define a single new data point (example values)
new_data_sgd = [[7, 14, 3, 0, 25.0, 50.0, 2000, 50, 0.7, 0.5, 0.3]]  # Example input values

# Scale the new data using the same scaler (important for SGD)
new_data_sgd_scaled = scaler.transform(new_data_sgd)

# Predict the EnergyConsumption for the new data point using the SGD model
predicted_value_sgd = sgd_model.predict(new_data_sgd_scaled)

# Print the predicted value
print(f"Predicted EnergyConsumption (SGD): {predicted_value_sgd[0]:.2f}")

print(' ')
print(' ')
