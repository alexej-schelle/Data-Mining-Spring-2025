import pandas as pd

def calculate_extreme_values(file_path, feature_name):
    """
    Calculate the extreme values (minimum and maximum) of a dataset for a specific feature.

    Parameters:
        file_path (str): Path to the CSV file containing the dataset.
        feature_name (str): The name of the feature/column to analyze.

    Returns:
        dict: A dictionary containing the minimum and maximum values of the feature.
    """
    # Load the dataset
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return f"Error loading file: {e}"
    
    # Check if the feature exists in the dataset
    if feature_name not in df.columns:
        return f"Feature '{feature_name}' not found in the dataset."
    
    # Calculate extreme values
    min_value = df[feature_name].min()
    max_value = df[feature_name].max()
    
    return {
        "Minimum Value": min_value,
        "Maximum Value": max_value
    }

print(calculate_extreme_values('Energy_consumption_dataset.csv', 'Hour'))
print(calculate_extreme_values('Energy_consumption_dataset.csv', 'Temperature'))
print(calculate_extreme_values('Energy_consumption_dataset.csv', 'Humidity'))
print(calculate_extreme_values('Energy_consumption_dataset.csv', 'SquareFootage'))
print(calculate_extreme_values('Energy_consumption_dataset.csv', 'Occupancy'))
print(calculate_extreme_values('Energy_consumption_dataset.csv', 'RenewableEnergy'))
print(calculate_extreme_values('Energy_consumption_dataset.csv', 'EnergyConsumption'))
