import pandas as pd

# Specify the path to your CSV file
file_path = "Energy_consumption_dataset.csv"  # Replace with your actual file path

# Load the data into a pandas DataFrame
try:
    df = pd.read_csv(file_path)
    print("Data successfully loaded!")
    print(df.head())  # Display the first 5 rows of the dataset
except FileNotFoundError:
    print(f"File not found: {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")

df_reduced = df[['DayOfWeek', 'Holiday', 'HVACUsage', 'LightingUsage']]

print(df.head())
print(df_reduced.head())

df_reduced = df_reduced.replace({'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7})
df_reduced = df_reduced.replace({'Yes': 1, 'No': 2})
df_reduced = df_reduced.replace({'On': 1, 'Off': 2})
df_reduced = df_reduced.replace({'On': 1, 'Off': 2})

print(' ')
print(' ')
print(df_reduced.head())
print(' ')
print(' ')
print('Mean values: ')
print(' ')
print(' ')
print(df_reduced.mean())
print(' ')
print(' ')
print('Standard deviations: ')
print(' ')
print(' ')
print(df_reduced.std())
print(' ')
print(' ')
print('Correlations: ')
print(' ')
print(' ')
print(df_reduced.corr())
print(' ')
print(' ')

# Also calculate the correlations within the whole dataset

# Specify the path to your CSV file
file_path = "Energy_consumption_dataset.csv"  # Replace with your actual file path

# Load the data into a pandas DataFrame
try:
    df = pd.read_csv(file_path)
    print("Data successfully loaded!")
    print(df.head())  # Display the first 5 rows of the dataset
except FileNotFoundError:
    print(f"File not found: {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")

df_reduced = df.replace({'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7})
df_reduced = df_reduced.replace({'Yes': 1, 'No': 2})
df_reduced = df_reduced.replace({'On': 1, 'Off': 2})
df_reduced = df_reduced.replace({'On': 1, 'Off': 2})

print(' ')
print(' ')
print(df_reduced.head())
print(' ')
print(' ')
print('Mean values: ')
print(' ')
print(' ')
print(df_reduced.mean())
print(' ')
print(' ')
print('Standard deviations: ')
print(' ')
print(' ')
print(df_reduced.std())
print(' ')
print(' ')
print('Correlations: ')
print(' ')
print(' ')
print(df_reduced.corr())
print(' ')
print(' ')
print(' ')
print(' ')
print(df_reduced.corr('pearson'))
print(' ')
print(' ')
print(' ')
print(' ')
print(df_reduced.corr('spearman'))
print(' ')
print(' ')
