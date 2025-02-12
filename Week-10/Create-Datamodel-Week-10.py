import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

# Load the data
df = pd.read_csv('Energy_consumption_dataset.csv')

# Step 1: Data Preprocessing
df_reduced = df.replace({'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7, 'Yes': 1, 'No': 0, 'On': 0, 'Off': 1, 'NaN': 0})

# Step 2: Evaluate distributions

# Create histograms
df_reduced.hist(bins=50, figsize=(8, 6))

# Add title and labels
plt.title('Histogram Example')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Show the plot
plt.show()

# Define distribution for variable Month

mean = df_reduced.mean()
variance = df_reduced.var()

minimum = min(df_reduced['Month'])
maximum = max(df_reduced['Month'])

Month = random.uniform(minimum, maximum)

# Define distribution for variable Hour

mean = df_reduced.mean()
variance = df_reduced.var()

minimum = min(df_reduced['Hour'])
maximum = max(df_reduced['Hour'])

Hour = random.uniform(minimum, maximum)

# Define distribution for variable DayOfWeek

mean = df_reduced.mean()
variance = df_reduced.var()

minimum = min(df_reduced['DayOfWeek'])
maximum = max(df_reduced['DayOfWeek'])

DayOfWeek = random.uniform(minimum, maximum)

# Define distribution for variable Holiday

mean = df_reduced.mean()
variance = df_reduced.var()

minimum = min(df_reduced['Holiday'])
maximum = max(df_reduced['Holiday'])

Holiday = random.uniform(minimum, maximum)

# Define distribution for variable Temperature

mean = df_reduced.mean()
variance = df_reduced.var()

minimum = min(df_reduced['Temperature'])
maximum = max(df_reduced['Temperature'])

Temperature = random.uniform(minimum, maximum)

# Define distribution for variable Humidity

mean = df_reduced.mean()
variance = df_reduced.var()

minimum = min(df_reduced['Humidity'])
maximum = max(df_reduced['Humidity'])

Humidity = random.uniform(minimum, maximum)

# Define distribution for variable SquareFootage

mean = df_reduced.mean()
variance = df_reduced.var()

minimum = min(df_reduced['SquareFootage'])
maximum = max(df_reduced['SquareFootage'])

SquareFootage = random.uniform(minimum, maximum)

# Define distribution for variable Occupancy

mean = df_reduced.mean()
variance = df_reduced.var()

minimum = min(df_reduced['Occupancy'])
maximum = max(df_reduced['Occupancy'])

Occupancy = random.uniform(minimum, maximum)

# Define distribution for variable HVSCUsage

mean = df_reduced.mean()
variance = df_reduced.var()

minimum = min(df_reduced['HVACUsage'])
maximum = max(df_reduced['HVACUsage'])

month = random.uniform(minimum, maximum)

# Define distribution for variable LightingUsage

mean = df_reduced.mean()
variance = df_reduced.var()

minimum = min(df_reduced['LightingUsage'])
maximum = max(df_reduced['LightingUsage'])

LightingUsage = random.uniform(minimum, maximum)

# Define distribution for variable RenewableEnergy

mean = df_reduced.mean()
variance = df_reduced.var()

minimum = min(df_reduced['RenewableEnergy'])
maximum = max(df_reduced['RenewableEnergy'])

RenewableEnergy = random.uniform(minimum, maximum)

# Define distribution for variable EnergyConsumption

mean = df_reduced.mean()
variance = df_reduced.var()

EnergyConsumption = random.gauss(mean, variance)

