import os
import sys

import pandas as pd
import numpy
import math

import matplotlib.pyplot as plt

# Replace 'your_file.csv' with the path to your CSV file
df = pd.read_csv('Energy_consumption_dataset.csv')

# Display the first few rows of the DataFrame
print(df.head())

# Show number of classes (elements per class)
print(df.count())

# Check for datatype of classes
print(df.dtypes)

# Show number of categories with each class 
# First class 1
df_class = df['Month']
print(df_class.value_counts()) 

# First class 2
df_class = df['Hour']
print(df_class.value_counts()) 

# First class 3
df_class = df['DayOfWeek']
print(df_class.value_counts()) 

# First class 4
df_class = df['Holiday']
print(df_class.value_counts()) 

# First class 8
df_class = df['Occupancy']
print(df_class.value_counts()) 

# First class 9
df_class = df['HVACUsage']
print(df_class.value_counts()) 

# First class 10
df_class = df['LightingUsage']
print(df_class.value_counts()) 

# Calculate mean values for numeric types
print('Month: ', df['Month'].mean())

# Calculate mean values for numeric types
print('Hour ', df['Hour'].mean())

# Calculate mean values for numeric types
print('Temperature:', df['Temperature'].mean())

# Calculate mean values for numeric types
print('Humidity: ', df['Humidity'].mean())

# Calculate mean values for numeric types
print('SquareFootage: ', df['SquareFootage'].mean())

# Calculate mean values for numeric types
print('RenewableEnergy: ', df['RenewableEnergy'].mean())

# Calculate mean values for numeric types
print('EnergyConsumption: ', df['EnergyConsumption'].mean())

# Create histograms
df.hist(bins=50, figsize=(8, 6))

# Show the plots
plt.show()
