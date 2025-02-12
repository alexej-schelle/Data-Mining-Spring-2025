import os
import sys

import pandas as pd
import numpy
import math

import matplotlib.pyplot as plt

# Replace 'your_file.csv' with the path to your CSV file
df = pd.read_csv('BigDataGenerated.csv')

# Display the first few rows of the DataFrame
print(df.head())

# Show number of classes (elements per class)
print(df.count())

# Check for datatype of classes
print(df.dtypes)

# Calculate mean value and standard deviations within each numeric class 

# Project on numeric classes

# Unterscheidung als Numerische und Logische VariablenTypen
reduced_numeric = df[['Hour', 'Temperature', 'Humidity', 'SquareFootage', 'Occupancy', 'RenewableEnergy', 'EnergyConsumption']].copy()
reduced_logic = df[['DayOfWeek', 'Holiday', 'HVACUsage', 'LightingUsage']].copy()

print(' ')
print('Mean Values : ')
print(' ')
print(reduced_numeric.mean())
print(' ')
print('Standard Deviations : ')
print(' ')
print(reduced_numeric.std())

print(' ')
print('Counting Statistcs: ')
print(' ')
print(reduced_logic['DayOfWeek'].value_counts())
print(' ')
print(' ')
print(reduced_logic['Holiday'].value_counts())
print(' ')
print(' ')
print(reduced_logic['HVACUsage'].value_counts())
print(' ')
print(' ')
print(reduced_logic['LightingUsage'].value_counts())
print(' ')

# Calcluate correlations of numeric variables

# Standard (quadratic) correlations:
print(' ')
print('Quadratic correlations: ')
print(' ')
print(reduced_numeric.corr())
print(' ')

# Standard (quadratic) correlations:
print(' ')
print('Pearson correlations: ')
print(' ')
print(reduced_numeric.corr('pearson'))
print(' ')

# Standard (quadratic) correlations:
print(' ')
print('Spearman correlations: ')
print(' ')
print(reduced_numeric.corr('spearman'))
print(' ')

# Calculate correlations between logic variables:
df_reduced_logic_mapped = reduced_logic.replace({'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7, 'Yes': 1, 'No': 0, 'On': 0, 'Off': 1})

print(df_reduced_logic_mapped.corr())
print(df_reduced_logic_mapped.corr('pearson'))
print(df_reduced_logic_mapped.corr('spearman'))
