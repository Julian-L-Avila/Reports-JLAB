# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = "./filiform-data.csv"

# Constants
n = 130
R = 0.2
especific_charge = 1.76
mu = np.pi * 4e-7

dU = 0.1
dI = 0.02
dr = 0.01e-2

# %%
# Load data
df = pd.read_csv(data, sep='\t')
df.info()

# %%
# Magnetic field calculation
def Magnetic_Field(I, n, R):
    return 8 * n * mu * I / (R * np.power(5, 3.0 / 2.0))

def Uncertainty_Magnetic_Field(dI, n, R):
    return 8 * n * mu * dI / (R * np.power(5, 3.0 / 2.0))

dB = Uncertainty_Magnetic_Field(dI, n, R)
print(f"Uncertainty in Magnetic Field: {dB}")

# %%
# Function for uncertainty in specific charge
def Uncertainty_Specific(U, r, B, dU, dr, dB):
    return np.sqrt((2.0 * dU / (r * B) ** 2) ** 2 + ((4.0 * U / (r * B) ** 3) ** 2) * ((B * dr) ** 2 + dB ** 2))

# Apply the function row by row
df['error'] = df.apply(
    lambda row: Uncertainty_Specific(row['U (V)'], row['r (cm)'] * 1e-2, row['B'], dU, dr, dB),
    axis=1
)

df['error'].mean() * 1e-10
