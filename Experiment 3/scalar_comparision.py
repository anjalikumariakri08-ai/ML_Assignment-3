import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Sample data
data = np.array([
    [20, 25000],
    [21, 30000],
    [22, 28000],
    [23, 35000],
    [24, 40000]
])

# StandardScaler
standard = StandardScaler()
standard_data = standard.fit_transform(data)

print("--- StandardScaler ---")
print(standard_data)
print("Minimum:", standard_data.min())
print("Maximum:", standard_data.max())

# MinMaxScaler
minmax = MinMaxScaler()
minmax_data = minmax.fit_transform(data)

print("\n--- MinMaxScaler ---")
print(minmax_data)
print("Minimum:", minmax_data.min())
print("Maximum:", minmax_data.max())