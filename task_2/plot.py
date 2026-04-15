import matplotlib.pyplot as plt
import numpy as np
from file_to_array import train_30s, features

pop_data = []
metal_data = []
disco_data = []
classical_data = []

for i in range(len(train_30s[1])):
    if train_30s[1][i] == 0:
        pop_data.append(train_30s[0][i])
    elif train_30s[1][i] == 1:
        metal_data.append(train_30s[0][i])
    elif train_30s[1][i] == 2:
        disco_data.append(train_30s[0][i])
    elif train_30s[1][i] == 5:
        classical_data.append(train_30s[0][i])

pop_data = np.array(pop_data)
metal_data = np.array(metal_data)
disco_data = np.array(disco_data)
classical_data = np.array(classical_data)

"""
# --- First plot: feature 0 vs feature 1 ---
plt.figure()
plt.scatter(pop_data[:,0], pop_data[:,1], color='red', label='Pop')
plt.scatter(metal_data[:,0], metal_data[:,1], color='blue', label='Metal')
plt.scatter(disco_data[:,0], disco_data[:,1], color='green', label='Disco')
plt.scatter(classical_data[:,0], classical_data[:,1], color='purple', label='Classical')
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.grid(True)
plt.legend()

# --- Second plot: feature 0 vs feature 2 ---
plt.figure()
plt.scatter(pop_data[:,0], pop_data[:,2], color='red', label='Pop')
plt.scatter(metal_data[:,0], metal_data[:,2], color='blue', label='Metal')
plt.scatter(disco_data[:,0], disco_data[:,2], color='green', label='Disco')
plt.scatter(classical_data[:,0], classical_data[:,2], color='purple', label='Classical')
plt.xlabel(features[0])
plt.ylabel(features[2])
plt.grid(True)
plt.legend()

# --- Third plot: feature 0 vs feature 3 ---
plt.figure()
plt.scatter(pop_data[:,0], pop_data[:,3], color='red', label='Pop')
plt.scatter(metal_data[:,0], metal_data[:,3], color='blue', label='Metal')
plt.scatter(disco_data[:,0], disco_data[:,3], color='green', label='Disco')
plt.scatter(classical_data[:,0], classical_data[:,3], color='purple', label='Classical')
plt.xlabel(features[0])
plt.ylabel(features[3])
plt.grid(True)
plt.legend()
"""

a = 0.4
bins = 12

plt.figure()
plt.hist(pop_data[:,0], alpha = a, label='Pop', bins = bins)
plt.hist(metal_data[:,0], alpha = a, label='Metal', bins = bins)
plt.hist(disco_data[:,0], alpha = a, label='Disco', bins = bins)
plt.hist(classical_data[:,0], alpha = a, label='Classical', bins = bins)
plt.title(features[0])
plt.xlabel(f"{features[0]}")
plt.ylabel("Frequency")
plt.legend()

plt.figure()
plt.hist(pop_data[:,1], alpha = a, label='Pop', bins = bins)
plt.hist(metal_data[:,1], alpha = a, label='Metal', bins = bins)
plt.hist(disco_data[:,1], alpha = a, label='Disco', bins = bins)
plt.hist(classical_data[:,1], alpha = a, label='Classical', bins = bins)
plt.title(features[1])
plt.xlabel(f"{features[1]}")
plt.ylabel("Frequency")
plt.legend()

plt.figure()
plt.hist(pop_data[:,2], alpha = a, label='Pop', bins = bins)
plt.hist(metal_data[:,2], alpha = a, label='Metal', bins = bins)
plt.hist(disco_data[:,2], alpha = a, label='Disco', bins = bins)
plt.hist(classical_data[:,2], alpha = a, label='Classical', bins = bins)
plt.title(features[2])
plt.xlabel(f"{features[2]}")
plt.ylabel("Frequency")
plt.legend()

plt.figure()
plt.hist(pop_data[:,3], alpha = a, label='Pop', bins = bins)
plt.hist(metal_data[:,3], alpha = a, label='Metal', bins = bins)
plt.hist(disco_data[:,3], alpha = a, label='Disco', bins = bins)
plt.hist(classical_data[:,3], alpha = a, label='Classical', bins = bins)
plt.title(features[3])
plt.xlabel(f"{features[3]}")
plt.ylabel("Frequency")
plt.legend()

plt.show()