import matplotlib.pyplot as plt
import numpy as np
from file_to_array import train, test, features

pop_data = []
metal_data = []
disco_data = []
classical_data = []

for i in range(len(train[1])):
    if train[1][i] == 0:
        pop_data.append(train[0][i])
    elif train[1][i] == 1:
        metal_data.append(train[0][i])
    elif train[1][i] == 2:
        disco_data.append(train[0][i])
    elif train[1][i] == 5:
        classical_data.append(train[0][i])

for i in range(len(test[1])):
    if test[1][i] == 0:
        pop_data.append(test[0][i])
    elif test[1][i] == 1:
        metal_data.append(test[0][i])
    elif test[1][i] == 2:
        disco_data.append(test[0][i])
    elif test[1][i] == 5:
        classical_data.append(test[0][i])

# Convert your class data lists to arrays for easy slicing
pop_data = np.array(pop_data)
metal_data = np.array(metal_data)
disco_data = np.array(disco_data)
classical_data = np.array(classical_data)

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

plt.show()