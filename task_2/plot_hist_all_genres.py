import matplotlib.pyplot as plt
import numpy as np
from file_to_array import train_30s, test_30s, features

# Map label → genre name
genre_names = {
    0: "Pop",
    1: "Metal",
    2: "Disco",
    3: "Blues",
    4: "Reggae",
    5: "Classical",
    6: "Rock",
    7: "Hiphop",
    8: "Country",
    9: "Jazz"
}

# Create dict to hold data
genre_data = {i: [] for i in range(10)}

# --- Fill from train ---
for i in range(len(train_30s[1])):
    label = train_30s[1][i]
    genre_data[label].append(train_30s[0][i])

# --- Fill from test ---
for i in range(len(test_30s[1])):
    label = test_30s[1][i]
    genre_data[label].append(test_30s[0][i])

# Convert to numpy arrays
for key in genre_data:
    genre_data[key] = np.array(genre_data[key])

# --- Plot histograms ---
a = 0.4
bins = 12

num_features = 4  # adjust if needed

for f in range(num_features):
    plt.figure()
    
    for label, data in genre_data.items():
        if len(data) > 0:  # avoid empty genres
            plt.hist(data[:, f], alpha=a, label=genre_names[label], bins=bins)
    
    plt.title(features[f])
    plt.xlabel(f"{features[f]} (normalized)")
    plt.ylabel("Frequency")
    plt.legend()

plt.show()
