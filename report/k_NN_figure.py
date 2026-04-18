import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


figure, ax = plt.subplots(figsize=(8, 6))

pop = np.array([
    [1.2, 5.4],
    [2.4, 5.5],
    [1.8, 3.8],
    [3.3, 4.1]
])

classical = np.array([
    [6.8, 1.9],
    [8.4, 2.8],
    [8.0, 4.7]
])

disco = np.array([
    [5.1, 5.0],
    [6.1, 5.9],
    [7.0, 5.3]
])

metal = np.array([
    [4.5, 2.2],
    [5.7, 3.0],
    [6.1, 1.2]
])


point_to_clssify = np.array([4.4, 3.6])

ax.scatter(pop[:, 0], pop[:, 1], s=180, label="Pop")
ax.scatter(classical[:, 0], classical[:, 1], s=180, label="Classical")
ax.scatter(disco[:, 0], disco[:, 1], s=180, label="Disco")
ax.scatter(metal[:, 0], metal[:, 1], s=180, label="Metal")
 
ax.scatter(
    point_to_clssify [0], point_to_clssify [1],
    s=260,
    edgecolor="black",
    linewidth=1.2,
    label="Unknown track"
)

circle_k3 = Circle(point_to_clssify , radius=1.45, fill=False, linestyle="--", linewidth=1.5)
circle_k5 = Circle(point_to_clssify , radius=2.00, fill=False, linestyle="--", linewidth=1.5)

ax.add_patch(circle_k3)
ax.add_patch(circle_k5)


ax.text(point_to_clssify[0] + 0.15, point_to_clssify[1] - 1.15, "k = 3", fontsize=12)
ax.text(point_to_clssify[0] + 0.15, point_to_clssify[1] - 1.75, "k = 5", fontsize=12)


ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")


ax.set_xlim(0.5, 9.2)
ax.set_ylim(0.5, 6.8)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))

plt.tight_layout()
plt.show()