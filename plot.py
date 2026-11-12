import numpy as np
import matplotlib.pyplot as plt

grid = np.loadtxt("grid.csv", delimiter=",")
plt.imshow(grid, cmap="hot")
plt.colorbar(label="temperature")
plt.title("2D heat equation (top edge = 100)")
plt.savefig("heat.png", dpi=150)
plt.show()
