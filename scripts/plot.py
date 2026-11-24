from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent.parent / "results"

grid = np.loadtxt(RESULTS / "grid.csv", delimiter=",")
plt.imshow(grid, cmap="hot")
plt.colorbar(label="temperature")
plt.title("2D heat equation (top edge = 100)")
plt.savefig(RESULTS / "heat.png", dpi=150)
plt.show()
