import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

def draw_phenotype(phenotype, id_individual, CUBE_FACE_SIZE, ranking, fitness, path, material_ids, material_colors):

    # Define color map for values in body
    color_map = {
        material_id: tuple(c / 255 for c in material_colors[name]) + (1.0,)
        for name, material_id in material_ids.items()
    }

    if phenotype.ndim != 2:
        raise ValueError(f"Expected 2D phenotype, got shape {phenotype.shape}")

    max_voxel_id = max(color_map.keys())
    colors = [(1, 1, 1, 0)]
    for voxel_id in range(1, max_voxel_id + 1):
        colors.append(color_map.get(voxel_id, (0.8, 0.8, 0.8, 0.5)))

    fig, ax = plt.subplots()
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, max_voxel_id + 1.5, 1), cmap.N)
    # EvoGym interprets NumPy arrays in row/column order: body[y, x].
    # Draw the same grid directly so the PNG matches the screen orientation.
    display_grid = phenotype
    ax.imshow(display_grid, cmap=cmap, norm=norm, origin="upper")
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(-0.5, display_grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, display_grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.5)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.tight_layout()
    plt.savefig(f"{path}/{ranking}_{fitness}_{id_individual}.png", dpi=300)
    plt.close(fig)
