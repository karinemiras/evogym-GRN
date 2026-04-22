import numpy as np
import sys
from pathlib import Path
import math
from sklearn.neighbors import KDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from experimental_setups.voxel_types import VOXEL_TYPES

METRICS_ABS = [
    # genotypic
    "genome_size",

    # behavioral: forward center-of-mass displacement along EvoGym x axis, in voxel-length units
    "displacement",
    "reward",
    "food_taken",
    "steps_until_food",

    # phenotypic
    "num_voxels",
    "bone_count",
    "bone_prop",
    "fat_count",
    "fat_prop",
    "muscle_h_count",
    "muscle_h_prop",
    "muscle_v_count",
    "muscle_v_prop",
    "muscle_count",
    "muscle_prop",
]

METRICS_REL = [
                "uniqueness",
                "fitness",
                "age",
                # "dominated_disp_age",
                # "dominated_disp_nov",
                # "novelty",
                # "novelty_weighted"
               ]

# metrics relative to other individuals or factors like time
def relative_metrics(population, args, generation, novelty_archive=None):
    uniqueness(population)
    # novelty(population, novelty_archive)
    # novelty_weighted(population)
    age(population, generation)
    # pareto_dominance_count( population,
    #                         objectives=(("age", "min"), ("displacement", "max")), out_attr="dominated_disp_age")
    set_fitness(population, args.fitness_metric)


def genopheno_abs_metrics(individual, args):

    # genome
    genome_size(individual)

    # phenotype
    num_voxels(individual)
    update_material_metrics(individual, args)
    test_validity(individual)


def behavior_abs_metrics(population):
    # TODO: implement others and treat for -inf
    pass

def update_material_metrics(individual, args):
    material_ids = VOXEL_TYPES

    grid = np.asarray(individual.phenotype, dtype=int)
    filled_total = int((grid != 0).sum())
    individual.filled_total = filled_total

    for name, mid in material_ids.items():

        count = int((grid == mid).sum())
        prop = (count / filled_total) if filled_total > 0 else 0.0

        setattr(individual, f"{name}_count", count)
        setattr(individual, f"{name}_prop", round(prop,2))

    # Aggregate actuator material regardless of controller phase.
    muscle_count = (
        individual.muscle_h_count
        + individual.muscle_v_count
    )
    individual.muscle_count = muscle_count
    individual.muscle_prop = round((muscle_count / filled_total) if filled_total > 0 else 0.0, 2)

def set_fitness(population, fitness_metric):
    for ind in population:
        ind.fitness = float(getattr(ind, fitness_metric, None))

def test_validity(individual):
    has_muscle = (individual.muscle_h_count + individual.muscle_v_count) >= 1
    phases = getattr(individual, "phenotype_phase_offsets", None)
    has_offphase_muscle = bool(phases is not None and np.any(np.asarray(phases) > 0))
    individual.valid = has_muscle and has_offphase_muscle

def age(population, generation):
    for ind in population:
        age = generation - ind.born_generation + 1
        ind.age = age

def genome_size(individual):
    individual.genome_size = len(individual.genome)


def num_voxels(individual):  # size / mass proxy
    individual.num_voxels = int((individual.phenotype != 0).sum())

def distance(g1, g2):
   #similar to hamming
    a = np.asarray(g1)
    b = np.asarray(g2)

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    one_zero = (a == 0) ^ (b == 0)  # 0 vs non-zero → 1.0 (different shape)
    both_nonzero_diff = (a != 0) & (b != 0) & (a != b)  # non-zero vs different non-zero → 0.5 (different material)

    return float(one_zero.sum() + 0.5 * both_nonzero_diff.sum())


def uniqueness(population):
    # average morphological distance to all current pop using Hamming distance
    for i, ind in enumerate(population):
        distances = []
        for j, other in enumerate(population):
            if i != j:
                d = distance(ind.phenotype, other.phenotype)
                distances.append(d / max(ind.num_voxels, other.num_voxels))
        ind.uniqueness = np.mean(distances)

# def novelty_weighted(population):
#     for ind in population:
#         novelty_weighted = ind.displacement * ind.novelty
#         ind.novelty_weighted = novelty_weighted

def novelty_weighted(population):
    beta = 0.05
    for ind in population:
        novelty_weighted = ind.displacement * ind.novelty + beta * ind.displacement
        ind.novelty_weighted = novelty_weighted

def novelty(population, novelty_archive, k=5, M=50, embed_fn=None):
    pool = list(population) + list(novelty_archive or [])

    if embed_fn is None:
        # minimal embedding: 1D vector
        embed_fn = lambda ind: np.array([ind.num_voxels], dtype=np.float32)

    X = np.vstack([embed_fn(ind) for ind in pool]).astype(np.float32)
    tree = KDTree(X)

    for ind in population:
        qi = embed_fn(ind).reshape(1, -1)
        _, idxs = tree.query(qi, k=min(M + 1, len(pool)))
        idxs = idxs[0]

        dists = []
        for j in idxs:
            other = pool[j]
            if other is ind:
                continue
            d = distance(ind.phenotype, other.phenotype)
            dists.append(d / max(ind.num_voxels, other.num_voxels))

        kk = min(k, len(dists))
        ind.novelty = float(np.partition(np.asarray(dists, dtype=np.float32), kk - 1)[:kk].mean()) if kk else 0.0


def pareto_dominance_count(
    population,
    objectives=(("age", "min"), ("displacement", "max")),
    out_attr="dominates_count",
):
    """
    For each individual, count how many others it Pareto-dominates
    Dominance rule:
      A dominates B iff
        - A is no worse than B in all objectives, AND
        - A is strictly better in at least one objective.
    """
    # Normalize directions and validate
    obj_specs = []
    for attr, direction in objectives:
        d = direction.strip().lower()
        obj_specs.append((attr, d))

    def dominates(a, b) -> bool:
        no_worse_all = True
        strictly_better_any = False

        for attr, d in obj_specs:
            av = getattr(a, attr)
            bv = getattr(b, attr)

            if d == "min":
                if av > bv:
                    no_worse_all = False
                    break
                if av < bv:
                    strictly_better_any = True
            else:  # "max"
                if av < bv:
                    no_worse_all = False
                    break
                if av > bv:
                    strictly_better_any = True

        return no_worse_all and strictly_better_any

    # Init output
    for ind in population:
        setattr(ind, out_attr, 0)

    # O(n^2) dominance counting
    n = len(population)
    for i in range(n):
        a = population[i]
        cnt = 0
        for j in range(n):
            if i == j:
                continue
            if dominates(a, population[j]):
                cnt += 1
        setattr(a, out_attr, cnt)
