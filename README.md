
# GRN Virtual Creatures in EvoGym 

## Setup

- Install EvoGym using the original instructions: https://evolutiongym.github.io/
- Clone current repo
- Install repo dependecies:

```bash
python3.9 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt

```

Run a test that uses a random GRN to generate the body+brain of a robot and then simulate it in EvoGym:

```bash
cd base
source ../.venv/bin/activate
python - <<'PY'
import random
import numpy as np
from types import SimpleNamespace
from experimental_setups.EA_classes import Individual
from experimental_setups.GRN_2D import GRN, initialization
from simulation.prepare_robot_files import prepare_robot_files
from simulation.offline_simulation import simulate_evogym_batch

rng = random.Random(3)
genome = initialization(rng, ini_genome_size=80)
phenotype_cells = GRN(
    max_voxels=25,
    cube_face_size=5,
    genotype=genome,
    env_conditions="",
    plastic=0,
).develop()

phenotype_materials = np.zeros(phenotype_cells.shape, dtype=int)
for idx, value in np.ndenumerate(phenotype_cells):
    phenotype_materials[idx] = value.voxel_type if value != 0 else 0

ind = Individual(genome=genome, id_counter=1)
ind.valid = 1
ind.phenotype = phenotype_materials

args = SimpleNamespace(
    out_path="/tmp",
    study_name="demo",
    experiment_name="smoke",
    run=1,
    evogym_steps=500,
    evogym_num_workers=1,
    evogym_init_x=3,
    evogym_init_y=1,
    evogym_action_bias=1.0,
    evogym_action_amplitude=0.4,
    evogym_period_steps=20,
    evogym_headless=0,
)

prepare_robot_files(ind, args)
simulate_evogym_batch([ind], args)
print("displacement:", ind.displacement)
PY
```

 
