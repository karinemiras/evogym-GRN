import os, sys
import argparse
import math
import importlib
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from algorithms.EA_classes import Robot, GenerationSurvivor
from utils.draw import draw_phenotype
from utils.config import Config
from algorithms.voxel_types import VOXEL_TYPES, VOXEL_TYPES_COLORS


def get_algorithm_class(module, algorithm_name):
    class_name_by_algorithm = {
        "basic_EA": "EA",
        "cmaes": "CMAES",
    }
    class_name = class_name_by_algorithm.get(algorithm_name)
    if class_name is None:
        raise ValueError(f"Unknown algorithm '{algorithm_name}'. Add its class name to snapshots_bests.py.")
    return getattr(module, class_name)


def main():
    args = Config()._get_params()

    study_name = args.study_name
    experiments = args.experiments.split(",")
    runs = list(map(int, args.runs.split(",")))
    generations = list(map(int, args.generations.split(",")))

    # instantiates the algorithm class with original params to develop the phenotypes
    module_name = f"algorithms.{args.algorithm}"
    module = importlib.import_module(module_name)
    algorithm = get_algorithm_class(module, args.algorithm)(args)

    numberrobots = 100  # top-N per generation

    for exp_idx, experiment_name in enumerate(experiments):
        print(experiment_name)

        material_ids = VOXEL_TYPES
        material_colors = VOXEL_TYPES_COLORS

        for run in runs:
            print(" run:", run)

            snapshot_root = f"{args.out_path}/{study_name}/analysis/snapshots/{experiment_name}/run_{run}"
            print(snapshot_root)
            os.makedirs(snapshot_root, exist_ok=True)

            db_path = f"{args.out_path}/{study_name}/{experiment_name}/run_{run}/run_{run}"
            print(db_path)
            if not os.path.exists(db_path):
                raise FileNotFoundError(
                    f"Database not found at '{db_path}'. "
                    "Make sure this matches the Experiment's DB location."
                )

            engine = create_engine(f"sqlite:///{db_path}", echo=False, future=True)
            Session = sessionmaker(bind=engine, expire_on_commit=False)

            with Session() as session:
                # Quick sanity check to avoid wasting time on empty DBs
                total_survivor_gens = session.query(func.count(GenerationSurvivor.generation.distinct())).scalar()
                if total_survivor_gens == 0:
                    print("  (no completed generations found in DB)")
                    continue

                for gen in generations:
                    print("  gen:", gen)

                    gen_dir = f"{snapshot_root}/gen_{gen}"
                    if os.path.exists(gen_dir):
                        print(f"    {gen_dir} already exists!")
                        continue
                    os.makedirs(gen_dir, exist_ok=True)

                    # Join survivors with their robots for this generation
                    rows = (
                        session.query(Robot, GenerationSurvivor)
                        .join(GenerationSurvivor, GenerationSurvivor.robot_id == Robot.robot_id)
                        .filter(GenerationSurvivor.generation == gen)
                        .order_by(GenerationSurvivor.fitness.desc().nullslast())
                        .all()
                    )

                    if not rows:
                        print("    (no survivors for this generation)")
                        continue

                    for idx, (robot_row, surv_row) in enumerate(rows[:numberrobots]):
                        genome = robot_row.genome

                        developed = algorithm.develop_phenotype(genome)
                        phenotype = developed[0] if isinstance(developed, tuple) else developed
                        draw_phenotype(phenotype, robot_row.robot_id, args.cube_face_size, idx,
                                       round(surv_row.fitness, 4), gen_dir, material_ids, material_colors)


if __name__ == "__main__":
    main()
