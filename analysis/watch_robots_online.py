#!/usr/bin/env python3
import argparse
import json
import shlex
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from experimental_setups.EA_classes import Individual
from experimental_setups.GRN_2D import GRN
from simulation.prepare_robot_files import prepare_robot_files_online
from simulation.foraging_ppo import replay_ppo_individual

PARAM_KEYS = [
    "out_path",
    "study_name",
    "experiments",
    "runs",
    "env_conditions",
    "plastic",
    "max_voxels",
    "cube_face_size",
    "fitness_metric",
    "evogym_steps",
    "evogym_action_bias",
    "evogym_action_amplitude",
    "evogym_period_steps",
    "evogym_render_mode",
    "evogym_freeze_first_frame_seconds",
    "evogym_add_walls",
    "evogym_add_ceiling",
    "evogym_env_width",
    "evogym_env_height",
    "ppo_timesteps",
    "ppo_n_steps",
    "ppo_batch_size",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Replay online foraging robots from saved DBs.")
    parser.add_argument(
        "--params_file",
        default=str(ROOT / "automation" / "setups" / "foraging.sh"),
        help="Path to experiment params .sh file.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=1,
        help="If generations are set: top_k per generation. Otherwise: top_k per run.",
    )
    parser.add_argument(
        "--generations",
        type=str,
        default="",
        help="CSV generation filter, e.g. '10,20,50'. Empty means use all generations.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["fitness", "reward", "food_taken", "steps_until_food"],
        default="reward",
        help="Metric used to rank saved robots.",
    )
    parser.add_argument(
        "--rank_mode",
        type=str,
        choices=["best", "worst"],
        default="best",
        help="Select best or worst metric values.",
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        default=None,
        help="Optional override for EvoGym render mode.",
    )
    parser.add_argument(
        "--robot_ids",
        type=str,
        default="",
        help="CSV robot IDs to replay manually. If set, ranking is ignored.",
    )
    parser.add_argument(
        "--freeze_first_frame_seconds",
        type=float,
        default=0,
        help="Render the initial pose for this many seconds before PPO replay starts.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    params = load_params(Path(args.params_file).resolve())
    generations = [int(x) for x in split_csv(args.generations)]
    manual_ids = [int(x) for x in split_csv(args.robot_ids)]
    worst_first = args.rank_mode == "worst"
    top_k = max(1, int(args.top_k))

    experiments, runs, selected = collect_selected_robots(
        params=params,
        generations=generations,
        metric=args.metric,
        top_k=top_k,
        worst_first=worst_first,
        manual_ids=manual_ids,
    )

    if not selected:
        print("No candidates found.")
        return

    vis_args = build_vis_args(params, args.render_mode, args.freeze_first_frame_seconds)

    print(f"Params file: {args.params_file}")
    print(f"Study: {params.get('study_name')}")
    print(f"Experiments: {experiments}")
    print(f"Runs: {runs}")
    print(f"Metric: {args.metric}")
    if manual_ids:
        print(f"Manual robot IDs: {manual_ids}")
    else:
        print(f"Rank mode: {args.rank_mode}")
    if manual_ids:
        print("Generations: ignored (manual ID mode)")
    elif generations:
        print(f"Generations: {generations}")
        print(f"top_k per generation: {top_k}")
    else:
        print("Generations: all")
        print(f"top_k per run: {top_k}")
    print(f"Selected robots total: {len(selected)}")

    last_query = None
    for idx, entry in enumerate(selected, start=1):
        if generations:
            query_key = (entry["experiment_name"], entry["run"], entry["queried_generation"])
            if query_key != last_query:
                print(
                    f"\nQuery generation: exp={entry['experiment_name']} "
                    f"run={entry['run']} generation={entry['queried_generation']}"
                )
                last_query = query_key
        replay_robot(entry, vis_args, params, idx, len(selected))


def collect_selected_robots(params, generations, metric, top_k, worst_first, manual_ids):
    experiments = split_csv(params.get("experiments"))
    runs = [int(x) for x in split_csv(params.get("runs"))]
    env_conditions_list = split_csv(params.get("env_conditions"))

    if not experiments:
        raise ValueError("No experiments found in params file.")
    if not runs:
        raise ValueError("No runs found in params file.")

    selected = []
    for exp_idx, experiment_name in enumerate(experiments):
        env_conditions = pick_for_experiment(env_conditions_list, exp_idx, "env_conditions")

        for run in runs:
            db_file = db_path(params["out_path"], params["study_name"], experiment_name, run)
            if not db_file.exists():
                print(f"[skip] DB missing: {db_file}")
                continue

            if manual_ids:
                rows = fetch_manual_ids(db_file, manual_ids, metric)
            elif generations:
                rows = fetch_best_or_worst_per_generation(db_file, generations, metric, top_k, worst_first)
            else:
                rows = fetch_best_or_worst_per_run(db_file, metric, top_k, worst_first)

            for row in rows:
                row["experiment_name"] = experiment_name
                row["run"] = run
                row["env_conditions"] = env_conditions
                selected.append(row)

    return experiments, runs, selected


def fetch_best_or_worst_per_run(db_file: Path, metric: str, top_k: int, worst_first: bool):
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    if metric == "fitness":
        direction = "ASC" if worst_first else "DESC"
        query = f"""
            SELECT
                ranked.robot_id AS robot_id,
                ranked.genome AS genome,
                ranked.metric_value AS metric_value,
                ranked.generation AS generation,
                ranked.born_generation AS born_generation,
                'survivor_generation' AS generation_label
            FROM (
                SELECT
                    r.robot_id AS robot_id,
                    r.genome AS genome,
                    r.born_generation AS born_generation,
                    gs.fitness AS metric_value,
                    gs.generation AS generation,
                    ROW_NUMBER() OVER (
                        PARTITION BY r.robot_id
                        ORDER BY gs.generation DESC
                    ) AS survivor_rank
                FROM generation_survivors gs
                JOIN all_robots r ON r.robot_id = gs.robot_id
                WHERE gs.fitness IS NOT NULL
            ) AS ranked
            WHERE ranked.survivor_rank = 1
            ORDER BY ranked.metric_value {direction}, ranked.robot_id ASC
            LIMIT ?
        """
    else:
        direction = "ASC" if worst_first else "DESC"
        query = f"""
            SELECT
                robot_id AS robot_id,
                genome AS genome,
                {metric} AS metric_value,
                born_generation AS generation,
                born_generation AS born_generation,
                'born_generation' AS generation_label
            FROM all_robots
            WHERE {metric} IS NOT NULL
            ORDER BY {metric} {direction}, robot_id ASC
            LIMIT ?
        """

    rows = cur.execute(query, (top_k,)).fetchall()
    conn.close()
    return rows_to_dicts(rows)


def fetch_best_or_worst_per_generation(db_file: Path, generations, metric: str, top_k: int, worst_first: bool):
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    placeholders = ",".join(["?"] * len(generations))
    direction = "ASC" if worst_first else "DESC"
    metric_expr = "gs.fitness" if metric == "fitness" else f"r.{metric}"
    rows = cur.execute(
        f"""
        SELECT
            r.robot_id AS robot_id,
            r.genome AS genome,
            {metric_expr} AS metric_value,
            gs.generation AS generation,
            r.born_generation AS born_generation,
            'survivor_generation' AS generation_label
        FROM generation_survivors gs
        JOIN all_robots r ON r.robot_id = gs.robot_id
        WHERE gs.generation IN ({placeholders})
          AND {metric_expr} IS NOT NULL
        ORDER BY gs.generation ASC, {metric_expr} {direction}, r.robot_id ASC
        """,
        tuple(generations),
    ).fetchall()
    conn.close()

    rows = rows_to_dicts(rows)
    selected = []
    for generation in generations:
        picked = 0
        for row in rows:
            if row["generation"] != generation:
                continue
            row["queried_generation"] = generation
            selected.append(row)
            picked += 1
            if picked >= top_k:
                break
    return selected


def fetch_manual_ids(db_file: Path, robot_ids, metric: str):
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    placeholders = ",".join(["?"] * len(robot_ids))
    if metric == "fitness":
        query = f"""
            SELECT
                r.robot_id AS robot_id,
                r.genome AS genome,
                (
                    SELECT gs.fitness
                    FROM generation_survivors gs
                    WHERE gs.robot_id = r.robot_id
                    ORDER BY gs.generation DESC
                    LIMIT 1
                ) AS metric_value,
                (
                    SELECT gs.generation
                    FROM generation_survivors gs
                    WHERE gs.robot_id = r.robot_id
                    ORDER BY gs.generation DESC
                    LIMIT 1
                ) AS generation,
                r.born_generation AS born_generation,
                'survivor_generation' AS generation_label
            FROM all_robots r
            WHERE r.robot_id IN ({placeholders})
            ORDER BY r.robot_id
        """
    else:
        query = f"""
            SELECT
                robot_id AS robot_id,
                genome AS genome,
                {metric} AS metric_value,
                born_generation AS generation,
                born_generation AS born_generation,
                'born_generation' AS generation_label
            FROM all_robots
            WHERE robot_id IN ({placeholders})
            ORDER BY robot_id
        """

    rows = cur.execute(query, tuple(robot_ids)).fetchall()
    conn.close()
    return rows_to_dicts(rows)


def build_vis_args(params, render_mode, freeze_first_frame_seconds):
    return SimpleNamespace(
        evogym_action_bias=float(params.get("evogym_action_bias") or 1.0),
        evogym_action_amplitude=float(params.get("evogym_action_amplitude") or 0.4),
        evogym_period_steps=int(params.get("evogym_period_steps") or 20),
        evogym_headless=0,
        evogym_render_mode=render_mode or params.get("evogym_render_mode") or "screen",
        evogym_freeze_first_frame_seconds=max(
            0.0,
            float(freeze_first_frame_seconds if freeze_first_frame_seconds is not None else (params.get("evogym_freeze_first_frame_seconds") or 0.0)),
        ),
        evogym_add_walls=int(params.get("evogym_add_walls") or 1),
        evogym_add_ceiling=int(params.get("evogym_add_ceiling") or 0),
        evogym_env_width=int(params.get("evogym_env_width") or 100),
        evogym_env_height=int(params.get("evogym_env_height") or 20),
        ppo_timesteps=int(params.get("ppo_timesteps") or 2048),
        ppo_n_steps=int(params.get("ppo_n_steps") or 256),
        ppo_batch_size=int(params.get("ppo_batch_size") or 64),
    )


def build_phenotype(genome, max_voxels, cube_face_size, env_conditions, plastic):
    cells = GRN(
        max_voxels=max_voxels,
        cube_face_size=cube_face_size,
        genotype=genome,
        env_conditions=env_conditions,
        plastic=plastic,
    ).develop()

    phenotype = np.zeros(cells.shape, dtype=int)
    phase_offsets = np.zeros(cells.shape, dtype=np.float32)
    for idx, value in np.ndenumerate(cells):
        if value != 0:
            phenotype[idx] = value.voxel_type
            phase_offsets[idx] = value.phase_offset
    return phenotype, phase_offsets


def replay_robot(entry, vis_args, params, rank, total):
    experiment_seed = load_experiment_seed(
        db_path(params["out_path"], params["study_name"], entry["experiment_name"], entry["run"])
    )

    print(
        f"\n[{rank}/{total}] exp={entry['experiment_name']} run={entry['run']} "
        f"robot_id={entry['robot_id']} metric_from_db={entry['metric_value']:.3f} "
        f"born_generation={entry['born_generation']}"
    )

    ind = Individual(genome=entry["genome"], id_counter=entry["robot_id"])
    ind.valid = 1
    ind.phenotype, ind.phenotype_phase_offsets = build_phenotype(
        genome=entry["genome"],
        max_voxels=int(params.get("max_voxels") or 64),
        cube_face_size=int(params.get("cube_face_size") or 4),
        env_conditions=entry["env_conditions"],
        plastic=int(params.get("plastic") or 0),
    )

    prepare_robot_files_online(ind, vis_args)
    reward, food_taken, steps_until_food = replay_ppo_individual(ind, vis_args, experiment_seed)
    print(
        f"Replay reward={reward:.3f} "
        f"food_taken={food_taken:.0f} "
        f"steps_until_food={steps_until_food:.0f}"
    )


def load_experiment_seed(db_file: Path) -> int:
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    row = cur.execute("SELECT seed FROM experiment_info LIMIT 1").fetchone()
    conn.close()
    if row is None:
        raise RuntimeError(f"No experiment seed found in {db_file}")
    return int(row[0])


def split_csv(text):
    if text is None:
        return []
    return [x.strip() for x in str(text).split(",") if x.strip()]


def load_params(params_path: Path):
    if not params_path.exists():
        raise FileNotFoundError(f"params file not found: {params_path}")

    keys = " ".join(PARAM_KEYS)
    cmd = (
        f"set -a; source {shlex.quote(str(params_path))}; "
        f"for k in {keys}; do printf '%s=%s\\n' \"$k\" \"${{!k-}}\"; done"
    )
    proc = subprocess.run(
        ["bash", "-lc", cmd],
        check=True,
        text=True,
        capture_output=True,
    )

    params = {}
    for line in proc.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        params[key] = value
    return params


def pick_for_experiment(values, exp_idx, field_name):
    if not values:
        raise ValueError(f"Missing required field: {field_name}")
    if len(values) == 1:
        return values[0]
    if exp_idx >= len(values):
        raise ValueError(f"Field '{field_name}' must have either one value or one per experiment.")
    return values[exp_idx]


def db_path(out_path, study_name, experiment_name, run):
    return Path(out_path) / study_name / experiment_name / f"run_{run}" / f"run_{run}"


def rows_to_dicts(rows):
    out = []
    for row in rows:
        genome = row["genome"]
        if isinstance(genome, str):
            genome = json.loads(genome)
        out.append(
            {
                "robot_id": int(row["robot_id"]),
                "genome": genome,
                "metric_value": float(row["metric_value"]),
                "generation": row["generation"],
                "born_generation": row["born_generation"],
                "generation_label": row["generation_label"],
                "queried_generation": None,
            }
        )
    return out


if __name__ == "__main__":
    main()
