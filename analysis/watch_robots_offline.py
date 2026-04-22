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
from simulation.prepare_robot_files import prepare_robot_files
from simulation.offline_simulation import simulate_evogym_batch

PARAM_KEYS = [
    "out_path",
    "study_name",
    "experiments",
    "runs",
    "env_conditions",
    "plastic",
    "max_voxels",
    "cube_face_size",
    "evogym_steps",
    "evogym_init_x",
    "evogym_init_y",
    "evogym_action_bias",
    "evogym_action_amplitude",
    "evogym_period_steps",
    "evogym_render_mode",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Replay robots from saved DBs in EvoGym.")
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
        "--rank_mode",
        type=str,
        choices=["best", "worst"],
        default="best",
        help="Select best or worst fitness.",
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
        help="Render the initial pose for this many seconds before simulation starts.",
    )
    # Backward-compatible no-op flags.
    parser.add_argument("--metric", type=str, default="fitness", help=argparse.SUPPRESS)
    parser.add_argument("--ascending", type=int, default=0, help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = parse_args()
    if str(args.metric).strip().lower() != "fitness":
        print("[info] --metric is ignored; this viewer ranks by fitness.")
    if int(args.ascending) != 0:
        print("[info] --ascending is ignored; use --rank_mode best|worst.")

    params = load_params(Path(args.params_file).resolve())
    generations = [int(x) for x in split_csv(args.generations)]
    manual_ids = [int(x) for x in split_csv(args.robot_ids)]
    worst_first = args.rank_mode == "worst"
    top_k = max(1, int(args.top_k))

    experiments, runs, selected = collect_selected_robots(
        params=params,
        generations=generations,
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


def collect_selected_robots(params, generations, top_k, worst_first, manual_ids):
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
                rows = fetch_manual_ids(db_file, manual_ids)
                if not rows:
                    print(f"[warn] None of the requested IDs found for exp={experiment_name} run={run}")
            elif generations:
                rows = fetch_best_or_worst_per_generation(db_file, generations, top_k, worst_first)
                if not rows:
                    print(f"[warn] No survivors found for exp={experiment_name} run={run}")
            else:
                rows = fetch_best_or_worst_per_run(db_file, top_k, worst_first)

            for row in rows:
                row["experiment_name"] = experiment_name
                row["run"] = run
                row["env_conditions"] = env_conditions
                selected.append(row)

    return experiments, runs, selected


def fetch_best_or_worst_per_run(db_file: Path, top_k: int, worst_first: bool):
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    columns = table_columns(cur, "generation_survivors")
    if "fitness" in columns:
        query = """
            SELECT
                ranked.robot_id AS robot_id,
                ranked.genome AS genome,
                ranked.fitness AS fitness,
                ranked.generation AS generation,
                ranked.born_generation AS born_generation,
                'survivor_generation' AS generation_label
            FROM (
                SELECT
                    r.robot_id AS robot_id,
                    r.genome AS genome,
                    r.born_generation AS born_generation,
                    gs.fitness AS fitness,
                    gs.generation AS generation,
                    ROW_NUMBER() OVER (
                        PARTITION BY r.robot_id
                        ORDER BY gs.fitness {direction}, gs.generation DESC
                    ) AS rank_in_robot
                FROM generation_survivors gs
                JOIN all_robots r ON r.robot_id = gs.robot_id
                WHERE gs.fitness IS NOT NULL
            ) AS ranked
            WHERE ranked.rank_in_robot = 1
            ORDER BY ranked.fitness {direction}, ranked.robot_id ASC
            LIMIT ?
        """.format(direction="ASC" if worst_first else "DESC")
    else:
        query = """
            SELECT
                robot_id AS robot_id,
                genome AS genome,
                displacement AS fitness,
                born_generation AS generation,
                born_generation AS born_generation,
                'born_generation' AS generation_label
            FROM all_robots
            WHERE displacement IS NOT NULL
            ORDER BY displacement {direction}
            LIMIT ?
        """.format(direction="ASC" if worst_first else "DESC")

    rows = cur.execute(query, (top_k,)).fetchall()
    conn.close()
    return rows_to_dicts(rows)


def fetch_best_or_worst_per_generation(db_file: Path, generations, top_k: int, worst_first: bool):
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    placeholders = ",".join(["?"] * len(generations))
    rows = cur.execute(
        f"""
        SELECT
            r.robot_id AS robot_id,
            r.genome AS genome,
            gs.fitness AS fitness,
            gs.generation AS generation,
            r.born_generation AS born_generation,
            'survivor_generation' AS generation_label
        FROM generation_survivors gs
        JOIN all_robots r ON r.robot_id = gs.robot_id
        WHERE gs.generation IN ({placeholders})
          AND gs.fitness IS NOT NULL
        ORDER BY gs.generation ASC, gs.fitness {"ASC" if worst_first else "DESC"}
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


def fetch_manual_ids(db_file: Path, robot_ids):
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    placeholders = ",".join(["?"] * len(robot_ids))
    columns = table_columns(cur, "generation_survivors")
    if "fitness" in columns:
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
                ) AS fitness,
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
                displacement AS fitness,
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
        out_path=params.get("out_path", "tmp_out"),
        study_name=params.get("study_name", "defaultstudy"),
        experiment_name="viz_replay",
        run=0,
        evogym_steps=int(params.get("evogym_steps") or 500),
        evogym_num_workers=1,
        evogym_init_x=int(params.get("evogym_init_x") or 3),
        evogym_init_y=int(params.get("evogym_init_y") or 1),
        evogym_action_bias=float(params.get("evogym_action_bias") or 1.0),
        evogym_action_amplitude=float(params.get("evogym_action_amplitude") or 0.4),
        evogym_period_steps=int(params.get("evogym_period_steps") or 20),
        evogym_headless=0,
        evogym_render_mode=render_mode or params.get("evogym_render_mode") or "screen",
        evogym_freeze_first_frame_seconds=max(0.0, float(freeze_first_frame_seconds)),
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
    print(
        f"\n[{rank}/{total}] exp={entry['experiment_name']} run={entry['run']} "
        f"robot_id={entry['robot_id']} fitness_from_db={entry['fitness']:.1f} "
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

    prepare_robot_files(ind, vis_args)
    simulate_evogym_batch([ind], vis_args)
    print(f"Replay displacement={ind.displacement:.1f}")


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
        raise ValueError(
            f"Field '{field_name}' must have either one value or one per experiment."
        )
    return values[exp_idx]


def db_path(out_path, study_name, experiment_name, run):
    return Path(out_path) / study_name / experiment_name / f"run_{run}" / f"run_{run}"


def table_columns(cur, table_name):
    rows = cur.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row[1] for row in rows}


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
                "fitness": float(row["fitness"]),
                "generation": row["generation"],
                "born_generation": row["born_generation"],
                "generation_label": row["generation_label"],
                "queried_generation": None,
            }
        )
    return out


if __name__ == "__main__":
    main()
