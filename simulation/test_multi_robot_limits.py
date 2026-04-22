#!/usr/bin/env python3
import argparse
import csv
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from evogym import EvoSim, EvoWorld, get_full_connectivity
from evogym.utils import VOXEL_TYPES as EVOGYM_VOXEL_TYPES
from evogym.viewer import EvoViewer

from experimental_setups.voxel_types import VOXEL_TYPES


GRID_SIZE = 6
INIT_Y = 3
START_X = 6
VIEWER_RESOLUTION = (2400, 1200)
VIEWER_LOCK_Y = 4
VIEWER_LOCK_HEIGHT = 12
VIEWER_PADDING = (4, 1)
VIEWER_SCALE = (0.03, 0.0)
VIEWER_MANUAL_PADDING = (3.0, 2.0)
WALL_THICKNESS = 1
WALL_MARGIN_X = 4
WALL_MARGIN_Y = 2

# python simulation/test_multi_robot_limits.py \
#   --robot_counts 10,20,30,40 \
#   --actuator_fractions 0.3,0.5,1.0 \
#   --steps 200 \
#   --headless 0

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stress-test EvoGym with many robots in one world."
    )
    parser.add_argument("--steps", type=int, default=300, help="Simulation steps per condition.")
    parser.add_argument("--spacing", type=int, default=6, help="Gap in voxel cells between robot origins.")
    parser.add_argument("--add_walls", type=int, default=1, help="1=surround robots with fixed walls, 0=open world.")
    parser.add_argument("--add_ceiling", type=int, default=0, help="1=also add a top wall, 0=leave top open.")
    parser.add_argument(
        "--robot_counts",
        type=str,
        default="10,20,30,40,50,60,70,80,90,100",
        help="CSV robot counts to test.",
    )
    parser.add_argument(
        "--actuator_fractions",
        type=str,
        default="0.3,0.5,1.0",
        help="CSV actuator fill fractions per 5x5 body.",
    )
    parser.add_argument("--seed", type=int, default=11, help="Random seed.")
    parser.add_argument("--headless", type=int, default=1, help="1=headless, 0=show viewer.")
    parser.add_argument(
        "--manual_camera",
        type=int,
        default=0,
        help="1=disable auto-tracking and use a fixed camera, 0=use auto-tracking.",
    )
    parser.add_argument("--camera_x", type=float, default=None, help="Optional manual camera center x in voxel units.")
    parser.add_argument("--camera_y", type=float, default=None, help="Optional manual camera center y in voxel units.")
    parser.add_argument("--camera_width", type=float, default=None, help="Optional manual camera width in voxel units.")
    parser.add_argument("--camera_height", type=float, default=None, help="Optional manual camera height in voxel units.")
    parser.add_argument("--render_robots", type=int, default=0, help="If >0 and headless=0, render this robot count.")
    parser.add_argument(
        "--render_actuator_fraction",
        type=float,
        default=-1.0,
        help="If >=0 and headless=0, render this actuator fraction (use 0.3, 0.5, or 1.0).",
    )
    parser.add_argument(
        "--render_condition",
        type=str,
        default="",
        help="Legacy render selector '<robots>:<fraction>' or '<robots>:<percent>', e.g. '40:0.5' or '40:50'.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="",
        help="Optional CSV output path for the summary table.",
    )
    return parser.parse_args()


def split_csv_numbers(text, cast):
    return [cast(x.strip()) for x in str(text).split(",") if x.strip()]


def normalize_fraction(value):
    value = float(value)
    if value > 1.0:
        value = value / 100.0
    return round(value, 4)


def should_render_condition(args, robot_count, actuator_fraction):
    if int(args.headless) != 0:
        return False

    target_fraction = normalize_fraction(actuator_fraction)

    if args.render_robots > 0 and args.render_actuator_fraction >= 0:
        return (
            int(args.render_robots) == int(robot_count)
            and normalize_fraction(args.render_actuator_fraction) == target_fraction
        )

    text = str(args.render_condition).strip()
    if text:
        try:
            robots_text, frac_text = [x.strip() for x in text.split(":", 1)]
            return int(robots_text) == int(robot_count) and normalize_fraction(frac_text) == target_fraction
        except Exception:
            return False

    # If graphics are on and no specific target was provided, render every condition.
    return True


def build_dense_body(rng, actuator_fraction):
    """
    Build a fully filled 5x5 body and control the fraction of actuator cells.
    """
    body = np.full((GRID_SIZE, GRID_SIZE), VOXEL_TYPES["fat"], dtype=np.int32)
    total_cells = GRID_SIZE * GRID_SIZE
    actuator_count = max(1, min(total_cells, int(round(total_cells * actuator_fraction))))

    flat_indices = list(range(total_cells))
    rng.shuffle(flat_indices)
    actuator_indices = set(flat_indices[:actuator_count])

    for flat_idx in range(total_cells):
        y, x = divmod(flat_idx, GRID_SIZE)
        if flat_idx in actuator_indices:
            body[y, x] = rng.choice([VOXEL_TYPES["muscle_h"], VOXEL_TYPES["muscle_v"]])
        else:
            body[y, x] = rng.choice([VOXEL_TYPES["fat"], VOXEL_TYPES["bone"]])

    return body


def sample_brain(rng, body):
    phase_offsets = np.zeros_like(body, dtype=np.float32)
    actuator_mask = (body == VOXEL_TYPES["muscle_h"]) | (body == VOXEL_TYPES["muscle_v"])
    actuator_positions = list(zip(*np.where(actuator_mask)))

    for idx in actuator_positions:
        phase_offsets[idx] = 0.0 if rng.random() < 0.5 else np.pi

    if actuator_positions and not np.any(phase_offsets[actuator_mask] > 0):
        phase_offsets[rng.choice(actuator_positions)] = np.pi

    return {
        "phase_offsets": phase_offsets,
        "action_bias": rng.uniform(0.95, 1.05),
        "action_amplitude": rng.uniform(0.25, 0.45),
        "period_steps": rng.randint(18, 30),
    }


def build_world(robot_count, actuator_fraction, spacing, seed):
    rng = random.Random(seed)
    world = EvoWorld()
    brains = {}
    robot_names = []

    for robot_idx in range(robot_count):
        name = f"robot_{robot_idx}"
        body = build_dense_body(rng, actuator_fraction)
        connections = get_full_connectivity(body).astype(np.int32)
        x = START_X + robot_idx * spacing
        y = INIT_Y

        world.add_from_array(
            name=name,
            structure=body,
            x=x,
            y=y,
            connections=connections,
        )
        brains[name] = sample_brain(rng, body)
        robot_names.append(name)

    return world, brains, robot_names


def add_walls(world, robot_count, spacing, add_ceiling):
    """
    Build a fixed container around the one-row robot layout so the occupied
    world stays bounded and easier to view.
    """
    inner_x_min = START_X - WALL_MARGIN_X
    inner_x_max = START_X + (robot_count - 1) * spacing + GRID_SIZE + WALL_MARGIN_X
    # Keep the arena itself strictly above y=0 so the floor can sit at y=0
    # while all object origins remain non-negative for EvoGym.
    inner_y_min = 1
    inner_y_max = INIT_Y + GRID_SIZE + WALL_MARGIN_Y

    wall_height = inner_y_max - inner_y_min
    floor_width = inner_x_max - inner_x_min

    fixed = EVOGYM_VOXEL_TYPES["FIXED"]

    left_wall = np.full((wall_height, WALL_THICKNESS), fixed, dtype=np.int32)
    right_wall = np.full((wall_height, WALL_THICKNESS), fixed, dtype=np.int32)
    floor = np.full((WALL_THICKNESS, floor_width + 2 * WALL_THICKNESS), fixed, dtype=np.int32)

    world.add_from_array("left_wall", left_wall, inner_x_min - WALL_THICKNESS, inner_y_min)
    world.add_from_array("right_wall", right_wall, inner_x_max, inner_y_min)
    world.add_from_array("floor", floor, inner_x_min - WALL_THICKNESS, inner_y_min - WALL_THICKNESS)

    if add_ceiling:
        ceiling = np.full((WALL_THICKNESS, floor_width + 2 * WALL_THICKNESS), fixed, dtype=np.int32)
        world.add_from_array("ceiling", ceiling, inner_x_min - WALL_THICKNESS, inner_y_max)

    return {
        "inner_x_min": inner_x_min,
        "inner_x_max": inner_x_max,
        "inner_y_min": inner_y_min,
        "inner_y_max": inner_y_max,
    }


def compute_manual_camera(robot_count, spacing, resolution, wall_bounds=None):
    """
    Build a stable camera that fits the full one-row layout for the current
    condition and matches the render aspect ratio to avoid visual distortion.
    """
    if wall_bounds is None:
        x_min = START_X - VIEWER_MANUAL_PADDING[0]
        x_max = START_X + (robot_count - 1) * spacing + GRID_SIZE + VIEWER_MANUAL_PADDING[0]
        y_min = INIT_Y - VIEWER_MANUAL_PADDING[1]
        y_max = INIT_Y + GRID_SIZE + VIEWER_MANUAL_PADDING[1]
    else:
        x_min = wall_bounds["inner_x_min"] - VIEWER_MANUAL_PADDING[0]
        x_max = wall_bounds["inner_x_max"] + VIEWER_MANUAL_PADDING[0]
        y_min = wall_bounds["inner_y_min"] - VIEWER_MANUAL_PADDING[1]
        y_max = wall_bounds["inner_y_max"] + VIEWER_MANUAL_PADDING[1]

    width = max(1.0, x_max - x_min)
    height = max(1.0, y_max - y_min)
    aspect = float(resolution[0]) / float(resolution[1])

    if width / height < aspect:
        width = height * aspect
    else:
        height = width / aspect

    pos = ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)
    size = (width, height)
    return pos, size


def run_condition(
    robot_count,
    actuator_fraction,
    spacing,
    steps,
    headless,
    should_render,
    seed,
    manual_camera=False,
    camera_pos=None,
    camera_size=None,
    add_walls_to_world=True,
    add_ceiling=False,
):
    condition_seed = seed + robot_count * 1000 + int(round(actuator_fraction * 100))
    started = time.time()

    try:
        world, brains, robot_names = build_world(robot_count, actuator_fraction, spacing, condition_seed)
        wall_bounds = None
        if add_walls_to_world:
            wall_bounds = add_walls(world, robot_count, spacing, add_ceiling)
        sim = EvoSim(world)
        sim.reset()
    except Exception as exc:
        return {
            "robots": robot_count,
            "actuator_fraction": actuator_fraction,
            "steps_requested": steps,
            "steps_completed": 0,
            "ok": 0,
            "unstable": 0,
            "error": f"{type(exc).__name__}: {exc}",
            "runtime_sec": round(time.time() - started, 2),
        }

    viewer = None
    if should_render and not headless:
        if manual_camera:
            auto_pos, auto_size = compute_manual_camera(
                robot_count,
                spacing,
                VIEWER_RESOLUTION,
                wall_bounds=wall_bounds,
            )
            camera_pos = (
                auto_pos[0] if camera_pos is None or camera_pos[0] is None else camera_pos[0],
                auto_pos[1] if camera_pos is None or camera_pos[1] is None else camera_pos[1],
            )
            camera_size = (
                auto_size[0] if camera_size is None or camera_size[0] is None else camera_size[0],
                auto_size[1] if camera_size is None or camera_size[1] is None else camera_size[1],
            )
        else:
            camera_pos = (12.0, 4.0)
            camera_size = (40.0, 20.0)

        viewer = EvoViewer(
            sim,
            resolution=VIEWER_RESOLUTION,
            pos=tuple(camera_pos),
            view_size=tuple(camera_size),
        )
        if not manual_camera:
            viewer.track_objects(*robot_names)
            # Auto-tracking is convenient for quick sweeps, but a fixed camera
            # is easier to reason about when edge clipping gets distracting.
            viewer.set_tracking_settings(
                padding=VIEWER_PADDING,
                scale=VIEWER_SCALE,
                lock_y=VIEWER_LOCK_Y,
                lock_height=VIEWER_LOCK_HEIGHT,
            )

    unstable = False
    error = ""
    steps_completed = 0

    try:
        for t in range(steps):
            for name in robot_names:
                actuator_indices = sim.get_actuator_indices(name).astype(int).flatten()
                if actuator_indices.size == 0:
                    continue

                brain = brains[name]
                phase_flat = brain["phase_offsets"].reshape(-1)
                actuator_phases = phase_flat[actuator_indices]
                angle = 2.0 * math.pi * (t / brain["period_steps"])
                action = brain["action_bias"] + brain["action_amplitude"] * np.sin(angle + actuator_phases)
                action = np.clip(action, 0.6, 1.6).astype(np.float64)
                sim.set_action(name, action)

            unstable = bool(sim.step())
            steps_completed = t + 1
            if viewer is not None:
                viewer.render("screen")
            if unstable:
                break

        if viewer is not None:
            freeze_until = time.monotonic() + 3.0
            while time.monotonic() < freeze_until:
                viewer.render("screen")
                time.sleep(1.0 / 30.0)

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if viewer is not None:
            viewer.close()

    return {
        "robots": robot_count,
        "actuator_fraction": actuator_fraction,
        "steps_requested": steps,
        "steps_completed": steps_completed,
        "ok": int(not unstable and error == ""),
        "unstable": int(unstable),
        "error": error,
        "runtime_sec": round(time.time() - started, 2),
    }


def main():
    args = parse_args()
    robot_counts = split_csv_numbers(args.robot_counts, int)
    actuator_fractions = split_csv_numbers(args.actuator_fractions, float)
    rows = []

    for robot_count in robot_counts:
        for actuator_fraction in actuator_fractions:
            should_render = should_render_condition(args, robot_count, actuator_fraction)
            print(
                f"[TEST] robots={robot_count} actuator_fraction={actuator_fraction:.2f} "
                f"spacing={args.spacing} steps={args.steps} render={int(should_render)}",
                flush=True,
            )
            row = run_condition(
                robot_count=robot_count,
                actuator_fraction=actuator_fraction,
                spacing=args.spacing,
                steps=args.steps,
                headless=bool(args.headless),
                should_render=should_render,
                seed=args.seed,
                manual_camera=bool(args.manual_camera),
                camera_pos=(args.camera_x, args.camera_y),
                camera_size=(args.camera_width, args.camera_height),
                add_walls_to_world=bool(args.add_walls),
                add_ceiling=bool(args.add_ceiling),
            )
            rows.append(row)
            print(
                f"[RESULT] robots={row['robots']} actuator_fraction={row['actuator_fraction']:.2f} "
                f"ok={row['ok']} unstable={row['unstable']} "
                f"steps={row['steps_completed']}/{row['steps_requested']} "
                f"runtime={row['runtime_sec']}s error={row['error'] or '-'}",
                flush=True,
            )

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "robots",
                    "actuator_fraction",
                    "steps_requested",
                    "steps_completed",
                    "ok",
                    "unstable",
                    "error",
                    "runtime_sec",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"[WROTE] {out_path}")


if __name__ == "__main__":
    main()
