#!/usr/bin/env python3
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from evogym import EvoSim, EvoWorld, get_full_connectivity
from evogym.viewer import EvoViewer

from experimental_setups.voxel_types import VOXEL_TYPES


GRID_SIZE = 5
N_ROBOTS = 10
SIM_STEPS = 1000
INIT_Y = 1
START_X = 2
X_SPACING = 9


def sample_connected_body(rng, grid_size=GRID_SIZE, min_voxels=6, max_voxels=14):
    target_voxels = rng.randint(min_voxels, max_voxels)
    body = np.zeros((grid_size, grid_size), dtype=np.int32)

    start = (rng.randrange(grid_size), rng.randrange(grid_size))
    occupied = {start}
    body[start] = rng.choice(
        [
            VOXEL_TYPES["bone"],
            VOXEL_TYPES["fat"],
            VOXEL_TYPES["muscle_h"],
            VOXEL_TYPES["muscle_v"],
        ]
    )

    while len(occupied) < target_voxels:
        x, y = rng.choice(tuple(occupied))
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        rng.shuffle(neighbors)

        placed = False
        for nx, ny in neighbors:
            if 0 <= nx < grid_size and 0 <= ny < grid_size and (nx, ny) not in occupied:
                occupied.add((nx, ny))
                body[nx, ny] = rng.choice(
                    [
                        VOXEL_TYPES["bone"],
                        VOXEL_TYPES["fat"],
                        VOXEL_TYPES["muscle_h"],
                        VOXEL_TYPES["muscle_v"],
                    ]
                )
                placed = True
                break

        if not placed:
            break

    # Ensure at least one actuator exists.
    if not np.any((body == VOXEL_TYPES["muscle_h"]) | (body == VOXEL_TYPES["muscle_v"])):
        ox, oy = rng.choice(tuple(occupied))
        body[ox, oy] = rng.choice([VOXEL_TYPES["muscle_h"], VOXEL_TYPES["muscle_v"]])

    return trim_body(body)


def trim_body(body):
    row_mask = np.any(body != 0, axis=1)
    col_mask = np.any(body != 0, axis=0)
    return body[row_mask][:, col_mask]


def sample_brain(rng, body):
    phase_offsets = np.zeros_like(body, dtype=np.float32)
    actuator_mask = (body == VOXEL_TYPES["muscle_h"]) | (body == VOXEL_TYPES["muscle_v"])

    for idx in zip(*np.where(actuator_mask)):
        phase_offsets[idx] = 0.0 if rng.random() < 0.5 else np.pi

    if np.any(actuator_mask) and not np.any(phase_offsets[actuator_mask] > 0):
        idx = rng.choice(list(zip(*np.where(actuator_mask))))
        phase_offsets[idx] = np.pi

    return {
        "phase_offsets": phase_offsets,
        "action_bias": rng.uniform(0.95, 1.05),
        "action_amplitude": rng.uniform(0.25, 0.45),
        "period_steps": rng.randint(18, 30),
    }


def build_world_and_brains(seed=7):
    rng = random.Random(seed)
    world = EvoWorld()
    brains = {}
    robot_names = []

    for robot_idx in range(N_ROBOTS):
        name = f"robot_{robot_idx}"
        body = sample_connected_body(rng)
        connections = get_full_connectivity(body).astype(np.int32)
        x = START_X + robot_idx * X_SPACING

        world.add_from_array(
            name=name,
            structure=body,
            x=x,
            y=INIT_Y,
            connections=connections,
        )
        brains[name] = sample_brain(rng, body)
        robot_names.append(name)

    return world, brains, robot_names


def main():
    world, brains, robot_names = build_world_and_brains()
    sim = EvoSim(world)
    sim.reset()

    viewer = EvoViewer(sim)
    viewer.track_objects(*robot_names)
    viewer.set_tracking_settings(padding=(6, 4), scale=(0.1, 0.15))

    print("Robots:", robot_names)
    print("Rendering all robots in one world. Close the window to stop.")

    freeze_until = time.monotonic() + 3.0
    while time.monotonic() < freeze_until:
        viewer.render("screen")
        time.sleep(1.0 / 30.0)

    for t in range(SIM_STEPS):
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

        unstable = sim.step()
        viewer.render("screen")
        if unstable:
            print("Simulation became unstable; stopping.")
            break

    freeze_until = time.monotonic() + 10.0
    while time.monotonic() < freeze_until:
        viewer.render("screen")
        time.sleep(1.0 / 30.0)

    viewer.close()


if __name__ == "__main__":
    main()
