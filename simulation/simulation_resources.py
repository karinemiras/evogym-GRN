#!/usr/bin/env python3
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np


def _resolve_steps(args) -> int:
    steps = int(getattr(args, "evogym_steps", 500))
    return max(1, steps)


def _resolve_workers(args, n_jobs: int) -> int:
    # Debug rendering should run in a single process to avoid multiple windows.
    if int(getattr(args, "evogym_headless", 1)) == 0:
        return 1
    requested = int(getattr(args, "evogym_num_workers", 0))
    if requested > 0:
        return max(1, min(requested, n_jobs))
    cpu = os.cpu_count() or 1
    return max(1, min(cpu, n_jobs))


def _simulate_one_robot(task: Dict) -> Tuple[int, float, str]:
    """
    Returns:
      (robot_id, displacement_x, error_msg)
    """
    from evogym import EvoWorld, EvoSim  # imported here for process safety
    from evogym.viewer import EvoViewer

    robot_id = int(task["id"])
    # structure is [y,x], meaning outer arrays arey y (rows) and inner arrays are x (columns)
    structure = task["structure"]
    connections = task["connections"]
    phase_offsets = task["phase_offsets"]
    bias = float(task["action_bias"])
    amplitude = float(task["action_amplitude"])
    period_steps = max(1, int(task["period_steps"]))
    sim_steps = int(task["sim_steps"])
    init_x = int(task["init_x"])
    init_y = int(task["init_y"])
    headless = bool(int(task["headless"]))
    render_mode = str(task["render_mode"])
    freeze_first_frame_seconds = float(task.get("freeze_first_frame_seconds", 0.0))

    try:
        world = EvoWorld()
        world.add_from_array(
            name="robot",
            structure=structure,
            x=init_x,
            y=init_y,
            connections=connections,
        )

        sim = EvoSim(world)
        sim.reset()
        viewer = None
        if not headless:
            viewer = EvoViewer(sim)
            # follow with camera
            viewer.track_objects("robot")
            if freeze_first_frame_seconds > 0:
                freeze_until = time.monotonic() + freeze_first_frame_seconds
                while time.monotonic() < freeze_until:
                    viewer.render(render_mode)
                    time.sleep(1.0 / 30.0)

        actuator_indices = sim.get_actuator_indices("robot").astype(int).flatten()
        phase_flat = phase_offsets.reshape(-1)
        actuator_phases = phase_flat[actuator_indices] if actuator_indices.size else np.array([])

        # Initial center-of-mass x position.
        # Get points of mass
        p0 = sim.object_pos_at_time(sim.get_time(), "robot")
        x0 = float(np.mean(p0[0]))

        for t in range(sim_steps):
            if actuator_indices.size:
                # oscillatory function
                angle = 2.0 * math.pi * (t / period_steps)
                action = bias + amplitude * np.sin(angle + actuator_phases)
                # default actuation bounds of evogym
                action = np.clip(action, 0.6, 1.6).astype(np.float64)
                sim.set_action("robot", action)

            unstable = sim.step()
            if viewer is not None:
                viewer.render(render_mode)
            if unstable:
                break

        # Final center-of-mass x position.
        p1 = sim.object_pos_at_time(sim.get_time(), "robot")
        x1 = float(np.mean(p1[0]))

        # Behavior metric exported to EA, in EvoSim voxel-length units:
        # forward displacement = final center-of-mass x minus initial center-of-mass x.
        # EvoSim.object_pos_at_time() returns coordinates normalized by VOXEL_SIZE.
        displacement_x = x1 - x0
        if viewer is not None:
            viewer.close()
        return robot_id, displacement_x, ""

    except Exception as exc:
        return robot_id, float("-inf"), f"{type(exc).__name__}: {exc}"


def simulate_evogym_batch(population, args):
    """
    Evaluate all valid individuals in EvoGym and write displacement into each individual.
    """
    sim_steps = _resolve_steps(args)
    init_x = int(getattr(args, "evogym_init_x", 3))
    init_y = int(getattr(args, "evogym_init_y", 1))
    default_bias = float(getattr(args, "evogym_action_bias", 1.0))
    default_amplitude = float(getattr(args, "evogym_action_amplitude", 0.4))
    default_period = int(getattr(args, "evogym_period_steps", 20))
    headless = int(getattr(args, "evogym_headless", 1))
    render_mode = str(getattr(args, "evogym_render_mode", "screen"))
    freeze_first_frame_seconds = float(getattr(args, "evogym_freeze_first_frame_seconds", 0.0))

    id_to_ind = {ind.id: ind for ind in population}
    tasks: List[Dict] = []

    for ind in population:
        if not getattr(ind, "valid", True):
            continue

        if not hasattr(ind, "evogym_structure"):
            raise RuntimeError(
                f"Robot {ind.id} missing EvoGym payload. "
                "Call prepare_robot_files(individual, args) before simulation."
            )

        ctrl = getattr(ind, "evogym_controller", {})
        task = {
            "id": ind.id,
            "structure": ind.evogym_structure,
            "connections": ind.evogym_connections,
            "phase_offsets": ind.evogym_phase_offsets,
            "action_bias": ctrl.get("action_bias", default_bias),
            "action_amplitude": ctrl.get("action_amplitude", default_amplitude),
            "period_steps": ctrl.get("period_steps", default_period),
            "sim_steps": sim_steps,
            "init_x": init_x,
            "init_y": init_y,
            "headless": headless,
            "render_mode": render_mode,
            "freeze_first_frame_seconds": freeze_first_frame_seconds,
        }
        tasks.append(task)

    if not tasks:
        print("[SIM-DONE] total=0 ok=0 failed=0")
        return

    n_workers = _resolve_workers(args, len(tasks))

    ok = 0
    failed = 0

    def _store_displacement(ind, disp):
        # Store displacement in voxel-length units at one decimal place for DB/fitness consistency.
        ind.displacement = round(float(disp), 1) if np.isfinite(disp) else float(disp)

    if n_workers == 1:
        for task in tasks:
            rid, disp, err = _simulate_one_robot(task)
            ind = id_to_ind[rid]
            _store_displacement(ind, disp)
            if err:
                failed += 1
                print(f"[SIM-FAIL] {rid}: {err}")
            else:
                ok += 1
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            fut_to_task = {ex.submit(_simulate_one_robot, t): t for t in tasks}
            for fut in as_completed(fut_to_task):
                task = fut_to_task[fut]
                try:
                    rid, disp, err = fut.result()
                except Exception as exc:
                    # EvoGym's C++ backend can occasionally terminate a worker
                    # abruptly for a malformed/unstable robot. Keep the EA run
                    # alive by assigning that individual -inf fitness.
                    rid = int(task["id"])
                    disp = float("-inf")
                    err = f"{type(exc).__name__}: {exc}"
                ind = id_to_ind[rid]
                _store_displacement(ind, disp)
                if err:
                    failed += 1
                    print(f"[SIM-FAIL] {rid}: {err}")
                else:
                    ok += 1

    print(
        f"[SIM-DONE] total={len(tasks)} ok={ok} failed={failed} "
        f"workers={n_workers} steps={sim_steps}"
    )
