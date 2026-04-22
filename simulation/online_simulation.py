#!/usr/bin/env python3
import math
from typing import Dict, Tuple

import numpy as np


def _rect_from_object(name: str, structure, x: int, y: int):
    arr = np.asarray(structure)
    if arr.ndim != 2:
        raise ValueError(f"Object {name} structure must be 2D, got shape={arr.shape}")
    h, w = arr.shape
    return {
        "name": name,
        "x0": int(x),
        "y0": int(y),
        "x1": int(x) + int(w) - 1,
        "y1": int(y) + int(h) - 1,
    }


def _overlap(a, b) -> bool:
    return not (a["x1"] < b["x0"] or b["x1"] < a["x0"] or a["y1"] < b["y0"] or b["y1"] < a["y0"])


def _validate_task_geometry(task: Dict) -> None:
    rects = []

    # Robot rectangle from its start position.
    robot_rect = _rect_from_object("robot", task["structure"], task["init_x"], task["init_y"])
    rects.append(robot_rect)

    # Each extra object must stay inside valid coordinates and avoid overlaps.
    for obj in task["extra_objects"]:
        name = str(obj["name"])
        x = int(obj["x"])
        y = int(obj["y"])
        if x < 0 or y < 0:
            raise ValueError(f"Object {name} has invalid negative origin ({x}, {y})")
        rect = _rect_from_object(name, obj["structure"], x, y)
        for other in rects:
            if _overlap(rect, other):
                raise ValueError(
                    f"Object {name} overlaps {other['name']} "
                    f"({rect['x0']},{rect['y0']})-({rect['x1']},{rect['y1']}) vs "
                    f"({other['x0']},{other['y0']})-({other['x1']},{other['y1']})"
                )
        rects.append(rect)


def _object_center_xy(sim, object_name: str) -> Tuple[float, float]:
    # EvoGym returns point coordinates as [x_coords, y_coords].
    points = sim.object_pos_at_time(sim.get_time(), object_name)
    return float(np.mean(points[0])), float(np.mean(points[1]))


def _extract_foraging_inputs(sim, task: Dict) -> Tuple[float, float]:
    """
    Return the basic online-controller inputs:
    signed dx, dy from robot center to food center.
    dx is horizontal and dy is vertical.
    """
    # Arena size used to normalize controller inputs.
    env_width = float(task["env_width"])
    env_height = float(task["env_height"])
    # Current robot center.
    robot_x, robot_y = _object_center_xy(sim, "robot")
    # Current food center.
    food_x, food_y = _object_center_xy(sim, "food")
    # Signed horizontal distance normalized by arena width.
    dx = (food_x - robot_x) / env_width
    # Signed vertical distance normalized by arena height.
    dy = (food_y - robot_y) / env_height
    return float(dx), float(dy)


def _center_distance(sim, a_name: str, b_name: str) -> float:
    ax, ay = _object_center_xy(sim, a_name)
    bx, by = _object_center_xy(sim, b_name)
    return math.hypot(bx - ax, by - ay)
