import numpy as np

from evogym.utils import VOXEL_TYPES as EVOGYM_VOXEL_TYPES


WALL_THICKNESS = 1
WALL_MARGIN_X = 4
WALL_MARGIN_Y = 2


def _bbox_distance(ax, ay, x_min, x_max, y_min, y_max):
    dx = max(x_min - ax, 0, ax - x_max)
    dy = max(y_min - ay, 0, ay - y_max)
    return dx + dy


def choose_foraging_robot_position(individual, args, rng):
    """
    Choose the robot start position inside the fixed arena.
    x is horizontal, so robot x is randomized.
    y is vertical, so robot y stays fixed on the floor.
    """
    structure = getattr(individual, "evogym_structure", None)
    if structure is None:
        raise RuntimeError("prepare_robot_files_online() must run before choose_foraging_robot_position().")

    env_width = int(args.evogym_env_width)
    env_height = int(args.evogym_env_height)
    robot_h, robot_w = structure.shape

    # Interior cells start at 1 so the floor and walls can sit just outside.
    inner_x_min = 1
    inner_y_min = 1

    # Robot width must fit inside the open interior width.
    if robot_w > env_width:
        raise ValueError(f"Robot width={robot_w} does not fit in env width={env_width}")
    if robot_h > env_height:
        raise ValueError(f"Robot height={robot_h} does not fit in env height={env_height}")

    x_low = inner_x_min
    # Keep the whole robot inside the open interior.
    x_high = inner_x_min + env_width - robot_w
    return rng.randint(x_low, x_high), inner_y_min


def build_foraging_food(individual, args, rng, robot_x, robot_y):
    """
    Create a two-voxel food object for foraging experiments only. (1-voxel food gives seg fault)
    x is horizontal, so food x is randomized.
    y is vertical, so food y stays fixed on the floor.
    """
    structure = getattr(individual, "evogym_structure", None)
    if structure is None:
        raise RuntimeError("prepare_robot_files_online() must run before build_foraging_food().")

    env_width = int(args.evogym_env_width)
    env_height = int(args.evogym_env_height)
    robot_h, robot_w = structure.shape
    robot_x_min = int(robot_x)
    robot_x_max = int(robot_x) + robot_w - 1
    robot_y_min = int(robot_y)
    robot_y_max = int(robot_y) + robot_h - 1

    min_distance = max(4, robot_w)
    # Interior cells start at 1 so the floor and walls can sit just outside.
    inner_x_min = 1
    inner_y_min = 1

    # Food structure is 1x2, so placement must leave room for both cells.
    food_width = 2
    inner_x_max = inner_x_min + env_width - 1
    inner_y_max = inner_y_min + env_height - 1

    # Robot must stay inside the open interior.
    if robot_x_min < inner_x_min or robot_x_max > inner_x_max:
        raise ValueError(
            f"Robot does not fit in env width={env_width}: robot x-range=({robot_x_min},{robot_x_max})"
        )
    if robot_y_min < inner_y_min or robot_y_max > inner_y_max:
        raise ValueError(
            f"Robot does not fit in env height={env_height}: robot y-range=({robot_y_min},{robot_y_max})"
        )

    x_low = inner_x_min
    # Keep the whole food object inside the open interior.
    x_high = inner_x_min + env_width - food_width
    food_x = min(x_high, robot_x_max + min_distance)
    food_y = inner_y_min
    found = False
    for _ in range(100):
        candidate_x = rng.randint(x_low, x_high)
        if _bbox_distance(
            candidate_x,
            food_y,
            robot_x_min,
            robot_x_max,
            robot_y_min,
            robot_y_max,
        ) >= min_distance:
            food_x = candidate_x
            found = True
            break

    if not found and _bbox_distance(
        food_x,
        food_y,
        robot_x_min,
        robot_x_max,
        robot_y_min,
        robot_y_max,
    ) < min_distance:
        raise ValueError(
            f"Could not place food inside env {env_width}x{env_height} with min_distance={min_distance}"
        )

    # EvoGym's object loader is fragile for singleton extra objects because
    # they have zero connections. Use the smallest connected target instead.
    return {
        "name": "food",
        "structure": np.array([[5, 5]], dtype=np.int32),
        "x": int(food_x),
        "y": int(food_y),
    }


def build_foraging_walls(args):
    """
    Build the fixed arena walls.
    """
    env_width = int(args.evogym_env_width)
    env_height = int(args.evogym_env_height)
    add_ceiling = bool(int(args.evogym_add_ceiling))

    # Interior cells start at 1 so the floor and walls can sit just outside.
    inner_x_min = 1
    inner_y_min = 1
    fixed = EVOGYM_VOXEL_TYPES["FIXED"]

    # Side walls span the full open interior height.
    left_wall = np.full((env_height, WALL_THICKNESS), fixed, dtype=np.int32)
    right_wall = np.full((env_height, WALL_THICKNESS), fixed, dtype=np.int32)
    # Floor spans the full open width plus both wall columns.
    floor = np.full((WALL_THICKNESS, env_width + 2 * WALL_THICKNESS), fixed, dtype=np.int32)

    objects = [
        {
            "name": "left_wall",
            "structure": left_wall,
            "x": int(inner_x_min - WALL_THICKNESS),
            "y": int(inner_y_min),
        },
        {
            "name": "right_wall",
            "structure": right_wall,
            "x": int(inner_x_min + env_width),
            "y": int(inner_y_min),
        },
        {
            "name": "floor",
            "structure": floor,
            "x": int(inner_x_min - WALL_THICKNESS),
            "y": int(inner_y_min - WALL_THICKNESS),
        },
    ]

    if add_ceiling:
        ceiling = np.full((WALL_THICKNESS, env_width + 2 * WALL_THICKNESS), fixed, dtype=np.int32)
        objects.append(
            {
                "name": "ceiling",
                "structure": ceiling,
                "x": int(inner_x_min - WALL_THICKNESS),
                "y": int(inner_y_min + env_height),
            }
        )

    return objects
