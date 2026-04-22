#!/usr/bin/env python3
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from simulation.online_simulation import _center_distance
from simulation.online_simulation import _extract_foraging_inputs
from simulation.online_simulation import _validate_task_geometry
from simulation.foraging_objects import build_foraging_food
from simulation.foraging_objects import build_foraging_walls
from simulation.foraging_objects import choose_foraging_robot_position


class ForagingEnv(gym.Env):
    metadata = {"render_modes": ["screen", "human"]}

    def __init__(self, task: Dict):
        from evogym import EvoWorld, EvoSim
        from evogym.viewer import EvoViewer

        self.EvoWorld = EvoWorld
        self.EvoSim = EvoSim
        self.EvoViewer = EvoViewer

        self.task = task
        self.structure = task["structure"]
        self.connections = task["connections"]
        self.phase_offsets = task["phase_offsets"]
        self.extra_objects = list(task.get("extra_objects", []))
        self.bias = float(task["action_bias"])
        self.amplitude = float(task["action_amplitude"])
        self.period_steps = max(1, int(task["period_steps"]))
        self.episode_steps = int(task["episode_steps"])
        self.init_x = int(task["init_x"])
        self.init_y = int(task["init_y"])
        self.headless = bool(int(task["headless"]))
        self.render_mode = str(task["render_mode"])
        self.freeze_first_frame_seconds = float(task["freeze_first_frame_seconds"])

        n_actuators = int(np.count_nonzero((self.structure == 3) | (self.structure == 4)))
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_actuators,), dtype=np.float32)

        self.sim = None
        self.viewer = None
        self.actuator_phases = None
        self.step_count = 0
        self.prev_distance = 0.0
        self.food_taken = 0.0
        self.steps_until_food = float(self.episode_steps)

    def _build_sim(self):
        _validate_task_geometry(self.task)

        world = self.EvoWorld()
        world.add_from_array(
            name="robot",
            structure=self.structure,
            x=self.init_x,
            y=self.init_y,
            connections=self.connections,
        )
        for obj in self.extra_objects:
            world.add_from_array(
                name=str(obj["name"]),
                structure=np.asarray(obj["structure"], dtype=np.int32),
                x=int(obj["x"]),
                y=int(obj["y"]),
                connections=obj.get("connections", None),
            )

        self.sim = self.EvoSim(world)
        self.sim.reset()

        if not self.headless:
            self.viewer = self.EvoViewer(self.sim)
            track_names = ["robot"] + [str(obj["name"]) for obj in self.extra_objects]
            self.viewer.track_objects(*track_names)
            if self.freeze_first_frame_seconds > 0:
                freeze_until = self.freeze_first_frame_seconds
                while freeze_until > 0.0:
                    self.viewer.render(self.render_mode)
                    freeze_until -= 1.0 / 30.0

        actuator_indices = self.sim.get_actuator_indices("robot").astype(int).flatten()
        phase_flat = self.phase_offsets.reshape(-1)
        self.actuator_phases = phase_flat[actuator_indices] if actuator_indices.size else np.array([])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

        self._build_sim()
        self.step_count = 0
        self.prev_distance = _center_distance(self.sim, "robot", "food")
        self.food_taken = 0.0
        self.steps_until_food = float(self.episode_steps)
        dx, dy = _extract_foraging_inputs(self.sim, self.task)
        # Policy input is the two signed distances to food.
        obs = np.array([dx, dy], dtype=np.float32)
        return obs, {}

    def step(self, action):
        if self.actuator_phases.size:
            # PPO outputs one gain per actuator voxel.
            actuator_gains = np.asarray(action, dtype=np.float64).flatten()
            # Angular frequency of the shared sine oscillator, in radians per simulation step.
            angular_frequency = 2.0 * math.pi / self.period_steps
            # Global oscillator phase at this simulation step.
            global_phase = angular_frequency * self.step_count
            # Phased sine wave from the current oscillator.
            phased_signal = np.sin(global_phase + self.actuator_phases)
            # PPO gains modulate each actuator's oscillator output.
            actuator_targets = self.bias + self.amplitude * actuator_gains * phased_signal
            # Keep actuator targets inside EvoGym's supported actuation range.
            actuator_targets = np.clip(actuator_targets, 0.6, 1.6).astype(np.float64)
            self.sim.set_action("robot", actuator_targets)

        unstable = self.sim.step()
        if self.viewer is not None:
            self.viewer.render(self.render_mode)

        current_distance = _center_distance(self.sim, "robot", "food")
        # reward for moving closer
        reward = self.prev_distance - current_distance
        self.prev_distance = current_distance

        terminated = False
        ended_by_timeout = False

        # reward for fetching food
        if current_distance <= 1.0:
            reward += 1.0
            self.food_taken = 1.0
            self.steps_until_food = float(self.step_count)
            print(f"[FOOD-TAKEN] robot_id={self.task['id']}")
            terminated = True
        if unstable:
            terminated = True
        self.step_count += 1
        if self.step_count >= self.episode_steps:
            ended_by_timeout = True

        dx, dy = _extract_foraging_inputs(self.sim, self.task)
        # Policy input is the two signed distances to food.
        obs = np.array([dx, dy], dtype=np.float32)
        info = {
            "distance": current_distance,
            "food_taken": self.food_taken,
            "steps_until_food": self.steps_until_food,
        }
        return obs, float(reward), terminated, ended_by_timeout, info

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


def _train_ppo_for_task(task: Dict, args, seed: int) -> Tuple[float, float, float]:
    import torch
    from stable_baselines3 import PPO

    # One environment for one robot body and one food target.
    env = ForagingEnv(task)
    # One hidden layer sized relative to the number of actuator voxels.
    hidden_size = max(4, env.action_space.shape[0] // 2)
    policy_kwargs = {
        "activation_fn": torch.nn.Tanh,
        "net_arch": {"pi": [hidden_size], "vf": [hidden_size]},
    }
    # PPO learns the policy that maps dx, dy to one gain per actuator.
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=seed,
        n_steps=int(args.ppo_n_steps),
        batch_size=int(args.ppo_batch_size),
    )
    # Lifetime learning budget for this robot.
    model.learn(total_timesteps=int(args.ppo_timesteps))

    # Run the learned policy once to get the final behavior score.
    obs, _ = env.reset()
    total_reward = 0.0
    terminated = False
    ended_by_timeout = False
    final_food_taken = 0.0
    final_steps_until_food = float(task["episode_steps"])
    while not (terminated or ended_by_timeout):
        # Policy forward pass.
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, ended_by_timeout, info = env.step(action)
        total_reward += reward
        final_food_taken = float(info["food_taken"])
        final_steps_until_food = float(info["steps_until_food"])
    env.close()
    print(total_reward)
    return float(total_reward), float(final_food_taken), float(final_steps_until_food)


def _resolve_workers(args, n_jobs: int) -> int:
    # Debug rendering should run in a single process to avoid multiple windows.
    if int(args.evogym_headless) == 0:
        return 1
    requested = int(args.evogym_num_workers)
    if requested > 0:
        return max(1, min(requested, n_jobs))
    cpu = os.cpu_count() or 1
    return max(1, min(cpu, n_jobs))


def _seed_for_robot(experiment_seed: int, robot_id: int) -> int:
    return (int(experiment_seed) * 1000003 + int(robot_id)) % (2**32)


def _build_foraging_task(ind, args, experiment_seed: int, batch_walls: List[Dict]) -> Tuple[Dict, int]:
    ctrl = getattr(ind, "evogym_controller", {})
    robot_seed = _seed_for_robot(experiment_seed, ind.id)
    robot_rng = random.Random(robot_seed)
    # Robot start position.
    robot_x, robot_y = choose_foraging_robot_position(ind, args, robot_rng)
    # Food is specific to this robot. Walls are shared across the batch.
    extra_objects = list(batch_walls)
    extra_objects.append(build_foraging_food(ind, args, robot_rng, robot_x, robot_y))
    # Everything needed to build one robot-specific PPO environment.
    task = {
        "id": ind.id,
        "structure": ind.evogym_structure,
        "connections": ind.evogym_connections,
        "phase_offsets": ind.evogym_phase_offsets,
        "action_bias": ctrl.get("action_bias", float(args.evogym_action_bias)),
        "action_amplitude": ctrl.get("action_amplitude", float(args.evogym_action_amplitude)),
        "period_steps": ctrl.get("period_steps", int(args.evogym_period_steps)),
        "episode_steps": int(args.ppo_timesteps),
        "init_x": int(robot_x),
        "init_y": int(robot_y),
        "extra_objects": extra_objects,
        "headless": int(args.evogym_headless),
        "render_mode": str(args.evogym_render_mode),
        "freeze_first_frame_seconds": float(args.evogym_freeze_first_frame_seconds),
        "env_width": int(args.evogym_env_width),
        "env_height": int(args.evogym_env_height),
    }
    return task, robot_seed


def replay_ppo_individual(ind, args, experiment_seed: int) -> Tuple[float, float, float]:
    batch_walls = []
    if int(args.evogym_add_walls) != 0:
        batch_walls = build_foraging_walls(args)
    task, robot_seed = _build_foraging_task(ind, args, experiment_seed, batch_walls)
    return _train_ppo_for_task(task, args, robot_seed)


def _train_task_wrapper(task: Dict, args, robot_seed: int) -> Tuple[int, float, float, float, str]:
    try:
        reward, food_taken, steps_until_food = _train_ppo_for_task(task, args, robot_seed)
        return int(task["id"]), reward, food_taken, steps_until_food, ""
    except Exception as exc:
        return int(task["id"]), float("-inf"), 0.0, float(task["episode_steps"]), f"{type(exc).__name__}: {exc}"


def train_ppo_batch(population, args, experiment_seed: int):
    tasks: List[Tuple[object, Dict, int]] = []
    # Walls are the same for every robot in this batch.
    batch_walls = []
    if int(args.evogym_add_walls) != 0:
        batch_walls = build_foraging_walls(args)

    for ind in population:
        if not getattr(ind, "valid", True):
            continue
        task, robot_seed = _build_foraging_task(ind, args, experiment_seed, batch_walls)
        tasks.append((ind, task, robot_seed))

    if not tasks:
        print("[PPO-DONE] total=0 ok=0 failed=0")
        return

    n_workers = _resolve_workers(args, len(tasks))
    ok = 0
    failed = 0
    id_to_ind = {ind.id: ind for ind, _, _ in tasks}

    if n_workers == 1:
        for ind, task, robot_seed in tasks:
            rid, reward, food_taken, steps_until_food, err = _train_task_wrapper(task, args, robot_seed)
            result_ind = id_to_ind[rid]
            result_ind.reward = round(reward, 3) if np.isfinite(reward) else float(reward)
            result_ind.food_taken = float(food_taken)
            result_ind.steps_until_food = float(steps_until_food)
            if err:
                failed += 1
                print(f"[PPO-FAIL] {rid}: {err}")
            else:
                ok += 1
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            fut_to_task = {
                ex.submit(_train_task_wrapper, task, args, robot_seed): (ind, task)
                for ind, task, robot_seed in tasks
            }
            for fut in as_completed(fut_to_task):
                ind, task = fut_to_task[fut]
                try:
                    rid, reward, food_taken, steps_until_food, err = fut.result()
                except Exception as exc:
                    rid = int(task["id"])
                    reward = float("-inf")
                    food_taken = 0.0
                    steps_until_food = float(task["episode_steps"])
                    err = f"{type(exc).__name__}: {exc}"
                result_ind = id_to_ind[rid]
                result_ind.reward = round(reward, 3) if np.isfinite(reward) else float(reward)
                result_ind.food_taken = float(food_taken)
                result_ind.steps_until_food = float(steps_until_food)
                if err:
                    failed += 1
                    print(f"[PPO-FAIL] {rid}: {err}")
                else:
                    ok += 1

    print(
        f"[PPO-DONE] total={len(tasks)} ok={ok} failed={failed} "
        f"workers={n_workers} timesteps={int(args.ppo_timesteps)}"
    )
