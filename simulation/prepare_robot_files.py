import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from evogym import get_full_connectivity


def trim_phenotype_materials(phenotype, phase_offsets=None):
    """
    Trim empty borders from a phenotype and return a 2D grid.
    """
    body = np.asarray(phenotype, dtype=int)
    phase = None if phase_offsets is None else np.asarray(phase_offsets, dtype=np.float32)

    if body.ndim != 2:
        raise ValueError(f"Expected 2D phenotype, got {body.shape}")
    if phase is not None and phase.shape != body.shape:
        raise ValueError(f"Phase-offset shape {phase.shape} does not match phenotype {body.shape}")

    x_mask = np.any(body != 0, axis=1)
    body = body[x_mask]
    if phase is not None:
        phase = phase[x_mask]
    y_mask = np.any(body != 0, axis=0)
    body = body[:, y_mask]
    if phase is not None:
        phase = phase[:, y_mask]
    return body, phase


def _material_maps():
    """
    Map GRN material IDs -> EvoGym voxel IDs.
    phase differences are carried by controller phase offsets.
    """
    EVOGYM = {
        "EMPTY": 0,
        "RIGID": 1,
        "SOFT": 2,
        "H_ACT": 3,
        "V_ACT": 4,
    }

    # bone, fat, horizontal muscle, vertical muscle
    material_to_evogym = {
        0: EVOGYM["EMPTY"],
        1: EVOGYM["RIGID"],
        2: EVOGYM["SOFT"],
        3: EVOGYM["H_ACT"],
        4: EVOGYM["V_ACT"],
    }

    return material_to_evogym


def _build_evogym_robot_data(body_materials, phase_offsets):
    material_to_evogym = _material_maps()

    structure = np.vectorize(lambda m: material_to_evogym.get(int(m), 0), otypes=[int])(body_materials)
    structure = structure.astype(np.int32)
    connections = get_full_connectivity(structure).astype(np.int32)

    # Per-voxel phase offsets from the GRN controller alternation rule.
    if phase_offsets is None:
        phase_offsets = np.zeros_like(structure, dtype=np.float32)
    phase_offsets = np.asarray(phase_offsets, dtype=np.float32)
    phase_offsets[body_materials == 0] = 0.0

    # Simple centralized sine controller defaults in EvoGym action space.
    controller = {
        "action_bias": 1.0,
        "action_amplitude": 0.4,
        "period_steps": 20,
    }

    return structure, connections, phase_offsets, controller


def prepare_robot_files(individual, args):
    """
    Prepare EvoGym robot artifacts from developed phenotype.
    """
    body, phase_offsets_body = trim_phenotype_materials(
        individual.phenotype,
        getattr(individual, "phenotype_phase_offsets", None),
    )
    structure, connections, phase_offsets, controller = _build_evogym_robot_data(body, phase_offsets_body)

    # Keep data in-memory for upcoming EvoGym simulation adapter.
    individual.evogym_structure = structure
    individual.evogym_connections = connections
    individual.evogym_phase_offsets = phase_offsets
    individual.evogym_controller = controller
