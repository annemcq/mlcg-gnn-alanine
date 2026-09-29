"""
Dataset + normalization helpers for training the CG force field by force
matching: given CG positions R and CG forces F (from src/cg_mapping.py),
learn a model such that -dU/dR ~= F.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class ForceMatchingDataset(Dataset):
    def __init__(self, positions_nm: np.ndarray, forces_kjmolnm: np.ndarray):
        assert positions_nm.shape == forces_kjmolnm.shape
        self.positions = torch.as_tensor(positions_nm, dtype=torch.float32)
        self.forces = torch.as_tensor(forces_kjmolnm, dtype=torch.float32)

    def __len__(self):
        return self.positions.shape[0]

    def __getitem__(self, idx):
        return self.positions[idx], self.forces[idx]


def force_rms_scale(forces_kjmolnm: np.ndarray) -> float:
    """
    A single scalar used to normalize the force-matching loss (root mean
    square force magnitude across the training set), so the loss is
    O(1) regardless of the raw force units/scale.
    """
    return float(np.sqrt(np.mean(np.sum(forces_kjmolnm ** 2, axis=-1))))
