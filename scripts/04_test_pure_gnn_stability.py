"""Test the stability of the pure-GNN model across Langevin seeds.

This reproduces the deliberately simplified pure-GNN experiment from notebook 03,
but evaluates several dynamics seeds so the reported instability is not tied to
one stochastic trajectory.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from src.cg_dataset import ForceMatchingDataset, force_rms_scale
from src.cg_dynamics import run_langevin
from src.cg_gnn_model import CGSchNetLike, ScaledForceModel

DATA_DIR = BASE / "data"
RESULTS_DIR = BASE / "results"

MASS_AMU = np.array([43.044, 15.018, 28.052, 28.010, 30.052])
TRAINING_SEED = 0
DYNAMICS_SEEDS = (1, 2, 3)
N_EPOCHS = 15
N_STEPS = 5_000
SAVE_EVERY = 20


def train_pure_gnn():
    torch.manual_seed(TRAINING_SEED)
    np.random.seed(TRAINING_SEED)

    positions = np.load(DATA_DIR / "cg_positions_nm.npy")
    forces = np.load(DATA_DIR / "cg_forces_kjmolnm.npy")
    centered = positions - positions.mean(axis=1, keepdims=True)
    rms_force = force_rms_scale(forces)

    dataset = ForceMatchingDataset(centered, forces / rms_force)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    model = CGSchNetLike(
        n_beads=5, n_features=32, n_rbf=32, n_interactions=2, d_max=1.0
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(N_EPOCHS):
        for pos, force in loader:
            optimizer.zero_grad()
            _, predicted_force = model(pos)
            loss = torch.mean((predicted_force - force) ** 2)
            loss.backward()
            optimizer.step()

    model.eval()
    return model, rms_force, centered[0]


def main():
    model, rms_force, start = train_pure_gnn()
    scaled_model = ScaledForceModel(model, rms_force)
    scaled_model.eval()

    rows = []
    for seed in DYNAMICS_SEEDS:
        traj = run_langevin(
            scaled_model, start, MASS_AMU, n_steps=N_STEPS,
            timestep_ps=0.002, temperature_k=300.0,
            friction_per_ps=1.0, save_every=SAVE_EVERY, seed=seed
        )
        centered = traj - traj.mean(axis=1, keepdims=True)
        rows.append({
            "seed": seed,
            "min_position_nm": centered.min(),
            "max_position_nm": centered.max(),
            "max_abs_position_nm": np.abs(centered).max(),
            "reference_extent_nm": 0.3,
        })

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "pure_gnn_stability_by_seed.csv"
    pd.DataFrame(rows).to_csv(out, index=False)

    print("=== Pure-GNN stability across dynamics seeds ===")
    for row in rows:
        print(
            f"seed {row['seed']}: {row['min_position_nm']:.3f} to "
            f"{row['max_position_nm']:.3f} nm "
            f"(max |position| = {row['max_abs_position_nm']:.3f} nm)"
        )
    print("Reference extent: approximately +/-0.300 nm")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
