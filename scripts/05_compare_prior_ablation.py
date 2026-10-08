"""Controlled comparison: GNN-only versus harmonic prior + GNN.

Both use the same mapped dataset, train/validation split, 60 epochs,
optimizer schedule, architecture, force-loss normalization and dynamics seeds.
Run from repo root: python scripts/05_compare_prior_ablation.py
This may take a while on CPU. Does not overwrite released model weights.
"""
from pathlib import Path
import sys
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from src.cg_dataset import ForceMatchingDataset, force_rms_scale
from src.cg_dynamics import run_langevin
from src.cg_gnn_model import (
    CGSchNetLike, CGForceField, HarmonicBondPrior, ScaledForceModel,
)
from src.cg_mapping import BEAD_NAMES, compute_backbone_dihedrals
from src.conformational_metrics import conformational_js_divergence

DATA = BASE / "data"
OUT = BASE / "results"
PAIRS = [(0, 1), (1, 2), (2, 3), (3, 4)]
MASSES = np.array([43.044, 15.018, 28.052, 28.010, 30.052])
EPOCHS = 60
TRAIN_SEED = 0
DYNAMICS_SEEDS = (1, 2, 3)
N_STEPS = 100_000
SAVE_EVERY = 20
BURN_IN = 300


def fit_model(use_prior, centered, forces, scale):
    torch.manual_seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    random.seed(TRAIN_SEED)

    dataset = ForceMatchingDataset(centered, forces)
    n_val = int(len(dataset) * 0.1)
    train_set, val_set = random_split(
        dataset, [len(dataset) - n_val, n_val],
        generator=torch.Generator().manual_seed(TRAIN_SEED),
    )
    train_loader = DataLoader(
        train_set, batch_size=256, shuffle=True,
        generator=torch.Generator().manual_seed(TRAIN_SEED),
    )
    val_loader = DataLoader(val_set, batch_size=1024)
    correction = CGSchNetLike(
        n_beads=5, n_features=32, n_rbf=32, n_interactions=2, d_max=1.0
    )
    if use_prior:
        prior = HarmonicBondPrior.fit_from_data(
            torch.tensor(centered, dtype=torch.float32), PAIRS, temperature_k=300.0
        )
        model = CGForceField(prior, correction)
    else:
        # ScaledForceModel maps the pure GNN's normalized outputs to physical units.
        model = ScaledForceModel(correction, scale)

    optimizer = torch.optim.Adam(correction.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    for epoch in range(EPOCHS):
        model.train()
        for pos, force in train_loader:
            optimizer.zero_grad()
            _, prediction = model(pos)
            loss = ((prediction / scale - force / scale) ** 2).mean()
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            model.eval()
            validation = []
            for pos, force in val_loader:
                _, prediction = model(pos)
                validation.append(
                    (((prediction - force) / scale) ** 2).mean().item()
                )
            print(
                f"{'prior+GNN' if use_prior else 'GNN-only'} "
                f"epoch {epoch+1}/{EPOCHS}: val loss {np.mean(validation):.4f}",
                flush=True,
            )
    model.eval()
    return model


def main():
    positions = np.load(DATA / "cg_positions_nm.npy")
    forces = np.load(DATA / "cg_forces_kjmolnm.npy")
    centered = positions - positions.mean(axis=1, keepdims=True)
    scale = force_rms_scale(forces)
    reference = np.load(DATA / "phi_psi_cg.npy")
    index = {name: i for i, name in enumerate(BEAD_NAMES)}
    ref_bonds = np.linalg.norm(
        positions[:, 1:] - positions[:, :-1], axis=-1
    )
    ref_min = ref_bonds.min(axis=0)
    ref_max = ref_bonds.max(axis=0)
    OUT.mkdir(exist_ok=True)
    rows = []

    for name, use_prior in (("gnn_only", False), ("prior_plus_gnn", True)):
        model = fit_model(use_prior, centered, forces, scale)
        dihedrals = []
        for seed in DYNAMICS_SEEDS:
            traj = run_langevin(
                model, centered[0], MASSES, n_steps=N_STEPS,
                timestep_ps=0.002, temperature_k=300.0,
                friction_per_ps=1.0, save_every=SAVE_EVERY, seed=seed,
            )
            eq = traj[BURN_IN:]
            phi, psi = compute_backbone_dihedrals(eq, index)
            angles = np.stack([phi, psi], axis=-1)
            dihedrals.append(angles)
            bonds = np.linalg.norm(eq[:, 1:] - eq[:, :-1], axis=-1)
            rows.append({
                "model": name, "dynamics_seed": seed,
                "training_seed": TRAIN_SEED, "training_epochs": EPOCHS,
                "steps": N_STEPS, "frames_after_burnin": len(eq),
                "jsd_bits": conformational_js_divergence(reference, angles),
                "bond_fraction_outside_reference_range": np.mean(
                    (bonds < ref_min) | (bonds > ref_max)
                ),
                "max_abs_com_coordinate_nm": np.max(np.abs(
                    eq - eq.mean(axis=1, keepdims=True)
                )),
            })
            print(name, seed, "JSD", rows[-1]["jsd_bits"], flush=True)
        # Also report JSD of pooled frames, matching the main project's metric.
        pooled = np.concatenate(dihedrals, axis=0)
        pooled_jsd = conformational_js_divergence(reference, pooled)
        for row in rows:
            if row["model"] == name:
                row["pooled_jsd_bits"] = pooled_jsd
        print(name, "pooled JSD bits:", pooled_jsd, flush=True)

    output = OUT / "controlled_prior_ablation.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print("Saved", output)


if __name__ == "__main__":
    main()
