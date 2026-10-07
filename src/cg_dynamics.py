"""
A minimal Langevin (BAOAB-style) integrator to run molecular dynamics
using the *learned* CG force field (CGSchNetLike) instead of an
analytical potential. Used purely to validate the model: does simulating
with the learned forces reproduce the same conformational distribution
(phi/psi basins) as the reference all-atom-derived CG trajectory?

Units follow OpenMM/GROMACS convention throughout: nm, kJ/mol, amu, ps,
kelvin, kJ/mol/K (kB).
"""

from __future__ import annotations

import numpy as np
import torch

KB_KJ_PER_MOL_K = 0.0083144621  # Boltzmann constant, kJ/mol/K


def run_langevin(
    model,
    initial_positions_nm: np.ndarray,
    masses_amu: np.ndarray,
    n_steps: int,
    timestep_ps: float = 0.002,
    temperature_k: float = 300.0,
    friction_per_ps: float = 1.0,
    save_every: int = 50,
    seed: int = 0,
) -> np.ndarray:
    """
    BAOAB Langevin integrator (Leimkuhler & Matthews 2013) driven by the
    forces of `model` (a CGSchNetLike instance in eval mode).

    Parameters
    ----------
    initial_positions_nm : (n_beads, 3)
    masses_amu : (n_beads,)

    Returns
    -------
    trajectory_nm : (n_steps // save_every, n_beads, 3)
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    model.eval()

    n_beads = initial_positions_nm.shape[0]
    m = torch.as_tensor(
        masses_amu,
        dtype=torch.float32,
    ).reshape(-1, 1)  # (n_beads, 1)

    kT = KB_KJ_PER_MOL_K * temperature_k
    gamma = friction_per_ps

    pos = torch.as_tensor(
        initial_positions_nm,
        dtype=torch.float32,
    ).unsqueeze(0)  # (1, n_beads, 3)

    vel = torch.zeros_like(pos)

    a = np.exp(-gamma * timestep_ps)
    b = np.sqrt(
        kT * (1 - a ** 2) / m.numpy()
    )  # (n_beads, 1), velocity noise scale

    def get_forces(p):
        _, f = model(p)
        return f.detach()

    forces = get_forces(pos)
    saved = []
    dt = timestep_ps

    for step in range(n_steps):
        # B: half-kick with current forces
        vel = vel + 0.5 * dt * forces / m

        # A: half-drift
        pos = pos + 0.5 * dt * vel

        # O: Ornstein-Uhlenbeck friction + noise
        noise = torch.randn(
            vel.shape,
            dtype=vel.dtype,
            device=vel.device,
            generator=generator,
        ) * torch.as_tensor(
            b,
            dtype=vel.dtype,
            device=vel.device,
        )

        vel = a * vel + noise

        # A: half-drift
        pos = pos + 0.5 * dt * vel

        # B: half-kick with new forces
        forces = get_forces(pos)
        vel = vel + 0.5 * dt * forces / m

        if (step + 1) % save_every == 0:
            saved.append(
                pos.squeeze(0).numpy().copy()
            )

    return np.array(saved)
