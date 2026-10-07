"""
Coarse-grained mapping for alanine dipeptide (ACE-ALA-NME).

The 22-atom system is mapped to five beads using the center of mass of
each atom group. CG forces are obtained by summing the atomic forces
belonging to each bead.
"""

from __future__ import annotations

import numpy as np


BEAD_NAMES = ["C_ACE", "N_ALA", "CA_ALA", "C_ALA", "N_NME"]

BEAD_ATOM_GROUPS = [
    list(range(0, 6)),    # C_ACE
    list(range(6, 8)),    # N_ALA
    list(range(8, 14)),   # CA_ALA
    list(range(14, 16)),  # C_ALA
    list(range(16, 22)),  # N_NME
]


def map_to_cg(
    positions_nm: np.ndarray,
    forces_kjmolnm: np.ndarray,
    masses_amu: np.ndarray,
):
    """
    Map atomistic positions and forces to the five-bead CG representation.

    Parameters
    ----------
    positions_nm : (T, 22, 3)
        Atomistic positions in nm.
    forces_kjmolnm : (T, 22, 3)
        Atomistic forces in kJ/mol/nm.
    masses_amu : (22,)
        Atomic masses in amu.

    Returns
    -------
    cg_positions : (T, 5, 3)
        Center-of-mass position of each bead.
    cg_forces : (T, 5, 3)
        Total force on each bead.
    """
    T = positions_nm.shape[0]
    n_beads = len(BEAD_ATOM_GROUPS)

    cg_positions = np.zeros((T, n_beads, 3))
    cg_forces = np.zeros((T, n_beads, 3))

    for b, atom_idx in enumerate(BEAD_ATOM_GROUPS):
        m = masses_amu[atom_idx]
        w = m / m.sum()

        cg_positions[:, b, :] = np.tensordot(
            positions_nm[:, atom_idx, :],
            w,
            axes=([1], [0]),
        )

        cg_forces[:, b, :] = forces_kjmolnm[:, atom_idx, :].sum(axis=1)

    return cg_positions, cg_forces


def dihedral(p0, p1, p2, p3):
    """Compute the dihedral angle in degrees for arrays of shape (..., 3)."""
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    b1 /= np.linalg.norm(b1, axis=-1, keepdims=True)

    v = b0 - np.sum(
        b0 * b1,
        axis=-1,
        keepdims=True,
    ) * b1

    w = b2 - np.sum(
        b2 * b1,
        axis=-1,
        keepdims=True,
    ) * b1

    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1, v) * w, axis=-1)

    return np.degrees(np.arctan2(y, x))


def compute_backbone_dihedrals(
    positions_nm: np.ndarray,
    atom_idx: dict,
):
    """
    Compute the backbone phi and psi dihedral angles.

    phi = dihedral(C_ACE, N_ALA, CA_ALA, C_ALA)
    psi = dihedral(N_ALA, CA_ALA, C_ALA, N_NME)
    """
    p = positions_nm

    phi = dihedral(
        p[:, atom_idx["C_ACE"]],
        p[:, atom_idx["N_ALA"]],
        p[:, atom_idx["CA_ALA"]],
        p[:, atom_idx["C_ALA"]],
    )

    psi = dihedral(
        p[:, atom_idx["N_ALA"]],
        p[:, atom_idx["CA_ALA"]],
        p[:, atom_idx["C_ALA"]],
        p[:, atom_idx["N_NME"]],
    )

    return phi, psi
