"""
Coarse-grained (CG) mapping for alanine dipeptide (ACE-ALA-NME, 22 atoms)
down to 5 beads centered on the backbone heavy atoms, following the
mapping popularized in the CGnet line of work (Wang, Chmiela, Clementi
et al. 2019) -- the same conceptual mapping used for CG force fields in
the Clementi group's own research, applied here to a toy public system.

Bead definition (index -> atom group, by atom index in
data/alanine-dipeptide.pdb as loaded by OpenMM):

    bead 0 ("C_ACE"):  atoms 0-5   (ACE methyl cap + carbonyl C=O)
    bead 1 ("N_ALA"):  atoms 6-7   (backbone amide N-H of ALA)
    bead 2 ("CA_ALA"): atoms 8-13  (alpha carbon + side chain CB, all H)
    bead 3 ("C_ALA"):  atoms 14-15 (backbone carbonyl C=O of ALA)
    bead 4 ("N_NME"):  atoms 16-21 (NME amide N-H + methyl cap)

Mapping convention (explicit choice):
we use a **center-of-mass (COM) mapping** rather than a "single-heavy-atom
slice" mapping. This is the choice that satisfies Noid et al.'s (2008)
force-matching consistency theorem exactly: for a linear mapping
R_I = sum_i M_Ii r_i / sum_i M_Ii, the force on the CG site that is
consistent with the many-body potential of mean force is
F_I = sum_i M_Ii f_i (the group's total atomistic force). A "slice"
mapping (bead position = one specific atom's raw position) does not
satisfy this exactly, so I use COM instead. Here, the tradeoff is that the CG
beads sit close to, but not exactly on top of, the backbone heavy atoms.
Notebooks/03 checks how much this matters for phi/psi by comparing
against the true atomistic dihedrals.
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


def map_to_cg(positions_nm: np.ndarray, forces_kjmolnm: np.ndarray, masses_amu: np.ndarray):
    """
    Parameters
    ----------
    positions_nm : (T, 22, 3) all-atom positions, nm
    forces_kjmolnm : (T, 22, 3) all-atom forces, kJ/mol/nm
    masses_amu : (22,) atomic masses, amu

    Returns
    -------
    cg_positions_nm : (T, 5, 3) center-of-mass position of each bead
    cg_forces_kjmolnm : (T, 5, 3) total force on each bead (sum over group)
    """
    T = positions_nm.shape[0]
    n_beads = len(BEAD_ATOM_GROUPS)
    cg_positions = np.zeros((T, n_beads, 3))
    cg_forces = np.zeros((T, n_beads, 3))

    for b, atom_idx in enumerate(BEAD_ATOM_GROUPS):
        m = masses_amu[atom_idx]
        w = m / m.sum()
        cg_positions[:, b, :] = np.tensordot(positions_nm[:, atom_idx, :], w, axes=([1], [0]))
        cg_forces[:, b, :] = forces_kjmolnm[:, atom_idx, :].sum(axis=1)

    return cg_positions, cg_forces


def dihedral(p0, p1, p2, p3):
    """
    Compute the dihedral angle (in degrees) defined by four points,
    for arrays of shape (..., 3). Standard praxeolitic formula.
    """
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    b1 /= np.linalg.norm(b1, axis=-1, keepdims=True)

    v = b0 - np.sum(b0 * b1, axis=-1, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, axis=-1, keepdims=True) * b1

    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1, v) * w, axis=-1)
    return np.degrees(np.arctan2(y, x))


def compute_backbone_dihedrals(positions_nm: np.ndarray, atom_idx: dict):
    """
    phi = dihedral(C_ACE, N_ALA, CA_ALA, C_ALA)
    psi = dihedral(N_ALA, CA_ALA, C_ALA, N_NME)

    atom_idx maps bead/atom name -> index into the last-but-one axis of
    positions_nm (works for both all-atom indices and CG bead indices).
    """
    p = positions_nm
    phi = dihedral(p[:, atom_idx["C_ACE"]], p[:, atom_idx["N_ALA"]],
                    p[:, atom_idx["CA_ALA"]], p[:, atom_idx["C_ALA"]])
    psi = dihedral(p[:, atom_idx["N_ALA"]], p[:, atom_idx["CA_ALA"]],
                    p[:, atom_idx["C_ALA"]], p[:, atom_idx["N_NME"]])
    return phi, psi
