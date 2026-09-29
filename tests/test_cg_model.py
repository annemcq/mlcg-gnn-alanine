"""
Fast, offline tests for the CG model components -- no training, no OpenMM,
so these run in a couple of seconds and don't depend on the (large,
generated) reference trajectory files being present.
"""

import numpy as np
import torch

from src.cg_gnn_model import CGSchNetLike, HarmonicBondPrior, CGForceField, ScaledForceModel
from src.cg_mapping import dihedral


def test_energy_is_translation_invariant():
    torch.manual_seed(0)
    model = CGSchNetLike(n_beads=5, n_features=16, n_rbf=16, n_interactions=2)
    pos = torch.randn(4, 5, 3) * 0.3
    shift = torch.randn(1, 1, 3)

    e1, _ = model(pos)
    e2, _ = model(pos + shift)
    assert torch.allclose(e1, e2, atol=1e-4)


def test_energy_is_rotation_invariant():
    torch.manual_seed(0)
    model = CGSchNetLike(n_beads=5, n_features=16, n_rbf=16, n_interactions=2)
    pos = torch.randn(2, 5, 3) * 0.3

    # a fixed 3D rotation matrix (90 degrees about z)
    R = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    pos_rotated = pos @ R.T

    e1, _ = model(pos)
    e2, _ = model(pos_rotated)
    assert torch.allclose(e1, e2, atol=1e-4)


def test_forces_are_negative_energy_gradient():
    """Finite-difference check that forces = -dE/dR for one coordinate."""
    torch.manual_seed(1)
    model = CGSchNetLike(n_beads=5, n_features=16, n_rbf=16, n_interactions=2)
    model.eval()
    pos = torch.randn(1, 5, 3) * 0.3

    _, forces = model(pos)

    eps = 1e-4
    pos_plus = pos.clone()
    pos_plus[0, 0, 0] += eps
    pos_minus = pos.clone()
    pos_minus[0, 0, 0] -= eps

    e_plus, _ = model(pos_plus)
    e_minus, _ = model(pos_minus)
    finite_diff_force = -(e_plus.item() - e_minus.item()) / (2 * eps)

    assert abs(finite_diff_force - forces[0, 0, 0].item()) < 1e-2


def test_harmonic_prior_fits_expected_equilibrium_distance():
    torch.manual_seed(0)
    # synthetic data: bead 1 fluctuates harmonically around 0.15 nm from bead 0
    n_frames = 5000
    true_d0 = 0.15
    true_std = 0.01
    pos = torch.zeros(n_frames, 2, 3)
    pos[:, 1, 0] = true_d0 + true_std * torch.randn(n_frames)

    prior = HarmonicBondPrior.fit_from_data(pos, bonded_pairs=[(0, 1)], temperature_k=300.0)
    assert abs(prior.d0[0].item() - true_d0) < 0.001
    # k = kT / Var(d); check it's positive and roughly matches kT/std^2
    kT = 0.0083144621 * 300.0
    expected_k = kT / true_std ** 2
    assert abs(prior.k[0].item() - expected_k) / expected_k < 0.1


def test_cgforcefield_combines_prior_and_correction_energy():
    torch.manual_seed(0)
    pos = torch.randn(3, 5, 3) * 0.2 + torch.tensor([0.2, 0.0, 0.0])
    prior = HarmonicBondPrior([(0, 1)], d0=torch.tensor([0.15]), k=torch.tensor([1000.0]))
    correction = CGSchNetLike(n_beads=5, n_features=8, n_rbf=8, n_interactions=1)
    combined = CGForceField(prior, correction)

    e_prior = prior(pos)
    e_correction = correction.compute_energy(pos)
    e_combined = combined.compute_energy(pos)

    assert torch.allclose(e_combined, e_prior + e_correction, atol=1e-5)


def test_scaled_force_model_scales_consistently():
    torch.manual_seed(0)
    base = CGSchNetLike(n_beads=5, n_features=8, n_rbf=8, n_interactions=1)
    scale = 37.5
    scaled = ScaledForceModel(base, scale)

    pos = torch.randn(2, 5, 3) * 0.2
    e_base, f_base = base(pos)
    e_scaled, f_scaled = scaled(pos)

    assert torch.allclose(e_scaled, e_base * scale, atol=1e-4)
    assert torch.allclose(f_scaled, f_base * scale, atol=1e-4)


def test_dihedral_matches_known_planar_case():
    # four points forming a perfect trans (180 deg) dihedral in the xy-plane
    p0 = np.array([[0.0, 1.0, 0.0]])
    p1 = np.array([[0.0, 0.0, 0.0]])
    p2 = np.array([[1.0, 0.0, 0.0]])
    p3 = np.array([[1.0, -1.0, 0.0]])
    angle = dihedral(p0, p1, p2, p3)
    assert abs(abs(angle[0]) - 180.0) < 1e-6
