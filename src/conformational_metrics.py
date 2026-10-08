"""Metrics for comparing atomistic and coarse-grained conformational distributions."""

import numpy as np

EPSILON = 1e-12


def js_divergence(p, q):
    """Return Jensen-Shannon divergence in bits for two probability vectors."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    p = p / p.sum()
    q = q / q.sum()
    p = np.clip(p, EPSILON, None)
    q = np.clip(q, EPSILON, None)
    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def conformational_js_divergence(phi_psi_aa, phi_psi_cg, n_bins=36):
    """Compare AA and CG phi/psi distributions with a common 2D histogram."""
    aa = np.asarray(phi_psi_aa, dtype=float)
    cg = np.asarray(phi_psi_cg, dtype=float)

    if aa.ndim != 2 or cg.ndim != 2 or aa.shape[1] != 2 or cg.shape[1] != 2:
        raise ValueError("Both inputs must have shape (n_samples, 2).")

    edges = np.linspace(-180.0, 180.0, n_bins + 1)
    aa_hist, _, _ = np.histogram2d(aa[:, 0], aa[:, 1], bins=[edges, edges])
    cg_hist, _, _ = np.histogram2d(cg[:, 0], cg[:, 1], bins=[edges, edges])

    return js_divergence(aa_hist.ravel(), cg_hist.ravel())
