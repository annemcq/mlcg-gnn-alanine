"""Quantify the difference between atomistic and CG conformational distributions.

The existing Ramachandran plot shows that the CG model samples the main
regions of phi/psi space but with different populations. This script turns
that qualitative observation into a single quantitative diagnostic using
Jensen-Shannon divergence between 2D phi/psi histograms.
"""

from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
AA_PATH = BASE / "data" / "phi_psi_aa.npy"
CG_PATH = BASE / "data" / "phi_psi_cg.npy"
RESULTS_DIR = BASE / "results"

N_BINS = 36
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


def conformational_js_divergence(phi_psi_aa, phi_psi_cg, n_bins=N_BINS):
    """Compare AA and CG phi/psi distributions with a common 2D histogram."""
    aa = np.asarray(phi_psi_aa, dtype=float)
    cg = np.asarray(phi_psi_cg, dtype=float)

    if aa.ndim != 2 or cg.ndim != 2 or aa.shape[1] != 2 or cg.shape[1] != 2:
        raise ValueError("Both inputs must have shape (n_samples, 2).")

    edges = np.linspace(-180.0, 180.0, n_bins + 1)

    aa_hist, _, _ = np.histogram2d(
        aa[:, 0], aa[:, 1], bins=[edges, edges]
    )
    cg_hist, _, _ = np.histogram2d(
        cg[:, 0], cg[:, 1], bins=[edges, edges]
    )

    return js_divergence(aa_hist.ravel(), cg_hist.ravel())


def main():
    phi_psi_aa = np.load(AA_PATH)
    phi_psi_cg = np.load(CG_PATH)

    jsd = conformational_js_divergence(phi_psi_aa, phi_psi_cg)

    RESULTS_DIR.mkdir(exist_ok=True)
    output = RESULTS_DIR / "conformational_distribution_comparison.csv"

    pd.DataFrame([{
        "metric": "Jensen-Shannon divergence",
        "value_bits": jsd,
        "n_bins_per_angle": N_BINS,
        "aa_samples": len(phi_psi_aa),
        "cg_samples": len(phi_psi_cg),
    }]).to_csv(output, index=False)

    print("=== AA vs CG conformational distribution ===")
    print(f"Jensen-Shannon divergence: {jsd:.4f} bits")
    print(f"Histogram bins per angle: {N_BINS}")
    print(f"AA samples: {len(phi_psi_aa)}")
    print(f"CG samples: {len(phi_psi_cg)}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
