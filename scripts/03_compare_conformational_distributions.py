"""Quantify the difference between atomistic and CG conformational distributions.

The existing Ramachandran plot shows that the CG model samples the main
regions of phi/psi space but with different populations. This script turns
that qualitative observation into a single quantitative diagnostic using
Jensen-Shannon divergence between 2D phi/psi histograms.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from src.conformational_metrics import conformational_js_divergence


BASE = Path(__file__).resolve().parents[1]
AA_PATH = BASE / "data" / "phi_psi_aa.npy"
CG_PATH = BASE / "data" / "phi_psi_cg.npy"
RESULTS_DIR = BASE / "results"

N_BINS = 36
EPSILON = 1e-12



def main():
    phi_psi_aa = np.load(AA_PATH)
    phi_psi_cg = np.load(CG_PATH)

    jsd = conformational_js_divergence(phi_psi_aa, phi_psi_cg)

    midpoint = len(phi_psi_aa) // 2
    aa_half_1 = phi_psi_aa[:midpoint]
    aa_half_2 = phi_psi_aa[midpoint:]
    reference_half_jsd = conformational_js_divergence(aa_half_1, aa_half_2)

    RESULTS_DIR.mkdir(exist_ok=True)
    output = RESULTS_DIR / "conformational_distribution_comparison.csv"

    pd.DataFrame([{
        "metric": "Jensen-Shannon divergence",
        "value_bits": jsd,
        "n_bins_per_angle": N_BINS,
        "aa_samples": len(phi_psi_aa),
        "cg_samples": len(phi_psi_cg),
        "reference_half_jsd_bits": reference_half_jsd,
    }]).to_csv(output, index=False)

    print("=== AA vs CG conformational distribution ===")
    print(f"Jensen-Shannon divergence: {jsd:.4f} bits")
    print(f"Histogram bins per angle: {N_BINS}")
    print(f"AA samples: {len(phi_psi_aa)}")
    print(f"CG samples: {len(phi_psi_cg)}")
    print(f"Reference half-vs-half JSD: {reference_half_jsd:.4f} bits")
    print(f"AA-vs-CG / reference baseline: {jsd / reference_half_jsd:.2f}x")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
