# Machine-Learned Coarse-Grained Force Field (GNN) for Alanine Dipeptide

[![Tests](https://github.com/annemcq/mlcg-gnn-alanine/actions/workflows/tests.yml/badge.svg)](https://github.com/annemcq/mlcg-gnn-alanine/actions/workflows/tests.yml)

This project presens a graph neural network (GNN) trained to reproduce a coarse-grained (CG) force field. This overall methodology as described in CGnet / CGSchNet (Wang, Chmiela, Clementi et al. 2019; Husic et al. 2020), and applied here to a small, entirely public toy system, independent of any specific
research project's data or results.

## System and method of choice

Alanine dipeptide (ACE-ALA-NME) is the standard toy system in the coarse-graining
literature: small, well characterized, with a known three-basin Ramachandran (phi/psi)
structure, and no overlap with any real research system. 

In addition, training a GNN to learn a CG force field, rather than say, sequence classifier or an embeddings-based predictor, targets a key intersection of statistical mechanics, molecular dynamics, and geometric deep learning. 

## Structure

```
mlcg-gnn-alanine/
├── README.md
├── environment.yml
├── pytest.ini
├── data/
│   ├── alanine-dipeptide.pdb        # reference structure (openmmtools test system)
│   └── *.npy, *.pt                  # generated trajectory, CG dataset, trained weights
├── src/
│   ├── cg_mapping.py                # 5-bead COM mapping, dihedral utilities
│   ├── cg_gnn_model.py              # SchNet-style GNN, harmonic prior, combined force field
│   ├── cg_dataset.py                # force-matching Dataset
│   └── cg_dynamics.py               # BAOAB Langevin integrator driven by a learned force field
├── scripts/
│   ├── 01_generate_reference_trajectory.py   # OpenMM: 2 ns, ff14SB + GBSA implicit solvent
│   └── 02_train_cgnet.py                     # builds the CG dataset, trains prior + GNN correction
├── notebooks/
│   └── 03_train_and_validate_cgnet.ipynb     # full narrative, honest validation and limitations
├── tests/
│   └── test_cg_model.py             # invariances, force = -dE/dR, prior fit, no OpenMM needed
└── results/figures/
```

Reproduce from scratch:

```bash
conda env create -f environment.yml
conda activate mlcg-gnn-alanine
python scripts/01_generate_reference_trajectory.py   # ~3-4 min
python scripts/02_train_cgnet.py                     # ~5 min
pytest tests/
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

## Method summary

1. **Reference trajectory**: 2 ns of Langevin dynamics (300 K, ff14SB + GBn2 implicit
   solvent) in OpenMM, saving positions and forces every 100 fs (20,000 frames).

2. **CG mapping**: 5 beads on the backbone heavy atoms (ACE-C, ALA-N, ALA-CA, ALA-C, NME-N),
   using a **center-of-mass mapping** — the mapping that satisfies Noid et al.'s (2008)
   force-matching consistency theorem exactly, as opposed to a "slice" (single-atom)
   mapping. Validated the mapping by checking that pseudo-phi/psi computed from the CG beads
   correlates strongly (cos/sin > 0.9) with the true all-atom backbone dihedrals.

3. **Model**: a small SchNet-style GNN (`CGSchNetLike`) predicts a rotation/translation-
   invariant scalar energy from the fully-connected 5-bead graph; forces come from
   autograd (`F = -dE/dR`), guaranteeing a conservative force field.

4. **A documented failure**: training a pure GNN with no physical prior and then simulating with it 

   makes the dynamics diverge (bead distances reaching several nm, when the real system stays within ~0.3 nm of its own center of mass). This is why the final model includes a fixed harmonic bonded 

   prior, fit from the training data's own bond-length statistics — the same design CGnet uses,
   with the GNN only learning the residual (non-bonded, conformational) energy surface.

5. **Validation**: simulating with the final (prior + GNN) model keeps bond lengths
   physical throughout. It reproduces the location of the dominant conformational basin
   seen in the reference trajectory, but as reported here, a short CG run does not resolve the second major basin, and instead visits a region the reference trajectory essentially never populates. This in turn, indicates that the learned energy surface has unconstrained structure outside the training distribution. In the final section of the notebook, I proceed to discuss what this means and what subsequent iteration it would require. 

## Notes

- Built with AI assistance (as disclosed for all projects in this portfolio); the
  methodology (GNN coarse-graining, force matching, physical priors) mirrors standard
  practice in the Clementi group's own published work on this exact class of models.
- Uses only public data and a public benchmark system — no unpublished research data
  or results.
