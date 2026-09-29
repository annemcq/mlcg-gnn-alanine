"""
A small SchNet-style graph neural network that learns a coarse-grained
(CG) energy function U(R) for a fully-connected graph of CG beads, from
which forces are obtained by automatic differentiation:

    F_i = -dU/dR_i

This is the same overall design as CGnet / CGSchNet (Wang et al. 2019;
Husic et al. 2020): a GNN predicts a scalar, rotation- and translation-
invariant energy from pairwise distances, and forces come from autograd
rather than being predicted directly. This guarantees the learned force
field is conservative (energy-consistent) by construction, which a model
that predicts forces directly would not guarantee.

The system here (5 CG beads) is tiny, so the graph is simply fully
connected; the architecture generalizes to larger systems by switching
to a distance cutoff for edges.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RBFExpansion(nn.Module):
    """Expand a scalar distance into a vector of Gaussian radial basis functions."""

    def __init__(self, d_min: float = 0.0, d_max: float = 1.2, n_gaussians: int = 32):
        super().__init__()
        centers = torch.linspace(d_min, d_max, n_gaussians)
        self.register_buffer("centers", centers)
        self.width = (centers[1] - centers[0]).item()

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        # d: (..., 1) -> (..., n_gaussians)
        return torch.exp(-((d - self.centers) ** 2) / (2 * self.width ** 2))


class InteractionBlock(nn.Module):
    """
    One SchNet-style continuous-filter convolution + node update, on a
    fully-connected graph of n_beads nodes.
    """

    def __init__(self, n_features: int, n_rbf: int):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, n_features),
            nn.Tanh(),
            nn.Linear(n_features, n_features),
        )
        self.update_net = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.Tanh(),
            nn.Linear(n_features, n_features),
        )

    def forward(self, node_feats: torch.Tensor, edge_rbf: torch.Tensor, edge_index: torch.Tensor):
        """
        node_feats : (B, n_beads, n_features)
        edge_rbf   : (B, n_edges, n_rbf)
        edge_index : (n_edges, 2) -- (source, target) bead indices, shared across batch
        """
        src, dst = edge_index[:, 0], edge_index[:, 1]
        filters = self.filter_net(edge_rbf)                    # (B, n_edges, n_features)
        messages = filters * node_feats[:, src, :]              # (B, n_edges, n_features)

        n_beads = node_feats.shape[1]
        agg = torch.zeros_like(node_feats)
        agg.index_add_(1, dst, messages)

        return node_feats + self.update_net(agg)


class CGSchNetLike(nn.Module):
    """
    Predicts a scalar CG energy for a batch of CG configurations, on a
    fully-connected graph of `n_beads` nodes.
    """

    def __init__(self, n_beads: int, n_features: int = 32, n_rbf: int = 32,
                 n_interactions: int = 2, d_max: float = 1.2):
        super().__init__()
        self.n_beads = n_beads
        self.embedding = nn.Embedding(n_beads, n_features)  # one "type" per bead index
        self.rbf = RBFExpansion(0.0, d_max, n_rbf)
        self.interactions = nn.ModuleList(
            [InteractionBlock(n_features, n_rbf) for _ in range(n_interactions)]
        )
        self.energy_head = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.Tanh(),
            nn.Linear(n_features, 1),
        )

        src, dst = torch.meshgrid(torch.arange(n_beads), torch.arange(n_beads), indexing="ij")
        mask = src != dst
        edge_index = torch.stack([src[mask], dst[mask]], dim=1)
        self.register_buffer("edge_index", edge_index)

    def compute_energy(self, positions: torch.Tensor) -> torch.Tensor:
        """positions: (B, n_beads, 3) -> energy: (B,)"""
        B = positions.shape[0]
        src, dst = self.edge_index[:, 0], self.edge_index[:, 1]
        diffs = positions[:, dst, :] - positions[:, src, :]     # (B, n_edges, 3)
        dists = torch.norm(diffs, dim=-1, keepdim=True)          # (B, n_edges, 1)
        edge_rbf = self.rbf(dists)                               # (B, n_edges, n_rbf)

        node_feats = self.embedding(torch.arange(self.n_beads, device=positions.device))
        node_feats = node_feats.unsqueeze(0).expand(B, -1, -1).clone()

        for block in self.interactions:
            node_feats = block(node_feats, edge_rbf, self.edge_index)

        per_bead_energy = self.energy_head(node_feats).squeeze(-1)  # (B, n_beads)
        return per_bead_energy.sum(dim=1)                            # (B,)

    def forward(self, positions: torch.Tensor):
        """
        positions: (B, n_beads, 3), requires_grad will be set internally.
        Returns (energy, forces) with forces = -dE/dR.
        """
        positions = positions.clone().requires_grad_(True)
        energy = self.compute_energy(positions)
        forces = -torch.autograd.grad(
            energy.sum(), positions, create_graph=self.training, retain_graph=True
        )[0]
        return energy, forces


class HarmonicBondPrior(nn.Module):
    """
    Fixed physical prior: a harmonic spring on each consecutive backbone
    bead pair (the 5 beads form a linear chain: bead i is bonded to
    bead i+1). Equilibrium distances and stiffnesses are fit directly
    from the training data's bond-length statistics (Boltzmann inversion
    of a single harmonic degree of freedom: k = kT / Var(d)), not learned
    by gradient descent.

    This is the same design used in CGnet (Wang et al. 2019): a
    classical "prior" energy handles the strongly-constrained bonded
    degrees of freedom, and the neural network only has to learn the
    (much softer) remaining non-bonded / conformational energy surface.
    A pure black-box GNN with no such prior is not guaranteed to keep
    bonded distances near their physical values when simulated outside
    the training distribution. This is precisely the failure mode observed
    (bead distances diverging to several nm within a short MD run)
    before adding this prior, which is why it's part of the final model
    rather than an optional extra.
    """

    def __init__(self, bonded_pairs: list[tuple[int, int]], d0: torch.Tensor, k: torch.Tensor):
        super().__init__()
        self.register_buffer("pairs", torch.tensor(bonded_pairs, dtype=torch.long))
        self.register_buffer("d0", d0)
        self.register_buffer("k", k)

    @classmethod
    def fit_from_data(cls, positions_nm: torch.Tensor, bonded_pairs: list[tuple[int, int]], temperature_k: float = 300.0):
        KB = 0.0083144621
        kT = KB * temperature_k
        d0s, ks = [], []
        for i, j in bonded_pairs:
            d = torch.norm(positions_nm[:, j, :] - positions_nm[:, i, :], dim=-1)
            d0s.append(d.mean())
            var = d.var()
            ks.append(kT / var)
        return cls(bonded_pairs, torch.stack(d0s), torch.stack(ks))

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """positions: (B, n_beads, 3) -> energy: (B,)"""
        src = self.pairs[:, 0]
        dst = self.pairs[:, 1]
        d = torch.norm(positions[:, dst, :] - positions[:, src, :], dim=-1)  # (B, n_pairs)
        e = 0.5 * self.k * (d - self.d0) ** 2
        return e.sum(dim=1)


class CGForceField(nn.Module):
    """
    Full CG force field = fixed harmonic bonded prior + learned GNN
    correction (CGSchNetLike). Forces are the gradient of the *combined*
    energy, so the GNN only has to learn what the harmonic prior leaves
    out (bond fluctuation anharmonicity, angles, and everything
    non-bonded).
    """

    def __init__(self, prior: HarmonicBondPrior, correction: "CGSchNetLike"):
        super().__init__()
        self.prior = prior
        self.correction = correction

    def compute_energy(self, positions: torch.Tensor) -> torch.Tensor:
        return self.prior(positions) + self.correction.compute_energy(positions)

    def forward(self, positions: torch.Tensor):
        positions = positions.clone().requires_grad_(True)
        energy = self.compute_energy(positions)
        forces = -torch.autograd.grad(
            energy.sum(), positions, create_graph=self.training, retain_graph=True
        )[0]
        return energy, forces


class ScaledForceModel(nn.Module):
    """
    Wraps a trained CGSchNetLike model that was fit to *normalized* forces
    (force / force_scale) and rescales its output back to physical units
    (kJ/mol/nm), for use in MD simulation. Since F = -dU/dR, scaling the
    output by a constant `force_scale` is equivalent to using the energy
    `force_scale * U`, so both energy and force are rescaled consistently.
    """

    def __init__(self, base_model: "CGSchNetLike", force_scale: float):
        super().__init__()
        self.base_model = base_model
        self.force_scale = force_scale

    def forward(self, positions: torch.Tensor):
        energy, forces = self.base_model(positions)
        return energy * self.force_scale, forces * self.force_scale
