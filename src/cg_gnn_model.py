"""
SchNet-style neural network for coarse-grained energy and force prediction.

The model predicts a scalar energy from pairwise bead distances, with
forces obtained as the negative gradient of the energy.
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
    """SchNet-style interaction block for the fully connected bead graph."""

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

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_rbf: torch.Tensor,
        edge_index: torch.Tensor,
    ):
        """
        node_feats : (B, n_beads, n_features)
        edge_rbf   : (B, n_edges, n_rbf)
        edge_index : (n_edges, 2)
        """
        src, dst = edge_index[:, 0], edge_index[:, 1]
        filters = self.filter_net(edge_rbf)
        messages = filters * node_feats[:, src, :]

        agg = torch.zeros_like(node_feats)
        agg.index_add_(1, dst, messages)

        return node_feats + self.update_net(agg)


class CGSchNetLike(nn.Module):
    """Predict a scalar CG energy from a fully connected graph of CG beads."""

    def __init__(
        self,
        n_beads: int,
        n_features: int = 32,
        n_rbf: int = 32,
        n_interactions: int = 2,
        d_max: float = 1.2,
    ):
        super().__init__()
        self.n_beads = n_beads

        self.embedding = nn.Embedding(n_beads, n_features)
        self.rbf = RBFExpansion(0.0, d_max, n_rbf)

        self.interactions = nn.ModuleList(
            [
                InteractionBlock(n_features, n_rbf)
                for _ in range(n_interactions)
            ]
        )

        self.energy_head = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.Tanh(),
            nn.Linear(n_features, 1),
        )

        src, dst = torch.meshgrid(
            torch.arange(n_beads),
            torch.arange(n_beads),
            indexing="ij",
        )
        mask = src != dst
        edge_index = torch.stack([src[mask], dst[mask]], dim=1)
        self.register_buffer("edge_index", edge_index)

    def compute_energy(self, positions: torch.Tensor) -> torch.Tensor:
        """positions: (B, n_beads, 3) -> energy: (B,)"""
        batch_size = positions.shape[0]

        src = self.edge_index[:, 0]
        dst = self.edge_index[:, 1]

        diffs = positions[:, dst, :] - positions[:, src, :]
        dists = torch.norm(diffs, dim=-1, keepdim=True)
        edge_rbf = self.rbf(dists)

        node_feats = self.embedding(
            torch.arange(self.n_beads, device=positions.device)
        )
        node_feats = (
            node_feats
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
            .clone()
        )

        for block in self.interactions:
            node_feats = block(node_feats, edge_rbf, self.edge_index)

        per_bead_energy = self.energy_head(node_feats).squeeze(-1)
        return per_bead_energy.sum(dim=1)

    def forward(self, positions: torch.Tensor):
        """Return energy and forces, with forces computed as -dE/dR."""
        positions = positions.clone().requires_grad_(True)
        energy = self.compute_energy(positions)

        forces = -torch.autograd.grad(
            energy.sum(),
            positions,
            create_graph=self.training,
            retain_graph=True,
        )[0]

        return energy, forces


class HarmonicBondPrior(nn.Module):
    """
    Harmonic prior for consecutive bonded beads.

    Equilibrium distances are estimated from the mapped reference data and the
    force constants are obtained from k = kT / Var(d).
    """

    def __init__(
        self,
        bonded_pairs: list[tuple[int, int]],
        d0: torch.Tensor,
        k: torch.Tensor,
    ):
        super().__init__()
        self.register_buffer(
            "pairs",
            torch.tensor(bonded_pairs, dtype=torch.long),
        )
        self.register_buffer("d0", d0)
        self.register_buffer("k", k)

    @classmethod
    def fit_from_data(
        cls,
        positions_nm: torch.Tensor,
        bonded_pairs: list[tuple[int, int]],
        temperature_k: float = 300.0,
    ):
        KB = 0.0083144621
        kT = KB * temperature_k

        d0s = []
        ks = []

        for i, j in bonded_pairs:
            d = torch.norm(
                positions_nm[:, j, :] - positions_nm[:, i, :],
                dim=-1,
            )

            d0s.append(d.mean())
            var = d.var()
            ks.append(kT / var)

        return cls(
            bonded_pairs,
            torch.stack(d0s),
            torch.stack(ks),
        )

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """positions: (B, n_beads, 3) -> energy: (B,)"""
        src = self.pairs[:, 0]
        dst = self.pairs[:, 1]

        d = torch.norm(
            positions[:, dst, :] - positions[:, src, :],
            dim=-1,
        )

        e = 0.5 * self.k * (d - self.d0) ** 2
        return e.sum(dim=1)


class CGForceField(nn.Module):
    """Combined force field consisting of a harmonic prior and GNN correction."""

    def __init__(
        self,
        prior: HarmonicBondPrior,
        correction: CGSchNetLike,
    ):
        super().__init__()
        self.prior = prior
        self.correction = correction

    def compute_energy(self, positions: torch.Tensor) -> torch.Tensor:
        return (
            self.prior(positions)
            + self.correction.compute_energy(positions)
        )

    def forward(self, positions: torch.Tensor):
        positions = positions.clone().requires_grad_(True)
        energy = self.compute_energy(positions)

        forces = -torch.autograd.grad(
            energy.sum(),
            positions,
            create_graph=self.training,
            retain_graph=True,
        )[0]

        return energy, forces


class ScaledForceModel(nn.Module):
    """Rescale the energy and forces of a model trained with normalized forces."""

    def __init__(
        self,
        base_model: CGSchNetLike,
        force_scale: float,
    ):
        super().__init__()
        self.base_model = base_model
        self.force_scale = force_scale

    def forward(self, positions: torch.Tensor):
        energy, forces = self.base_model(positions)

        return (
            energy * self.force_scale,
            forces * self.force_scale,
        )
