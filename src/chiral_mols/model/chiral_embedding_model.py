import torch
from chiral_mols.model.irreps_utils import (
    get_invariant_indices,
    get_equivariant_irreps,
    split_invariants_equivariants,
)
from e3nn import o3
from chiral_mols.model.rms_norm import RMSLayerNorm
from e3nn.o3 import Irreps
from e3nn.nn import NormActivation, Gate


class ChiGate(torch.nn.Module):
    """
    FiLM-style invariant→gate mapping.
    Returns a (…, K) tensor in (0, 1).
    """
    def __init__(self, inv_dim: int, K: int, hidden: int | None = None):
        super().__init__()
        if hidden is None:
            hidden = 2* inv_dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(inv_dim, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, K)
        )
        self.act = torch.nn.Sigmoid()           # keep outputs positive

    def forward(self, inv):
        gate = self.act(self.net(inv))    # (..., K)
        return gate



class ChiralEmbeddingModel(torch.nn.Module):

    def __init__(
        self,
        input_irreps: Irreps,
        pseudoscalar_irreps: Irreps,
        output_embedding_dim,
        mean_inv_atomic_embedding,
        std_inv_atomic_embedding,
        low_dim_equivariant: int
    ):
        super().__init__()

        self.invariant_indices, self.invariant_irreps = get_invariant_indices(
            input_irreps
        )

        self.equivariant_irreps = get_equivariant_irreps(input_irreps)
        print(self.equivariant_irreps)
        self.pseudoscalar_irreps = pseudoscalar_irreps

        self.register_buffer(
            "mean_inv_atomic_embedding",
            mean_inv_atomic_embedding,
            persistent=True,
        )
        self.register_buffer(
            "std_inv_atomic_embedding",
            std_inv_atomic_embedding,
            persistent=True,
        )
        
        self.eq_embedding_irrep= Irreps(f"{low_dim_equivariant}x1o")

        self.lin0 = o3.Linear(self.equivariant_irreps,  self.eq_embedding_irrep)
        self.lin1 = o3.Linear(self.equivariant_irreps,  self.eq_embedding_irrep)
        self.lin2 = o3.Linear(self.equivariant_irreps,  self.eq_embedding_irrep)

        self.eq_norm = NormActivation(self.eq_embedding_irrep, torch.sigmoid )

        self.tp_cross = o3.TensorProduct(
             self.eq_embedding_irrep,
             self.eq_embedding_irrep,
            o3.Irreps(f"{low_dim_equivariant}x1e"),
            # single CG path, weights = CG only
            instructions=[(0, 0, 0, "uvu", True)],
            internal_weights=True,
            shared_weights=True,
            irrep_normalization="component",
        )

        # 2) dot: (1e ⊗ 1o) → 0o
        self.tp_dot = o3.TensorProduct(
            self.tp_cross.irreps_out,
            self.eq_embedding_irrep,
            self.pseudoscalar_irreps,  # final pseudoscalar
            instructions=[(0, 0, 0, "uvu", True)],
            internal_weights=True,
            shared_weights=True,
            irrep_normalization="component",
        )

        self.rms_norm = RMSLayerNorm(self.equivariant_irreps.num_irreps)
        self.ln = torch.nn.LayerNorm(output_embedding_dim, dtype=torch.float32)
        #self.ps_layer_norm1 = torch.nn.LayerNorm(self.pseudoscalar_irreps.dim)
        #self.ps_layer_norm2 = torch.nn.LayerNorm(self.pseudoscalar_irreps.dim)
        #self.ps_layer_norm3 = torch.nn.LayerNorm(self.pseudoscalar_irreps.dim)

        self.chi_gate = ChiGate(inv_dim = self.invariant_irreps.dim, K = self.pseudoscalar_irreps.dim)

#        self.linear_out = torch.nn.Linear(self.pseudoscalar_irreps.dim, output_embedding_dim)

    def rescale_invariant(self, invariants):

        invariants = (
            invariants - self.mean_inv_atomic_embedding
        ) / self.std_inv_atomic_embedding

        invariants = invariants.float()
        return invariants

    def forward(self, atomic_embeddings):

        invariant_features, equivariant_features = split_invariants_equivariants(
            atomic_embeddings, self.invariant_indices
        )

        invariant_features = self.rescale_invariant(invariant_features)

        equivariant_features = self.rms_norm(equivariant_features)
        

        x0 = self.lin0(equivariant_features)

        x1 = self.lin1(equivariant_features)
        x2 = self.lin2(equivariant_features)
        

        cross = self.tp_cross(x0, x1)  # v₂ x v₃
        chi = self.tp_dot(cross, x2)  # v₁ · (v₂ x v₃)

        #cross01 = self.tp_cross(x0, x1)
        #cross12 = self.tp_cross(x1, x2)
        #cross20 = self.tp_cross(x2, x0)
#
        #chi0 = self.ps_layer_norm1(self.tp_dot(cross01, x2))
        #chi1 = self.ps_layer_norm1(self.tp_dot(cross12, x0))
        #chi2 = self.ps_layer_norm1(self.tp_dot(cross20, x1))



        #chi = torch.cat([chi0, chi1, chi2], dim=-1)


        out = chi.to(torch.float32)

        out = self.ln(out)
        out = self.chi_gate(invariant_features) * out
        #out = self.linear_out(out)
        return out
