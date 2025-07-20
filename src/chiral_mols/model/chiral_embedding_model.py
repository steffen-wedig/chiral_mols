import torch
from chiral_mols.model.irreps_utils import get_invariant_indices, get_equivariant_irreps, split_invariants_equivariants
from torch import nn 
from e3nn import o3
from chiral_mols.model.rms_norm import RMSLayerNorm
from e3nn.o3 import Irreps



class ChiralEmbeddingModel(nn.Module):


    def __init__(self, input_irreps : Irreps, pseudoscalar_irreps : Irreps, output_embedding_dim, mean_inv_atomic_embedding, std_inv_atomic_embedding):
        super().__init__()


        self.invariant_indices, self.invariant_irreps = get_invariant_indices(
            input_irreps
        )

        self.equivariant_irreps = get_equivariant_irreps(input_irreps)

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


        self.tp_cross = o3.TensorProduct(
            self.equivariant_irreps,
            self.equivariant_irreps,
            o3.Irreps("128x1e"),
            # single CG path, weights = CG only
            instructions=[(0, 0, 0, "uvu", True)],
            internal_weights=True,
            shared_weights=True,
            irrep_normalization="component",
        )

        # 2) dot: (1e ⊗ 1o) → 0o
        self.tp_dot = o3.TensorProduct(
            self.tp_cross.irreps_out,
            self.equivariant_irreps,
            self.pseudoscalar_irreps,  # final pseudoscalar
            instructions=[(0, 0, 0, "uvu", True)],
            internal_weights=True,
            shared_weights=True,
            irrep_normalization="component",
        )

        self.rms_norm = RMSLayerNorm(self.equivariant_irreps.num_irreps)
        self.ln = nn.LayerNorm(self.pseudoscalar_irreps.dim, dtype=torch.float32)


        self.linear_out = nn.Linear(self.pseudoscalar_irreps.dim, output_embedding_dim)



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
        #
        cross = self.tp_cross(equivariant_features, equivariant_features)  # v₂ x v₃
        chi = self.tp_dot(equivariant_features, cross)  # v₁ · (v₂ x v₃)


        chi = chi.to(torch.float32)

        out = self.linear_out(chi)

        return out