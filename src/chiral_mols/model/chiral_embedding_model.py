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
            torch.nn.Linear(inv_dim, hidden,bias=None),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, K, bias= None)
        )
        self.act = torch.nn.Sigmoid()           # keep outputs positive

    def forward(self, inv):
        gate = self.act(self.net(inv))    # (..., K)
        return gate



class OddMLP(torch.nn.Module):


    def __init__(self, pseudoscalar_dim : int, hidden_dim : int, chiral_embedding_dim : int):

        super().__init__()

        self.pseudoscalar_dim = pseudoscalar_dim
        self.hidden_dim = hidden_dim
        self.chiral_embedding_dim = chiral_embedding_dim

        self.model = torch.nn.Sequential(
            torch.nn.Linear(pseudoscalar_dim, hidden_dim, bias = False),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, chiral_embedding_dim, bias = False),
        )

    def forward(self, x):
        return self.model(x)


class ChiralEmbeddingModel(torch.nn.Module):

    def __init__(
        self,
        input_irreps: Irreps,
        pseudoscalar_dimension: Irreps,
        chiral_embedding_dim : int,
        mean_inv_atomic_embedding,
        std_inv_atomic_embedding,
        gated: bool = True,
        equivariant_rms_norm: bool = True,
        dtype = torch.float64
    ):
        super().__init__()


        self.dtype = dtype
        self.gated = gated
        self.equivariant_rms_norm = equivariant_rms_norm

        self.input_irreps = Irreps(input_irreps)
        print(self.input_irreps)
        self.invariant_indices, self.invariant_irreps = get_invariant_indices(
            self.input_irreps
        )
        self.equivariant_irreps = get_equivariant_irreps(input_irreps)
        self.pseudoscalar_irreps = Irreps(f"{pseudoscalar_dimension}x0o")



        if mean_inv_atomic_embedding is None:
            mean_inv_atomic_embedding = torch.zeros(self.invariant_irreps.dim, dtype = dtype)

        if std_inv_atomic_embedding is None:
            std_inv_atomic_embedding = torch.ones(self.invariant_irreps.dim, dtype = dtype)


        self.register_buffer(
            "mean_inv_atomic_embedding",
            mean_inv_atomic_embedding,
            persistent=True)
        self.register_buffer(
            "std_inv_atomic_embedding",
            std_inv_atomic_embedding,
            persistent=True,
        )
        
        self.eq_embedding_irrep= Irreps(f"{pseudoscalar_dimension}x1o")

        self.lin0 = o3.Linear(self.equivariant_irreps,  self.eq_embedding_irrep)
        self.lin1 = o3.Linear(self.equivariant_irreps,  self.eq_embedding_irrep)
        self.lin2 = o3.Linear(self.equivariant_irreps,  self.eq_embedding_irrep)

        #self.eq_norm = NormActivation(self.eq_embedding_irrep, torch.sigmoid )

        self.tp_cross = o3.TensorProduct(
             self.eq_embedding_irrep,
             self.eq_embedding_irrep,
            o3.Irreps(f"{pseudoscalar_dimension}x1e"),
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
        self.ln = torch.nn.LayerNorm(pseudoscalar_dimension, dtype=dtype, bias = False)
        #self.ps_layer_norm1 = torch.nn.LayerNorm(self.pseudoscalar_irreps.dim)
        #self.ps_layer_norm2 = torch.nn.LayerNorm(self.pseudoscalar_irreps.dim)
        #self.ps_layer_norm3 = torch.nn.LayerNorm(self.pseudoscalar_irreps.dim)

        self.chi_gate = ChiGate(inv_dim = self.invariant_irreps.dim, K = self.pseudoscalar_irreps.dim)

        self.linear_out = torch.nn.Linear(pseudoscalar_dimension, chiral_embedding_dim, bias = False)


        #self.mlp_out = OddMLP(
        #    pseudoscalar_dim = pseudoscalar_dimension,
        #    hidden_dim = pseudoscalar_dimension * 2,
        #    chiral_embedding_dim = chiral_embedding_dim
        #)


    def rescale_invariant(self, invariants):

        invariants = (
            invariants - self.mean_inv_atomic_embedding
        ) / self.std_inv_atomic_embedding

        invariants = invariants
        return invariants

    def forward(self, atomic_embeddings):

        invariant_features, equivariant_features = split_invariants_equivariants(
            atomic_embeddings, self.invariant_indices
        )

        invariant_features = self.rescale_invariant(invariant_features)

        if self.equivariant_rms_norm:
            equivariant_features = self.rms_norm(equivariant_features)
        

        x0 = self.lin0(equivariant_features)

        x1 = self.lin1(equivariant_features)
        x2 = self.lin2(equivariant_features)
        
        #x0 = equivariant_features
        #x1 = equivariant_features
        #x2 = equivariant_features
    
        cross = self.tp_cross(x0, x1)  # v₂ x v₃
        out = self.tp_dot(cross, x2)  # v₁ · (v₂ x v₃)


        out = self.ln(out)

        if self.gated:
            out = self.chi_gate(invariant_features) * out

        out = self.linear_out(out)
        #out = self.mlp_out(out)
        out = out.to(torch.float32)
        return out
