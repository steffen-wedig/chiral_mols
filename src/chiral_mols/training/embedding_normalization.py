from chiral_mols.model.irreps_utils import get_invariant_indices
from e3nn.o3 import Irreps
from torch import Tensor
from jaxtyping import Float




def get_mean_std_invariant_indices(embeddings: Float[Tensor, "batch embedding_dim"],input_irreps : Irreps):

    invariant_indices, _ = get_invariant_indices(input_irreps)

    invariants = embeddings[:, invariant_indices]


    mean_inv = invariants.mean(dim = 0)
    std_inv = invariants.std(dim = 0)

    return mean_inv, std_inv