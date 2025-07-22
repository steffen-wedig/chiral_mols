import torch
from e3nn.o3 import Irreps
from mace.calculators import MACECalculator


def get_mace_calculator_irrep_signature(mace_calculator: MACECalculator) -> Irreps:
    signature = None

    for products in mace_calculator.models[0].products: # type: ignore
        if signature is None:
            signature = Irreps(str(products.linear.__dict__["irreps_out"]))
        else:
            signature = signature + Irreps(str(products.linear.__dict__["irreps_out"]))

    return signature

def slices_to_index_list(slices, sequence_length):
    # Using a list comprehension to flatten the list of indices
    return [i for s in slices for i in range(*s.indices(sequence_length))]


def get_invariant_indices(irreps: Irreps):
    """
    Gets the slices for all irreps indices, and only returns those with degree 0
    """
    total_dim = irreps.dim
    slices = irreps.slices()
    invariant_slices = []
    out_irrep = []
    for idx, irrep_slice in enumerate(irreps):
        if irrep_slice.ir[0] == 0:
            out_irrep.append(irrep_slice)
            invariant_slices.append(slices[idx])

    index_list = slices_to_index_list(
        slices=invariant_slices, sequence_length=total_dim
    )
    return index_list, Irreps(out_irrep)



def get_equivariant_irreps(irreps: Irreps):

    irreps = Irreps(irreps)                          # normalise input
    filtered = [(mul, ir) for mul, ir in irreps       # keep l>0
                if ir.l > 0]
    return Irreps(filtered)



def split_invariants_equivariants(emb, invariant_indices):
    # emb: [B*N, C]
    C = emb.shape[1]    # ensure tensor of long indices on the correct device
    if not torch.is_tensor(invariant_indices):
        invariant_indices = torch.tensor(list(invariant_indices), dtype=torch.long, device=emb.device)
    # build mask
    mask = torch.zeros(C, dtype=torch.bool, device=emb.device)
    mask[invariant_indices] = True

    invariants   = emb[:, mask]     # picks out the True positions
    equivariants = emb[:, ~mask]     # picks out the False positions
    return invariants, equivariants
