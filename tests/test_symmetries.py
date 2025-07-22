from chiral_mols.data.create_dataset import get_ase_atoms
import torch 
from mace.calculators import MACECalculator
from chiral_mols.model.chiral_embedding_model import ChiralEmbeddingModel
from e3nn.o3 import Irreps
from chiral_mols.data.mlip_utils import relax_atoms
from chiral_mols.model.irreps_utils import get_mace_calculator_irrep_signature


def test_pseudoscalars():
    smiles = "CC(N)O"
    atoms = get_ase_atoms(smiles)


    MACE_PATH = "/share/snw30/projects/mace_model/MACE-OFF24_medium.model"
    mace_calc = MACECalculator(MACE_PATH, device= "cuda", default_dtype= "float32")

    print(get_mace_calculator_irrep_signature(mace_calc))

    relax_atoms(atoms, mace_calc)

    pos = atoms.get_positions()
    atoms2 = atoms.copy()
    pos2 = pos * -1.0
    atoms2.set_positions(pos2)

    des1 = mace_calc.get_descriptors(atoms, invariants_only=False)
    des1 = torch.Tensor(des1)
    des2 = mace_calc.get_descriptors(atoms2, invariants_only=False)
    des2 = torch.Tensor(des2)

    chiral_embedding_model = ChiralEmbeddingModel(
    input_irreps=Irreps("128x0e+128x1o+128x0e"),
    pseudoscalar_irreps=Irreps("128x0o"),
    output_embedding_dim=128,
    mean_inv_atomic_embedding=None,
    std_inv_atomic_embedding=None,
    low_dim_equivariant=128,
    dtype = torch.float32
    )

    ces1 = chiral_embedding_model(des1)
    ces2 = chiral_embedding_model(des2)
    print(ces1)
    print(ces2)

    print((ces1 + ces2).max())

    assert torch.allclose(ces1, -ces2)

    