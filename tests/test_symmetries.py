from chiral_mols.data.create_dataset import get_ase_atoms
import torch 
from mace.calculators import MACECalculator
from chiral_mols.model.chiral_embedding_model import ChiralEmbeddingModel
from e3nn.o3 import Irreps
from chiral_mols.data.mlip_utils import relax_atoms
from chiral_mols.model.irreps_utils import get_mace_calculator_irrep_signature
from chiral_mols.model.configuration import ChiralEmbeddingConfig
from chiral_mols.model.chiral_embedding_model import ChiralEmbeddingModel
from mace.calculators import MACECalculator
import torch
import pydantic_yaml as pydyaml
from chiral_mols.model.molecular_embedding_model import MolecularEmbeddingModel
from rdkit import Chem
from rdkit.Chem.rdMolTransforms import GetDihedralDeg, SetDihedralDeg
import numpy as np
from rdkit.Chem import AllChem, Draw
from ase import Atoms
from rdkit.Geometry import Point3D
from pathlib import Path
from scipy.spatial.transform import Rotation
from chiral_mols.model.reload_model import reload_model_from_dir


def test_pseudoscalars():
    smiles = "CC(N)O"
    atoms = get_ase_atoms(smiles)


    MACE_PATH = "/share/snw30/projects/mace_model/MACE-OFF24_medium.model"
    mace_calc = MACECalculator(MACE_PATH, device= "cuda", default_dtype= "float64")

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

    chiral_embedding_model_config = ChiralEmbeddingConfig(
    input_irreps="128x0e+128x1o+128x0e",
    pseudoscalar_dimension=64,
    chiral_embedding_dim=24,
    gated=True,
    equivariant_rms_norm=True,
)

    chiral_embedding_model = ChiralEmbeddingModel(
    **chiral_embedding_model_config.model_dump(exclude_none= True),
    mean_inv_atomic_embedding=None,
    std_inv_atomic_embedding=None,
    dtype=torch.float64,
)

    ces1 = chiral_embedding_model(des1)
    ces2 = chiral_embedding_model(des2)

    assert torch.allclose(ces1, -ces2)




def test_rotation_invariance():
    smiles = "CC(N)O"
    atoms = get_ase_atoms(smiles)


    MACE_PATH = "/share/snw30/projects/mace_model/MACE-OFF24_medium.model"
    mace_calc = MACECalculator(MACE_PATH, device= "cuda", default_dtype= "float64")


    # Create a random 3D rotation matrix
    R = Rotation.random().as_matrix()  # shape (3, 3)

    relax_atoms(atoms, mace_calc)

    pos = atoms.get_positions()
    atoms2 = atoms.copy()
    atoms2.set_positions(np.dot(pos, R.T) + np.array([0, 0, 1.2]))

    des1 = mace_calc.get_descriptors(atoms, invariants_only=False)
    des1 = torch.Tensor(des1)
    des2 = mace_calc.get_descriptors(atoms2, invariants_only=False)
    des2 = torch.Tensor(des2)

    chiral_embedding_model_config = ChiralEmbeddingConfig(
    input_irreps="128x0e+128x1o+128x0e",
    pseudoscalar_dimension=64,
    chiral_embedding_dim=24,
    gated=True,
    equivariant_rms_norm=True,
)

    chiral_embedding_model = ChiralEmbeddingModel(
    **chiral_embedding_model_config.model_dump(exclude_none= True),
    mean_inv_atomic_embedding=None,
    std_inv_atomic_embedding=None,
    dtype=torch.float64,
)

    ces1 = chiral_embedding_model(des1)
    ces2 = chiral_embedding_model(des2)
    
    assert torch.allclose(ces1, ces2)

def test_pseudoscalars_with_reload():
    smiles = "CC(N)O"
    atoms = get_ase_atoms(smiles)


    MACE_PATH = "/share/snw30/projects/mace_model/MACE-OFF24_medium.model"
    mace_calc = MACECalculator(MACE_PATH, device= "cuda", default_dtype= "float64")

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

    model_dir = Path("/share/snw30/projects/chiral_mols/training_runs/2-2025_07_23_08_27_50-Nobiases_WithLinearOut_24_dim")
    chiral_embedding_model_config = pydyaml.parse_yaml_file_as(ChiralEmbeddingConfig, file = model_dir / "chiral_embedding_model_config.yaml")


    chiral_embedding_model = ChiralEmbeddingModel(
        **chiral_embedding_model_config.model_dump(exclude="reload_state_dict"),
        mean_inv_atomic_embedding=None,
        std_inv_atomic_embedding=None,
        dtype=torch.float64,
    )
    state_dict = torch.load(chiral_embedding_model_config.reload_state_dict)
    chiral_embedding_model.load_state_dict(state_dict)
    
    
    ces1 = chiral_embedding_model(des1)
    ces2 = chiral_embedding_model(des2)

    assert torch.allclose(ces1, -ces2)

    

def test_rotation_invariance_with_reload():
    smiles = "CC(N)O"
    atoms = get_ase_atoms(smiles)


    MACE_PATH = "/share/snw30/projects/mace_model/MACE-OFF24_medium.model"
    mace_calc = MACECalculator(MACE_PATH, device= "cuda", default_dtype= "float64")


    # Create a random 3D rotation matrix
    R = Rotation.random().as_matrix()  # shape (3, 3)

    relax_atoms(atoms, mace_calc)

    pos = atoms.get_positions()
    atoms2 = atoms.copy()
    atoms2.set_positions(np.dot(pos, R.T) + np.array([0, 0, 1.2]))

    des1 = mace_calc.get_descriptors(atoms, invariants_only=False)
    des1 = torch.Tensor(des1)
    des2 = mace_calc.get_descriptors(atoms2, invariants_only=False)
    des2 = torch.Tensor(des2)
    model_dir = Path("/share/snw30/projects/chiral_mols/training_runs/2-2025_07_23_08_27_50-Nobiases_WithLinearOut_24_dim")
    chiral_embedding_model = reload_model_from_dir(model_dir)

    ces1 = chiral_embedding_model(des1)
    ces2 = chiral_embedding_model(des2)
    
    assert torch.allclose(ces1, ces2)