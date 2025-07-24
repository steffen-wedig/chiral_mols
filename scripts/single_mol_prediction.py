import warnings
from ase.data import chemical_symbols
from ase import Atoms
from rdkit.Chem import AllChem

warnings.filterwarnings(
    "ignore",
    message=r"The TorchScript type system doesn't support instance-level annotations.*",
    category=UserWarning,
)
from chiral_mols.model.molecular_embedding_model import MolecularEmbeddingModel
import torch
from chiral_mols.data.ptr_dataset import PtrMoleculeDataset
from rdkit import Chem
from torch.utils.data import DataLoader
from chiral_mols.data.sample import concat_collate

from pathlib import Path
from e3nn.o3 import Irreps
from chiral_mols.model.chiral_embedding_model import ChiralEmbeddingModel
from chiral_mols.model.classifier import ChiralityClassifier

from chiral_mols.training.evaluation import build_metric_collection
from chiral_mols.training.training import evaluate
from chiral_mols.training.loss import FocalLoss
from chiral_mols.model.configuration import (
    ChiralEmbeddingConfig,
    ChiralityClassifierConfig,
)
import pydantic_yaml as pydyaml
from chiral_mols.evaluation.evaluation import UMAPPlotter, compute_channel_stats
import yaml

import matplotlib.pyplot as plt
from chiral_mols.evaluation.evaluation import get_misclassified_structure_ids
from mace.calculators import MACECalculator
from ase.visualize.plot import plot_atoms
from matplotlib import pyplot as plt

device = "cuda"
model_dir = Path(
    "/share/snw30/projects/chiral_mols/training_runs/13-2025_07_23_15_08_22-OutLinear"
)


classifier_config = pydyaml.parse_yaml_file_as(
    ChiralityClassifierConfig, file=model_dir / "classifier_config.yaml"
)
chiral_embedding_model_config = pydyaml.parse_yaml_file_as(
    ChiralEmbeddingConfig, file=model_dir / "chiral_embedding_model_config.yaml"
)


chiral_embedding_model = ChiralEmbeddingModel(
    **chiral_embedding_model_config.model_dump(exclude="reload_state_dict"),
    mean_inv_atomic_embedding=None,
    std_inv_atomic_embedding=None,
    dtype=torch.float32,
)

classifier = ChiralityClassifier(
    **classifier_config.model_dump(exclude="reload_state_dict")
)
classifier.load_state_dict(torch.load(classifier_config.reload_state_dict))


state_dict = torch.load(chiral_embedding_model_config.reload_state_dict)
chiral_embedding_model.load_state_dict(state_dict)


MACE_PATH = "/share/snw30/projects/mace_model/MACE-OFF24_medium.model"
mace_calc = MACECalculator(MACE_PATH, default_dtype="float32")

smiles = "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"

mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol)

atom_symbols = [a.GetSymbol() for a in mol.GetAtoms()]
atoms = Atoms(atom_symbols, positions=mol.GetConformer().GetPositions(), pbc=False)


model = MolecularEmbeddingModel(
    mace_calc=mace_calc, chiral_embeding_model=chiral_embedding_model
)


chiral_embeddings = model(atoms)

chirality_logits = classifier(chiral_embeddings)


predictions = chirality_logits.argmax(dim=1)


print(predictions)


dataset_dir = Path("/share/snw30/projects/chiral_mols/dataset/chiral_atoms_120k")
N_classes = 3

print("Reloading Dataset")
dataset = PtrMoleculeDataset.reload_dataset_from_dir(dataset_dir, reload_atoms=True)
atoms = dataset.get_atoms_from_structure_id(1000)
fig, ax = plt.subplots()
plot_atoms(atoms, ax, rotation=('0x,0y,0z'))
plt.savefig('molecule.png')
plt.close()


print( (dataset.labels != 0) & (dataset.atomic_numbers == 1))

print( any((dataset.labels != 0) & (dataset.atomic_numbers == 1)))

