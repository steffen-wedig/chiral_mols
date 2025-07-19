from mace.calculators import MACECalculator
from ase import Atoms
from typing import List
from tqdm import tqdm

MACE_OFF_ELEMENTS = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}




def get_mace_embeddings(atoms : List[Atoms], mace_calc : MACECalculator):

    mace_embeddings = []

    for m in tqdm(atoms, desc= "Loading Mace Embeddings"):

        mace_embeddings.append(mace_calc.get_descriptors(m, invariants_only=False))
    return mace_embeddings