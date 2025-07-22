from mace.calculators import MACECalculator
from ase import Atoms
from typing import List
from tqdm import tqdm
from ase.optimize import LBFGS

MACE_OFF_ELEMENTS = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}




def get_mace_embeddings(atoms : List[Atoms], mace_calc : MACECalculator):

    mace_embeddings = []

    for m in tqdm(atoms, desc= "Loading Mace Embeddings"):

        mace_embeddings.append(mace_calc.get_descriptors(m, invariants_only=False))
    return mace_embeddings



def relax_atoms(atoms: Atoms, calculator: MACECalculator, BFGS_tol=0.05, max_steps=100):
    atoms.calc = calculator
    dyn = LBFGS(atoms, logfile=None)
    converged = dyn.run(fmax=BFGS_tol, steps=max_steps)
    if not converged:
        raise ValueError("LBFGS did not converge")
