from rdkit import Chem
from rdkit.Chem import AllChem
import polars as pl
import random 
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers, StereoEnumerationOptions
)
from chiral_mols.mlip_utils import MACE_OFF_ELEMENTS
from typing import Generator

random.seed(0) 

def stream_ligand_smiles(tsv_file: str, batch_size: int = 10):
    """
    Generator yielding 'Ligand SMILES' values from a TSV file in streaming batches.
    """
    # 1) Build a lazy plan that only reads the one column
    lazy_smiles = pl.scan_csv(tsv_file, separator="\t", has_header=True).select(
        "Ligand SMILES"
    )

    # 2) Execute in the streaming engine (memory‐bounded)
    df = lazy_smiles.collect(engine="streaming")

    # 3) Slice into batches and yield one SMILES at a time
    for batch_df in df.iter_slices(n_rows=batch_size):
        for smi in batch_df["Ligand SMILES"]:
            yield smi


def stream_random_ligand_smiles_batches(
    tsv_file: str,
    read_batch_size: int = 100,
    pool_size: int = 10_000,
    yield_batch_size: int = 100
) -> Generator[list[str], None, None]:
    """
    Approximate random-order streaming in *batches*:
      - Maintains a rolling pool of up to `pool_size` valid SMILES.
      - Once full, shuffle it and drain in chunks of `yield_batch_size`.
      - At EOF, shuffle & drain any remainder in batches.
    """
    pool: list[str] = []
    for smi in stream_ligand_smiles(tsv_file, read_batch_size):

        pool.append(smi)
        if len(pool) >= pool_size:
            random.shuffle(pool)
            # drain the pool in batches

            while pool:
                batch = [pool.pop() for _ in range(min(yield_batch_size, len(pool)))]
                yield batch

    # final flush
    if pool:
        random.shuffle(pool)
        while pool:
            batch = [pool.pop() for _ in range(min(yield_batch_size, len(pool)))]
            yield batch




def get_chiral_smiles(smiles : list[str]):

    smiles_collection = []

    for smi in smiles:

        mol = Chem.MolFromSmiles(smi)

        if mol is None or mol.GetNumAtoms() < 3:
            continue
        if any(a.GetSymbol() not in MACE_OFF_ELEMENTS for a in mol.GetAtoms()):
            continue
        try:
            mol2 = Chem.AddHs(mol)
        except:
            continue
        if mol2.GetNumAtoms() > 47:
            continue
        if any(
            a.GetNumRadicalElectrons() != 0
            or a.GetIsotope() != 0
            or a.GetFormalCharge() != 0
            for a in mol2.GetAtoms()
        ):
            continue

        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        if len(chiral_centers) == 1:
            smiles_collection.append(Chem.MolToSmiles(mol))


    return smiles_collection




def add_enantiomer_pair(smiles):
    opts = StereoEnumerationOptions(unique=True, onlyUnassigned=False)

    updated_smiles = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        updated_smiles.extend([Chem.MolToSmiles(m) for m in EnumerateStereoisomers(mol, options=opts)]
)

    return updated_smiles


