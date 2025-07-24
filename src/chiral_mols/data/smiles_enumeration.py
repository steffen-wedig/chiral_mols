from rdkit import Chem
from rdkit.Chem import AllChem
import polars as pl
import random 
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers, StereoEnumerationOptions
)
from chiral_mols.data.mlip_utils import MACE_OFF_ELEMENTS
from typing import Generator
from rdkit.Chem import rdmolops, rdchem

random.seed(0) 




def stream_ligand_smiles(tsv_file: str, batch_size: int = 10):
    """
    Generator yielding 'Ligand SMILES' values from a TSV file in streaming batches.
    """
    # 1) Build a lazy plan (scan instead of read) that only reads the smiles column
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


def has_any_ez_stereo(mol):
    # ensure RDKit has marked explicit & potential stereo
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    rdmolops.FindPotentialStereoBonds(mol, cleanIt=True)
    for b in mol.GetBonds():
        if b.GetBondType() == rdchem.BondType.DOUBLE and \
           b.GetStereo() in (
               rdchem.BondStereo.STEREOANY,
               rdchem.BondStereo.STEREOE,
               rdchem.BondStereo.STEREOZ
           ):
            return True
    return False


def get_chiral_smiles(smiles : list[str]):

    out = []

    for smi in smiles:
        try:

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


            if len(Chem.GetMolFrags(mol, asMols=True)) > 1:
                continue

            if any(
                a.GetNumRadicalElectrons() != 0
                or a.GetIsotope() != 0
                or a.GetFormalCharge() != 0
                for a in mol2.GetAtoms()
            ):
                continue
            
            # eject any E/Z stereo (potential or explicit)
            if has_any_ez_stereo(mol):
                continue
            
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
            centers = Chem.FindMolChiralCenters(
                mol,
                includeUnassigned=True,
                useLegacyImplementation=False
            )
            if len(centers) == 1:
                # produce a canonical isomeric SMILES
                out.append(Chem.MolToSmiles(mol, isomericSmiles=True))
        except Exception as e:
            print(f"Filtering Error for smiles {smi}: {e}")
            continue

    return out

def filter_smiles(smiles_list: list[str]) -> list[str]:
    return sorted(set(smiles_list))

def add_enantiomer_pair(smiles: list[str]) -> list[str]:
    opts = StereoEnumerationOptions(unique=True, onlyUnassigned=False)
    out = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        for stereo_mol in EnumerateStereoisomers(mol, options=opts):
            smi = Chem.MolToSmiles(stereo_mol, isomericSmiles=True)
            out.append(smi)
    return out


