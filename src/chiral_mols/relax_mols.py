import time
from rdkit import Chem
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from typing import List

from rdkit.Chem import AllChem


def _process_one_smiles(args):
    """Embed + MMFF-optimize one SMILES."""
    smiles, N_conformers = args
    warnings = []

    # prepare your ETKDG3 parameters
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.enforceChirality = True
    params.useRandomCoords = False

    # 0) Parse & add Hs
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        warnings.append(f"SMILES '{smiles}' could not be parsed; skipped.")
        return None , warnings
    
    mol = Chem.AddHs(mol)

    try:
        # 3) Embed multiple conformers
        conf_ids = AllChem.EmbedMultipleConfs(mol, N_conformers, params)

        if not conf_ids:
            raise ValueError(f"Embedding failed for SMILES '{smiles}'")

        # 4) MMFF optimize & get convergence flags
        results = AllChem.MMFFOptimizeMoleculeConfs(
            mol, maxIters=500, nonBondedThresh=500.0, numThreads=1
        )

        for i, (flag, E) in enumerate(results):
            if flag != 0:
                warnings.append(
                    f"SMILES '{smiles}', conformer {i}: MMFF did not converge."
                )

        return mol, warnings

    except Exception as e:
        warnings.append(f"SMILES '{smiles}': unexpected error {e!r}")
        return None, warnings


def convert_smiles_to_mols(
    smiles_list,
    N_conformers: int = 1,
    n_workers: int | None = None,
    verbose: bool = True,
)-> List[Chem.Mol]:
    """Parallel version that filters out unconverged conformers."""
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    jobs = [(sm, N_conformers) for sm in smiles_list]
    all_mols = []

    t0 = time.time()
    with Pool(processes=n_workers) as pool:
        for mol, warnings in tqdm(
            pool.imap_unordered(_process_one_smiles, jobs),
            total=len(jobs),
            desc="SMILES → ASE",
            disable=not verbose,
        ):
            if verbose:
                for w in warnings:
                    tqdm.write(f"Warning: {w}")
            if mol is not None:
                all_mols.append(mol)

    dt = time.time() - t0
    if verbose:
        print(
            f"Processed {len(smiles_list)} SMILES → {len(all_mols)} converged conformers in {dt:.1f}s"
        )
    return all_mols
